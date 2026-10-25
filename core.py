"""RippleChain core: pure python + ffmpeg (no ComfyUI imports) so it can be tested alone.

Plan: decode the source at a fixed fps, find hard cuts (gray histogram distance), and inside every scene start
a chunk at the cut and again every `chunk` frames. Chunk k is rendered with edited still k.

Bundle / chain files are one mp4 each (stream 0 = video, extra video streams = attached pictures = the edited
stills, comment tag = JSON state), same trick as VoiceChain.
"""
import json
import math
import os
import subprocess

import numpy as np

TAG = "ripplechain/1"
MAX_CHUNKS = 256


def _run(cmd, **kw):
    return subprocess.run(cmd, capture_output=True, **kw)


def probe(path):
    r = _run(["ffprobe", "-v", "error", "-show_streams", "-show_format", "-of", "json", path])
    if r.returncode:
        raise RuntimeError(f"ffprobe failed on {os.path.basename(path)}: {r.stderr[-300:]!r}")
    return json.loads(r.stdout)


def _rate(s):
    try:
        a, b = str(s).split("/")
        return float(a) / float(b) if float(b) else 0.0
    except Exception:
        return 0.0


def source_info(path):
    """-> (w, h, fps, has_audio) of the first real video stream."""
    info = probe(path)
    w = h = 0
    fps = 0.0
    has_audio = any(s["codec_type"] == "audio" for s in info["streams"])
    for s in info["streams"]:
        if s["codec_type"] == "video" and not s.get("disposition", {}).get("attached_pic"):
            w, h = int(s["width"]), int(s["height"])
            rot = (s.get("tags") or {}).get("rotate")
            for sd in s.get("side_data_list", []) or []:
                if "rotation" in sd:
                    rot = sd["rotation"]
            if rot is not None and abs(int(float(rot))) % 180 == 90:
                w, h = h, w
            fps = _rate(s.get("avg_frame_rate")) or _rate(s.get("r_frame_rate"))
            break
    if w <= 0:
        raise RuntimeError("no video stream in the source")
    return w, h, fps, has_audio


def pick_fps(src_fps, fps_in):
    f = float(fps_in) if fps_in and fps_in > 0 else (src_fps or 24.0)
    return round(min(30.0, max(8.0, f)), 3)


def pick_size(src_w, src_h, width, height, target_pixels=393216, lo=256, hi=1280):
    def snap(v, a, b):
        return int(min(b, max(a, round(v / 32.0) * 32)))
    if width <= 0 and height <= 0:
        s = math.sqrt(target_pixels / float(src_w * src_h))
        return snap(src_w * s, lo, hi), snap(src_h * s, lo, hi)
    if width <= 0:
        width = height * src_w / float(src_h)
    elif height <= 0:
        height = width * src_h / float(src_w)
    return snap(width, 64, 16384), snap(height, 64, 16384)


def decode(src, fps, w, h, start, count):
    """Frames [start, start+count) of the source resampled to fps, cover-scaled + centre-cropped -> uint8 [n,h,w,3]."""
    vf = (f"fps={fps},select='between(n\\,{start}\\,{start + count - 1})',"
          f"scale={w}:{h}:force_original_aspect_ratio=increase:flags=lanczos,crop={w}:{h}")
    r = _run(["ffmpeg", "-v", "error", "-i", src, "-vf", vf, "-fps_mode", "passthrough", "-frames:v", str(count),
              "-pix_fmt", "rgb24", "-f", "rawvideo", "-"])
    if r.returncode:
        raise RuntimeError(f"decode failed: {r.stderr[-300:]!r}")
    a = np.frombuffer(r.stdout, np.uint8)
    n = a.size // (w * h * 3)
    if n == 0:
        raise RuntimeError("no frames decoded (start beyond the end of the video?)")
    return a[: n * w * h * 3].reshape(n, h, w, 3)


def find_cuts(src, fps, thr=0.35, min_scene=8):
    """-> (total_frames, [frame index where a new scene starts]) using a 32-bin gray histogram distance."""
    r = _run(["ffmpeg", "-v", "error", "-i", src, "-vf", f"fps={fps},scale=160:90", "-pix_fmt", "gray",
              "-fps_mode", "passthrough", "-f", "rawvideo", "-"])
    if r.returncode:
        raise RuntimeError(f"scan failed: {r.stderr[-300:]!r}")
    a = np.frombuffer(r.stdout, np.uint8)
    n = a.size // (160 * 90)
    if n == 0:
        raise RuntimeError("no frames in the source")
    a = a[: n * 160 * 90].reshape(n, 160 * 90)
    cuts, last = [], 0
    prev = None
    for i in range(n):
        h = np.bincount(a[i] >> 3, minlength=32) / float(160 * 90)
        if prev is not None and 0.5 * np.abs(h - prev).sum() > thr and i - last >= min_scene:
            cuts.append(i)
            last = i
        prev = h
    return n, cuts


def make_plan(total, cuts, chunk):
    """-> [[start, count, scene], ...]: a chunk at every cut and every `chunk` frames inside a scene."""
    bounds = [0] + list(cuts) + [total]
    plan = []
    for sc in range(len(bounds) - 1):
        s, e = bounds[sc], bounds[sc + 1]
        for st in range(s, e, chunk):
            plan.append([st, min(chunk, e - st), sc])
    if len(plan) > MAX_CHUNKS:
        raise RuntimeError(f"{len(plan)} chunks needed (max {MAX_CHUNKS}): raise the chunk length or the cut threshold")
    return plan


def grid_len(count):
    """Render length N (8k+1) whose N-1 source frames cover `count` frames."""
    return int(math.ceil(count / 8.0)) * 8 + 1


def read_state(path):
    info = probe(path)
    tags = {k.lower(): v for k, v in info.get("format", {}).get("tags", {}).items()}
    raw = tags.get("comment", "")
    st = json.loads(raw) if raw.startswith("{") else {}
    if st.get("tag") != TAG:
        raise RuntimeError(f"{os.path.basename(path)} is not a ripple-chain bundle/chain video (no state tag)")
    return st


def extract_covers(path, out_dir):
    info = probe(path)
    out = []
    for s in info["streams"]:
        if s["codec_type"] == "video" and s.get("disposition", {}).get("attached_pic"):
            p = os.path.join(out_dir, f"cover{len(out):03d}.png")
            r = _run(["ffmpeg", "-y", "-v", "error", "-i", path, "-map", f"0:{s['index']}", "-frames:v", "1", p])
            if r.returncode:
                raise RuntimeError(f"cover extract failed: {r.stderr[-300:]!r}")
            out.append(p)
    return out


def write_clip(frames_rgb, fps, out_path, size=None):
    h, w = frames_rgb.shape[1:3]
    cmd = ["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-r", str(fps), "-i", "-"]
    if size and (w, h) != tuple(size):
        cmd += ["-vf", f"scale={size[0]}:{size[1]}:flags=lanczos"]
    cmd += ["-an", "-c:v", "libx264", "-preset", "veryfast", "-crf", "17", "-pix_fmt", "yuv420p", "-r", str(fps),
            "-video_track_timescale", "30000", "-movflags", "+faststart", out_path]
    r = subprocess.run(cmd, input=np.ascontiguousarray(frames_rgb).tobytes(), capture_output=True)
    if r.returncode:
        raise RuntimeError(f"clip encode failed: {r.stderr[-400:]!r}")
    return out_path


def package(work, video, covers, state, out_path, audio_src=None, audio_seconds=None):
    """Write the final mp4: `video` main stream (copied), optional audio from audio_src, covers as attached pics."""
    cmd = ["ffmpeg", "-y", "-v", "error", "-i", video]
    idx = 1
    if audio_src:
        cmd += ["-i", audio_src]
        idx = 2
    for p in covers:
        cmd += ["-i", p]
    cmd += ["-map", "0:v:0"]
    if audio_src:
        cmd += ["-map", "1:a:0"]
    for i in range(len(covers)):
        cmd += ["-map", f"{idx + i}:v:0"]
    cmd += ["-c:v:0", "copy"]
    if audio_src:
        cmd += ["-c:a", "aac", "-b:a", "192k"]
    for i in range(len(covers)):
        cmd += [f"-c:v:{1 + i}", "png", f"-disposition:v:{1 + i}", "attached_pic"]
    cmd += ["-metadata", "comment=" + json.dumps(state, separators=(",", ":")), "-movflags", "+faststart"]
    if audio_src:
        cmd += ["-t", f"{audio_seconds:.4f}"]
    cmd += [out_path]
    r = _run(cmd)
    if r.returncode:
        raise RuntimeError(f"mux failed: {r.stderr[-400:]!r}")
    return out_path


def stitch(work, new_clip, prev_video):
    parts = []
    if prev_video:
        prev_v = os.path.join(work, "prev_video.mp4")
        r = _run(["ffmpeg", "-y", "-v", "error", "-i", prev_video, "-map", "0:v:0", "-c", "copy", prev_v])
        if r.returncode:
            raise RuntimeError(f"prev video split failed: {r.stderr[-300:]!r}")
        parts.append(prev_v)
    parts.append(new_clip)
    joined = os.path.join(work, "joined.mp4")
    lst = os.path.join(work, "concat.txt")
    with open(lst, "w", encoding="utf-8") as f:
        for p in parts:
            f.write("file '" + p.replace("\\", "/").replace("'", r"'\''") + "'\n")
    r = _run(["ffmpeg", "-y", "-v", "error", "-f", "concat", "-safe", "0", "-i", lst, "-c", "copy", joined])
    if r.returncode:
        raise RuntimeError(f"concat failed: {r.stderr[-300:]!r}")
    return joined


def black_clip(work, w=64, h=64):
    p = os.path.join(work, "black.mp4")
    write_clip(np.zeros((2, h, w, 3), np.uint8), 24, p)
    return p
