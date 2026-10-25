import asyncio
import json
import os
import shutil
import tempfile
import time

import numpy as np
from PIL import Image

from . import core


def _input_dir():
    try:
        import folder_paths
        return folder_paths.get_input_directory()
    except Exception:
        return ""


def _local(name, work, allow_url=True):
    """Graydient stages media as a local filename (or full path); a http(s) url is downloaded."""
    name = (name or "").strip()
    if not name:
        return ""
    if allow_url and name.startswith(("http://", "https://")):
        import requests
        p = os.path.join(work, f"dl_{abs(hash(name))}.mp4")
        with requests.get(name, timeout=120, stream=True) as r:
            r.raise_for_status()
            with open(p, "wb") as fh:
                for c in r.iter_content(65536):
                    fh.write(c)
        return p
    for cand in (name, os.path.join(_input_dir(), name)):
        if os.path.isfile(cand):
            return cand
    raise FileNotFoundError(f"media file not found: {name}")


def _tensor(arr):
    import torch
    return torch.from_numpy(arr.astype(np.float32) / 255.0)


def _fit_png(path, w, h):
    im = Image.open(path).convert("RGB")
    sc = max(w / im.width, h / im.height)
    im = im.resize((max(w, round(im.width * sc)), max(h, round(im.height * sc))), Image.LANCZOS)
    l, t = (im.width - w) // 2, (im.height - h) // 2
    return np.asarray(im.crop((l, t, l + w, t + h)))


def _out_path(prefix, step):
    import folder_paths
    out_dir = folder_paths.get_output_directory()
    sub, pre = os.path.split(prefix)
    os.makedirs(os.path.join(out_dir, sub), exist_ok=True)
    name = f"{pre}_{int(time.time())}_{step:03d}.mp4"
    return os.path.join(out_dir, sub, name), name, sub


def _analyse(src, width, height, fps_in, chunk, thr):
    sw, sh, sfps, has_audio = core.source_info(src)
    fps = core.pick_fps(sfps, fps_in)
    w, h = core.pick_size(sw, sh, int(width), int(height))
    total, cuts = core.find_cuts(src, fps, thr)
    plan = core.make_plan(total, cuts, max(8, int(chunk)))
    return fps, w, h, plan, has_audio


class RippleChainPrep:
    """Scan a video: first frame of every scene and of every `chunk` frames inside a scene -> image batch."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "url": ("STRING", {"default": ""}),
            "width": ("INT", {"default": 0, "min": 0, "max": 16384}),
            "height": ("INT", {"default": 0, "min": 0, "max": 16384}),
            "fps": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1000.0}),
            "chunk": ("INT", {"default": 160, "min": 8, "max": 100000}),
            "cut_threshold": ("FLOAT", {"default": 0.35, "min": 0.05, "max": 1.0}),
        }, "optional": {"filename": ("STRING", {"default": ""})}}

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("stills",)
    FUNCTION = "run"
    CATEGORY = "RippleChain"

    async def run(self, url, width, height, fps, chunk, cut_threshold, filename=""):
        return await asyncio.to_thread(self._run, url, width, height, fps, chunk, cut_threshold, filename)

    def _run(self, url, width, height, fps, chunk, cut_threshold, filename):
        work = tempfile.mkdtemp(prefix="ripplechain_")
        try:
            src = _local(url or filename, work)
            if not src:
                raise RuntimeError("reply to a video to scan")
            f, w, h, plan, _ = _analyse(src, width, height, fps, chunk, cut_threshold)
            stills = [core.decode(src, f, w, h, s, 1)[0] for s, _, _ in plan]
            scenes = sorted({p[2] for p in plan})
            print(f"[RippleChainPrep] {len(plan)} stills, {len(scenes)} scenes, {w}x{h} @ {f} fps, chunk {chunk}: "
                  + ", ".join(f"{s}+{c}(scene {sc})" for s, c, sc in plan))
            return (_tensor(np.stack(stills)),)
        finally:
            shutil.rmtree(work, ignore_errors=True)


class RippleBundle:
    """Pack up to 8 images (appended after the images of an earlier bundle) into one self-contained mp4."""

    @classmethod
    def INPUT_TYPES(cls):
        req = {"prev_bundle": ("STRING", {"default": ""})}
        for i in range(1, 9):
            req[f"image{i}"] = ("STRING", {"default": ""})
        req["filename_prefix"] = ("STRING", {"default": "video/RippleBundle"})
        return {"required": req}

    RETURN_TYPES = ()
    FUNCTION = "run"
    OUTPUT_NODE = True
    CATEGORY = "RippleChain"

    async def run(self, prev_bundle, image1, image2, image3, image4, image5, image6, image7, image8, filename_prefix):
        return await asyncio.to_thread(self._run, prev_bundle,
                                       [image1, image2, image3, image4, image5, image6, image7, image8], filename_prefix)

    def _run(self, prev_bundle, images, filename_prefix):
        work = tempfile.mkdtemp(prefix="ripplebundle_")
        prev = _local(prev_bundle, work, allow_url=False)
        covers = []
        if prev:
            st = core.read_state(prev)
            if st.get("kind") != "bundle":
                raise RuntimeError("video1 must be an earlier ripple bundle (or leave it empty to start a new one)")
            covers = core.extract_covers(prev, work)
        for n in images:
            p = _local(n, work, allow_url=False)
            if not p:
                continue
            im = Image.open(p).convert("RGB")
            im.thumbnail((1280, 1280))
            q = os.path.join(work, f"new{len(covers):03d}.png")
            im.save(q)
            covers.append(q)
        if not covers:
            raise RuntimeError("nothing to bundle: attach image1..image8")
        if len(covers) > core.MAX_CHUNKS:
            raise RuntimeError(f"a bundle holds at most {core.MAX_CHUNKS} images")
        state = {"tag": core.TAG, "kind": "bundle", "count": len(covers)}
        out, name, sub = _out_path(filename_prefix, len(covers))
        core.package(work, core.black_clip(work), covers, state, out)
        print(f"[RippleBundle] bundle of {len(covers)} images -> {name}")
        shutil.rmtree(work, ignore_errors=True)
        return {"ui": {"images": [{"filename": name, "subfolder": sub, "type": "output"}], "animated": (True,)}}


class RippleChainLoad:
    """Start (prev = bundle) or continue (prev = previous chain result) an edit chain on the source video.

    Outputs chunk k's N-1 source frames (padded with the last frame), the edited still k, and the render length N.
    Same first five outputs as LoadVideoFromURLMatched so the Ripple graph plugs in unchanged."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "url": ("STRING", {"default": ""}),
            "prev": ("STRING", {"default": ""}),
            "width": ("INT", {"default": 0, "min": 0, "max": 16384}),
            "height": ("INT", {"default": 0, "min": 0, "max": 16384}),
            "length": ("INT", {"default": 0, "min": 0, "max": 100000}),
            "fps": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1000.0}),
            "cut_threshold": ("FLOAT", {"default": 0.35, "min": 0.05, "max": 1.0}),
        }, "optional": {"filename": ("STRING", {"default": ""}), "bundle_image": ("STRING", {"default": ""})}}

    RETURN_TYPES = ("IMAGE", "INT", "INT", "INT", "FLOAT", "IMAGE", "STRING")
    RETURN_NAMES = ("frames", "width", "height", "length", "fps", "edited_frame", "ctx")
    FUNCTION = "run"
    CATEGORY = "RippleChain"

    async def run(self, url, prev, width, height, length, fps, cut_threshold, filename="", bundle_image=""):
        return await asyncio.to_thread(self._run, url, prev, width, height, length, fps, cut_threshold, filename,
                                       bundle_image)

    def _run(self, url, prev, width, height, length, fps, cut_threshold, filename, bundle_image):
        work = tempfile.mkdtemp(prefix="ripplechain_")
        src = _local(url or filename, work)
        if not src:
            raise RuntimeError("reply to the source video every step")
        prev_path = _local(prev or bundle_image, work, allow_url=False)
        if not prev_path:
            raise RuntimeError("video1 is required: the bundle (first step) or the previous chain result (later steps)")
        st = core.read_state(prev_path)
        covers = core.extract_covers(prev_path, work)
        if st["kind"] == "bundle":
            f, w, h, plan, has_audio = _analyse(src, width, height, fps, length or 160, cut_threshold)
            idx, step, prev_video = 0, 1, None
            params = {"fps": f, "w": w, "h": h, "plan": plan, "has_audio": has_audio}
            if len(covers) < len(plan):
                raise RuntimeError(f"this video needs {len(plan)} edited frames (one per chunk), the bundle has "
                                   f"{len(covers)}: run ripple-chain-prep again and bundle every still")
        else:
            if st.get("done"):
                raise RuntimeError("this chain is already complete: nothing left to render")
            params = st["params"]
            idx, step, prev_video = int(st["next"]), int(st["step"]) + 1, prev_path
        start, count, scene = params["plan"][idx]
        n = core.grid_len(count)
        w, h, f = params["w"], params["h"], params["fps"]
        fr = core.decode(src, f, w, h, start, count)
        if fr.shape[0] < n - 1:
            fr = np.concatenate([fr, np.repeat(fr[-1:], n - 1 - fr.shape[0], axis=0)])
        edited = _fit_png(covers[idx], w, h)
        ctx = {"work": work, "src": src, "prev": prev_video, "covers": covers, "step": step, "idx": idx,
               "count": count, "params": params}
        print(f"[RippleChainLoad] step {step}: chunk {idx + 1}/{len(params['plan'])} frames {start}+{count} "
              f"scene {scene}, render N={n}, {w}x{h} @ {f}")
        return (_tensor(fr[: n - 1]), int(w), int(h), int(n), float(f), _tensor(edited[None]), json.dumps(ctx))


class RippleChainFinish:
    """Stitch the decoded clip after the previous chain video; on the last chunk mux the source audio."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "images": ("IMAGE",), "ctx": ("STRING", {"forceInput": True}),
            "filename_prefix": ("STRING", {"default": "video/RippleChain"}),
        }}

    RETURN_TYPES = ()
    FUNCTION = "run"
    OUTPUT_NODE = True
    CATEGORY = "RippleChain"

    async def run(self, images, ctx, filename_prefix):
        return await asyncio.to_thread(self._run, images, ctx, filename_prefix)

    def _run(self, images, ctx, filename_prefix):
        c = json.loads(ctx)
        p = c["params"]
        work = c["work"]
        frames = (images.cpu().numpy().clip(0, 1) * 255).round().astype(np.uint8)[: c["count"]]
        clip = core.write_clip(frames, p["fps"], os.path.join(work, "newclip.mp4"), size=(p["w"], p["h"]))
        joined = core.stitch(work, clip, c["prev"])
        nxt = c["idx"] + 1
        done = nxt >= len(p["plan"])
        state = {"tag": core.TAG, "kind": "chain", "step": c["step"], "next": nxt, "done": done, "params": p}
        out, name, sub = _out_path(filename_prefix, c["step"])
        audio = c["src"] if (done and p.get("has_audio")) else None
        secs = sum(x[1] for x in p["plan"]) / p["fps"]
        core.package(work, joined, c["covers"], state, out, audio, secs)
        print(f"[RippleChainFinish] step {c['step']} -> {name} done={done}")
        return {"ui": {"images": [{"filename": name, "subfolder": sub, "type": "output"}], "animated": (True,)}}


NODE_CLASS_MAPPINGS = {"RippleChainPrep": RippleChainPrep, "RippleBundle": RippleBundle,
                       "RippleChainLoad": RippleChainLoad, "RippleChainFinish": RippleChainFinish}
NODE_DISPLAY_NAME_MAPPINGS = {"RippleChainPrep": "Ripple Chain - Scan Video for Stills",
                              "RippleBundle": "Ripple Chain - Bundle Images",
                              "RippleChainLoad": "Ripple Chain - Load Chunk",
                              "RippleChainFinish": "Ripple Chain - Stitch and Save"}
