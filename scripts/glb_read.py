"""Minimal GLB reader for the layer exports: meshes -> {name: dict(V, F, uv, uv1, uv2, uv3, tex RGBA uint8, mips)}
(mips: the texture's mip levels, level 0 = tex; only KTX2 pages have more than one).

Reads PNG pages and KTX2 pages (KHR_texture_basisu, as written by glb_ktx2.py / EXPORT_KTX2): those are transcoded
to RGBA8 (every mip level), which is what the GPU samples (UASTC -> ASTC is lossless), so a _ktx2.glb renders with its
compression error. Needs `ktx` from KTX-Software for KTX2 (KTX_HOME, as in glb_ktx2.py).
"""
import io, json, os, platform, struct, subprocess, tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
from PIL import Image

CT = {5126: np.float32, 5125: np.uint32, 5123: np.uint16, 5121: np.uint8}
NC = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4}
KTX_HOME = Path(os.environ.get("KTX_HOME", Path.home() / ".local/opt/ktx-4.4.2" / (
    "KTX-Software-4.4.2-Linux-" + {"aarch64": "arm64"}.get(platform.machine(), platform.machine()))))


def ktx2_rgba(data):
    """KTX2 bytes -> [(h, w, 4) uint8 per mip level], level 0 first."""
    vkf = struct.unpack_from("<I", data, 12)[0]                   # vkFormat: 0 = Basis (UASTC), else e.g. ASTC
    w, h = struct.unpack_from("<II", data, 20)                    # pixelWidth, pixelHeight
    nlev = max(1, struct.unpack_from("<I", data, 40)[0])          # levelCount
    out = []
    env = dict(os.environ, LD_LIBRARY_PATH=str(KTX_HOME / "lib"))
    with tempfile.TemporaryDirectory() as td:
        src = Path(td) / "p.ktx2"
        src.write_bytes(data)
        for lv in range(nlev):
            if vkf == 0:                                          # Basis: transcode to raw RGBA8
                dst = Path(td) / f"p{lv}.raw"
                subprocess.run([str(KTX_HOME / "bin" / "ktx"), "extract", "--transcode", "rgba8", "--level", str(lv),
                                "--raw", str(src), str(dst)], check=True, env=env, capture_output=True)
                out.append(np.fromfile(dst, np.uint8).reshape(max(1, h >> lv), max(1, w >> lv), 4))
            else:                                                 # block-compressed (ASTC): decoded to PNG
                dst = Path(td) / f"p{lv}.png"
                subprocess.run([str(KTX_HOME / "bin" / "ktx"), "extract", "--level", str(lv), str(src), str(dst)],
                               check=True, env=env, capture_output=True)
                out.append(np.asarray(Image.open(dst).convert("RGBA")))
    return out


def read_glb(path):
    b = open(path, "rb").read()
    jl = struct.unpack_from("<I", b, 12)[0]
    g = json.loads(b[20:20 + jl])
    bin_ = b[20 + jl + 8:]
    def view(i):
        v = g["bufferViews"][i]; o = v.get("byteOffset", 0)
        return bin_[o:o + v["byteLength"]]
    def acc(i):
        a = g["accessors"][i]
        arr = np.frombuffer(view(a["bufferView"]), CT[a["componentType"]], count=a["count"] * NC[a["type"]],
                            offset=a.get("byteOffset", 0))
        return arr.reshape(a["count"], NC[a["type"]]) if NC[a["type"]] > 1 else arr
    def image(si):
        im = g["images"][si]; data = view(im["bufferView"])
        if im.get("mimeType") == "image/ktx2":
            return ktx2_rgba(data)
        return [np.asarray(Image.open(io.BytesIO(data)).convert("RGBA"))]
    def source(ti):
        t = g["textures"][ti]
        return t["extensions"]["KHR_texture_basisu"]["source"] if "source" not in t else t["source"]
    srcs = {}
    for m in g["meshes"]:
        mat = g["materials"][m["primitives"][0]["material"]]
        srcs[m["name"]] = source(mat["pbrMetallicRoughness"]["baseColorTexture"]["index"])
    uniq = sorted(set(srcs.values()))
    with ThreadPoolExecutor(len(uniq) or 1) as ex:                 # pages in parallel
        imgs = dict(zip(uniq, ex.map(image, uniq)))
    out = {}
    for m in g["meshes"]:
        pr = m["primitives"][0]; at = pr["attributes"]
        out[m["name"]] = dict(V=acc(at["POSITION"]), F=acc(pr["indices"]).reshape(-1, 3), uv=acc(at["TEXCOORD_0"]),
                              uv1=acc(at["TEXCOORD_1"]) if "TEXCOORD_1" in at else None,
                              uv2=acc(at["TEXCOORD_2"]) if "TEXCOORD_2" in at else None,
                              uv3=acc(at["TEXCOORD_3"]) if "TEXCOORD_3" in at else None, tex=imgs[srcs[m["name"]]][0],
                              mips=imgs[srcs[m["name"]]])
    return out
