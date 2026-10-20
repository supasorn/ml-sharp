"""Compress the textures of an exported layer .glb (Hybrid-MPI / MPI-only) to KTX2 (Basis UASTC).

Usage:
  python scripts/glb_ktx2.py IN.glb OUT.glb [--quality 2] [--zstd 18] [--decoded DECODED.glb]

Each embedded PNG atlas page is encoded with KTX-Software's `ktx create --encode uastc` (linear transfer
function: the pages hold premultiplied, display-space values that the viewer samples raw) and Zstandard
supercompression; OUT.glb references them through the glTF extension KHR_texture_basisu (three.js
GLTFLoader + KTX2Loader transcode to ASTC on the Quest, BC7 on desktops). --decoded additionally writes a
.glb whose PNG pages are the KTX2 pages decoded back to RGBA8 (what the GPU samples; UASTC -> ASTC is
lossless), for offline quality checks.

Needs `ktx` from KTX-Software (KTX_HOME, default ~/.local/opt/ktx-4.4.2/KTX-Software-4.4.2-Linux-<x86_64|arm64>).
"""
import argparse
import json
import os
import struct
import subprocess
import platform
import tempfile
import time
from pathlib import Path

KTX_HOME = Path(os.environ.get("KTX_HOME", Path.home() / ".local/opt/ktx-4.4.2" / (
    "KTX-Software-4.4.2-Linux-" + {"aarch64": "arm64"}.get(platform.machine(), platform.machine()))))


def ktx(*args):
    env = dict(os.environ, LD_LIBRARY_PATH=str(KTX_HOME / "lib"))
    subprocess.run([str(KTX_HOME / "bin" / "ktx"), *map(str, args)], check=True, env=env,
                   stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)


def ktx_args(quality=2, zstd=18, rdo=0.0, levels=3, astc=None):
    """`ktx create` arguments for the layer pages: UASTC, linear transfer (premultiplied display-space values
    sampled raw), box mipmaps, Zstandard. astc="fast" (etc.): plain ASTC 4x4 instead (the Quest's native format, no
    transcoding; ~6x faster to encode than UASTC and more accurate, but most desktop GPUs can't sample it)."""
    if astc:
        a = ["create", "--format", "ASTC_4x4_UNORM_BLOCK", "--astc-quality", astc, "--assign-tf", "linear"]
    else:
        a = ["create", "--format", "R8G8B8A8_UNORM", "--encode", "uastc", "--uastc-quality", quality, "--assign-tf", "linear"]
    if levels > 1:
        a += ["--generate-mipmap", "--mipmap-filter", "box", "--levels", levels]
    if rdo > 0:
        a += ["--uastc-rdo", "--uastc-rdo-l", rdo]
    if zstd > 0:
        a += ["--zstd", zstd]
    return a


def encode_pages(pngs, td, **kw):
    """PNG bytes per page -> KTX2 bytes per page, all pages encoded in parallel."""
    from concurrent.futures import ThreadPoolExecutor
    def one(i):
        src, dst = Path(td) / f"p{i}.png", Path(td) / f"p{i}.ktx2"
        src.write_bytes(pngs[i])
        ktx(*ktx_args(**kw), src, dst)
        return dst.read_bytes()
    with ThreadPoolExecutor(len(pngs) or 1) as ex:
        return list(ex.map(one, range(len(pngs))))


def read_glb(path):
    b = Path(path).read_bytes()
    n = struct.unpack("<I", b[12:16])[0]
    g = json.loads(b[20:20 + n])
    bn = struct.unpack("<I", b[20 + n:24 + n])[0]
    return g, b[28 + n:28 + n + bn]


def write_glb(path, g, views):
    """Rebuild the BIN chunk from per-bufferView bytes (4-byte aligned) and write the .glb."""
    out = bytearray()
    for i, data in enumerate(views):
        out += b"\0" * (-len(out) % 4)
        g["bufferViews"][i]["byteOffset"] = len(out)
        g["bufferViews"][i]["byteLength"] = len(data)
        out += data
    out += b"\0" * (-len(out) % 4)
    g["buffers"] = [{"byteLength": len(out)}]
    js = json.dumps(g, separators=(",", ":")).encode()
    js += b" " * (-len(js) % 4)
    total = 12 + 8 + len(js) + 8 + len(out)
    Path(path).write_bytes(struct.pack("<III", 0x46546C67, 2, total) + struct.pack("<II", len(js), 0x4E4F534A) + js
                           + struct.pack("<II", len(out), 0x004E4942) + bytes(out))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("src", type=Path)
    ap.add_argument("dst", type=Path)
    ap.add_argument("--quality", type=int, default=2, help="UASTC level 0-4 (slower = better)")
    ap.add_argument("--zstd", type=int, default=18, help="Zstandard level (0 = none)")
    ap.add_argument("--rdo", type=float, default=0, help="UASTC RDO lambda (0 = off; ~1 = smaller files)")
    ap.add_argument("--levels", type=int, default=3,
                    help="mipmap levels (box filter; 1 = none). The Quest shrinks full-resolution cells 2-3x; "
                         "with cells' 4-texel borders, levels 0-2 stay free of neighbouring cells")
    ap.add_argument("--decoded", type=Path, help="also write a .glb with the decoded pages as PNG")
    a = ap.parse_args()
    g, bin_ = read_glb(a.src)
    views = [bin_[v.get("byteOffset", 0):v.get("byteOffset", 0) + v["byteLength"]] for v in g["bufferViews"]]
    gk = json.loads(json.dumps(g))
    dec_views = list(views)
    t0 = time.time()
    png_bytes = ktx_bytes = 0
    with tempfile.TemporaryDirectory() as td:
        pngs = [views[im["bufferView"]] for im in g["images"]]
        png_bytes = sum(len(p) for p in pngs)
        kts = encode_pages(pngs, td, quality=a.quality, zstd=a.zstd, rdo=a.rdo, levels=a.levels)   # pages in parallel
        ktx_bytes = sum(len(k) for k in kts)
        for i, (im, kb) in enumerate(zip(g["images"], kts)):
            bv = im["bufferView"]
            gk["images"][i] = {"bufferView": bv, "mimeType": "image/ktx2"}
            if a.decoded:
                enc, dec = Path(td) / f"e{i}.ktx2", Path(td) / f"d{i}.png"
                enc.write_bytes(kb)
                ktx("extract", "--transcode", "rgba8", enc, dec)
                dec_views[bv] = dec.read_bytes()
            views[bv] = kb
    for t in gk["textures"]:
        t["extensions"] = {"KHR_texture_basisu": {"source": t.pop("source")}}
    gk["extensionsUsed"] = sorted(set(gk.get("extensionsUsed", [])) | {"KHR_texture_basisu"})
    gk["extensionsRequired"] = sorted(set(gk.get("extensionsRequired", [])) | {"KHR_texture_basisu"})
    write_glb(a.dst, gk, views)
    if a.decoded:
        write_glb(a.decoded, json.loads(json.dumps(g)), dec_views)
    n = len(g["images"])
    print(f"{a.src.name}: {n} pages, PNG {png_bytes / 2**20:.1f} MB -> KTX2 {ktx_bytes / 2**20:.1f} MB, "
          f"GPU {n * 4096 * 4096 * 4 / 2**20:.0f} MB RGBA8 -> {n * 4096 * 4096 / 2**20 * (4 / 3 if a.levels > 1 else 1):.0f} MB ASTC 4x4, "
          f"{time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
