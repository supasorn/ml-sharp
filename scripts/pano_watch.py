"""Watch a collection's pano/inputs and turn each new 360 panorama (equirectangular .jpg/.jpeg/.png) into what
the gallery shows (docs/pano-360.md):
  1. scripts/pano_predict.py  -> pano/out_dr/<name>.ply (SHARP splats on DAP-corrected depth, --depth dap_ratio,
                                 up-view layout; ~1 min on the GB10)
  2. scripts/pano_spz.py      -> pano/out_dr/<name>.spz (the viewer's "360 splats")
  3. scripts/pano_msi.py      -> pano/out_dr_photo4/<name>_msi[_astc].glb (photo-detail v4 multi-sphere image,
                                 the viewer's default "360 MSI")
web.js puts uploaded panoramas into pano/inputs (scripts/start_server.sh runs this watcher next to the others).

python scripts/pano_watch.py --root data/public [--once] [names...]
A panorama is picked up once its file has been unchanged for a few seconds; one whose steps failed is retried only
after the file changes. Logs per panorama go to stdout. pano/.watch_status.json tells the gallery what is being
built ({"current": name, "step": "splats" | "spz" | "msi", "writing": [names], "failed": [names]}).

One process for all panoramas, as the photo bake's persistent worker (scripts/bake_mpi.py): the scripts are imported
and the SHARP model loaded once (fresh processes cost ~15 s of imports and loading per panorama), and a panorama's
MSI files (PNG pages, ASTC copy) are written on a background thread while the next one is predicted.
"""
import argparse, gc, json, os, sys, time, traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CHECKPOINT = ROOT / "sharp_2572gikvuh.pt"
EXTS = (".jpg", ".jpeg", ".png")
SPLAT_DIR, MSI_DIR = "out_dr", "out_dr_photo4"
PREDICT_ARGS = ["--layout", "icoup", "--depth", "dap_ratio"]   # docs/pano-360.md
MSI_ARGS = ["--maxpages", "16", "--texfull-thr", "0.0015", "--detail-alpha", "0.95", "--detail-erode", "2",
            "--tex", "astc", "--cull", "0.05"]   # cull: 23% fewer cells, renders within 61+ dB from moved eyes


def newer(out: Path, src: Path) -> bool:
    return out.exists() and out.stat().st_mtime >= src.stat().st_mtime


def write_status(pano: Path, **st):
    tmp = pano / ".watch_status.json.tmp"
    tmp.write_text(json.dumps(st))
    os.replace(tmp, pano / ".watch_status.json")


def process(img: Path, pano: Path, status, writer):
    """The missing steps for one panorama, in this process; returns the future of its MSI files (written on
    `writer`), or None."""
    import pano_msi, pano_predict, pano_spz
    name = img.stem
    ply, spz = pano / SPLAT_DIR / f"{name}.ply", pano / SPLAT_DIR / f"{name}.spz"
    msi = pano / MSI_DIR / f"{name}_msi_astc.glb"
    steps = [("splats", ply, lambda: pano_predict.main([str(img), "-o", str(ply.parent),
                                                        "--checkpoint", str(CHECKPOINT), *PREDICT_ARGS])),
             ("spz", spz, lambda: pano_spz.main([str(ply)])),
             ("msi", msi, lambda: pano_msi.main([str(ply), "-o", str(msi.parent), *MSI_ARGS,
                                                 "--photo-detail", str(img)], writer=writer))]
    fut = None
    for label, out, fn in steps:
        if newer(out, img):
            continue
        status(current=name, step=label)
        out.parent.mkdir(parents=True, exist_ok=True)
        t = time.time()
        print(f"  {out.name} ...", flush=True)
        fut = fn()
        print(f"  {out.name}: {time.time() - t:.0f} s{' (files being written)' if fut is not None else ''}", flush=True)
    return fut


def todo(pano: Path, names):
    inputs = pano / "inputs"
    for img in sorted(inputs.glob("*")) if inputs.is_dir() else []:
        if img.suffix.lower() not in EXTS or (names and img.stem not in names):
            continue
        try:
            if not any(newer(pano / MSI_DIR / f"{img.stem}_msi_{k}.glb", img) for k in ("astc", "ktx2")):
                yield img
        except OSError:                                        # removed while listing
            continue


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=ROOT / "data/public", help="collection folder (holds pano/)")
    ap.add_argument("--once", action="store_true", help="process what is there and exit")
    ap.add_argument("--poll", type=float, default=10, help="seconds between scans")
    ap.add_argument("names", nargs="*", help="only these panoramas (file names without extension)")
    args = ap.parse_args()
    pano = args.root.resolve() / "pano"
    sys.path.insert(0, str(ROOT / "scripts"))
    os.chdir(ROOT)
    failed = {}                                                # name -> input mtime that failed
    writing = {}                                               # name -> (future of its MSI files, start, mtime)
    writer = ThreadPoolExecutor(1)
    print(f"watching {pano / 'inputs'} for 360 panoramas (every {args.poll:.0f} s)", flush=True)
    current = {"current": None}
    lock = __import__("threading").Lock()                     # status and `writing` change on the writer thread too

    def status(**st):
        with lock:
            current.update(st)
            write_status(pano, failed=sorted(failed), writing=sorted(writing), **current)

    def fail(name, mtime, e):
        failed[name] = mtime
        print(f"  {name} FAILED ({type(e).__name__}: {e}); retried when the file changes", flush=True)
        traceback.print_exc()

    def written(name, t, mtime, fut):                          # on the writer thread, when the files are written
        with lock:
            del writing[name]
        try:
            fut.result()
            print(f"  {name}: done in {time.time() - t:.0f} s", flush=True)
        except Exception as e:
            fail(name, mtime, e)
        status()

    status(current=None)
    while True:
        for img in todo(pano, set(args.names)):
            try:
                mtime = img.stat().st_mtime
            except OSError:                                    # removed since the scan
                continue
            if (img.stem in writing or failed.get(img.stem) == mtime
                    or (not args.once and time.time() - mtime < 5)):
                continue
            t = time.time()
            print(f"[{time.strftime('%H:%M:%S')}] {img.name}", flush=True)
            try:
                fut = process(img, pano, status, writer)
                if fut is not None:
                    with lock:
                        writing[img.stem] = fut
                    fut.add_done_callback(lambda f, n=img.stem, t=t, m=mtime: written(n, t, m, f))
                else:
                    print(f"  done in {time.time() - t:.0f} s", flush=True)
            except Exception as e:
                if not img.exists():                           # removed while being processed
                    print("  input removed; skipped", flush=True)
                else:
                    fail(img.stem, mtime, e)
            finally:
                gc.collect()
                __import__("torch").cuda.empty_cache()         # the next panorama (and the other watchers) need it
            status(current=None, step=None)
        if args.once:
            writer.shutdown(wait=True)
            break
        time.sleep(args.poll)


if __name__ == "__main__":
    main()
