"""Watch a collection's pano/inputs and turn each new 360 panorama (equirectangular .jpg/.jpeg/.png) into what
the gallery shows (docs/pano-360.md):
  1. scripts/pano_predict.py  -> pano/out_ico2/<name>.ply (SHARP splats; ~2 min on the GB10)
  2. scripts/pano_spz.py      -> pano/out_ico2/<name>.spz (the viewer's "360 splats")
  3. scripts/pano_msi.py      -> pano/out_ico2_photo4/<name>_msi[_ktx2].glb (photo-detail v4 multi-sphere image,
                                 the viewer's default "360 MSI")
web.js puts uploaded panoramas into pano/inputs (scripts/start_server.sh runs this watcher next to the others).

python scripts/pano_watch.py --root data/public [--once] [names...]
A panorama is picked up once its file has been unchanged for a few seconds; one whose steps failed is retried only
after the file changes. Logs per panorama go to stdout. pano/.watch_status.json tells the gallery what is being
built ({"current": name, "step": "splats" | "spz" | "msi", "failed": [names]}).
"""
import argparse, json, os, subprocess, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CHECKPOINT = ROOT / "sharp_2572gikvuh.pt"
EXTS = (".jpg", ".jpeg", ".png")
MSI_DIR = "out_ico2_photo4"
MSI_ARGS = ["--maxpages", "16", "--texfull-thr", "0.0015", "--detail-alpha", "0.95", "--detail-erode", "2"]


def newer(out: Path, src: Path) -> bool:
    return out.exists() and out.stat().st_mtime >= src.stat().st_mtime


def write_status(pano: Path, **st):
    tmp = pano / ".watch_status.json.tmp"
    tmp.write_text(json.dumps(st))
    os.replace(tmp, pano / ".watch_status.json")


def process(img: Path, pano: Path, status):
    name = img.stem
    ply, spz = pano / "out_ico2" / f"{name}.ply", pano / "out_ico2" / f"{name}.spz"
    msi = pano / MSI_DIR / f"{name}_msi_ktx2.glb"
    py = [sys.executable, "-u"]
    steps = [("splats", ply, py + [str(ROOT / "scripts/pano_predict.py"), str(img), "-o", str(ply.parent),
                         "--checkpoint", str(CHECKPOINT)]),
             ("spz", spz, py + [str(ROOT / "scripts/pano_spz.py"), str(ply)]),
             ("msi", msi, py + [str(ROOT / "scripts/pano_msi.py"), str(ply), "-o", str(msi.parent), *MSI_ARGS,
                         "--photo-detail", str(img)])]
    for label, out, cmd in steps:
        if newer(out, img):
            continue
        status(current=name, step=label)
        out.parent.mkdir(parents=True, exist_ok=True)
        t = time.time()
        print(f"  {out.name} ...", flush=True)
        subprocess.run(cmd, check=True, cwd=ROOT, stdout=sys.stdout, stderr=subprocess.STDOUT)
        print(f"  {out.name}: {time.time() - t:.0f} s", flush=True)


def todo(pano: Path, names):
    inputs = pano / "inputs"
    for img in sorted(inputs.glob("*")) if inputs.is_dir() else []:
        if img.suffix.lower() not in EXTS or (names and img.stem not in names):
            continue
        try:
            if not newer(pano / MSI_DIR / f"{img.stem}_msi_ktx2.glb", img):
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
    failed = {}                                                # name -> input mtime that failed
    print(f"watching {pano / 'inputs'} for 360 panoramas (every {args.poll:.0f} s)", flush=True)
    status = lambda **st: write_status(pano, failed=sorted(failed), **st)
    status(current=None)
    while True:
        for img in todo(pano, set(args.names)):
            try:
                mtime = img.stat().st_mtime
            except OSError:                                    # removed since the scan
                continue
            if failed.get(img.stem) == mtime or (not args.once and time.time() - mtime < 5):
                continue
            t = time.time()
            print(f"[{time.strftime('%H:%M:%S')}] {img.name}", flush=True)
            try:
                process(img, pano, status)
                print(f"  done in {time.time() - t:.0f} s", flush=True)
            except (subprocess.CalledProcessError, OSError) as e:
                if not img.exists():                           # removed while being processed
                    print("  input removed; skipped", flush=True)
                    status(current=None)
                    continue
                failed[img.stem] = mtime
                print(f"  FAILED ({e}); "
                      "retried when the file changes", flush=True)
            status(current=None)
        if args.once:
            break
        time.sleep(args.poll)


if __name__ == "__main__":
    main()
