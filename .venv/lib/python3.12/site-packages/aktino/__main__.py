# aktino/__main__.py  (pure Python)
import argparse, os, time
from PIL import Image
from .core import add_chromatic, add_jitter, blend_images   # functions *are* exported

def main() -> None:
    p = argparse.ArgumentParser(description="Apply chromatic aberration (Cython)")
    p.add_argument("filename")
    p.add_argument("-s", "--strength", type=float, default=1.0)
    p.add_argument("-j", "--jitter",   type=int,   default=0)
    p.add_argument("-y", "--overlay",  type=float, default=0.0)
    p.add_argument("-n", "--noblur",   action="store_true")
    p.add_argument("-o", "--out")
    args = p.parse_args()

    im  = Image.open(args.filename).convert("RGB")
    og  = im.copy()
    im  = add_chromatic(im, strength=args.strength, no_blur=args.noblur)
    im  = add_jitter(im,   pixels=args.jitter)
    im  = blend_images(im, og, alpha=args.overlay, strength=args.strength)
    out = args.out or os.path.splitext(args.filename)[0] + "_chromatic.jpg"
    im.save(out, quality=99)
    print("✔ saved", out)

if __name__ == "__main__":
    main()
