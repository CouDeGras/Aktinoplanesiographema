Forked from yoonsikp/kromo:master

Rewritten in Cython for performance (negligible)


# Aktinoplanesiographema


## Quick-start (Linux / WSL)

```bash
# 1. Clone or unpack the source
git clone https://github.com/CouDeGras/Aktinoplanesiographema.git
cd Aktinoplanesiographema        # project root (contains setup.py, pyproject.toml)

# 2. Create and activate an isolated environment
python3 -m venv .venv
source .venv/bin/activate       # on Windows use: .venv\Scripts\activate.bat

# 3. Upgrade packaging tools
pip install --upgrade pip setuptools wheel

# 4. Build + install (compiles the Cython extension)
pip install .

# 5. Test-drive
python -m aktino -h                             # CLI help
python -m aktino -s 0.3 -j 1 -o out.jpg path/to/photo.jpg
````

> **Result:** `out.jpg` contains a film-style red/green/blue fringe +
> polar Gaussian blur, computed with a C extension for 4-10× speed-up.

---

## Requirements
idk

On Debian/Ubuntu/WSL:

```bash
sudo apt update
sudo apt install build-essential python3-dev
```

Pip pulls the pure-Python deps automatically during step 4.


## Uninstall / clean

```bash
deactivate          # leave venv
rm -rf .venv build dist *.egg-info \
       aktino/core*.c aktino/core*.so
```

---

## Troubleshooting

| Symptom                                | Fix                                                                                                          |
| -------------------------------------- | ------------------------------------------------------------------------------------------------------------ |
| **`ModuleNotFoundError: aktino.core`** | You’re running from the project folder. Move outside, or delete the local `aktino/` directory after install. |
| **`No module named PIL`**              | `pip install Pillow` inside the venv.                                                                        |
| **Slow build on ARM boards**           | Use `pip install --no-build-isolation .` (assumes Cython & NumPy already in the venv).                       |





# kromo
`kromo` is a play on words, combining "chromatic aberration" and "lo-mo photography". I made `kromo` because perfect optics are overrated.
## Before & After
<p align="center">
  <img src=https://github.com/yoonsikp/kromo/blob/master/beforeafter.gif?raw=true width=60%>
 </p>
 <p align="center">
  Image of Berries, 1.0 strength
</p>

## [More Images](https://github.com/yoonsikp/kromo/blob/master/gallery.md)

## Quick Start
```
$ pip3 install -r requirements.txt
$ python3 kromo.py -v flower.jpg 

Original Image: JPEG (1962, 2615) RGB
Dimensions must be odd numbers, cropping...
New Dimensions: (1961, 2615)
Completed in:  43.85s

```

## Usage
```
$ python3 kromo.py --help

usage: kromo.py [-h] [-s STRENGTH] [-j JITTER] [-y OVERLAY] [-n] [-o OUT] [-v]
                filename

Apply chromatic aberration and lens blur to images

positional arguments:
  filename              input filename

optional arguments:
  -h, --help            show this help message and exit
  -s STRENGTH, --strength STRENGTH
                        set blur/aberration strength, defaults to 1.0
  -j JITTER, --jitter JITTER
                        set color channel offset pixels, defaults to 0
  -y OVERLAY, --overlay OVERLAY
                        alpha of original image overlay, defaults to 0.0
  -n, --noblur          disable radial blur
  -o OUT, --out OUT     write to OUTPUT (supports multiple formats)
  -v, --verbose         print status messages
```

## Runtime
`kromo` is slow, just like how film photography used to be. Clone the repo for a blast from the past.

The time complexity is O(n), so a 12MP picture takes 4 times longer than a 3MP picture.

## See also
[Circular & radial blur](http://chemaguerra.com/circular-radial-blur/)

[Use depth-of-field and other lens effects](https://doc.babylonjs.com/how_to/using_depth-of-field_and_other_lens_effects)
