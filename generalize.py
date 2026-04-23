"""
generalize.py  –  Synthesise unique handwritten-style glyphs per character.

Personalization
---------------
If a user_writings/ folder exists alongside default_writings/, the pipeline
blends the user's samples with the default ones. The blend is controlled by
USER_WEIGHT (0.0 = all default, 1.0 = all user). At the default of 0.75, 75%
of generated glyphs are drawn from the user's samples and 25% from the default.

Folder layout expected
----------------------
  <project>/
    default_writings/
      letter_images/          ← produced by classify_and_store on default data
        a/  b/  ...  A/  B/  ...  0/  1/  ...
    user_writings/            ← YOU create this from your filled template
      letter_images/          ← produced by classify_and_store on user photos
        a/  b/  ...  A/  B/  ...  0/  1/  ...
    generalize.py
    classify_and_store.py

How to add your handwriting
---------------------------
1. Print handwriting_template.pdf, fill it in with a dark pen.
2. Photograph or scan it (good lighting, flat, no shadows).
3. Place the photo(s) in  user_writings/  (create the folder).
4. Run:  python generalize.py
   The script calls classify_and_store on user_writings/ automatically,
   then blends and generates.

Architecture (single-image augmentation)
-----------------------------------------
Each output glyph = one real handwritten source + rich augmentation:
  affine warp → elastic distortion → stroke-width jitter → tiny noise
This guarantees a structurally complete glyph every time.
"""

import pathlib
import cv2
import numpy as np
from typing import Optional

from classify_and_store import classify_and_store, LETTER_IMAGES_DIR

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif"}

TARGET = 32
PAD    = 10

# Fraction of generated glyphs drawn from user samples (rest from default).
# 0.0 = ignore user,  1.0 = ignore default entirely.
USER_WEIGHT = 0.75

HERE                 = pathlib.Path(__file__).resolve().parent
DEFAULT_LETTER_DIR   = HERE / "default_writings"  / "letter_images"
USER_WRITINGS_DIR    = HERE / "user_writings"
USER_LETTER_DIR      = HERE / "user_writings"     / "letter_images"
GENERATED_CHARS_DIR  = HERE / "default_writings"  / "generated_chars"


# ---------------------------------------------------------------------------
# Binarise + normalise
# ---------------------------------------------------------------------------

def _prepare(img: np.ndarray) -> Optional[np.ndarray]:
    """Hard-binarise, drop dust, centre on padded canvas (ink=255, bg=0)."""
    if img is None or img.size == 0:
        return None
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    ink = 255 - binary

    n, labels, stats, _ = cv2.connectedComponentsWithStats(ink, connectivity=8)
    if n <= 1:
        return None
    areas    = stats[1:, cv2.CC_STAT_AREA]
    max_area = np.max(areas)
    keep = np.zeros_like(ink)
    for i, area in enumerate(areas):
        if area >= max_area * 0.05:
            keep[labels == i + 1] = 255
    if not keep.any():
        return None

    rows = np.where(keep.any(axis=1))[0]
    cols = np.where(keep.any(axis=0))[0]
    crop = keep[rows[0]:rows[-1]+1, cols[0]:cols[-1]+1]

    gh, gw = crop.shape
    scale = min(TARGET / gh, TARGET / gw)
    nh = max(1, int(round(gh * scale)))
    nw = max(1, int(round(gw * scale)))
    scaled = cv2.resize(crop, (nw, nh), interpolation=cv2.INTER_NEAREST)
    _, scaled = cv2.threshold(scaled, 127, 255, cv2.THRESH_BINARY)

    size   = TARGET + 2 * PAD
    canvas = np.zeros((size, size), dtype=np.uint8)
    top    = PAD + (TARGET - nh) // 2
    left   = PAD + (TARGET - nw) // 2
    canvas[top:top+nh, left:left+nw] = scaled
    return canvas


# ---------------------------------------------------------------------------
# Augmentation
# ---------------------------------------------------------------------------

def _augment(ink_canvas: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Affine + elastic + stroke-width jitter + noise. Returns white-bg black-ink."""
    h, w = ink_canvas.shape
    cx, cy = w / 2.0, h / 2.0

    angle = rng.uniform(-12, 12)
    tx    = rng.uniform(-3, 3)
    ty    = rng.uniform(-3, 3)
    scale = rng.uniform(0.88, 1.12)
    M = cv2.getRotationMatrix2D((cx, cy), angle, scale)
    M[0, 2] += tx
    M[1, 2] += ty
    warped = cv2.warpAffine(ink_canvas, M, (w, h),
                            flags=cv2.INTER_NEAREST,
                            borderMode=cv2.BORDER_CONSTANT,
                            borderValue=0)

    alpha = rng.uniform(2.0, 6.0)
    sigma = rng.uniform(2.5, 5.0)
    dx = cv2.GaussianBlur(rng.uniform(-1, 1, (h, w)).astype(np.float32), (0, 0), sigma) * alpha
    dy = cv2.GaussianBlur(rng.uniform(-1, 1, (h, w)).astype(np.float32), (0, 0), sigma) * alpha
    xs = (np.meshgrid(np.arange(w), np.arange(h))[0] + dx).astype(np.float32)
    ys = (np.meshgrid(np.arange(w), np.arange(h))[1] + dy).astype(np.float32)
    warped = cv2.remap(warped, xs, ys,
                       interpolation=cv2.INTER_NEAREST,
                       borderMode=cv2.BORDER_CONSTANT,
                       borderValue=0)

    sw = rng.uniform(0, 1)
    if sw < 0.30:
        k = int(rng.integers(2, 4))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        warped = cv2.erode(warped, kernel, iterations=1)
    elif sw > 0.70:
        k = int(rng.integers(2, 4))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        warped = cv2.dilate(warped, kernel, iterations=1)

    noise = rng.random((h, w))
    warped[noise < 0.015] = 255
    warped[noise > 0.985] = 0

    return _remove_dust(_trim_resize(255 - warped))


def _remove_dust(img: np.ndarray, min_area: int = 2) -> np.ndarray:
    """Remove ink blobs with fewer than min_area pixels (isolated specks/dots).

    Every surviving ink pixel is guaranteed to have at least one neighbour
    that is also ink (since single-pixel and tiny isolated clusters are erased).
    """
    ink = (img < 128).astype(np.uint8) * 255
    n, labels, stats, _ = cv2.connectedComponentsWithStats(ink, connectivity=8)
    out = img.copy()
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] < min_area:
            out[labels == i] = 255  # erase to background
    return out


def _trim_resize(img: np.ndarray) -> np.ndarray:
    ink = img < 128
    if not ink.any():
        return np.full((TARGET, TARGET), 255, dtype=np.uint8)
    rows = np.where(ink.any(axis=1))[0]
    cols = np.where(ink.any(axis=0))[0]
    r0 = max(0, rows[0] - 1);  r1 = min(img.shape[0], rows[-1] + 2)
    c0 = max(0, cols[0] - 1);  c1 = min(img.shape[1], cols[-1] + 2)
    resized = cv2.resize(img[r0:r1, c0:c1], (TARGET, TARGET), interpolation=cv2.INTER_NEAREST)
    _, out = cv2.threshold(resized, 127, 255, cv2.THRESH_BINARY)
    return out


# ---------------------------------------------------------------------------
# Load sources from a letter_images dir
# ---------------------------------------------------------------------------

def _load_sources(letter_dir: pathlib.Path) -> dict[str, list[np.ndarray]]:
    """Return {char_label: [grayscale arrays]} from a letter_images directory."""
    sources: dict[str, list[np.ndarray]] = {}
    if not letter_dir.exists():
        return sources
    for char_dir in sorted(letter_dir.iterdir()):
        if not char_dir.is_dir():
            continue
        imgs = []
        for p in char_dir.iterdir():
            if p.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
            if img is not None:
                imgs.append(img)
        if imgs:
            sources[char_dir.name] = imgs
    return sources


# ---------------------------------------------------------------------------
# Blend pools
# ---------------------------------------------------------------------------

def _build_pool(char: str,
                default_srcs: dict[str, list[np.ndarray]],
                user_srcs:    dict[str, list[np.ndarray]],
                rng: np.random.Generator,
                n_samples: int) -> list[np.ndarray]:
    """
    Build a list of (source_image, is_user) pairs for `n_samples` glyphs,
    respecting USER_WEIGHT.  Returns prepared (normalised) ink canvases.
    """
    d_imgs = default_srcs.get(char, [])
    u_imgs = user_srcs.get(char, [])

    d_prep = [p for p in (_prepare(m) for m in d_imgs) if p is not None]
    u_prep = [p for p in (_prepare(m) for m in u_imgs) if p is not None]

    if not d_prep and not u_prep:
        return []

    pool = []
    for i in range(n_samples):
        # Decide source based on weight; fall back if one pool is empty
        use_user = u_prep and (not d_prep or rng.random() < USER_WEIGHT)
        src_pool = u_prep if use_user else d_prep
        pool.append(src_pool[i % len(src_pool)])
    return pool


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_font_variations(prepared: list[np.ndarray],
                              n_samples: int,
                              seed: Optional[int] = None) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    return [_augment(prepared[i % len(prepared)], rng) for i in range(n_samples)]


def generalize(n_samples_per_char: int = 20,
               seed: Optional[int] = 42,
               run_classify: bool = True) -> dict[str, list[np.ndarray]]:

    # --- classify default writings ---
    if run_classify:
        classify_and_store()   # uses its own DEFAULT_WRITINGS_DIR

    # --- classify user writings if present ---
    if USER_WRITINGS_DIR.exists():
        print(f"\nUser writings found at {USER_WRITINGS_DIR} — classifying …")
        classify_and_store(
            writings_dir=USER_WRITINGS_DIR,
            output_dir=USER_LETTER_DIR,
        )
    else:
        print(f"\nNo user_writings/ folder found at {USER_WRITINGS_DIR}.")
        print("To personalise, create that folder and place your handwriting photos in it.")

    default_srcs = _load_sources(DEFAULT_LETTER_DIR)
    user_srcs    = _load_sources(USER_LETTER_DIR)

    all_chars = set(default_srcs.keys()) | set(user_srcs.keys())
    master_rng = np.random.default_rng(seed)
    results: dict[str, list[np.ndarray]] = {}

    for char in sorted(all_chars):
        char_seed = int(master_rng.integers(0, 2**31))
        rng = np.random.default_rng(char_seed)

        pool = _build_pool(char, default_srcs, user_srcs, rng, n_samples_per_char)
        if not pool:
            continue

        variations = generate_font_variations(pool, n_samples_per_char, seed=char_seed)
        if variations:
            results[char] = variations
            u_count = sum(1 for i in range(n_samples_per_char)
                          if user_srcs.get(char) and rng.random() < USER_WEIGHT)
            src_note = ""
            if char in user_srcs and char in default_srcs:
                src_note = f" [blended, user_weight={USER_WEIGHT}]"
            elif char in user_srcs:
                src_note = " [user only]"
            print(f"  [{char}]  default={len(default_srcs.get(char,[]))}  "
                  f"user={len(user_srcs.get(char,[]))}  → {len(variations)} glyphs{src_note}")

    return results


# ---------------------------------------------------------------------------
# Saving
# ---------------------------------------------------------------------------

def _folder_name(ch: str, needs_suffix: set) -> str:
    if ch in needs_suffix:
        return f"{ch}_{'upper' if ch.isupper() else 'lower'}"
    return ch


def save_generated_fonts(results: dict[str, list[np.ndarray]],
                         output_dir: pathlib.Path = GENERATED_CHARS_DIR) -> None:
    output_path = pathlib.Path(output_dir)
    all_chars    = set(results.keys())
    needs_suffix = {ch for ch in all_chars if ch.isalpha() and ch.swapcase() in all_chars}

    for char, images in results.items():
        folder = output_path / _folder_name(char, needs_suffix)
        folder.mkdir(parents=True, exist_ok=True)
        for i, img in enumerate(images):
            cv2.imwrite(str(folder / f"{char}_{i:04d}.png"), img)

    print(f"\nSaved to {output_path}")
    for char in sorted(results.keys()):
        print(f"  {_folder_name(char, needs_suffix)}/  ({len(results[char])} images)")


if __name__ == "__main__":
    results = generalize()
    print(f"\nGenerated variations for {len(results)} characters.")
    save_generated_fonts(results)