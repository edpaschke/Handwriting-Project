import pathlib
import cv2
import numpy as np
from typing import Union

from recognize import load_model, segment, classify


# ---------------------------------------------------------------------------
# Folder naming helpers (imported by generalize.py, app.py, write_cat.py)
# ---------------------------------------------------------------------------

def _folder_name(char: str) -> str:
    """Map a character to its storage folder name.

    Alphabetic characters always get an _upper / _lower suffix so that 'A'
    and 'a' never collide on case-insensitive file systems (macOS default).
    Digits and non-alpha characters are stored as-is.
    """
    if len(char) == 1 and char.isalpha():
        return f"{char}_{'upper' if char.isupper() else 'lower'}"
    return char


def _folder_to_char(folder: str) -> str:
    """Reverse of _folder_name — returns the original character."""
    if folder.endswith("_upper") or folder.endswith("_lower"):
        return folder.rsplit("_", 1)[0]
    return folder


# ---------------------------------------------------------------------------
# Crop cleaning
# ---------------------------------------------------------------------------

def _clean_crop(crop_bgr: np.ndarray,
                min_area_fraction: float = 0.08) -> np.ndarray:
    """Return a white-background binary image.

    Keeps all ink blobs >= min_area_fraction of the largest blob so that
    dots (i, j) and crossbars (t, f) are preserved.
    """
    gray = (cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
            if crop_bgr.ndim == 3 else crop_bgr)
    _, binary = cv2.threshold(gray, 0, 255,
                              cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary, connectivity=8)
    if num_labels <= 1:
        return np.full(gray.shape, 255, dtype=np.uint8)
    areas = stats[1:, cv2.CC_STAT_AREA]
    min_area = int(np.max(areas) * min_area_fraction)
    result = np.full(gray.shape, 255, dtype=np.uint8)
    for i, area in enumerate(areas):
        if area >= min_area:
            result[labels == i + 1] = 0
    return result


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

DEFAULT_WRITINGS_DIR = pathlib.Path(__file__).resolve().parent / "default_writings"
LETTER_IMAGES_DIR    = pathlib.Path(__file__).resolve().parent / "default_writings" / "letter_images"

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif"}


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def classify_and_store(
    writings_dir:      Union[str, pathlib.Path] = DEFAULT_WRITINGS_DIR,
    output_dir:        Union[str, pathlib.Path] = LETTER_IMAGES_DIR,
    weights:           str   = "EMNIST-Classifier/model_hub/fine_tuned/vggnet_tuned.pth",
    unknown_threshold: float = 0.15,
    threshold_mode:    str   = "single",
) -> None:
    """Segment every handwriting image in writings_dir and store each
    character crop in output_dir/<folder_name>/."""
    writings_dir = pathlib.Path(writings_dir)
    output_dir   = pathlib.Path(output_dir)

    images = [p for p in sorted(writings_dir.iterdir())
              if p.suffix.lower() in IMAGE_EXTENSIONS]
    if not images:
        print(f"No images found in {writings_dir}")
        return

    model = load_model(weights)
    char_counts: dict[str, int] = {}

    for image_path in images:
        print(f"Processing {image_path.name} …")
        try:
            img, binaries, lines = segment(str(image_path))
        except FileNotFoundError as exc:
            print(f"  Skipping: {exc}")
            continue

        candidates = (
            [binaries["detect"]] if threshold_mode == "single"
            else [binaries["otsu"], binaries["adaptive_mean"],
                  binaries["adaptive_gaussian"], binaries["detect"]]
        )

        for line in lines:
            for x, y, w, h in line:
                best_char, best_conf = "?", -1.0
                for binary in candidates:
                    char, conf = classify(model, binary[y:y+h, x:x+w])
                    if conf > best_conf:
                        best_char, best_conf = char, conf

                if best_conf < unknown_threshold:
                    best_char = "unknown"

                folders = [_folder_name(best_char)]
                if best_char == 'C':
                    folders.append('c_lower')

                crop  = img[y:y+h, x:x+w]
                clean = _clean_crop(crop)

                for folder in folders:
                    char_dir = output_dir / folder
                    char_dir.mkdir(parents=True, exist_ok=True)
                    count = char_counts.get(folder, 0)
                    char_counts[folder] = count + 1
                    filename = f"{image_path.stem}_{folder}_{count:04d}.png"
                    cv2.imwrite(str(char_dir / filename), clean)

    total = sum(char_counts.values())
    print(f"\nStored {total} crops across {len(char_counts)} character folders:")
    for folder, n in sorted(char_counts.items()):
        print(f"  {folder}/  →  {n} crops")


if __name__ == "__main__":
    classify_and_store()
