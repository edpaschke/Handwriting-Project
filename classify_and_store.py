import pathlib
import cv2
import numpy as np
from typing import Union

from recognize import load_model, segment, classify

DEFAULT_WRITINGS_DIR = pathlib.Path(__file__).resolve().parent / "default_writings"
LETTER_IMAGES_DIR    = pathlib.Path(__file__).resolve().parent / "default_writings" / "letter_images"

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif"}


def classify_and_store(
    writings_dir: Union[str, pathlib.Path] = DEFAULT_WRITINGS_DIR,
    output_dir:   Union[str, pathlib.Path] = LETTER_IMAGES_DIR,
    weights: str = "EMNIST-Classifier/model_hub/fine_tuned/vggnet_tuned.pth",
    unknown_threshold: float = 0.15,
    threshold_mode: str = "single",
):
    """Run recognition on every image in writings_dir and save each character
    crop into output_dir/<character>/ named by source image and bounding box."""
    writings_dir = pathlib.Path(writings_dir)
    output_dir   = pathlib.Path(output_dir)

    images = [p for p in sorted(writings_dir.iterdir())
              if p.suffix.lower() in IMAGE_EXTENSIONS]
    if not images:
        print(f"No images found in {writings_dir}")
        return

    model = load_model(weights)

    for image_path in images:
        print(f"Processing {image_path.name} ...")
        try:
            img, binaries, lines = segment(str(image_path))
        except FileNotFoundError as exc:
            print(f"  Skipping: {exc}")
            continue

        candidate_binaries = (
            [binaries["detect"]] if threshold_mode == "single"
            else [binaries["otsu"], binaries["adaptive_mean"],
                  binaries["adaptive_gaussian"], binaries["detect"]]
        )

        char_counts: dict[str, int] = {}

        for line in lines:
            for x, y, w, h in line:
                best_char, best_conf = "?", -1.0
                for binary in candidate_binaries:
                    char, conf = classify(model, binary[y:y+h, x:x+w])
                    if conf > best_conf:
                        best_char, best_conf = char, conf

                if best_conf < unknown_threshold:
                    best_char = "unknown"

                char_dir = output_dir / best_char
                char_dir.mkdir(parents=True, exist_ok=True)

                count = char_counts.get(best_char, 0)
                char_counts[best_char] = count + 1

                crop = img[y:y+h, x:x+w]
                filename = f"{image_path.stem}_{best_char}_{count:04d}.png"
                cv2.imwrite(str(char_dir / filename), crop)

        total = sum(char_counts.values())
        print(f"  Stored {total} crops across {len(char_counts)} character folders.")


if __name__ == "__main__":
    classify_and_store()
