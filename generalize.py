import pathlib
import cv2
import numpy as np

from classify_and_store import classify_and_store, LETTER_IMAGES_DIR

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif"}

def sample_writer_styles(n_samples: int) -> list[dict]:
    """Pre-sample writer-level style parameters shared across all characters."""
    styles = []
    for _ in range(n_samples):
        styles.append({
            "brightness":      np.random.uniform(-0.12, 0.12),   # lighter/darker hand
            "noise_scale":     np.random.uniform(0.6, 1.4),      # shaky vs steady
            "thickness_shift": np.random.uniform(-0.06, 0.06),   # thin vs thick strokes
        })
    return styles


def generate_font_variations(matrices: list[np.ndarray],
                              styles: list[dict]) -> list[np.ndarray]:
    """Generate one image per style entry, applying the shared writer style to this character."""
    matrices = np.array(matrices) / 255.0

    mean         = np.mean(matrices, axis=0)
    std          = np.std(matrices, axis=0)
    support_mask = np.any(matrices > 0.1, axis=0)

    generated = []
    for style in styles:
        prob      = np.clip(np.mean(matrices > 0.3, axis=0) + style["thickness_shift"], 0, 1)
        structure = (np.random.rand(*mean.shape) < prob).astype(float)
        noise     = np.random.normal(0, std * style["noise_scale"], mean.shape)

        img = (mean + style["brightness"] + noise) * structure
        img[~support_mask] = 0
        img = np.clip(img, 0, 1) * 255
        generated.append(img.astype(np.uint8))

    return generated


def generalize(n_samples_per_char: int = 20) -> dict[str, list[np.ndarray]]:
    classify_and_store()

    # Sample writer styles once so every character shares the same handwriting personality
    styles = sample_writer_styles(n_samples_per_char)

    results: dict[str, list[np.ndarray]] = {}
    letter_images_dir = pathlib.Path(LETTER_IMAGES_DIR)

    for char_dir in sorted(letter_images_dir.iterdir()):
        if not char_dir.is_dir():
            continue

        matrices = []
        for img_path in char_dir.iterdir():
            if img_path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            resized = cv2.resize(img, (32, 32), interpolation=cv2.INTER_AREA)
            matrices.append(resized.astype(np.float64))

        if matrices:
            results[char_dir.name] = generate_font_variations(matrices, styles)

    return results

GENERATED_CHARS_DIR = pathlib.Path(__file__).resolve().parent / "default_writings" / "generated_chars"


def save_generated_fonts(results: dict[str, list[np.ndarray]],
                         output_dir: pathlib.Path = GENERATED_CHARS_DIR) -> None:
    output_path = pathlib.Path(output_dir)
    for char, images in results.items():
        char_dir = output_path / char
        char_dir.mkdir(parents=True, exist_ok=True)
        for i, img in enumerate(images):
            cv2.imwrite(str(char_dir / f"{char}_{i}.png"), img)
    print(f"Saved generated characters to {output_path}")


if __name__ == "__main__":
    print("Running classify_and_store + generalize...")
    results = generalize()
    print(f"Generated variations for {len(results)} characters.")
    save_generated_fonts(results)