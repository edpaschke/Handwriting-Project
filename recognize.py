import sys
import pathlib
import cv2
import torch
import numpy as np
from torchvision import transforms
from PIL import Image

sys.path.insert(0, str(pathlib.Path(__file__).parent / "EMNIST-Classifier"))
from charclf.models import VGGNet

EMNIST_CLASSES = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)

TRANSFORM = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.1736,), (0.3248,)),
])


def load_model(weights: str = "EMNIST-Classifier/model_hub/fine_tuned/vggnet_tuned.pth") -> VGGNet:
    model = VGGNet()
    model.load_state_dict(torch.load(weights, map_location=DEVICE))
    model.to(DEVICE).eval()
    return model


def _group_boxes_into_lines(boxes: list[tuple[int, int, int, int]]) -> list[list[tuple[int, int, int, int]]]:
    """Group boxes into text lines and sort each line left-to-right."""
    if not boxes:
        return []

    heights = np.array([h for _, _, _, h in boxes], dtype=np.float32)
    line_threshold = max(12, int(np.median(heights) * 0.7))

    boxes_by_center = sorted(boxes, key=lambda b: b[1] + b[3] // 2)
    lines: list[list[tuple[int, int, int, int]]] = []
    line_centers: list[int] = []

    for box in boxes_by_center:
        cy = box[1] + box[3] // 2
        if not lines:
            lines.append([box])
            line_centers.append(cy)
            continue

        idx = int(np.argmin([abs(cy - lc) for lc in line_centers]))
        if abs(cy - line_centers[idx]) <= line_threshold:
            lines[idx].append(box)
            centers = [b[1] + b[3] // 2 for b in lines[idx]]
            line_centers[idx] = int(sum(centers) / len(centers))
        else:
            lines.append([box])
            line_centers.append(cy)

    line_pairs = sorted(zip(lines, line_centers), key=lambda item: item[1])
    ordered_lines: list[list[tuple[int, int, int, int]]] = []
    for line_boxes, _ in line_pairs:
        ordered_lines.append(sorted(line_boxes, key=lambda b: b[0]))
    return ordered_lines


def _build_binary_images(
    gray: np.ndarray,
    adaptive_block_size: int = 31,
    adaptive_c: int = 12,
    morph_kernel_size: int = 1,
) -> dict[str, np.ndarray]:
    """Build multiple binary images for robust character extraction and classification."""
    blur = cv2.GaussianBlur(gray, (3, 3), 0)

    otsu = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    adaptive_mean = cv2.adaptiveThreshold(
        blur,
        255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY_INV,
        adaptive_block_size,
        adaptive_c,
    )
    adaptive_gaussian = cv2.adaptiveThreshold(
        blur,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        adaptive_block_size,
        adaptive_c,
    )

    if morph_kernel_size > 1:
        kernel = np.ones((morph_kernel_size, morph_kernel_size), dtype=np.uint8)

        def _clean(img: np.ndarray) -> np.ndarray:
            return cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
    else:
        # No morphology by default: preserves thin strokes better.
        def _clean(img: np.ndarray) -> np.ndarray:
            return img

    detect = cv2.bitwise_or(otsu, adaptive_gaussian)

    return {
        "detect": _clean(detect),
        "otsu": _clean(otsu),
        "adaptive_mean": _clean(adaptive_mean),
        "adaptive_gaussian": _clean(adaptive_gaussian),
    }


def segment(
    image_path: str,
    adaptive_block_size: int = 31,
    adaptive_c: int = 12,
    morph_kernel_size: int = 1,
) -> tuple[np.ndarray, dict[str, np.ndarray], list[list[tuple[int, int, int, int]]]]:
    """Return (original BGR image, binary variants, line-grouped bounding boxes)."""
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Could not open image: {image_path}")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    binaries = _build_binary_images(
        gray,
        adaptive_block_size=adaptive_block_size,
        adaptive_c=adaptive_c,
        morph_kernel_size=morph_kernel_size,
    )
    detect_binary = binaries["detect"]

    boxes = []
    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(detect_binary, connectivity=8)
    for i in range(1, num_labels):
        x, y, w, h, area = stats[i]
        if w > 4 and h > 8 and area > 20:
            boxes.append((int(x), int(y), int(w), int(h)))

    lines = _group_boxes_into_lines(boxes)
    return img, binaries, lines


def _normalize_glyph(binary_crop: np.ndarray, output_size: int = 32) -> Image.Image:
    """Center and pad a binary character crop into a square image for stable inference."""
    ys, xs = np.where(binary_crop > 0)
    if len(xs) == 0 or len(ys) == 0:
        canvas = np.zeros((output_size, output_size), dtype=np.uint8)
        return Image.fromarray(canvas)

    x1, x2 = int(xs.min()), int(xs.max()) + 1
    y1, y2 = int(ys.min()), int(ys.max()) + 1
    glyph = binary_crop[y1:y2, x1:x2]

    h, w = glyph.shape
    side = max(h, w)
    pad = max(2, side // 6)
    canvas_side = side + 2 * pad
    canvas = np.zeros((canvas_side, canvas_side), dtype=np.uint8)

    y_off = (canvas_side - h) // 2
    x_off = (canvas_side - w) // 2
    canvas[y_off:y_off + h, x_off:x_off + w] = glyph
    resized = cv2.resize(canvas, (output_size, output_size), interpolation=cv2.INTER_AREA)
    return Image.fromarray(resized)


def classify(model: VGGNet, binary_crop: np.ndarray) -> tuple[str, float]:
    pil = _normalize_glyph(binary_crop, output_size=32)
    tensor = TRANSFORM(pil).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        output = model(tensor)
        probs = torch.softmax(output, dim=1)
    idx = output.argmax(dim=1).item()
    confidence = probs[0, idx].item()
    return EMNIST_CLASSES[idx], confidence


def _line_to_text(
    line_items: list[tuple[tuple[int, int, int, int], str, float]],
    letter_spacing: int = 0,
    word_gap_scale: float = 1.8,
) -> tuple[str, list[float]]:
    """Build a readable text line by inserting spaces from gaps and optional extra letter spacing."""
    if not line_items:
        return "", []

    if len(line_items) == 1:
        _, ch, conf = line_items[0]
        return ch, [conf]

    gaps = []
    widths = []
    for i, (box, _, _) in enumerate(line_items):
        x, _, w, _ = box
        widths.append(w)
        if i > 0:
            prev_x, _, prev_w, _ = line_items[i - 1][0]
            gaps.append(max(0, x - (prev_x + prev_w)))

    median_gap = float(np.median(gaps)) if gaps else 0.0
    median_width = float(np.median(widths)) if widths else 0.0
    # Conservative threshold to avoid over-inserting spaces.
    space_threshold = max(6.0, median_gap * word_gap_scale, median_width * 0.35)

    chars: list[str] = []
    confidences: list[float] = []
    for i, (_, ch, conf) in enumerate(line_items):
        if i > 0:
            prev_x, _, prev_w, _ = line_items[i - 1][0]
            x, _, _, _ = line_items[i][0]
            gap = max(0, x - (prev_x + prev_w))
            if gap >= space_threshold:
                chars.append(" ")
                confidences.append(1.0)
        chars.append(ch)
        confidences.append(conf)
        if i < len(line_items) - 1 and letter_spacing > 0:
            chars.append(" " * letter_spacing)
            confidences.extend([1.0] * letter_spacing)

    return "".join(chars), confidences


def recognize(image_path: str, weights: str = "EMNIST-Classifier/model_hub/fine_tuned/vggnet_tuned.pth",
              save_annotated: bool = False,
              return_confidences: bool = False,
              unknown_threshold: float = 0.0,
              threshold_mode: str = "single",
              adaptive_block_size: int = 31,
              adaptive_c: int = 12,
              letter_spacing: int = 0,
              word_gap_scale: float = 1.8,
              morph_kernel_size: int = 1):
    model = load_model(weights)
    img, binaries, lines = segment(
        image_path,
        adaptive_block_size=adaptive_block_size,
        adaptive_c=adaptive_c,
        morph_kernel_size=morph_kernel_size,
    )

    if threshold_mode not in {"single", "multi"}:
        raise ValueError("threshold_mode must be 'single' or 'multi'")

    candidate_binaries = [binaries["detect"]] if threshold_mode == "single" else [
        binaries["otsu"],
        binaries["adaptive_mean"],
        binaries["adaptive_gaussian"],
        binaries["detect"],
    ]

    result_lines: list[str] = []
    confidences: list[float] = []
    for line in lines:
        line_items: list[tuple[tuple[int, int, int, int], str, float]] = []
        for x, y, w, h in line:
            best_char = "?"
            best_conf = -1.0
            for binary in candidate_binaries:
                crop = binary[y:y+h, x:x+w]
                char, conf = classify(model, crop)
                if conf > best_conf:
                    best_char = char
                    best_conf = conf

            char, conf = best_char, best_conf
            if conf < unknown_threshold:
                char = "?"
            line_items.append(((x, y, w, h), char, conf))

            if save_annotated:
                cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 1)
                cv2.putText(img, f"{char}:{conf:.2f}", (x, max(0, y - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)

        line_text, line_conf = _line_to_text(
            line_items,
            letter_spacing=letter_spacing,
            word_gap_scale=word_gap_scale,
        )
        if line_text:
            result_lines.append(line_text)
            confidences.extend(line_conf)

    text = "\n".join(result_lines)

    if save_annotated:
        out_path = pathlib.Path(image_path).stem + "_annotated.png"
        cv2.imwrite(out_path, img)
        print(f"Annotated image saved to {out_path}")

    if return_confidences:
        return text, confidences
    return text


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Recognize handwritten characters in a document image.")
    parser.add_argument("image", help="Path to the input image")
    parser.add_argument("--weights", default="EMNIST-Classifier/model_hub/fine_tuned/vggnet_tuned.pth",
                        help="Path to model weights")
    parser.add_argument("--annotate", action="store_true",
                        help="Save a copy of the image with bounding boxes and predictions overlaid")
    parser.add_argument("--confidence", action="store_true",
                        help="Print per-character confidence scores")
    parser.add_argument("--unknown-threshold", type=float, default=0.0,
                        help="Replace low-confidence characters with '?' (0.0 disables)")
    parser.add_argument("--threshold-mode", choices=["single", "multi"], default="single",
                        help="Use one threshold ('single') or best-of-multiple thresholds ('multi')")
    parser.add_argument("--adaptive-block-size", type=int, default=31,
                        help="Adaptive threshold neighborhood size (must be odd)")
    parser.add_argument("--adaptive-c", type=int, default=12,
                        help="Adaptive threshold subtraction constant")
    parser.add_argument("--morph-kernel-size", type=int, default=1,
                        help="Morphology open kernel size (1 disables morphology)")
    parser.add_argument("--letter-spacing", type=int, default=0,
                        help="Number of spaces inserted between adjacent characters")
    parser.add_argument("--word-gap-scale", type=float, default=1.8,
                        help="Lower values insert word spaces more aggressively")
    args = parser.parse_args()

    if args.adaptive_block_size < 3 or args.adaptive_block_size % 2 == 0:
        raise ValueError("--adaptive-block-size must be an odd integer >= 3")
    if args.morph_kernel_size < 1 or args.morph_kernel_size % 2 == 0:
        raise ValueError("--morph-kernel-size must be an odd integer >= 1")
    if args.letter_spacing < 0:
        raise ValueError("--letter-spacing must be >= 0")
    if args.word_gap_scale <= 0:
        raise ValueError("--word-gap-scale must be > 0")

    recognized = recognize(
        args.image,
        weights=args.weights,
        save_annotated=args.annotate,
        return_confidences=args.confidence,
        unknown_threshold=args.unknown_threshold,
        threshold_mode=args.threshold_mode,
        adaptive_block_size=args.adaptive_block_size,
        adaptive_c=args.adaptive_c,
        letter_spacing=args.letter_spacing,
        word_gap_scale=args.word_gap_scale,
        morph_kernel_size=args.morph_kernel_size,
    )

    if args.confidence:
        text, confidences = recognized
    else:
        text = recognized

    print("Recognized text:")
    print(text)

    if args.confidence:
        print("Per-character confidence:")
        print(" ".join(f"{c}:{p:.2f}" for c, p in zip(text, confidences)))
