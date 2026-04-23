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

# The training pkl files contain raw EMNIST data which is stored mirrored
# horizontally. The model learned mirrored characters, so we must flip every
# crop before inference to match what the model was trained on.
TRANSFORM = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.RandomHorizontalFlip(p=1.0),   # deterministic mirror — matches training data
    transforms.Lambda(lambda img: transforms.functional.rotate(img, 90)),  # CCW 90° to fix CW rotation
    transforms.ToTensor(),
    transforms.Normalize((0.1736,), (0.3248,)),
])


def load_model(weights: str = "EMNIST-Classifier/model_hub/fine_tuned/vggnet_tuned.pth") -> VGGNet:
    model = VGGNet()
    model.load_state_dict(torch.load(weights, map_location=DEVICE))
    model.to(DEVICE).eval()
    return model


def _group_boxes_into_lines(boxes):
    if not boxes:
        return []

    heights = np.array([h for _, _, _, h in boxes], dtype=np.float32)
    line_threshold = max(12, int(np.median(heights) * 0.7))

    boxes_by_center = sorted(boxes, key=lambda b: b[1] + b[3] // 2)
    lines, line_centers = [], []

    for box in boxes_by_center:
        cy = box[1] + box[3] // 2
        if not lines:
            lines.append([box]); line_centers.append(cy); continue

        idx = int(np.argmin([abs(cy - lc) for lc in line_centers]))
        if abs(cy - line_centers[idx]) <= line_threshold:
            lines[idx].append(box)
            centers = [b[1] + b[3] // 2 for b in lines[idx]]
            line_centers[idx] = int(sum(centers) / len(centers))
        else:
            lines.append([box]); line_centers.append(cy)

    line_pairs = sorted(zip(lines, line_centers), key=lambda item: item[1])
    return [sorted(lb, key=lambda b: b[0]) for lb, _ in line_pairs]


def _build_binary_images(
    gray,
    adaptive_block_size=31,
    adaptive_c=18,
    morph_kernel_size=1,
    remove_horizontal_lines=True,
    use_background_filter=True,
    background_blur_kernel=41,
    suppress_bleedthrough=True,
    bleedthrough_diff_threshold=40,
):
    if suppress_bleedthrough:
        bg_large = cv2.GaussianBlur(gray, (background_blur_kernel, background_blur_kernel), 0)
        diff = bg_large.astype(np.int16) - gray.astype(np.int16)
        weak = diff < bleedthrough_diff_threshold
        cleaned = gray.copy().astype(np.int16)
        cleaned[weak] = np.minimum(255, bg_large.astype(np.int16)[weak] + 5)
        gray = np.clip(cleaned, 0, 255).astype(np.uint8)

    if use_background_filter:
        bg = cv2.GaussianBlur(gray, (background_blur_kernel, background_blur_kernel), 0)
        normalized = cv2.divide(gray, bg, scale=255)
        base = cv2.normalize(normalized, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    else:
        base = gray

    blur = cv2.GaussianBlur(base, (3, 3), 0)
    otsu = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    adaptive_mean = cv2.adaptiveThreshold(
        blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV,
        adaptive_block_size, adaptive_c)
    adaptive_gaussian = cv2.adaptiveThreshold(
        blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,
        adaptive_block_size, adaptive_c)

    clean = ((lambda img: cv2.morphologyEx(img, cv2.MORPH_OPEN,
              np.ones((morph_kernel_size, morph_kernel_size), dtype=np.uint8)))
             if morph_kernel_size > 1 else (lambda img: img))

    def _remove_lines(img):
        if not remove_horizontal_lines:
            return img
        width = max(25, gray.shape[1] // 18)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (width, 1))
        return cv2.subtract(img, cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel))

    detect = cv2.bitwise_or(otsu, adaptive_gaussian)
    return {
        "detect":            _remove_lines(clean(detect)),
        "otsu":              _remove_lines(clean(otsu)),
        "adaptive_mean":     _remove_lines(clean(adaptive_mean)),
        "adaptive_gaussian": _remove_lines(clean(adaptive_gaussian)),
    }


def segment(
    image_path,
    adaptive_block_size=31,
    adaptive_c=18,
    morph_kernel_size=1,
    remove_horizontal_lines=True,
    use_background_filter=True,
    background_blur_kernel=41,
    suppress_bleedthrough=True,
    bleedthrough_diff_threshold=40,
    min_height_fraction=0.03,
    max_count=200,
):
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Could not open image: {image_path}")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    binaries = _build_binary_images(
        gray,
        adaptive_block_size=adaptive_block_size,
        adaptive_c=adaptive_c,
        morph_kernel_size=morph_kernel_size,
        remove_horizontal_lines=remove_horizontal_lines,
        use_background_filter=use_background_filter,
        background_blur_kernel=background_blur_kernel,
        suppress_bleedthrough=suppress_bleedthrough,
        bleedthrough_diff_threshold=bleedthrough_diff_threshold,
    )
    detect_binary = cv2.morphologyEx(
        binaries["detect"], cv2.MORPH_CLOSE, np.ones((2, 2), dtype=np.uint8))

    candidates = []
    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(detect_binary, connectivity=8)
    for i in range(1, num_labels):
        x, y, w, h, area = stats[i]
        if w > 1 and h > 4 and area > 6:
            candidates.append((int(x), int(y), int(w), int(h), int(area)))

    boxes = []
    if candidates:
        hs    = np.array([h for _, _, _, h, _ in candidates], dtype=np.float32)
        areas = np.array([a for _, _, _, _, a in candidates], dtype=np.float32)
        h_ref    = float(np.percentile(hs,    90))
        area_ref = float(np.percentile(areas, 90))
        img_h, img_w = gray.shape[:2]

        abs_min_h = img_h * min_height_fraction
        min_h    = max(abs_min_h, 0.35 * h_ref)
        max_h    = min(0.80 * img_h, 2.0 * h_ref)
        min_area = max(100.0, 0.40 * area_ref)
        max_area = min(0.05 * img_h * img_w, max(500.0, 5.0 * area_ref))

        for x, y, w, h, area in candidates:
            aspect  = w / max(h, 1)
            density = area / max(1.0, float(w * h))
            touches_border = (x <= 2 or y <= 2 or
                              (x + w) >= (img_w - 2) or (y + h) >= (img_h - 2))
            if touches_border:                         continue
            if h < min_h or h > max_h:                continue
            if w > 0.25 * img_w:                      continue
            if area < min_area or area > max_area:    continue
            if aspect < 0.10 or aspect > 3.0:         continue
            if density < 0.08:                         continue
            boxes.append((x, y, w, h))

        if len(boxes) > max_count:
            boxes = sorted(boxes, key=lambda b: b[2]*b[3], reverse=True)[:max_count]

    return img, binaries, _group_boxes_into_lines(boxes)


def _normalize_glyph(binary_crop, output_size=32):
    ys, xs = np.where(binary_crop > 0)
    if len(xs) == 0 or len(ys) == 0:
        return Image.fromarray(np.zeros((output_size, output_size), dtype=np.uint8))

    x1, x2 = int(xs.min()), int(xs.max()) + 1
    y1, y2 = int(ys.min()), int(ys.max()) + 1
    glyph = binary_crop[y1:y2, x1:x2]
    h, w = glyph.shape
    side = max(h, w)
    pad = max(2, side // 6)
    cs = side + 2 * pad
    canvas = np.zeros((cs, cs), dtype=np.uint8)
    canvas[(cs-h)//2:(cs-h)//2+h, (cs-w)//2:(cs-w)//2+w] = glyph
    return Image.fromarray(
        cv2.resize(canvas, (output_size, output_size), interpolation=cv2.INTER_AREA))


def classify(model: VGGNet, binary_crop: np.ndarray) -> tuple[str, float]:
    """Classify a character crop using the horizontally-flipped transform that matches training data."""
    pil = _normalize_glyph(binary_crop, output_size=32)
    tensor = TRANSFORM(pil).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        output = model(tensor)
        probs  = torch.softmax(output, dim=1)
    idx  = output.argmax(dim=1).item()
    return EMNIST_CLASSES[idx], probs[0, idx].item()


def _line_to_text(line_items, letter_spacing=0, word_gap_scale=1.8):
    if not line_items:
        return "", []
    if len(line_items) == 1:
        _, ch, conf = line_items[0]
        return ch, [conf]

    gaps, widths = [], []
    for i, (box, _, _) in enumerate(line_items):
        x, _, w, _ = box
        widths.append(w)
        if i > 0:
            px, _, pw, _ = line_items[i-1][0]
            gaps.append(max(0, x - (px + pw)))

    median_gap   = float(np.median(gaps))   if gaps   else 0.0
    median_width = float(np.median(widths)) if widths else 0.0
    space_threshold = max(6.0, median_gap * word_gap_scale, median_width * 0.25)

    chars, confidences = [], []
    for i, (_, ch, conf) in enumerate(line_items):
        if i > 0:
            px, _, pw, _ = line_items[i-1][0]
            x = line_items[i][0][0]
            if max(0, x - (px + pw)) >= space_threshold:
                chars.append(" "); confidences.append(1.0)
        chars.append(ch); confidences.append(conf)
        if i < len(line_items) - 1 and letter_spacing > 0:
            chars.extend([" "] * letter_spacing)
            confidences.extend([1.0] * letter_spacing)

    return "".join(chars), confidences


def recognize(
    image_path: str,
    weights: str             = "EMNIST-Classifier/model_hub/fine_tuned/vggnet_tuned.pth",
    save_annotated: bool     = True,
    return_confidences: bool = True,
    unknown_threshold: float = 0.15,
    threshold_mode: str      = "single",
    adaptive_block_size: int = 31,
    adaptive_c: int          = 18,
    letter_spacing: int      = 0,
    word_gap_scale: float    = 1.8,
    morph_kernel_size: int   = 1,
    remove_horizontal_lines: bool = True,
    annotations_dir: str     = "Pracmages/annotations",
    use_background_filter: bool  = True,
    background_blur_kernel: int  = 41,
    suppress_bleedthrough: bool  = True,
    bleedthrough_diff_threshold: int = 40,
    min_height_fraction: float   = 0.03,
):
    model = load_model(weights)
    img, binaries, lines = segment(
        image_path,
        adaptive_block_size=adaptive_block_size,
        adaptive_c=adaptive_c,
        morph_kernel_size=morph_kernel_size,
        remove_horizontal_lines=remove_horizontal_lines,
        use_background_filter=use_background_filter,
        background_blur_kernel=background_blur_kernel,
        suppress_bleedthrough=suppress_bleedthrough,
        bleedthrough_diff_threshold=bleedthrough_diff_threshold,
        min_height_fraction=min_height_fraction,
    )

    if threshold_mode not in {"single", "multi"}:
        raise ValueError("threshold_mode must be 'single' or 'multi'")

    candidate_binaries = ([binaries["detect"]] if threshold_mode == "single"
                          else [binaries["otsu"], binaries["adaptive_mean"],
                                binaries["adaptive_gaussian"], binaries["detect"]])

    result_lines, confidences = [], []
    for line in lines:
        line_items = []
        for x, y, w, h in line:
            best_char, best_conf = "?", -1.0
            for binary in candidate_binaries:
                char, conf = classify(model, binary[y:y+h, x:x+w])
                if conf > best_conf:
                    best_char, best_conf = char, conf

            char, conf = best_char, best_conf
            if conf < unknown_threshold:
                char = "?"
            line_items.append(((x, y, w, h), char, conf))

            if save_annotated:
                cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 1)
                cv2.putText(img, f"{char}:{conf:.2f}", (x, max(0, y-4)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)

        line_text, line_conf = _line_to_text(
            line_items, letter_spacing=letter_spacing, word_gap_scale=word_gap_scale)
        if line_text:
            result_lines.append(line_text)
            confidences.extend(line_conf)

    text = "\n".join(result_lines)

    if save_annotated:
        out_path = pathlib.Path(image_path).stem + "_annotated.png"
        cv2.imwrite(out_path, img)
        print(f"Annotated image saved to {out_path}")
        ann_dir = pathlib.Path(__file__).resolve().parent / annotations_dir
        ann_dir.mkdir(parents=True, exist_ok=True)
        ann_path = ann_dir / f"{pathlib.Path(image_path).stem}_annotated.png"
        cv2.imwrite(str(ann_path), img)
        print(f"Annotated image also saved to {ann_path}")

    return (text, confidences) if return_confidences else text


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("image")
    parser.add_argument("--weights",                  default="EMNIST-Classifier/model_hub/fine_tuned/vggnet_tuned.pth")
    parser.add_argument("--annotate",                 dest="annotate", action="store_true")
    parser.add_argument("--no-annotate",              dest="annotate", action="store_false")
    parser.set_defaults(annotate=True)
    parser.add_argument("--confidence",               action="store_true")
    parser.add_argument("--unknown-threshold",        type=float, default=0.15)
    parser.add_argument("--threshold-mode",           choices=["single", "multi"], default="single")
    parser.add_argument("--adaptive-block-size",      type=int,   default=31)
    parser.add_argument("--adaptive-c",               type=int,   default=18)
    parser.add_argument("--morph-kernel-size",        type=int,   default=1)
    parser.add_argument("--keep-horizontal-lines",    action="store_true")
    parser.add_argument("--no-bg-filter",             action="store_true")
    parser.add_argument("--bg-blur-kernel",           type=int,   default=41)
    parser.add_argument("--letter-spacing",           type=int,   default=0)
    parser.add_argument("--word-gap-scale",           type=float, default=1.8)
    parser.add_argument("--annotations-dir",          default="Pracmages/annotations")
    parser.add_argument("--no-suppress-bleedthrough", action="store_true")
    parser.add_argument("--bleedthrough-threshold",   type=int,   default=40)
    parser.add_argument("--min-height-fraction",      type=float, default=0.03)
    args = parser.parse_args()

    for name, val, check in [
        ("--adaptive-block-size", args.adaptive_block_size, lambda v: v >= 3 and v % 2 == 1),
        ("--morph-kernel-size",   args.morph_kernel_size,   lambda v: v >= 1 and v % 2 == 1),
        ("--bg-blur-kernel",      args.bg_blur_kernel,      lambda v: v >= 3 and v % 2 == 1),
        ("--letter-spacing",      args.letter_spacing,      lambda v: v >= 0),
        ("--word-gap-scale",      args.word_gap_scale,      lambda v: v > 0),
    ]:
        if not check(val):
            raise ValueError(f"Invalid value for {name}: {val}")

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
        remove_horizontal_lines=not args.keep_horizontal_lines,
        annotations_dir=args.annotations_dir,
        use_background_filter=not args.no_bg_filter,
        background_blur_kernel=args.bg_blur_kernel,
        suppress_bleedthrough=not args.no_suppress_bleedthrough,
        bleedthrough_diff_threshold=args.bleedthrough_threshold,
        min_height_fraction=args.min_height_fraction,
    )

    if args.confidence:
        text, confidences = recognized
    else:
        text = recognized

    print("Recognized text:")
    print(text)
    if args.confidence:
        print("\nPer-character confidence:")
        print(" ".join(f"{c}:{p:.2f}" for c, p in zip(text, confidences)))