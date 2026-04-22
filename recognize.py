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


def segment(image_path: str) -> tuple[np.ndarray, list[tuple]]:
    """Return (original BGR image, list of (x, y, w, h) bounding boxes) sorted left-to-right, top-to-bottom."""
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Could not open image: {image_path}")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    boxes = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        # Filter out noise (very small regions)
        if w > 5 and h > 10:
            boxes.append((x, y, w, h))

    # Sort top-to-bottom by row, then left-to-right within each row
    boxes.sort(key=lambda b: (b[1] // 20, b[0]))
    return img, boxes


def classify(model: VGGNet, crop: np.ndarray) -> str:
    pil = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY))
    tensor = TRANSFORM(pil).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        output = model(tensor)
    idx = output.argmax(dim=1).item()
    return EMNIST_CLASSES[idx]


def recognize(image_path: str, weights: str = "EMNIST-Classifier/model_hub/fine_tuned/vggnet_tuned.pth",
              save_annotated: bool = False) -> str:
    model = load_model(weights)
    img, boxes = segment(image_path)

    result = []
    for x, y, w, h in boxes:
        crop = img[y:y+h, x:x+w]
        char = classify(model, crop)
        result.append(char)

        if save_annotated:
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 1)
            cv2.putText(img, char, (x, y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    text = "".join(result)

    if save_annotated:
        out_path = pathlib.Path(image_path).stem + "_annotated.png"
        cv2.imwrite(out_path, img)
        print(f"Annotated image saved to {out_path}")

    return text


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Recognize handwritten characters in a document image.")
    parser.add_argument("image", help="Path to the input image")
    parser.add_argument("--weights", default="EMNIST-Classifier/model_hub/fine_tuned/vggnet_tuned.pth",
                        help="Path to model weights")
    parser.add_argument("--annotate", action="store_true",
                        help="Save a copy of the image with bounding boxes and predictions overlaid")
    args = parser.parse_args()

    recognized = recognize(args.image, weights=args.weights, save_annotated=args.annotate)
    print("Recognized text:")
    print(recognized)
