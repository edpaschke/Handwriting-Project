import random
import glob
from PIL import Image

from classify_and_store import _folder_name

base = "default_writings/generated_chars"

def pick(letter):
    folder = _folder_name(letter)
    files = glob.glob(f"{base}/{folder}/*.png")
    return random.choice(files)

images = [Image.open(pick(ch)).convert("RGB") for ch in ("C", "a", "t")]

h = max(img.height for img in images)
images = [img.resize((img.width, h)) for img in images]

total_w = sum(img.width for img in images)
result = Image.new("RGB", (total_w, h), (255, 255, 255))

x = 0
for img in images:
    result.paste(img, (x, 0))
    x += img.width

result.save("cat.png")
print("Saved cat.png")
