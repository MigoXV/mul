import PIL
from pathlib import Path
from PIL import Image

input_path = "data-bin/celeba/CelebAMask-HQ/CelebA-HQ-img"
input_path = Path(input_path)
input_path = input_path.iterdir().__next__()

print(input_path)

image = Image.open(input_path)
print(image.size)
image = image.resize((256, 256))
print(image.size)
image.save("data-bin/0.jpg")