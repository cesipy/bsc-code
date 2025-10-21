import torchvision
from torchvision.transforms import Compose
import os
from PIL import Image


SIZE = (224, 224)


resize_transform = Compose([
    torchvision.transforms.Resize(SIZE),
])

def main():
    img_path = "/mnt/ifi/iis/cedric.sillaber/hateful_memes/img_old"
    dest_path ="/mnt/ifi/iis/cedric.sillaber/hateful_memes/img"

    for filename in os.listdir(img_path):
        if filename.endswith(".png"):
            full_filename = os.path.join(img_path, filename)
            print(full_filename)

            img = Image.open(full_filename).convert("RGB")
            img = resize_transform(img)

            save_path = os.path.join(dest_path, filename)
            img.save(save_path)




main()
