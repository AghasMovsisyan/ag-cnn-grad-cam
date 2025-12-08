# dataset.py
import os
from PIL import Image, UnidentifiedImageError
from torch.utils.data import Dataset
import torchvision.transforms as T


class RadioDataset(Dataset):
    def __init__(self, root_dir, train=True):
        self.root = root_dir
        self.files = []
        self.labels = []

        classes = sorted(os.listdir(root_dir))

        for label, cls in enumerate(classes):
            cls_dir = os.path.join(root_dir, cls)
            for f in os.listdir(cls_dir):
                if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp")):
                    fullpath = os.path.join(cls_dir, f)
                    try:
                        Image.open(fullpath)
                        self.files.append(fullpath)
                        self.labels.append(label)
                    except UnidentifiedImageError:
                        print("Skipping corrupted image:", fullpath)
                    except:
                        print("Unknown error reading:", fullpath)

        if train:
            self.transform = T.Compose(
                [
                    T.ToTensor(),
                ]
            )
        else:
            self.transform = T.Compose(
                [
                    T.ToTensor(),
                ]
            )

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img_path = self.files[idx]

        try:
            img = Image.open(img_path)
        except:
            print("Bad image during __getitem__:", img_path)
            return self.__getitem__((idx + 1) % len(self.files))

        img = self.transform(img)
        label = self.labels[idx]
        return img, label
