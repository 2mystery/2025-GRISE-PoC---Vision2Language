import os, json
from PIL import Image
import torch
from torch.utils.data import Dataset

class AIHubJSONL(Dataset):
    def __init__(self, root, jsonl_path, transforms=None):
        self.root = root
        self.transforms = transforms
        self.samples = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                self.samples.append(json.loads(line))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        img_path = os.path.join(self.root, s["image"].replace("/", os.sep))
        img = Image.open(img_path).convert("RGB")

        x, y, w, h = s["bbox_xywh"]
        boxes = torch.tensor([[x, y, x + w, y + h]], dtype=torch.float32)  # xyxy
        labels = torch.ones((1,), dtype=torch.int64)

        target = {
            "boxes": boxes,
            "labels": labels,
            "caption": s["caption"],
            "image_id": torch.tensor([s.get("image_id", idx)]),
            "orig_size": torch.tensor([s["height"], s["width"]]),
            "size": torch.tensor([s["height"], s["width"]]),
        }

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target
