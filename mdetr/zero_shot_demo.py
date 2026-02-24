import pathlib
import torch
from PIL import Image
import torchvision.transforms as T

import pathlib
import os

# 리눅스 환경에서 WindowsPath를 요청할 경우 PosixPath로 대체하게 함
if os.name != 'nt':
    pathlib.WindowsPath = pathlib.PosixPath

# 🔥 Windows에서 Linux checkpoint(PosixPath) 로드 우회
pathlib.PosixPath = pathlib.WindowsPath


# --------------------
# 설정
# --------------------
IMAGE_PATH = "/root/mdetr/black-isolated-car_23-2151852894.avif"
TEXT_QUERY = "a car"   # 바꿔도 됨: "a car", "a dog", "the red object"

CKPT_PATH = "/root/mdetr/checkpoints/pretrained_resnet101_checkpoint.pth"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --------------------
# 모델 로드
# --------------------
model = torch.hub.load(
    ".", 
    "mdetr_resnet101", 
    pretrained=False, 
    source="local"
)

checkpoint = torch.load(CKPT_PATH, map_location="cpu", weights_only=False)
state_dict = checkpoint.get("model", checkpoint)
model.load_state_dict(state_dict, strict=False)

model.to(DEVICE)
model.eval()

print("Model loaded successfully")

# --------------------
# 이미지 로드
# --------------------
transform = T.Compose([T.ToTensor()])
image = Image.open(IMAGE_PATH).convert("RGB")
image_tensor = transform(image).to(DEVICE)

# --------------------
# Zero-shot forward
# --------------------
with torch.no_grad():
    outputs = model([image_tensor], [TEXT_QUERY])

# --------------------
# 출력 확인
# --------------------
print("Zero-shot inference success!")
print("Output type:", type(outputs))

if isinstance(outputs, dict):
    for k, v in outputs.items():
        if torch.is_tensor(v):
            print(f"{k}: {v.shape}")
        elif isinstance(v, (list, tuple)) and len(v) > 0 and torch.is_tensor(v[0]):
            print(f"{k}: list[{len(v)}], tensor shape {v[0].shape}")
