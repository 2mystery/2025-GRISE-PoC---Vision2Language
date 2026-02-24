import torch
from torch.utils.data import Dataset
import json
import os
from PIL import Image

class MDETRDataset(Dataset):
    def __init__(self, json_file, transform=None):
        """
        Args:
            json_file (string): train.json 경로
            transform (callable, optional): 이미지 사이즈 조절 등을 위한 전처리 함수
        """
        print(f"📂 데이터 로딩 중: {json_file}")
        with open(json_file, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        
        # 1. 이미지 로드 (파일 경로에서 이미지를 엽니다)
        img_path = item['file_name']
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"⚠️ 이미지 로드 에러: {img_path} ({e})")
            # 에러 발생 시 0번 인덱스 데이터로 대체 (학습 중단 방지)
            return self.__getitem__(0)

        # 2. 정답(Target) 준비
        # 전처리한 bbox는 이미 [cx, cy, w, h] (0~1 정규화) 되어 있음(MDETR에 적합)
        bbox = torch.tensor(item['bbox'], dtype=torch.float32)
        
        # 라벨 (카테고리 ID)
        label = torch.tensor(item['label'], dtype=torch.long) 
        
        # 질문 (Caption) - 모델이 "이게 뭐냐?"고 물어볼 텍스트
        caption = item['caption'] 

        target = {
            "boxes": bbox.unsqueeze(0), # (1, 4) 형태로 차원 추가
            "labels": label.unsqueeze(0),
        }

        # 3. 이미지 전처리 (Resize 등 - 팀원이 transforms 설정해서 넣을 것임)
        if self.transform:
            image, target = self.transform(image, target)
            
        return image, caption, target

'''
# 1. 데이터 로더 가져오기
from dataset import MDETRDataset # dataset.py 파일
import torch
from torch.utils.data import DataLoader

# 2. 데이터셋 연결 
train_dataset = MDETRDataset("/root/dataset/train.json", transform=make_transforms("train"))
val_dataset = MDETRDataset("/root/dataset/val.json", transform=make_transforms("val"))

# 3. 배치 만들기 (Collate Function)
# MDETR은 이미지 크기가 다르면 합칠 때 에러가 날 수 있어서 해당 함수가 필요
def collate_fn(batch):
    return tuple(zip(*batch))

# 4. 로더 생성
data_loader_train = DataLoader(train_dataset, batch_size=2, shuffle=True, collate_fn=collate_fn)
data_loader_val = DataLoader(val_dataset, batch_size=2, shuffle=False, collate_fn=collate_fn)

print(f"학습 준비 완료! 데이터 {len(train_dataset)}개 로딩 성공")

# 이후에 모델 불러와서 loop 돌리면 끝
'''