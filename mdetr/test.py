import argparse
import json
import os
import random
import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import RobertaTokenizerFast
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import importlib.util

# ---------------------------------------------------------
# 기존 학습 코드에서 가져온 모듈들
# ---------------------------------------------------------
from main import get_args_parser as base_get_args_parser
import util.misc as utils
from models import build_model
from datasets.coco import create_positive_map

# 사용자 데이터셋 로드
DATASET_PY = "/root/dataset/dataset.py"
spec = importlib.util.spec_from_file_location("mdetr_custom_dataset", DATASET_PY)
md = importlib.util.module_from_spec(spec)
spec.loader.exec_module(md)
MDETRDataset = md.MDETRDataset

# ---------------------------------------------------------
# 시각화 및 유틸리티 함수
# ---------------------------------------------------------
def build_simple_transform():
    from torchvision import transforms
    to_tensor = transforms.ToTensor()
    def _tf(image, target):
        image = to_tensor(image)
        return image, target
    return _tf

def mdetr_collate_with_captions(batch):
    images, captions, targets = zip(*batch)
    base = utils.collate_fn(False, list(zip(images, targets)))
    base["captions"] = list(captions)
    return base

def ensure_tokens_positive(target, caption):
    if "tokens_positive" in target and target["tokens_positive"] is not None:
        return target
    n_boxes = int(target["boxes"].shape[0]) if "boxes" in target else 1
    L = len(caption)
    target["tokens_positive"] = [ [[0, L]] for _ in range(n_boxes) ]
    return target

# ---------------------------------------------------------
# 시각화 함수 (Tensor -> Image 변환 추가)
# ---------------------------------------------------------
def plot_visualization(image_tensor, model_outputs, caption, save_path):
    # 1. Tensor -> PIL Image 변환
    img_np = image_tensor.permute(1, 2, 0).cpu().numpy()
    
    # 2. Plotting
    plt.figure(figsize=(16, 10))
    plt.imshow(img_np)
    ax = plt.gca()
    
    # 제목에 캡션 표시
    plt.title(f"Prompt: {caption}", fontsize=15)

    # 3. 예측값 그리기
    probas = 1 - model_outputs['pred_logits'].softmax(-1)[0, :, -1].cpu()
    keep = (probas > 0.7).cpu() # Confidence Threshold 0.7
    
    if keep.sum() == 0:
        print(f"⚠️ No box detected for: {save_path}")
        plt.close()
        return

    boxes = model_outputs['pred_boxes'][0, keep].cpu()
    scores = probas[keep]
    
    # Box 좌표 변환 (cx, cy, w, h) -> (xmin, ymin, w, h) for plotting
    H, W, _ = img_np.shape
    
    for score, box in zip(scores, boxes):
        cx, cy, w, h = box.tolist()
        # 상대좌표(0~1)를 픽셀좌표로 변환
        cx, cy, w, h = cx * W, cy * H, w * W, h * H
        xmin = cx - 0.5 * w
        ymin = cy - 0.5 * h
        
        # 사각형 그리기
        rect = patches.Rectangle((xmin, ymin), w, h, linewidth=3, edgecolor='red', facecolor='none')
        ax.add_patch(rect)
        ax.text(xmin, ymin, f"{score:.2f}", color='white', fontsize=12, bbox=dict(facecolor='red', alpha=0.5))

    plt.axis('off')
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.close()
    print(f"🖼️ Saved visualization: {save_path}")

# ---------------------------------------------------------
# Main Test Logic
# ---------------------------------------------------------
def main():
    # 인자 설정
    parser = base_get_args_parser()
    parser.add_argument("--val_json", required=True)
    # --resume 은 base_get_args_parser에 이미 있으므로 중복 정의 삭제함
    args = parser.parse_args()

    # 1. Config 로드
    if args.dataset_config is not None:
        with open(args.dataset_config, "r") as f:
            cfg = json.load(f)
        for k, v in cfg.items():
            setattr(args, k, v)

    # 2. Output Dir 설정 (테스트 결과 저장용)
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        vis_dir = os.path.join(args.output_dir, "visualizations")
        os.makedirs(vis_dir, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"🚀 Start Testing on {device}...")

    # 3. 모델 빌드 & 체크포인트 로드
    model, criterion, contrastive_criterion, qa_criterion, weight_dict = build_model(args)
    model.to(device)

    if args.resume:
        print(f"📥 Loading Checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location="cpu")
        if "model" in checkpoint:
            model.load_state_dict(checkpoint["model"], strict=False)
        else:
            model.load_state_dict(checkpoint, strict=False)
    else:
        print("⚠️ Warning: No checkpoint loaded! Testing with random weights.")

    model.eval()

    # 4. 데이터셋 로드
    val_ds = MDETRDataset(args.val_json, transform=build_simple_transform())
    val_loader = DataLoader(
        val_ds,
        batch_size=1, # 시각화를 위해 배치 1로 설정
        shuffle=False, # 순서대로 평가
        num_workers=2,
        collate_fn=mdetr_collate_with_captions,
        drop_last=False,
    )

    print(f"✅ Validation Dataset Len: {len(val_ds)}")
    
    # 5. Tokenizer
    tokenizer = RobertaTokenizerFast.from_pretrained(args.text_encoder_type)

    # 6. 평가 루프 (Loss 계산 + 시각화)
    total_loss = 0.0
    count = 0
    
    print("=== Evaluation Start ===")
    
    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            samples = batch["samples"].to(device)
            captions = batch["captions"]
            targets = list(batch["targets"])

            # Target 전처리 (Loss 계산용)
            new_targets = []
            for cap, t in zip(captions, targets):
                t = ensure_tokens_positive(t, cap)
                tt = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in t.items()}
                new_targets.append(tt)
            targets = new_targets

            # Positive Map 생성
            positive_maps = []
            for cap, t in zip(captions, targets):
                tokenized = tokenizer(cap, padding="max_length", truncation=True, max_length=256, return_offsets_mapping=True)
                pm = create_positive_map(tokenized, t["tokens_positive"])
                positive_maps.append(pm)
            positive_map = torch.cat(positive_maps, dim=0).to(device)

            # Forward
            memory_cache = model(samples, captions, encode_and_save=True)
            outputs = model(samples, captions, encode_and_save=False, memory_cache=memory_cache)

            # Loss 계산
            loss_dict = criterion(outputs, targets, positive_map)
            losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)
            
            total_loss += losses.item()
            count += 1
            
            if i % 10 == 0:
                print(f"[{i}/{len(val_loader)}] Current Loss: {losses.item():.4f}")

            # 🌟 시각화 이미지 저장 (처음 20장만 저장)
            if i < 20:
                save_path = os.path.join(vis_dir, f"result_{i:03d}.png")
                plot_visualization(samples.tensors[0], outputs, captions[0], save_path)

    if count > 0:
        avg_loss = total_loss / count
        print("=" * 30)
        print(f"🏆 Final Average Validation Loss: {avg_loss:.4f}")
        print(f"📂 Visualizations saved to: {vis_dir}")
        print("=" * 30)
    else:
        print("⚠️ No data evaluated.")

if __name__ == "__main__":
    main()