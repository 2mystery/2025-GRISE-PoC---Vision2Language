import argparse, json, os, random
import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import RobertaTokenizerFast

from main import get_args_parser as base_get_args_parser
import util.misc as utils
from models import build_model
from datasets.coco import create_positive_map

import importlib.util

# ---------------------------------------------------------
# 사용자 데이터셋 로드 (수정 X)
# ---------------------------------------------------------
DATASET_PY = "/root/dataset/dataset.py"

spec = importlib.util.spec_from_file_location("mdetr_custom_dataset", DATASET_PY)
md = importlib.util.module_from_spec(spec)
spec.loader.exec_module(md)

MDETRDataset = md.MDETRDataset

# ---------------------------------------------------------
# Transform & Utils
# ---------------------------------------------------------
def build_simple_transform():
    from torchvision import transforms
    to_tensor = transforms.ToTensor()
    def _tf(image, target):
        image = to_tensor(image)
        return image, target
    return _tf

def get_args_parser():
    parser = base_get_args_parser()
    parser.add_argument("--train_json", required=True)
    parser.add_argument("--val_json", default="")
    return parser

def ensure_tokens_positive(target, caption):
    if "tokens_positive" in target and target["tokens_positive"] is not None:
        return target
    n_boxes = int(target["boxes"].shape[0]) if "boxes" in target else 1
    L = len(caption)
    target["tokens_positive"] = [ [[0, L]] for _ in range(n_boxes) ]
    return target

def mdetr_collate_with_captions(batch):
    images, captions, targets = zip(*batch)
    base = utils.collate_fn(False, list(zip(images, targets)))
    base["captions"] = list(captions)
    return base

# ---------------------------------------------------------
# Main Execution
# ---------------------------------------------------------
def main():
    args = get_args_parser().parse_args()

    # 1. Config 로드
    if args.dataset_config is not None:
        with open(args.dataset_config, "r") as f:
            cfg = json.load(f)
        for k, v in cfg.items():
            setattr(args, k, v)

    args.no_detection = False

    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)

    # 2. Seed 설정
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # 3. Dataset & Loader
    train_ds = MDETRDataset(args.train_json, transform=build_simple_transform())
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=mdetr_collate_with_captions,
        drop_last=True,
    )

    print("✅ dataset len =", len(train_ds))
    print("🚀 building model...")

    # 4. Model Build
    model, criterion, contrastive_criterion, qa_criterion, weight_dict = build_model(args)
    model.to(device)

    # 5. Tokenizer
    tokenizer_name = getattr(args, "text_encoder_type", "roberta-base")
    print("📝 using tokenizer:", tokenizer_name)
    tokenizer = RobertaTokenizerFast.from_pretrained(tokenizer_name)

    # 6. Optimizer 설정
    param_dicts = [
        {"params": [p for n, p in model.named_parameters()
                    if "backbone" not in n and "text_encoder" not in n and p.requires_grad]},
        {"params": [p for n, p in model.named_parameters()
                    if "backbone" in n and p.requires_grad], "lr": args.lr_backbone},
        {"params": [p for n, p in model.named_parameters()
                    if "text_encoder" in n and p.requires_grad], "lr": args.text_encoder_lr},
    ]
    optimizer = torch.optim.AdamW(param_dicts, lr=args.lr, weight_decay=args.weight_decay)

    # =======================================================
    # [핵심 수정] 체크포인트 로드 로직 (Resume 우선 -> Pretrained 차선)
    # =======================================================
    start_epoch = 0
    best_loss = float('inf') # 최고 성능 기록용
    
    # 경로 설정
    my_checkpoint_path = os.path.join(args.output_dir, "checkpoint.pth")
    pretrained_path = "/root/mdetr/checkpoints/pretrained_resnet101_checkpoint.pth"
    
    # 1) 내 학습 기록(Resume)이 있는지 확인
    if os.path.exists(my_checkpoint_path):
        print(f"♻️ Found existing checkpoint: {my_checkpoint_path}")
        print("   -> Resuming training from where it stopped...")
        
        checkpoint = torch.load(my_checkpoint_path, map_location="cpu")
        
        # 모델 복구
        if "model" in checkpoint:
            model.load_state_dict(checkpoint["model"], strict=False)
        else:
            model.load_state_dict(checkpoint, strict=False)
            
        # 옵티마이저, 에폭, Best Loss 복구
        if "optimizer" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer"])
        if "epoch" in checkpoint:
            start_epoch = checkpoint["epoch"] + 1
        if "best_loss" in checkpoint:
            best_loss = checkpoint["best_loss"]
            
        print(f"   -> Resume success! Start Epoch: {start_epoch}, Current Best Loss: {best_loss:.4f}")

    # 2) 없으면 Pretrained 로드
    elif os.path.exists(pretrained_path):
        print(f"📥 No resume file found. Loading Pre-trained weights: {pretrained_path}")
        checkpoint = torch.load(pretrained_path, map_location="cpu")
        if "model" in checkpoint:
            model.load_state_dict(checkpoint["model"], strict=False)
        else:
            model.load_state_dict(checkpoint, strict=False)
    else:
        print("⚠️ No checkpoint found at all. Training from scratch (Random Init).")

    # =======================================================

    print(f"=== Start Training (Epoch {start_epoch} ~ {args.epochs}) ===")

    # start_epoch부터 시작
    for epoch in range(start_epoch, args.epochs):
        model.train()
        running_loss = 0.0

        for it, batch in enumerate(train_loader):
            samples = batch["samples"].to(device)
            captions = batch["captions"]
            targets = list(batch["targets"])

            # Targets device 이동 및 보정
            new_targets = []
            for cap, t in zip(captions, targets):
                t = ensure_tokens_positive(t, cap)
                tt = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in t.items()}
                new_targets.append(tt)
            targets = new_targets

            # Positive Map
            positive_maps = []
            for cap, t in zip(captions, targets):
                tokenized_i = tokenizer(
                    cap, padding="max_length", truncation=True, max_length=256, return_offsets_mapping=True
                )
                pm = create_positive_map(tokenized_i, t["tokens_positive"])
                positive_maps.append(pm)
            positive_map = torch.cat(positive_maps, dim=0).to(device)

            # Forward
            memory_cache = model(samples, captions, encode_and_save=True)
            outputs = model(samples, captions, encode_and_save=False, memory_cache=memory_cache)

            # Loss
            loss_dict = {}
            loss_dict.update(criterion(outputs, targets, positive_map))

            losses = 0.0
            for k, v in loss_dict.items():
                if k in weight_dict:
                    losses += v * weight_dict[k]

            optimizer.zero_grad(set_to_none=True)
            losses.backward()
            if args.clip_max_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_max_norm)
            optimizer.step()

            running_loss += float(losses.item())
            if it % 20 == 0:
                print(f"[Epoch {epoch} | Iter {it}] loss={losses.item():.4f}")

        avg_loss = running_loss / max(1, len(train_loader))
        print(f"=== Epoch {epoch} done | avg_loss={avg_loss:.4f} ===")

        # =======================================================
        # [핵심 수정] 저장 로직 (Latest & Best 만 저장)
        # =======================================================
        if args.output_dir:
            # 저장할 데이터
            checkpoint_dict = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
                "args": vars(args),
                "loss": avg_loss,
                "best_loss": best_loss # 기존 best 유지 (갱신 전 값)
            }

            # 1. Latest 저장 (항상 덮어쓰기)
            latest_path = os.path.join(args.output_dir, "checkpoint.pth")
            torch.save(checkpoint_dict, latest_path)
            
            # 2. Best 저장 (성능 갱신 시에만 저장)
            if avg_loss < best_loss:
                print(f"🎉 Performance Improved! (Loss: {best_loss:.4f} -> {avg_loss:.4f})")
                best_loss = avg_loss
                
                # best_loss 갱신해서 저장
                checkpoint_dict["best_loss"] = best_loss 
                best_path = os.path.join(args.output_dir, "checkpoint_best.pth")
                torch.save(checkpoint_dict, best_path)
                print(f"💾 Saved Best Model to: {best_path}")
            else:
                print(f"Running Checkpoint Saved: {latest_path}")

    print("=== Training Finished ===")

if __name__ == "__main__":
    main()