import os, json, glob, random, argparse

def iter_json_files(root):
    pattern = os.path.join(root, "group_*", "annotation", "*.json")
    return sorted(glob.glob(pattern))

def make_rel_image_path(json_path):
    # ...\group_010120\annotation\010120_000002.json
    # -> group_010120\rgb\010120_000002.png
    group_dir = os.path.dirname(os.path.dirname(json_path))   # group_010120
    fname = os.path.splitext(os.path.basename(json_path))[0] + ".png"
    img_path = os.path.join(group_dir, "rgb", fname)
    return img_path

def main(root, out_dir, val_ratio, seed):
    random.seed(seed)
    os.makedirs(out_dir, exist_ok=True)

    samples_by_group = {}
    for jp in iter_json_files(root):
        group = os.path.basename(os.path.dirname(os.path.dirname(jp)))  # group_010120
        with open(jp, "r", encoding="utf-8") as f:
            data = json.load(f)

        # 이미지 메타
        w = data["images"]["width"]
        h = data["images"]["height"]

        img_abs = make_rel_image_path(jp)  # absolute
        # root 기준 상대경로로 저장 (path만 바꾸게 만들기 위함)
        img_rel = os.path.relpath(img_abs, root).replace("\\", "/")

        # annotations 여러 개면 각각 한 줄 샘플로 풀어냄
        for i, ann in enumerate(data["annotations"]):
            cap = ann["referring_expression"]
            bbox = ann["bbox"]  # [x,y,w,h]
            sample = {
                "image": img_rel,
                "width": w,
                "height": h,
                "caption": cap,
                "bbox_xywh": bbox,
                "image_id": data["images"]["id"],
                "ann_id": i,
                "category_id": ann.get("category_id", -1),
            }
            samples_by_group.setdefault(group, []).append(sample)

    groups = list(samples_by_group.keys())
    random.shuffle(groups)

    n_val = max(1, int(len(groups) * val_ratio))
    val_groups = set(groups[:n_val])
    train_groups = set(groups[n_val:])

    train_samples, val_samples = [], []
    for g, s in samples_by_group.items():
        (val_samples if g in val_groups else train_samples).extend(s)

    def write_jsonl(path, samples):
        with open(path, "w", encoding="utf-8") as f:
            for s in samples:
                f.write(json.dumps(s, ensure_ascii=False) + "\n")

    write_jsonl(os.path.join(out_dir, "train.jsonl"), train_samples)
    write_jsonl(os.path.join(out_dir, "val.jsonl"), val_samples)

    print("Done.")
    print("train samples:", len(train_samples))
    print("val samples:", len(val_samples))
    print("val groups:", len(val_groups), "train groups:", len(train_groups))

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="원본 데이터 root (group_*가 있는 폴더)")
    ap.add_argument("--out", default="mdetr_annotations", help="출력 폴더")
    ap.add_argument("--val_ratio", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    main(args.root, args.out, args.val_ratio, args.seed)
