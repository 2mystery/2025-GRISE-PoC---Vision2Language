import torch
import torch.nn.functional as F

# repo마다 util.misc에 nested_tensor_from_tensor_list가 없을 수 있어서
# 여기서 직접 정의한다.
def nested_tensor_from_tensor_list(tensor_list):
    """
    tensor_list: list of Tensor[C,H,W]
    return: samples (NestedTensor or tuple-like)
    - MDETR 원본 util.misc에 NestedTensor가 있으면 그걸 쓰고,
      없으면 (tensor, mask) 튜플로라도 반환해서 다음 에러 위치를 확인할 수 있게 한다.
    """
    # max size 계산
    max_c = max(img.shape[0] for img in tensor_list)
    max_h = max(img.shape[1] for img in tensor_list)
    max_w = max(img.shape[2] for img in tensor_list)

    batch_size = len(tensor_list)
    dtype = tensor_list[0].dtype
    device = tensor_list[0].device

    # padded tensor + mask 만들기
    tensor = torch.zeros((batch_size, max_c, max_h, max_w), dtype=dtype, device=device)
    mask = torch.ones((batch_size, max_h, max_w), dtype=torch.bool, device=device)

    for i, img in enumerate(tensor_list):
        c, h, w = img.shape
        tensor[i, :c, :h, :w] = img
        mask[i, :h, :w] = False  # False = valid region

    # 가능하면 repo의 NestedTensor 사용
    try:
        from util.misc import NestedTensor
        return NestedTensor(tensor, mask)
    except Exception:
        # NestedTensor가 없으면 일단 (tensor, mask)로 넘겨서 다음 에러 확인
        return (tensor, mask)


def collate_team_mdetr(batch, tokenizer, max_text_len=256):
    images = []
    targets = []
    captions = []

    for i, (img, cap, tgt) in enumerate(batch):
        images.append(img)
        captions.append(cap)

        h, w = img.shape[-2], img.shape[-1]
        tgt = dict(tgt)

        tgt["caption"] = cap
        nbox = tgt["boxes"].shape[0]
        tgt["tokens_positive"] = [[(0, len(cap))] for _ in range(nbox)]

        tgt["image_id"] = torch.as_tensor(i, dtype=torch.int64)
        tgt["orig_size"] = torch.as_tensor([h, w], dtype=torch.int64)
        tgt["size"] = torch.as_tensor([h, w], dtype=torch.int64)

        targets.append(tgt)

    samples = nested_tensor_from_tensor_list(images)

    tokenized = tokenizer(
        captions,
        padding="longest",
        truncation=True,
        max_length=max_text_len,
        return_tensors="pt",
    )
    L = tokenized["input_ids"].shape[1]
    total_boxes = sum(t["boxes"].shape[0] for t in targets)

    positive_map = torch.ones((total_boxes, L), dtype=torch.float32)

    return {
        "samples": samples,
        "targets": targets,
        "positive_map": positive_map,
    }
