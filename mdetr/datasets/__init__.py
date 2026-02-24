# Copyright (c) Aishwarya Kamath & Nicolas Carion. Licensed under the Apache License 2.0. All Rights Reserved
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import torch.utils.data
import torchvision
import os


from .mixed import CustomCocoDetection
from .clevr import build as build_clevr
from .clevrref import build as build_clevrref
from .coco import build as build_coco
from .flickr import build as build_flickr
from .gqa import build as build_gqa
from .lvis import LvisDetectionBase
from .lvis import build as build_lvis
from .lvis_modulation import build as build_modulated_lvis
from .mixed import build as build_mixed
from .phrasecut import build as build_phrasecut
from .refexp import build as build_refexp
from .vg import build as build_vg


def get_coco_api_from_dataset(dataset):
    for _ in range(10):
        # if isinstance(dataset, torchvision.datasets.CocoDetection):
        #     break
        if isinstance(dataset, torch.utils.data.Subset):
            dataset = dataset.dataset
    if isinstance(dataset, LvisDetectionBase):
        return dataset.lvis
    if isinstance(dataset, (torchvision.datasets.CocoDetection, CustomCocoDetection)):
        return dataset.coco


def build_dataset(dataset_file: str, image_set: str, args):
    if "clevrref" in dataset_file:
        return build_clevrref(image_set, args)
    if "clevr" in dataset_file:
        return build_clevr(dataset_file, image_set, args)
    if dataset_file == "coco":
        return build_coco(image_set, args)
    if dataset_file == "flickr":
        return build_flickr(image_set, args)
    if dataset_file == "gqa":
        return build_gqa(image_set, args)
    if dataset_file == "lvis":
        return build_lvis(image_set, args)
    if dataset_file == "modulated_lvis":
        return build_modulated_lvis(image_set, args)
    if dataset_file == "mixed":
        return build_mixed(image_set, args)
    if dataset_file == "refexp":
        return build_refexp(image_set, args)
    if dataset_file == "vg":
        return build_vg(image_set, args)
    if dataset_file == "phrasecut":
        return build_phrasecut(image_set, args)
    if dataset_file == "aihub":
        from .aihub_jsonl import AIHubJSONL
        from .coco import make_coco_transforms

        jsonl = os.path.join(args.aihub_ann_dir, "train.jsonl" if image_set == "train" else "val.jsonl")
        dataset = AIHubJSONL(
            root=args.aihub_root,
            jsonl_path=jsonl,
            transforms=make_coco_transforms(image_set, cautious=args.cautious),
        )
        return dataset

    raise ValueError(f"dataset {dataset_file} not supported")
