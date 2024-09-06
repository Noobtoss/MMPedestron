# Copyright (c) OpenMMLab. All rights reserved.
import copy
import os
import warnings

import mmcv
import torch
from mmcv.utils import get_git_hash
from mmdet import __version__
from mmdet.datasets import build_dataset

import mmdet_custom  # noqa: F401,F403
import argparse
import os

import mmcv
from mmdet import __version__
# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import os.path as osp
import time

import mmcv
import torch
from mmcv import Config, DictAction
from mmdet.datasets import build_dataset, build_dataloader
from mmdet.utils import replace_cfg_vals, update_data_root

from pycocotools.coco import COCO
import os
import cv2


def load_coco_dataset(annotation_file, image_dir):
    # Load COCO annotations
    coco = COCO(annotation_file)

    # Get all image ids
    img_ids = coco.getImgIds()

    for img_id in img_ids:
        # Load image metadata
        img_info = coco.loadImgs(img_id)[0]
        img_path = os.path.join(image_dir, img_info['file_name'])

        # Load image
        image = cv2.imread(img_path)
        if image is None:
            print(f"Image not found: {img_path}")
            continue


def parse_args():
    parser = argparse.ArgumentParser(description='Test a dataset in MMDetection')
    parser.add_argument('config', help='train config file path')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=mmcv.DictAction,
        help='override some settings in the used config, the key-value pair '
             'in xxx=yyy format will be merged into config file.')
    return parser.parse_args()


def main():
    args = parse_args()

    # Load and update config
    cfg = Config.fromfile(args.config)
    cfg = replace_cfg_vals(cfg)
    update_data_root(cfg)
    cfg.data.workers_per_gpu = 0  # Set num_workers=0 to avoid multiprocessing issues

    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    # Set cudnn_benchmark if enabled in config
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    # Output directory
    work_dir = cfg.get('work_dir', './work_dirs/test_dataset')
    mmcv.mkdir_or_exist(osp.abspath(work_dir))
    log_file = osp.join(work_dir, f'test_dataset_{time.strftime("%Y%m%d_%H%M%S", time.localtime())}.log')

    print(f'Config:\n{cfg.pretty_text}')


    annotation_file = cfg.data.train.ann_file[0]  # Path to COCO annotation JSON file
    image_dir = cfg.data.train.img_prefix[0]  # Directory containing images#
    print(annotation_file)
    print(image_dir)
    load_coco_dataset(annotation_file, image_dir)


    # Build dataset and dataloader
    try:
        dataset = build_dataset(cfg.data.train)
        dataloader = build_dataloader(
            dataset,
            samples_per_gpu=1,  # Set batch size
            workers_per_gpu=cfg.data.workers_per_gpu,
            dist=False,
            shuffle=False  # Set shuffle=True if you want to shuffle the dataset
        )
        print('Dataset and DataLoader built successfully.')

        # Print batches from DataLoader
        for i, batch in enumerate(dataloader):
            print(f"Batch {i + 1}:")
            for key, value in batch.items():
                print(f"  {key}: {value.shape if isinstance(value, torch.Tensor) else value}")
            if i >= 4:  # Print only first 5 batches for brevity
                break

    except Exception as e:
        print(f"Error loading dataset or DataLoader: {e}")


if __name__ == '__main__':
    main()
