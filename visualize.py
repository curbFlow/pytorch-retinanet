import argparse
import configparser
import json
import os
import sys
import time

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from retinanet import model
from retinanet.dataloader import CSVDataset, collater, Resizer, AspectRatioBasedSampler, UnNormalizer, \
    Normalizer

assert torch.__version__.split('.')[0] == '1'

print('CUDA available: {}'.format(torch.cuda.is_available()))


def draw_caption(image, box, caption):
    b = np.array(box).astype(int)
    cv2.putText(image, caption, (b[0], b[1] - 10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 3)
    cv2.putText(image, caption, (b[0], b[1] - 10), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 2)


def main(args=None):
    parser = argparse.ArgumentParser(
        description='Simple training script for visualizing results from a RetinaNet network.')

    parser.add_argument('--csv', help='Path to file containing annotations (optional, see readme)')
    parser.add_argument('--csv_classes', help='Path to file containing class list (see readme)')
    parser.add_argument('--model_path', help='Path to model (.pt) file.')
    parser.add_argument('--configfile', help='Path to the config file.')
    parser.add_argument('--out_path', help='Path to the folder where to save the images.')

    parser = parser.parse_args(args)
    configs = configparser.ConfigParser()
    configs.read(parser.configfile)

    try:

        depth = int(configs['TRAINING']['depth'])
        maxside = int(configs['TRAINING']['maxside'])
        minside = int(configs['TRAINING']['minside'])
        try:
            ratios = json.loads(configs['MODEL']['ratios'])
            scales = json.loads(configs['MODEL']['scales'])
        except Exception as e:
            print(e)
            print('USING DEFAULT RATIOS AND SCALES')
            ratios = None
            scales = None
    except Exception as e:
        print(e)
        print('CONFIG FILE IS INVALID. PLEASE REFER TO THE EXAMPLE CONFIG FILE AT config.txt')
        sys.exit()

    if parser.csv is None:
        dataset_eval = None
        print('No validation annotations provided.')
    else:
        dataset_eval = CSVDataset(train_file=parser.csv, class_list=parser.csv_classes,
                                  transform=transforms.Compose([Normalizer(), Resizer(min_side=minside,
                                                                                      max_side=maxside)]))

    dataloader_val = None
    if dataset_eval is not None:
        sampler_val = AspectRatioBasedSampler(dataset_eval, batch_size=1, drop_last=False)
        dataloader_val = DataLoader(dataset_eval, num_workers=1, collate_fn=collater, batch_sampler=sampler_val)

    # Create the model
    if depth == 18:
        retinanet = model.resnet18(num_classes=dataset_eval.num_classes(), pretrained=False, ratios=ratios,
                                   scales=scales)
    elif depth == 50:
        retinanet = model.resnet50(num_classes=dataset_eval.num_classes(), pretrained=True, ratios=ratios,
                                   scales=scales)
    else:
        print(f"DEPTH FROM : {parser.configfile} INACCURATE. MUST BE 18 or 50")
        sys.exit(0)

    if torch.cuda.is_available():
        retinanet = retinanet.cuda()
        retinanet = torch.nn.DataParallel(retinanet).cuda()
        retinanet.load_state_dict(torch.load(parser.model_path))

    else:
        retinanet = torch.nn.DataParallel(retinanet)
        retinanet.load_state_dict(torch.load(parser.model_path, map_location=torch.device('cpu')))

    retinanet.eval()

    unnormalize = UnNormalizer()

    if not os.path.exists(parser.out_path):
        os.makedirs(parser.out_path, exist_ok=True)

    for idx, data in enumerate(dataloader_val):

        with torch.no_grad():
            st = time.time()
            if torch.cuda.is_available():
                scores, classification, transformed_anchors = retinanet(data['img'].cuda().float())
            else:
                scores, classification, transformed_anchors = retinanet(data['img'].float())
            print('Elapsed time: {}'.format(time.time() - st))
            idxs = np.where(scores.cpu() > 0.5)
            img = np.array(255 * unnormalize(data['img'][0, :, :, :])).copy()

            img[img < 0] = 0
            img[img > 255] = 255

            img = np.transpose(img, (1, 2, 0))

            img = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_RGB2BGR)

            for j in range(idxs[0].shape[0]):
                bbox = transformed_anchors[idxs[0][j], :]
                x1 = int(bbox[0])
                y1 = int(bbox[1])
                x2 = int(bbox[2])
                y2 = int(bbox[3])
                label_name = dataset_eval.labels[int(classification[idxs[0][j]])]
                draw_caption(img, (x1, y1, x2, y2), label_name)
                cv2.rectangle(img, (x1, y1), (x2, y2), color=(0, 0, 255), thickness=2)

            cv2.imwrite(os.path.join(parser.out_path, f'image_{idx}.png'), img)


if __name__ == '__main__':
    main()
