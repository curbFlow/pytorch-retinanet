import argparse
import configparser
import json
import os
import sys
import time

import cv2
import numpy as np
import torch

from tools import Preprocessor, load_model
from utils.drawing_utils import draw_caption
from utils.label_utils import load_classes_from_configfile


def detect_images(image_path, model_path, configfile, output_dir):
    # Load model
    configs = configparser.ConfigParser()
    configs.read(configfile)

    try:
        input_shape = json.loads(configs['MODEL']['input_shape'])
    except:
        print("CONFIG FILE DOES NOT HAVE INPUT_SHAPE")
        sys.exit()

    retinanet = load_model(model_path, configfile)

    preprocessor = Preprocessor(input_width=input_shape[2], input_height=input_shape[1],
                                mean=np.array([[[0.485, 0.456, 0.406]]]), std=np.array([[[0.229, 0.224, 0.225]]]))
    # Get labelmap
    labels = load_classes_from_configfile(configfile)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    assert (os.path.abspath(output_dir) != os.path.abspath(image_path))

    for img_name in os.listdir(image_path):

        image = cv2.imread(os.path.join(image_path, img_name))
        if image is None:
            continue

        image_orig = image.copy()
        image, scales = preprocessor(image)
        image = np.expand_dims(image, 0)
        image = np.transpose(image, (0, 3, 1, 2))

        with torch.no_grad():
            image = torch.from_numpy(image)
            if torch.cuda.is_available():
                image = image.cuda()

            st = time.time()
            if (torch.cuda.is_available()):
                scores, classification, transformed_anchors = retinanet(image.cuda().float())
            else:
                scores, classification, transformed_anchors = retinanet(image.float())

            print('Elapsed time: {}'.format(time.time() - st))
            idxs = np.where(scores.cpu() > 0.35)

            for j in range(idxs[0].shape[0]):
                bbox = transformed_anchors[idxs[0][j], :]

                x1 = int(bbox[0] / scales[0])
                y1 = int(bbox[1] / scales[1])
                x2 = int(bbox[2] / scales[0])
                y2 = int(bbox[3] / scales[1])
                label_name = labels[int(classification[idxs[0][j]])]
                score = scores[j]
                caption = '{} {:.3f}'.format(label_name, score)
                draw_caption(image_orig, (x1, y1, x2, y2), caption)
                cv2.rectangle(image_orig, (x1, y1), (x2, y2), color=(0, 0, 255), thickness=2)

            cv2.imwrite(os.path.join(output_dir, img_name), image_orig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Simple script for visualizing result of training.')

    parser.add_argument('--image_dir', help='Path to directory containing images')
    parser.add_argument('--model_path', help='Path to model')
    parser.add_argument('--configfile', help='Path to the config file of the model')
    parser.add_argument('--out_dir', help='Path to the output directory', default='output_dir', required=False)

    parser = parser.parse_args()

    detect_images(parser.image_dir, parser.model_path, parser.configfile, parser.out_dir)
