import configparser
import json
import sys

import cv2
import numpy as np
import torch

from retinanet import model


class Preprocessor(object):

    def __init__(self, input_width, input_height, mean, std):
        self.input_width = input_width
        self.input_height = input_height
        self.mean = mean
        self.std = std

        assert (self.input_width % 32 == 0 and self.input_height % 32 == 0)

    def __call__(self, image, resized_only=False):
        image = cv2.resize(image, (self.input_width, self.input_height))
        if (resized_only):
            return image
        image = image.astype(np.float32) / 255.0
        return ((image - self.mean) / self.std)


def load_model(model_path, configfile, no_nms=False):
    configs = configparser.ConfigParser()
    configs.read(configfile)

    try:
        depth = int(configs['TRAINING']['depth'])
        input_shape = json.loads(configs['MODEL']['input_shape'])
        num_classes = int(configs['TRAINING']['num_classes'])
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

    # Create the model
    if depth == 18:
        retinanet = model.resnet18(num_classes=num_classes, pretrained=False, ratios=ratios,
                                   scales=scales, no_nms=no_nms)
    elif depth == 50:
        retinanet = model.resnet50(num_classes=num_classes, pretrained=True, ratios=ratios,
                                   scales=scales, no_nms=no_nms)
    else:
        print(f"DEPTH FROM : {configfile} INACCURATE. MUST BE 18 or 50")
        sys.exit(0)

    if torch.cuda.is_available():
        retinanet = retinanet.cuda()
        retinanet = torch.nn.DataParallel(retinanet).cuda()
        retinanet.load_state_dict(torch.load(model_path))

    else:
        retinanet = torch.nn.DataParallel(retinanet)
        retinanet.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))

    retinanet.training = False
    retinanet.eval()

    return retinanet


def draw_caption(image, box, caption):
    b = np.array(box).astype(int)
    cv2.putText(image, caption, (b[0], b[1] - 10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 2)
    cv2.putText(image, caption, (b[0], b[1] - 10), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1)