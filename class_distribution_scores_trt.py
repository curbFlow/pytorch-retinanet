import argparse
import configparser
import json
import os
import sys
import time

import cv2
import numpy as np
import torch

from retinanet.model import PostProcessor
from tools import Preprocessor
from trttools import common
from trttools.engine_utils import gpu_warmup, get_engine
from utils.label_utils import load_classes_from_configfile


def preprocess_image(image_path, preprocessor):
    image = cv2.imread(image_path)
    if image is None:
        return None, None, None, None

    image_orig = image.copy()
    image, scales = preprocessor(image)
    image = np.expand_dims(image, 0)
    image = np.transpose(image, (0, 3, 1, 2))
    image = image.astype(np.float32)
    batch = image.copy()

    return image, image_orig, batch, scales


def detect_images(image_path, model_path, configfile, output_dir):
    # Load model
    configs = configparser.ConfigParser()
    configs.read(configfile)

    configs = configparser.ConfigParser()
    configs.read(configfile)

    try:
        input_shape = json.loads(configs['MODEL']['input_shape'])
        try:
            ratios = json.loads(configs['MODEL']['ratios'])
            scales = json.loads(configs['MODEL']['scales'])
        except Exception as e:
            print(e)
            print('USING DEFAULT RATIOS AND SCALES')
            ratios = None
            scales = None

    except:
        print("CONFIG FILE DOES NOT HAVE INPUT_SHAPE")
        sys.exit()

    preprocessor = Preprocessor(input_width=input_shape[2], input_height=input_shape[1],
                                mean=np.array([[[0.485, 0.456, 0.406]]]), std=np.array([[[0.229, 0.224, 0.225]]]))
    postprocessor = PostProcessor(ratios=ratios, scales=scales)

    # Get labelmap
    labels = load_classes_from_configfile(configfile)

    # GPU WARMUP
    img_name = [i for i in os.listdir(image_path) if i.endswith(('.jpg', '.png',))][0]
    image, image_orig, batch, scales = preprocess_image(os.path.join(image_path, img_name), preprocessor)
    gpu_warmup(image_orig, trt_engine_path=model_path)

    print('Getting engine')
    engine = get_engine(model_path)
    print('Engine retrieved')
    context = engine.create_execution_context()
    print('Execution context created')

    inputs, outputs, bindings, stream = common.allocate_buffers(engine)
    print('Buffers allocated')

    input_shapes = []
    output_shapes = []

    for binding in engine:
        if engine.binding_is_input(binding):
            input_shapes.append(engine.get_binding_shape(binding))
        else:  # and one output
            output_shapes.append(engine.get_binding_shape(binding))

    print(f'INPUT SHAPES:{input_shapes}, OUTPUT SHAPES:{output_shapes}')

    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    assert (os.path.abspath(output_dir) != os.path.abspath(image_path))

    class_scores_list = {i: [] for i in labels.values()}
    for img_name in os.listdir(image_path):

        image, image_orig, batch, scales = preprocess_image(os.path.join(image_path, img_name), preprocessor)
        if (image is None):
            continue
        st = time.time()

        # TRT INFERENCE
        inputs[0].host = batch

        trt_outputs = common.do_inference(
            context=context,
            bindings=bindings,
            inputs=inputs,
            outputs=outputs,
            stream=stream,
            batch_size=1,
        )

        regression, classification = [output.reshape(shape) for output, shape in zip(trt_outputs, output_shapes)]

        if torch.cuda.is_available():
            regression = torch.from_numpy(regression).cuda()
            classification = torch.from_numpy(classification).cuda()
        else:
            regression = torch.from_numpy(regression)
            classification = torch.from_numpy(classification)

        scores, classification, transformed_anchors = postprocessor(batch, regression, classification)

        print('Elapsed time: {}'.format(time.time() - st))
        idxs = np.where(scores.cpu() > 0.01)

        for j in range(idxs[0].shape[0]):
            label_name = labels[int(classification[idxs[0][j]])]
            score = scores[j]
            class_scores_list[label_name].append(score.cpu().numpy())

    # Get quartile distribution of all the scores.
    for label_name, scores_list in class_scores_list.items():
        print(
            f'{label_name} class has mean_score: {np.mean(scores_list)}, '
            f'0.25:{np.quantile(scores_list, 0.25)}, '
            f'0.5:{np.quantile(scores_list, 0.5)},'
            f'0.75:{np.quantile(scores_list, 0.75)}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Simple script for visualizing result of training.')

    parser.add_argument('--image_dir', help='Path to directory containing images')
    parser.add_argument('--model_path', help='Path to trt model')
    parser.add_argument('--configfile', help='Path to the config file of the model')
    parser.add_argument('--out_dir', help='Path to the output directory', default='output_dir', required=False)

    parser = parser.parse_args()

    detect_images(parser.image_dir, parser.model_path, parser.configfile, parser.out_dir)
