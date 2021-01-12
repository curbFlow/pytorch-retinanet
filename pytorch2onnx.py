import argparse
import configparser
import importlib
import json
import sys
import traceback

import torch

from tools import load_model

jtop_found = importlib.util.find_spec('jtop')
assert jtop_found, 'jetson-stats must be installed! Do so by running pip3 install jetson-stats'


def load_model_weight(model, model_path):
    model.load_state_dict(torch.load(model_path))
    model = model.eval()
    return model


def export_onnx_model(model, input_shape, onnx_path, input_names=None, output_names=None, dynamic_axes=None):
    inputs = torch.ones(*input_shape).cuda()
    out = model(inputs)
    torch.onnx.export(
        model, inputs, onnx_path, input_names=input_names,
        output_names=output_names, dynamic_axes=dynamic_axes,
    )


def main(args=None):
    parser = argparse.ArgumentParser(description='Simple script for converting pytorch models to onnx.')

    parser.add_argument(
        '--out_name', help='Name of resulting onnx file',
        required=True,
        nargs='?',
        const=1,
        type=str,
        default='model.onnx',
    )

    parser.add_argument(
        '--model', help='Path to pt model state dict to be converted',
        required=True,
        nargs='?',
        const=1,
        type=str,
        default='resnet18-retinanet.pt',
    )

    parser.add_argument(
        '--configfile', help='Config File of the PT Model',
        required=False,
        nargs='?',
        const=1,
        type=str,
        default='config.txt',
    )

    parser = parser.parse_args(args)
    configs = configparser.ConfigParser()
    configs.read(parser.configfile)
    try:
        input_shape = json.loads(configs['MODEL']['input_shape'])
    except:
        print("CONFIG FILE DOES NOT HAVE INPUT_SHAPE")
        sys.exit()

    retinanet = load_model(parser.model, parser.configfile)

    input_shape = (1, 3, input_shape[1], input_shape[2])
    onnx_path = parser.out_name

    try:
        export_onnx_model(retinanet, input_shape, onnx_path, output_names=['regression', 'classification'])
        print('Model conversion finished')
    except Exception as e:
        print (e)
        traceback.print_exc(file=sys.stdout)
        print('Error converting')

    print('Before converting model to TRT, simplify the model using onnx-simplifier')
    print(f'The command is: python3 -m onnxsim {parser.out_name} new_model_name.onnx')
    exit(0)


if __name__ == '__main__':
    main()
