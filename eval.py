import argparse
import configparser
import json
import sys

import torch
from torchvision import transforms

from retinanet import csv_eval
from retinanet import model
from tools import load_model
from retinanet.dataloader import CSVDataset, Resizer, Normalizer

assert torch.__version__.split('.')[0] == '1'

print('CUDA available: {}'.format(torch.cuda.is_available()))


def main(args=None):
    parser = argparse.ArgumentParser(description='Simple training script for training a RetinaNet network.')

    parser.add_argument('--csv', help='Path to dataset file you would like to evaluate')
    parser.add_argument('--csv_classes', help='Path to file containing class list (see readme)')
    parser.add_argument('--model_path', help='Path to the model file.')
    parser.add_argument('--configfile', help='Path to the config file.')

    parser = parser.parse_args(args)
    configs = configparser.ConfigParser()
    configs.read(parser.configfile)

    try:
        maxside = int(configs['TRAINING']['maxside'])
        minside = int(configs['TRAINING']['minside'])
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
    retinanet = load_model(parser.model_path,parser.configfile)

    mAP = csv_eval.evaluate(dataset_eval, retinanet)
    print('-----------------')
    print(mAP)
    print('-----------------')


if __name__ == '__main__':
    main()
