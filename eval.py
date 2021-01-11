import argparse
import configparser
import json
import sys

import torch
from torchvision import transforms

from retinanet import csv_eval
from retinanet import model
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

    mAP = csv_eval.evaluate(dataset_eval, retinanet)
    print('-----------------')
    print(mAP)
    print('-----------------')


if __name__ == '__main__':
    main()
