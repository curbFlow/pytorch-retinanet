import argparse

import torch
from torchvision import transforms

from retinanet import csv_eval
from retinanet.dataloader import CSVDataset, Resizer, Normalizer

assert torch.__version__.split('.')[0] == '1'

print('CUDA available: {}'.format(torch.cuda.is_available()))


def main(args=None):
    parser = argparse.ArgumentParser(description='Simple training script for training a RetinaNet network.')

    parser.add_argument('--csv', help='Path to dataset file you would like to evaluate')
    parser.add_argument('--csv_classes', help='Path to file containing class list (see readme)')
    parser.add_argument('--model_path', help='Path to the model file.')
    parser.add_argument('--minside', help='The minside to scale the image to')
    parser.add_argument('--maxside', help='The maxside to scale the image to')

    parser = parser.parse_args(args)

    if parser.csv is None:
        dataset_eval = None
        print('No validation annotations provided.')
    else:
        dataset_eval = CSVDataset(train_file=parser.csv, class_list=parser.csv_classes,
                                  transform=transforms.Compose([Normalizer(), Resizer(min_side=int(parser.minside),
                                                                                      max_side=int(parser.maxside))]))

    if torch.cuda.is_available():
        retinanet = torch.load(parser.model_path)
        retinanet = retinanet.cuda()
        retinanet = torch.nn.DataParallel(retinanet).cuda()
    else:
        retinanet = torch.load(parser.model_path, map_location=torch.device('cpu'))
        retinanet = torch.nn.DataParallel(retinanet)


    retinanet.eval()

    mAP = csv_eval.evaluate(dataset_eval, retinanet)
    print('-----------------')
    print(mAP)
    print('-----------------')


if __name__ == '__main__':
    main()
