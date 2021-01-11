import argparse
import configparser
import json
import sys

import torch

from retinanet import model

assert torch.__version__.split('.')[0] == '1'

print('CUDA available: {}'.format(torch.cuda.is_available()))


def main(args=None):
    parser = argparse.ArgumentParser(description='Simple training script for training a RetinaNet network.')

    parser.add_argument('--model_path', help='Path to the model file.')
    parser.add_argument('--configfile', help='Path to the config file.')
    parser.add_argument('--model_out_path', help='Path to the output model file')
    parser = parser.parse_args(args)
    configs = configparser.ConfigParser()
    configs.read(parser.configfile)

    try:

        depth = int(configs['TRAINING']['depth'])
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
                                   scales=scales)
    elif depth == 50:
        retinanet = model.resnet50(num_classes=num_classes, pretrained=True, ratios=ratios,
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

    torch.save(retinanet.state_dict(), parser.model_out_path, _use_new_zipfile_serialization=False)


if __name__ == '__main__':
    main()
