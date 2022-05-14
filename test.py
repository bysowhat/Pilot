import argparse

from datasets.kitty import Kitti
from tools.configs.parser import parser

def parse_args():
  parser = argparse.ArgumentParser(description='Train a detector')
  parser.add_argument('--config', type=str, help='train config file path')

  args = parser.parse_args()

  return args



if __name__ == '__main__':
  args = parse_args()

  cfg = parser(args.config)

  k = Kitti(cfg, phase='train')