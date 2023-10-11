import os
import argparse

import tqdm
import numpy as np
import xml.etree.ElementTree as ET
from scipy.io import loadmat


def parse_args():
    parser = argparse.ArgumentParser(
        description='MMDet test (and eval) a model')
    parser.add_argument('--gt_path', help='path of mat file of ground truth')
    parser.add_argument('--ann_path', help='path of generated xml files')

    args = parser.parse_args()
    return args

def main():
    args = parse_args()

if __name__ == '__main__':
    main()