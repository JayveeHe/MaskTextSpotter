# coding=utf-8

"""
Created by Jayvee_He on 2020-03-10.
"""
import io
import os
import pickle
import sys

import requests
import torch
from PIL import Image

from maskrcnn_benchmark.textspotter import MaskTextSpotter

PROJECT_PATH = os.path.dirname(os.path.dirname(__file__))
print('current file:%s, PROJECT_PATH: %s' % (22, PROJECT_PATH))
sys.path.append(PROJECT_PATH)


if __name__ == '__main__':
    DATAPATH = sys.argv[1]
    test_image_url = sys.argv[2]
    device = torch.device("cpu")
    print(device)

    # from maskrcnn_benchmark.config import cfg
    #
    # cfg.merge_from_file('%s/models/OCR/batch.yaml' % DATAPATH)
    cfg = pickle.load(open('%s/models/OCR/config.pkl' % DATAPATH,'rb'))
    print('initing ocr model')
    mts = MaskTextSpotter(
        cfg,
        min_image_size=800,
        confidence_threshold=0.7,
        output_polygon=True
    )
    img_obj = Image.open(io.BytesIO(
        requests.get(test_image_url).content))
    res = mts.run_on_pillow_image(img_obj)
    print(res)
