# coding=utf-8

import io
import os
import pickle
import sys

import requests
import torch
import torch.multiprocessing as mp
from PIL import Image
from deepcat.models.ImageVecExtractor import ImageVecExtractor
from deepcat.models.MultiModalClassifier import MultiModalClassifer

PROJECT_PATH = os.path.dirname(os.path.dirname(__file__))
print('current file:%s, PROJECT_PATH: %s' % (22, PROJECT_PATH))
sys.path.append(PROJECT_PATH)
from tools.mask_text_spotter_util import MaskTextSpotter

if __name__ == '__main__':
    DATAPATH = sys.argv[1]
    # nsfw_model_path = '%s/models/NSFW/nsfw.299x299.h5' % DATAPATH
    # nsfw_model = NSFWModel(None)
    # nsfw_model.nsfw_model = nm
    # print('nsfw_model', nsfw_model)
    device = torch.device("cpu")
    print(device)
    # print('initing ImageVecExtractor ')
    # ive = ImageVecExtractor(device)
    # print('initing multimodal')
    # multimodal_classifier = MultiModalClassifer(model_path='./')
    # multimodal_classifier.load_model('%s/models/MultiModal/multimodal_newrules_20200225.mod.inst' % DATAPATH,
    #                                  location_device=device)
    # print(multimodal_classifier)

    from maskrcnn_benchmark.config import cfg

    # image seeker

    # cfg.merge_from_file('%s/models/OCR/batch.yaml' % DATAPATH)
    cfg.merge_from_file('%s/models/OCR/batch.yaml' % DATAPATH)
    # print('cfg', cfg)
    print('initing ocr model')
    mts = MaskTextSpotter(
        cfg,
        min_image_size=800,
        confidence_threshold=0.7,
        output_polygon=True
    )
    # pickle.dump(mts, open('./masktextspotter.pkl', 'wb'))
    img_obj = Image.open(io.BytesIO(
        requests.get('http://www.cbdwarehouseusa.com/wp-content/uploads/2018/05/choicepowder-300x300.jpg').content))
    res = mts.run_on_pillow_image(img_obj)
    print(res)
    # new_mts = pickle.load(open('./masktextspotter.pkl', 'rb'))
    # new_res = mts.run_on_pillow_image(img_obj)
    # print(new_res)
