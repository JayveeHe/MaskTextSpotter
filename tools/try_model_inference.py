# coding=utf-8

"""
Created by Jayvee_He on 2020-03-10.
"""
import copy
import datetime
import io
import os
import pickle
import random
import sys

import requests
import torch
from PIL import Image
from shapely.geometry import Polygon, Point, LineString

from maskrcnn_benchmark.textspotter import MaskTextSpotter

PROJECT_PATH = os.path.dirname(os.path.dirname(__file__))
print('current file:%s, PROJECT_PATH: %s' % (22, PROJECT_PATH))
sys.path.append(PROJECT_PATH)
import cv2
import torch
# from torchvision import transforms as T
#
# from maskrcnn_benchmark.modeling.detector import build_detection_model
# from maskrcnn_benchmark.utils.checkpoint import DetectronCheckpointer
# from maskrcnn_benchmark.structures.image_list import to_image_list
# from maskrcnn_benchmark.utils.chars import getstr_grid, get_tight_rect

from PIL import Image
import numpy as np

# from symspellpy import SymSpell, Verbosity
#
# sym_spell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)


# class MaskTextSpotter(object):
#     def __init__(
#             self,
#             cfg,
#             confidence_threshold=0.7,
#             min_image_size=224,
#             output_polygon=True
#     ):
#         self.cfg = cfg.clone()
#         self.model = build_detection_model(cfg)
#         self.model.eval()
#         self.device = torch.device(cfg.MODEL.DEVICE)
#         self.model.to(self.device)
#         self.min_image_size = min_image_size
#
#         checkpointer = DetectronCheckpointer(cfg, self.model)
#         _ = checkpointer.load(cfg.MODEL.WEIGHT)
#
#         self.transforms = self.build_transform()
#         self.cpu_device = torch.device("cpu")
#         self.confidence_threshold = confidence_threshold
#         self.output_polygon = output_polygon
#
#     def build_transform(self):
#         """
#         Creates a basic transformation that was used to train the models
#         """
#         cfg = self.cfg
#         # we are loading images with OpenCV, so we don't need to convert them
#         # to BGR, they are already! So all we need to do is to normalize
#         # by 255 if we want to convert to BGR255 format, or flip the channels
#         # if we want it to be in RGB in [0-1] range.
#         if cfg.INPUT.TO_BGR255:
#             to_bgr_transform = T.Lambda(lambda x: x * 255)
#         else:
#             to_bgr_transform = T.Lambda(lambda x: x[[2, 1, 0]])
#
#         normalize_transform = T.Normalize(
#             mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD
#         )
#
#         transform = T.Compose(
#             [
#                 T.ToPILImage(),
#                 T.Resize(self.min_image_size),
#                 T.ToTensor(),
#                 to_bgr_transform,
#                 normalize_transform,
#             ]
#         )
#         return transform
#
#     def run_on_opencv_image(self, image):
#         """
#         Arguments:
#             image (np.ndarray): an image as returned by OpenCV
#         Returns:
#             result_polygons (list): detection results
#             result_words (list): recognition results
#         """
#         result_polygons, result_words, result_dict = self.compute_prediction(image)
#         return result_polygons, result_words, result_dict
#
#     def run_on_pillow_image(self, image):
#         arr = np.array(image)
#         result_polygons, result_words, result_dict = self.run_on_opencv_image(arr)
#         return result_polygons, result_words, result_dict
#
#     def compute_prediction(self, original_image):
#         def chunks(l, n):
#             for i in range(0, len(l), n):
#                 yield l[i: i + n]
#
#         def mk_direction(char_polygons):
#
#             def centroid(char_polygon):
#                 centroid = Polygon(list(chunks(char_polygon, 2))).centroid.coords
#                 return list(centroid)[0]
#
#             first, last = char_polygons[0], char_polygons[-1]
#             start, end = centroid(first), centroid(last)
#             if start[0] == end[0]:
#                 end = (end[0] + 1, end[1])
#             return start, end
#
#         # apply pre-processing to image
#         import datetime, time
#         start_time = time.time()
#         # print('transform', datetime.datetime.now())
#         image = self.transforms(original_image)
#         # convert to an ImageList, padded so that it is divisible by
#         # cfg.DATALOADER.SIZE_DIVISIBILITY
#         # print('to image list', datetime.datetime.now())
#         image_list = to_image_list(image, self.cfg.DATALOADER.SIZE_DIVISIBILITY)
#         image_list = image_list.to(self.device)
#         # compute predictions
#         with torch.no_grad():
#             # print('predict', datetime.datetime.now())
#             predictions = self.model(image_list)
#             if not predictions or len(predictions) < 1:
#                 # print('no text detected')
#                 return [], []
#         # print('post process', datetime.datetime.now())
#         global_predictions = predictions[0]
#         char_predictions = predictions[1]
#         char_mask = char_predictions['char_mask']
#         char_boxes = char_predictions['boxes']
#         words, rec_scores, rec_char_scores, char_polygons = self.process_char_mask(char_mask, char_boxes)
#         detailed_seq_scores = char_predictions['detailed_seq_scores']
#         seq_words = char_predictions['seq_outputs']
#         seq_scores = char_predictions['seq_scores']
#         global_predictions = [o.to(self.cpu_device) for o in global_predictions]
#
#         # always single image is passed at a time
#         global_prediction = global_predictions[0]
#
#         # reshape prediction (a BoxList) into the original image size
#         height, width = original_image.shape[:-1]
#         test_image_width, test_image_height = global_prediction.size
#         global_prediction = global_prediction.resize((width, height))
#         resize_ratio = float(height) / test_image_height
#         boxes = global_prediction.bbox.tolist()
#         scores = global_prediction.get_field("scores").tolist()
#         masks = global_prediction.get_field("mask").cpu().numpy()
#
#         result_polygons = []
#         result_words = []
#         result_dicts = []
#
#         for k, box in enumerate(boxes):
#             box = list(map(int, box))
#             mask = masks[k, 0, :, :]
#             polygon = self.mask2polygon(mask, box, original_image.shape, threshold=0.5,
#                                         output_polygon=self.output_polygon)
#             if polygon is None:
#                 polygon = [box[0], box[1], box[2], box[1], box[2], box[3], box[0], box[3]]
#             result_polygons.append(polygon)
#             score = scores[k]
#             if score < self.confidence_threshold:
#                 continue
#             word = words[k]
#             rec_score = rec_scores[k]
#             char_score = rec_char_scores[k]
#             seq_word = seq_words[k]
#             seq_char_scores = seq_scores[k]
#             seq_score = sum(seq_char_scores) / float(len(seq_char_scores))
#             spell_fix = lambda word: \
#                 [s.term for s in sym_spell.lookup(word, Verbosity.CLOSEST, max_edit_distance=2, include_unknown=True)][
#                     0]
#             detailed_seq_score = detailed_seq_scores[k]
#             detailed_seq_score = np.squeeze(np.array(detailed_seq_score), axis=1)
#             # if 'total_text' in output_folder or 'cute80' in output_folder:
#             #     result_log = [int(x * 1.0) for x in box[:4]] + polygon + [word] + [seq_word] + [score] + [rec_score] + [
#             #         seq_score] + [char_score] + [detailed_seq_score] + [len(polygon)]
#             # else:
#             result_log = [int(x * 1.0) for x in box[:4]] + polygon + [word] + [seq_word] + [score] + [rec_score] + [
#                 seq_score] + [char_score] + [detailed_seq_score]
#             # result_logs.append(result_log)
#             if len(seq_word) > 0 and len(char_polygons[k]) > 0:
#                 d = {
#                     "seq_word": seq_word if len(seq_word) < 4 else spell_fix(seq_word),
#                     "seq_word_orig": seq_word,
#                     "direction": mk_direction([[int(c * resize_ratio) for c in p] for p in char_polygons[k]]),
#                     "word": word if len(word) < 4 else spell_fix(word),
#                     "word_orig": word,
#                     "box": [int(x * 1.0) for x in box[:4]],
#                     "polygon": polygon,
#                     "prob": score * seq_score
#                 }
#                 result_words.append(d['word'])
#                 result_dicts.append(d)
#
#         # default_logger.debug('done', datetime.datetime.now())
#         end_time = time.time()
#         # default_logger.debug('cost time: %s' % (end_time - start_time))
#
#         return result_polygons, result_words, result_dicts
#
#     # def process_char_mask(self, char_masks, boxes, threshold=192):
#     #     texts, rec_scores = [], []
#     #     for index in range(char_masks.shape[0]):
#     #         box = list(boxes[index])
#     #         box = list(map(int, box))
#     #         text, rec_score, _, _ = getstr_grid(char_masks[index, :, :, :].copy(), box, threshold=threshold)
#     #         texts.append(text)
#     #         rec_scores.append(rec_score)
#     #     return texts, rec_scores
#
#     def process_char_mask(self, char_masks, boxes, threshold=192):
#         texts, rec_scores, rec_char_scores, char_polygons = [], [], [], []
#         for index in range(char_masks.shape[0]):
#             box = list(boxes[index])
#             box = list(map(int, box))
#             text, rec_score, rec_char_score, char_polygon = getstr_grid(char_masks[index, :, :, :].copy(), box,
#                                                                         threshold=threshold)
#             texts.append(text)
#             rec_scores.append(rec_score)
#             rec_char_scores.append(rec_char_score)
#             char_polygons.append(char_polygon)
#             # segmss.append(segms)
#         return texts, rec_scores, rec_char_scores, char_polygons
#
#     def mask2polygon(self, mask, box, im_size, threshold=0.5, output_polygon=True):
#         # mask 32*128
#         image_width, image_height = im_size[1], im_size[0]
#         box_h = box[3] - box[1]
#         box_w = box[2] - box[0]
#         cls_polys = (mask * 255).astype(np.uint8)
#         poly_map = np.array(Image.fromarray(cls_polys).resize((box_w, box_h)))
#         poly_map = poly_map.astype(np.float32) / 255
#         poly_map = cv2.GaussianBlur(poly_map, (3, 3), sigmaX=3)
#         ret, poly_map = cv2.threshold(poly_map, 0.5, 1, cv2.THRESH_BINARY)
#         if output_polygon:
#             SE1 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
#             poly_map = cv2.erode(poly_map, SE1)
#             poly_map = cv2.dilate(poly_map, SE1);
#             poly_map = cv2.morphologyEx(poly_map, cv2.MORPH_CLOSE, SE1)
#             try:
#                 _, contours, _ = cv2.findContours((poly_map * 255).astype(np.uint8), cv2.RETR_LIST,
#                                                   cv2.CHAIN_APPROX_NONE)
#             except:
#                 contours, _ = cv2.findContours((poly_map * 255).astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
#             if len(contours) == 0:
#                 print(contours)
#                 print(len(contours))
#                 return None
#             max_area = 0
#             max_cnt = contours[0]
#             for cnt in contours:
#                 area = cv2.contourArea(cnt)
#                 if area > max_area:
#                     max_area = area
#                     max_cnt = cnt
#             perimeter = cv2.arcLength(max_cnt, True)
#             epsilon = 0.01 * cv2.arcLength(max_cnt, True)
#             approx = cv2.approxPolyDP(max_cnt, epsilon, True)
#             pts = approx.reshape((-1, 2))
#             pts[:, 0] = pts[:, 0] + box[0]
#             pts[:, 1] = pts[:, 1] + box[1]
#             polygon = list(pts.reshape((-1,)))
#             polygon = list(map(int, polygon))
#             if len(polygon) < 6:
#                 return None
#         else:
#             SE1 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
#             poly_map = cv2.erode(poly_map, SE1)
#             poly_map = cv2.dilate(poly_map, SE1);
#             poly_map = cv2.morphologyEx(poly_map, cv2.MORPH_CLOSE, SE1)
#             idy, idx = np.where(poly_map == 1)
#             xy = np.vstack((idx, idy))
#             xy = np.transpose(xy)
#             hull = cv2.convexHull(xy, clockwise=True)
#             # reverse order of points.
#             if hull is None:
#                 return None
#             hull = hull[::-1]
#             # find minimum area bounding box.
#             rect = cv2.minAreaRect(hull)
#             corners = cv2.boxPoints(rect)
#             corners = np.array(corners, dtype="int")
#             pts = get_tight_rect(corners, box[0], box[1], image_height, image_width, 1)
#             polygon = [x * 1.0 for x in pts]
#             polygon = list(map(int, polygon))
#         return polygon
#
#     def visualization(self, img, polygons, words):
#         cur_img = copy.deepcopy(img)
#         for polygon, word in zip(polygons, words):
#             pts = np.array(polygon, np.int32)
#             pts = pts.reshape((-1, 1, 2))
#             xmin = min(pts[:, 0, 0])
#             ymin = min(pts[:, 0, 1])
#             color_tuple = (int(255 * random.random()), int(255 * random.random()), int(255 * random.random()))
#             cv2.polylines(cur_img, [pts], True, color_tuple)
#
#             cv2.putText(cur_img, word, (xmin, ymin), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.75, color_tuple, 1)
#         return cur_img

if __name__ == '__main__':
    DATAPATH = sys.argv[1]
    test_image_url = sys.argv[2]
    device = torch.device("cpu")
    print(device)

    from maskrcnn_benchmark.config import cfg

    #
    cfg.merge_from_file('%s/models/OCR/batch.yaml' % DATAPATH)
    # cfg = pickle.load(open('%s/models/OCR/config.pkl' % DATAPATH, 'rb'))
    cfg['MODEL']['WEIGHT'] = '%s/models/OCR/model_pretrain.pth' % DATAPATH
    cfg['DEVICE'] = 'cpu'
    print('initing ocr model')
    mts = MaskTextSpotter(
        cfg,
        min_image_size=800,
        confidence_threshold=0.7,
        output_polygon=True
    )
    print('[%s]predicting' % datetime.datetime.now())
    img_obj = Image.open(io.BytesIO(
        requests.get(test_image_url, verify=False).content))
    img_obj = img_obj.convert('RGB')
    res = mts.run_on_pillow_image(img_obj)
    print('[%s]done' % datetime.datetime.now())
    # print(res)
    import json

    line_result = {'label': res[2]['label'],
                   'details': [{'idx': b[0],
                                'word_list': [{'word': c['word'],
                                               'word_orig': c['word_orig'],
                                               'prob': c['prob'],
                                               'polygon': c['polygon']}
                                              for c in b[1]]}
                               for b in res[2]['details']]}
    json_line_result = json.dumps(line_result)
    print(line_result)
    img = cv2.cvtColor(np.asarray(img_obj), cv2.COLOR_RGB2BGR)

    vis_image = mts.visualization(img, res[0], res[1])
    Image.fromarray(cv2.cvtColor(vis_image, cv2.COLOR_BGR2RGB)).show()
