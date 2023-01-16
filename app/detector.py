import sys

import numpy as np
import torch

sys.path.append("..")

from models.experimental import attempt_load
from utils.datasets import letterbox
from utils.general import check_img_size, non_max_suppression, scale_coords
from utils.torch_utils import select_device

from . import settings


class Detector(object):
    def __init__(self, weights):
        self.dict_conf_treh = settings.DICT_CONF_TRESH
        self.dict_iou_treh = settings.DICT_IOU_TRESHOLDS
        self.weights = weights
        self.current_imgsz = settings.IMG_SIZE
        self.current_device = settings.DEVICE
        self.conf_thres = settings.CONFIDENCE
        self.iou_thres = settings.IOU_TRESHOLD
        self.classes = None
        self.agnostic_nms = False
        self.device = select_device(self.current_device)
        self.model = attempt_load(self.weights, map_location=self.device)  # load FP32 model
        self.imgsz = check_img_size(self.current_imgsz, s=self.model.stride.max())  # check img_size
        self.names = self.model.module.names if hasattr(self.model, 'module') else self.model.names

        # if 'p' in self.names:
        #     self.names.remove('p')

    def detect(self, im0):
        with torch.no_grad():
            img = letterbox(im0, new_shape=self.imgsz)[0]
            # Convert
            img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
            img = np.ascontiguousarray(img)
            img = torch.from_numpy(img).to(self.device)
            img = img.float()  # uint8 to fp16/32
            img /= 255.0  # 0 - 255 to 0.0 - 1.0
            if img.ndimension() == 3:
                img = img.unsqueeze(0)
            # adaptive thresholds
            pred1 = self.model(img, augment=False)[0]
            output_list = []
            output = None
            for i, item in enumerate(self.names):
                output_list = non_max_suppression(
                    pred1,
                    iou_thres=self.dict_iou_treh[i],
                    conf_thres=self.dict_conf_treh[i],
                    classes=self.classes,
                    agnostic=self.agnostic_nms
                    )

                for xi, pred in enumerate(output_list):
                    if pred is not None:
                        pred = pred[(pred[:, 5:6] == torch.tensor(i)).any(1)]
                        if len(pred) > 0:
                            if output is None:
                                output = pred
                            else:
                                output = torch.cat((output, pred))
                    else:
                        return None
            pred = [output]

            for i, det in enumerate(pred):  # detections per image
                if det is not None and len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
            return det
