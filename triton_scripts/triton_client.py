import logging
import sys
import time

import numpy as np
import torch
import tritonclient.grpc as grpcclient
from grpc import aio
from tritonclient.grpc import service_pb2_grpc as prediction_service_pb2_grpc

sys.path.append("..")

from app import settings
from triton_postprocessing import Postprocess
from utils.datasets import letterbox
from utils.general import non_max_suppression, scale_coords

log = logging.getLogger()


class Client(object):

    def __init__(self, url, model_name) -> None:
        # triton server settings
        self.url = url
        self.model = model_name
        self.model_version = str(settings.MODEL_VERSION)
        self.input_name = 'images'
        self.output_name = ['output1', 'output2', 'output3']
        self.dtype = 'FP32'

        self.triton_client = grpcclient.InferenceServerClient(
                url=self.url,
                verbose=False,
                )

        # init postprocess for model output
        self.Postproc = Postprocess()

        # class names in weights
        self.model_names = settings.MODEL_NAMES
        # if 'p' in self.model_names:
        #     self.model_names.remove('p')

        # porstproc for output anchors
        self.agnostic_nms = False
        self.classes = None
        self.dict_conf_treh = settings.DICT_CONF_TRESH
        self.dict_iou_treh = settings.DICT_IOU_TRESH

    def preprocess(self, img, IMG_SIZE=640):
        """
        Preprocessed image
        """
        img = letterbox(img, new_shape=(IMG_SIZE, IMG_SIZE))[0]
        # Convert
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)
        img = img.astype('float32')

        for i, item in enumerate(img.copy()):
            img[i, :, :] = item/255

        img = np.expand_dims(img, axis=0)
        self.preproc_img_shape = img.shape
        log.info('Image shape after preproc:\n%s', self.preproc_img_shape)
        return img

    def NMS(self, model_res, img_shape):

        output_list = []
        output = None
        for i, item in enumerate(self.model_names):

            output_list = non_max_suppression(
                model_res,
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
                det[:, :4] = scale_coords(self.preproc_img_shape[2:], det[:, :4], img_shape).round()
        return det

    async def predict(self, image, req_id):
        start = time.time()
        img_orig_shape = image.shape
        image = self.preprocess(image)
        inputs = [
            grpcclient.InferInput(self.input_name, image.shape, self.dtype)
        ]

        inputs[0].set_data_from_numpy(image)

        outputs = [
            grpcclient.InferRequestedOutput(self.output_name[0]),
            grpcclient.InferRequestedOutput(self.output_name[1]),
            grpcclient.InferRequestedOutput(self.output_name[2])
            ]

        async with aio.insecure_channel(self.url) as channel:
            stub = prediction_service_pb2_grpc.GRPCInferenceServiceStub(channel)
            request = grpcclient._get_inference_request(
                    model_name=self.model,
                    model_version=self.model_version,
                    inputs=inputs,
                    outputs=outputs,
                    timeout=10,
                    request_id="",
                    sequence_id=0,
                    sequence_start=0,
                    sequence_end=0,
                    priority=0,
                )
            response = await stub.ModelInfer(request)

        res = grpcclient.InferResult(response)

        res_outputs = []
        for out_name in self.output_name:
            res_outputs.append(res.as_numpy(out_name))

        eval_anchors = self.Postproc.get_final_ancors(res_outputs)
        res = self.NMS(eval_anchors, img_orig_shape)
        end = time.time() - start
        return res, end
