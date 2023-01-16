# Detector Net
WEIGHTS_PT = '../weights/documents_pretrain_2mln.pt'
WEIGHTS_ONNX = '../triton_models/detector/1/model.onnx'
TEST_IMAGE = '../test_image.jpg'
CONFIDENCE = 0.5
IOU_TRESHOLD = 0.5
IMG_SIZE = 640
DEVICE = '0'

MODEL_NAMES = ['text', 'title', 'list', 'table', 'figure']

# triton config
TRITON_INFERENCE = True
TRITON_URL = '127.0.0.1:9005'
MODEL_NAME = 'detector'
MODEL_VERSION = '1'

# you can set different trashholds for different classes (defaults - equal)
DICT_CONF_TRESH = {
    0: CONFIDENCE,
    1: CONFIDENCE,
    2: CONFIDENCE,
    3: CONFIDENCE,
    4: CONFIDENCE
    }

DICT_IOU_TRESH = {
    0: IOU_TRESHOLD,
    1: IOU_TRESHOLD,
    2: IOU_TRESHOLD,
    3: IOU_TRESHOLD,
    4: IOU_TRESHOLD
    }
