import sys

import cv2
import numpy as np
import torch

sys.path.append("..")

import settings
from utils.datasets import letterbox

if __name__ == '__main__':
    img_size = [settings.IMG_SIZE] * 2
    test_image = settings.TEST_IMAGE
    # Input
    im0 = cv2.imread(test_image)
    # img = torch.zeros((opt.batch_size, 3, *opt.img_size))  # image size(1,3,320,192) iDetection
    img = letterbox(im0, new_shape=img_size)[0]
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
    img = np.ascontiguousarray(img)
    img = torch.from_numpy(img).to(torch.device('cpu'))
    img = img.float()  # uint8 to fp16/32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    # Load PyTorch model
    model = torch.load(settings.WEIGHTS_PT, map_location=torch.device('cpu'))['model'].float()
    model.eval()
    model.model[-1].export = True  # set Detect() layer export=True
    y = model(img)  # dry run
    print('anchor grid: ', model.eval().model[-1].anchor_grid.cpu().numpy().reshape((9, 2)))

    # ONNX export
    try:
        import onnx

        print('\nStarting ONNX export with onnx %s...' % onnx.__version__)
        f = settings.WEIGHTS_ONNX  # filename
        # f = '../triton_models/detector/1/model.onnx'
        model.fuse()  # only for ONNX
        torch.onnx.export(model, img, f, verbose=False, opset_version=12, input_names=['images'],
                          output_names=['output1', 'output2','output3'],
                          dynamic_axes={
                                        'images': {0: 'batch_size', 2: 'image_height', 3: 'image_width'},
                                        'output1': {0: 'batch_size'},
                                        'output2': {0: 'batch_size'},
                                        'output3': {0: 'batch_size'},
                                        }
                          )

        # Checks
        onnx_model = onnx.load(f)  # load onnx model
        onnx.checker.check_model(onnx_model)  # check onnx model
        print(onnx.helper.printable_graph(onnx_model.graph))  # print a human readable model
        print('ONNX export success, saved as %s' % f)
    except Exception as e:
        print('ONNX export failure: %s' % e)

    # # Finish
    print('\nExport complete. Visualize with https://github.com/lutzroeder/netron.')
