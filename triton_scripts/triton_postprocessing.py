import sys

import numpy as np
import torch

sys.path.append("..")

from app import settings


def make_grid(nx=20, ny=20):
    yv, xv = torch.meshgrid([torch.arange(ny), torch.arange(nx)])
    return torch.stack((xv, yv), 2).view((1, 1, ny, nx, 2)).float()


class Postprocess(object):
    def __init__(self):
        detection_layer = torch.load(settings.WEIGHTS_PT,
                                     map_location=torch.device('cpu'))['model'].float().eval().model[-1]

        self.anchors = detection_layer.anchor_grid.cpu().numpy().reshape((9, 2))
        self.nl = detection_layer.nl  # number of detection layers
        self.no = detection_layer.no  # number of outputs per anchor
        self.na = detection_layer.na  # number of anchors
        self.stride = detection_layer.stride  # strides computed during build
        self.grid = [torch.zeros(1)] * self.nl  # init grid
        self.a = torch.tensor(self.anchors).float().view(self.nl, -1, 2)
        self.anchor_grid = self.a.clone().view(self.nl, 1, -1, 1, 1, 2)
        self.device = torch.device('cpu')

    def get_final_ancors(self, model_output):
        print(self.anchors)
        print(self.nl)
        print(self.no)
        print(self.na)
        print(self.stride)
        z = []
        for i in range(len(model_output)):
            if isinstance(model_output[i], np.ndarray):
                model_output[i] = torch.from_numpy(model_output[i]).to(torch.float32).to(self.device)
            bs, _, ny, nx, _ = model_output[i].shape  # x(bs,255,20,20) to x(bs,3,20,20,85)
            # inference
            if self.grid[i].shape[2:4] != model_output[i].shape[2:4]:
                self.grid[i] = make_grid(nx, ny).to(self.device)

            y_i = model_output[i].sigmoid()
            y_i[..., 0:2] = (y_i[..., 0:2] * 2. - 0.5 + self.grid[i].to(self.device)) * self.stride[i]  # xy
            y_i[..., 2:4] = (y_i[..., 2:4] * 2) ** 2 * self.anchor_grid[i].to(self.device)  # wh
            z.append(y_i.view(bs, -1, self.no))
        z = torch.cat(z, 1)
        return z
