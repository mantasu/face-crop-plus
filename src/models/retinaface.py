import os
import cv2
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.models._utils as _utils

from math import ceil
from itertools import product

import sys
sys.path.append("src")

# def prep_img(
#     image: np.ndarray,
#     padding_mode: str = "constant",
#     size: int = 512,
#     offset: tuple[int] = (-104, -117, -123),
#     as_rgb: bool = False,
# ) -> tuple[torch.Tensor, torch.Tensor]:
#     padding_modes = {
#         "reflect": cv2.BORDER_REFLECT,
#         "replicate": cv2.BORDER_REPLICATE,
#         "constant": cv2.BORDER_CONSTANT,
#         "reflect_101": cv2.BORDER_REFLECT_101
#     }
#     if isinstance(size, int):
#         size = (size, size)
    
#     (h, w), m = image.shape[:2], max(*image.shape[:2])
#     interpolation = cv2.INTER_AREA if m > max(size) else cv2.INTER_CUBIC

#     padding = (m - h) // 2, (m - h + 1) // 2, (m - w) // 2, (m - w + 1) // 2
#     image = cv2.copyMakeBorder(image, *padding, padding_modes[padding_mode])
#     image = cv2.resize(image, size, interpolation=interpolation)
#     image = image + offset

#     transform_back = [m / size[0], m / size[1], padding[2], padding[0]]

#     if as_rgb:
#         image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

#     return torch.from_numpy(image).float(), torch.tensor(transform_back)

def py_cpu_nms(dets: torch.Tensor, thresh):
    """Pure Python NMS baseline."""
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort(descending=True)

    keep = []
    while order.numel() > 0:
        i = order[0]
        keep.append(i)
        xx1 = torch.maximum(x1[i], x1[order[1:]])
        yy1 = torch.maximum(y1[i], y1[order[1:]])
        xx2 = torch.minimum(x2[i], x2[order[1:]])
        yy2 = torch.minimum(y2[i], y2[order[1:]])

        w = torch.maximum(torch.tensor(0.0), xx2 - xx1 + 1)
        h = torch.maximum(torch.tensor(0.0), yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = torch.where(ovr <= thresh)[0]
        order = order[inds + 1]

    return keep

# def decode(loc, priors, variances):
#     """Decode locations from predictions using priors to undo
#     the encoding we did for offset regression at train time.
#     Args:
#         loc (tensor): location predictions for loc layers,
#             Shape: [num_priors,4]
#         priors (tensor): Prior boxes in center-offset form.
#             Shape: [num_priors,4].
#         variances: (list[float]) Variances of priorboxes
#     Return:
#         decoded bounding box predictions
#     """
#     boxes = torch.cat((
#         priors[:, :2] + loc[..., :2] * variances[0] * priors[:, 2:],
#         priors[:, 2:] * torch.exp(loc[..., 2:] * variances[1])), 2)
#     boxes[..., :2] -= boxes[..., 2:] / 2
#     boxes[..., 2:] += boxes[..., :2]
#     return boxes

# def decode_landm(pre, priors, variances):
#     """Decode landm from predictions using priors to undo
#     the encoding we did for offset regression at train time.
#     Args:
#         pre (tensor): landm predictions for loc layers,
#             Shape: [num_priors,10]
#         priors (tensor): Prior boxes in center-offset form.
#             Shape: [num_priors,4].
#         variances: (list[float]) Variances of priorboxes
#     Return:
#         decoded landm predictions
#     """
#     landms = torch.cat((priors[..., :2] + pre[..., :2] * variances[0] * priors[..., 2:],
#                         priors[..., :2] + pre[..., 2:4] * variances[0] * priors[..., 2:],
#                         priors[..., :2] + pre[..., 4:6] * variances[0] * priors[..., 2:],
#                         priors[..., :2] + pre[..., 6:8] * variances[0] * priors[..., 2:],
#                         priors[..., :2] + pre[..., 8:10] * variances[0] * priors[..., 2:],
#                         ), dim=2)
#     return landms

def conv_bn(inp, oup, stride = 1, leaky = 0):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.LeakyReLU(negative_slope=leaky, inplace=True)
    )

def conv_bn_no_relu(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
    )

def conv_bn1X1(inp, oup, stride, leaky=0):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, stride, padding=0, bias=False),
        nn.BatchNorm2d(oup),
        nn.LeakyReLU(negative_slope=leaky, inplace=True)
    )

def conv_dw(inp, oup, stride, leaky=0.1):
    return nn.Sequential(
        nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
        nn.BatchNorm2d(inp),
        nn.LeakyReLU(negative_slope= leaky,inplace=True),

        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.LeakyReLU(negative_slope= leaky,inplace=True),
    )


class PriorBox(object):
    def __init__(self, image_size=None):
        super(PriorBox, self).__init__()
        # self.min_sizes = cfg['min_sizes']
        # self.steps = cfg['steps']
        # self.clip = cfg['clip']
        self.min_sizes = [[16, 32], [64, 128], [256, 512]]
        self.steps = [8, 16, 32]
        self.clip = False
        self.image_size = image_size
        self.feature_maps = [[ceil(self.image_size[0]/step), ceil(self.image_size[1]/step)] for step in self.steps]
        self.name = "s"

    def forward(self):
        anchors = []
        for k, f in enumerate(self.feature_maps):
            min_sizes = self.min_sizes[k]
            for i, j in product(range(f[0]), range(f[1])):
                for min_size in min_sizes:
                    s_kx = min_size / self.image_size[1]
                    s_ky = min_size / self.image_size[0]
                    dense_cx = [x * self.steps[k] / self.image_size[1] for x in [j + 0.5]]
                    dense_cy = [y * self.steps[k] / self.image_size[0] for y in [i + 0.5]]
                    for cy, cx in product(dense_cy, dense_cx):
                        anchors += [cx, cy, s_kx, s_ky]

        # back to torch land
        output = torch.Tensor(anchors).view(-1, 4)
        if self.clip:
            output.clamp_(max=1, min=0)
        return output


class SSH(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(SSH, self).__init__()
        assert out_channel % 4 == 0
        leaky = 0
        if (out_channel <= 64):
            leaky = 0.1
        self.conv3X3 = conv_bn_no_relu(in_channel, out_channel//2, stride=1)

        self.conv5X5_1 = conv_bn(in_channel, out_channel//4, stride=1, leaky = leaky)
        self.conv5X5_2 = conv_bn_no_relu(out_channel//4, out_channel//4, stride=1)

        self.conv7X7_2 = conv_bn(out_channel//4, out_channel//4, stride=1, leaky = leaky)
        self.conv7x7_3 = conv_bn_no_relu(out_channel//4, out_channel//4, stride=1)

    def forward(self, input):
        conv3X3 = self.conv3X3(input)

        conv5X5_1 = self.conv5X5_1(input)
        conv5X5 = self.conv5X5_2(conv5X5_1)

        conv7X7_2 = self.conv7X7_2(conv5X5_1)
        conv7X7 = self.conv7x7_3(conv7X7_2)

        out = torch.cat([conv3X3, conv5X5, conv7X7], dim=1)
        out = F.relu(out)
        return out


class FPN(nn.Module):
    def __init__(self,in_channels_list,out_channels):
        super(FPN,self).__init__()
        leaky = 0
        if (out_channels <= 64):
            leaky = 0.1
        self.output1 = conv_bn1X1(in_channels_list[0], out_channels, stride = 1, leaky = leaky)
        self.output2 = conv_bn1X1(in_channels_list[1], out_channels, stride = 1, leaky = leaky)
        self.output3 = conv_bn1X1(in_channels_list[2], out_channels, stride = 1, leaky = leaky)

        self.merge1 = conv_bn(out_channels, out_channels, leaky = leaky)
        self.merge2 = conv_bn(out_channels, out_channels, leaky = leaky)

    def forward(self, input):
        # names = list(input.keys())
        input = list(input.values())

        output1 = self.output1(input[0])
        output2 = self.output2(input[1])
        output3 = self.output3(input[2])

        up3 = F.interpolate(output3, size=[output2.size(2), output2.size(3)], mode="nearest")
        output2 = output2 + up3
        output2 = self.merge2(output2)

        up2 = F.interpolate(output2, size=[output1.size(2), output1.size(3)], mode="nearest")
        output1 = output1 + up2
        output1 = self.merge1(output1)

        out = [output1, output2, output3]
        return out


# class MobileNetV1(nn.Module):
#     def __init__(self):
#         super(MobileNetV1, self).__init__()
#         self.stage1 = nn.Sequential(
#             conv_bn(3, 8, 2, leaky = 0.1),    # 3
#             conv_dw(8, 16, 1),   # 7
#             conv_dw(16, 32, 2),  # 11
#             conv_dw(32, 32, 1),  # 19
#             conv_dw(32, 64, 2),  # 27
#             conv_dw(64, 64, 1),  # 43
#         )
#         self.stage2 = nn.Sequential(
#             conv_dw(64, 128, 2),  # 43 + 16 = 59
#             conv_dw(128, 128, 1), # 59 + 32 = 91
#             conv_dw(128, 128, 1), # 91 + 32 = 123
#             conv_dw(128, 128, 1), # 123 + 32 = 155
#             conv_dw(128, 128, 1), # 155 + 32 = 187
#             conv_dw(128, 128, 1), # 187 + 32 = 219
#         )
#         self.stage3 = nn.Sequential(
#             conv_dw(128, 256, 2), # 219 +3 2 = 241
#             conv_dw(256, 256, 1), # 241 + 64 = 301
#         )
#         self.avg = nn.AdaptiveAvgPool2d((1,1))
#         self.fc = nn.Linear(256, 1000)

#     def forward(self, x):
#         x = self.stage1(x)
#         x = self.stage2(x)
#         x = self.stage3(x)
#         x = self.avg(x)
#         # x = self.model(x)
#         x = x.view(-1, 256)
#         x = self.fc(x)
#         return x


class ClassHead(nn.Module):
    def __init__(self,inchannels=512,num_anchors=3):
        super(ClassHead,self).__init__()
        self.num_anchors = num_anchors
        self.conv1x1 = nn.Conv2d(inchannels,self.num_anchors*2,kernel_size=1)

    def forward(self,x):
        out = self.conv1x1(x)
        out = out.permute(0,2,3,1).contiguous()
        
        return out.view(out.shape[0], -1, 2)


class BboxHead(nn.Module):
    def __init__(self,inchannels=512,num_anchors=3):
        super(BboxHead,self).__init__()
        self.conv1x1 = nn.Conv2d(inchannels,num_anchors*4,kernel_size=1)

    def forward(self,x):
        out = self.conv1x1(x)
        out = out.permute(0,2,3,1).contiguous()

        return out.view(out.shape[0], -1, 4)


class LandmarkHead(nn.Module):
    def __init__(self,inchannels=512,num_anchors=3):
        super(LandmarkHead,self).__init__()
        self.conv1x1 = nn.Conv2d(inchannels,num_anchors*10,kernel_size=1)

    def forward(self,x):
        out = self.conv1x1(x)
        out = out.permute(0,2,3,1).contiguous()

        return out.view(out.shape[0], -1, 10)


class RetinaFace(nn.Module):
    def __init__(
        self,
        backbone_name: str = "resnet50",
        size: int | tuple[int, int] = 512,
        vis_threshold: float = 0.6,
        nms_threshold: float = 0.4,
        strategy: str = "largest",
        variance: list[int] = [0.1, 0.2],
        padding_mode: str = "constant",
        device: str | torch.device = "cpu",
    ):
        super().__init__()
        self.size = size
        self.vis_threshold = vis_threshold
        self.nms_threshold = nms_threshold
        self.strategy = strategy
        self.variance = variance
        self.padding_mode = padding_mode
        self.device = device

        backbone = None

        # if backbone_name == "mobilenet0.25":
        #     backbone = MobileNetV1()
        #     return_layers = {'stage1': 1, 'stage2': 2, 'stage3': 3}
        #     in_channels = 32
        #     out_channels = 64
        #     weights_path = "weights/retinaface-pytorch-mobilenet0.25.pth"

        if backbone_name == "resnet50":
            backbone = models.resnet50()
            return_layers = {'layer2': 1, 'layer3': 2, 'layer4': 3}
            in_channels = 256
            out_channels = 256
            # weights_path = "weights/retinaface-pytorch-resnet50.pth"
            weights_path = "weights/RetinaFace-R50.pth"

        self.body = _utils.IntermediateLayerGetter(backbone, return_layers)

        in_channels_list = [
            in_channels * 2,
            in_channels * 4,
            in_channels * 8,
        ]
        # out_channels = cfg['out_channel']
        self.fpn = FPN(in_channels_list, out_channels)
        self.ssh1 = SSH(out_channels, out_channels)
        self.ssh2 = SSH(out_channels, out_channels)
        self.ssh3 = SSH(out_channels, out_channels)

        self.ClassHead = self._make_class_head(fpn_num=3, inchannels=out_channels)
        self.BboxHead = self._make_bbox_head(fpn_num=3, inchannels=out_channels)
        self.LandmarkHead = self._make_landmark_head(fpn_num=3, inchannels=out_channels)

        # self.load_state_dict(torch.load(weights_path))
        self.load(weights_path)
    
    def remove_prefix(self, state_dict, prefix):
        ''' Old style model is stored with all names of parameters sharing common prefix 'module.' '''
        f = lambda x: x.split(prefix, 1)[-1] if x.startswith(prefix) else x
        return {f(key): value for key, value in state_dict.items()}

    def load(self, weights_path, load_to_cpu=False):
        if load_to_cpu:
            pretrained_dict = torch.load(weights_path, map_location=lambda storage, loc: storage)
        else:
            pretrained_dict = torch.load(weights_path, map_location=lambda storage, loc: storage.cuda())
        if "state_dict" in pretrained_dict.keys():
            pretrained_dict = self.remove_prefix(pretrained_dict['state_dict'], 'module.')
        else:
            pretrained_dict = self.remove_prefix(pretrained_dict, 'module.')
        
        self.load_state_dict(pretrained_dict, strict=False)

    def _make_class_head(self,fpn_num=3,inchannels=64,anchor_num=2):
        classhead = nn.ModuleList()
        for i in range(fpn_num):
            classhead.append(ClassHead(inchannels,anchor_num))
        return classhead
    
    def _make_bbox_head(self,fpn_num=3,inchannels=64,anchor_num=2):
        bboxhead = nn.ModuleList()
        for i in range(fpn_num):
            bboxhead.append(BboxHead(inchannels,anchor_num))
        return bboxhead

    def _make_landmark_head(self,fpn_num=3,inchannels=64,anchor_num=2):
        landmarkhead = nn.ModuleList()
        for i in range(fpn_num):
            landmarkhead.append(LandmarkHead(inchannels,anchor_num))
        return landmarkhead

    def forward(self,inputs):
        out = self.body(inputs)

        # FPN
        fpn = self.fpn(out)

        # SSH
        feature1 = self.ssh1(fpn[0])
        feature2 = self.ssh2(fpn[1])
        feature3 = self.ssh3(fpn[2])
        features = [feature1, feature2, feature3]

        bbox_regressions = torch.cat([self.BboxHead[i](feature) for i, feature in enumerate(features)], dim=1)
        classifications = torch.cat([self.ClassHead[i](feature) for i, feature in enumerate(features)],dim=1)
        ldm_regressions = torch.cat([self.LandmarkHead[i](feature) for i, feature in enumerate(features)], dim=1)

        # if self.phase == 'train':
        #     output = (bbox_regressions, classifications, ldm_regressions)
        # else:
        output = (bbox_regressions, F.softmax(classifications, dim=-1), ldm_regressions)
        
        return output
    
    def decode_bboxes(self, loc, priors):
        """Decode locations from predictions using priors to undo
        the encoding we did for offset regression at train time.
        Args:
            loc (tensor): location predictions for loc layers,
                Shape: [num_priors,4]
            priors (tensor): Prior boxes in center-offset form.
                Shape: [num_priors,4].
            variances: (list[float]) Variances of priorboxes
        Return:
            decoded bounding box predictions
        """
        boxes = torch.cat((
            priors[:, :2] + loc[..., :2] * self.variance[0] * priors[:, 2:],
            priors[:, 2:] * torch.exp(loc[..., 2:] * self.variance[1])), 2)
        boxes[..., :2] -= boxes[..., 2:] / 2
        boxes[..., 2:] += boxes[..., :2]
        return boxes

    def decode_landms(self, pre, priors):
        """Decode landm from predictions using priors to undo
        the encoding we did for offset regression at train time.
        Args:
            pre (tensor): landm predictions for loc layers,
                Shape: [num_priors,10]
            priors (tensor): Prior boxes in center-offset form.
                Shape: [num_priors,4].
            variances: (list[float]) Variances of priorboxes
        Return:
            decoded landm predictions
        """
        landms = torch.cat((priors[..., :2] + pre[..., :2] * self.variance[0] * priors[..., 2:],
                            priors[..., :2] + pre[..., 2:4] * self.variance[0] * priors[..., 2:],
                            priors[..., :2] + pre[..., 4:6] * self.variance[0] * priors[..., 2:],
                            priors[..., :2] + pre[..., 6:8] * self.variance[0] * priors[..., 2:],
                            priors[..., :2] + pre[..., 8:10] * self.variance[0] * priors[..., 2:],
                            ), dim=2)
        return landms
    
    def parse_landmarks(
        self,
        scores: torch.Tensor,
        bboxes: torch.Tensor,
        landms: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, list[int]]:
        """Parses predictions for identified faces for each sample
        
        This method works as follows:
            1. First, it filters out bad predictions based on
               `self.vis_threshold`.
            2. Then it gathers all the remaining predictions across the
               batch dimension, i.e., the batch dimension becomes not
               the number of samples but the number of filtered out
               predictions.
            3. It loops for each set of filtered predictions per sample
               sorting each set of confidence scores from best to worst.
            4. For each set of confidence scores, it identifies distinct 
               faces and keeps the record of which indices to keep. At 
               this stage it uses `self.nms_threshold` to remove the 
               duplicate face predictions.
            5. Finally, it applies the kept indices for each person
               (each face) to select corresponding bounding boxes and
               landmarks.

        Note:
            N corresponds to batch size and `out_dim` corresponds to
            the total guesses that the model made about each sample.
            Within those guesses, there typically exists at least 1
            face but can be more. By default, it should be 43,008.

        Args:
            scores: The confidence score predictions of shape
                (N, out_dim).
            bboxes: The bounding boxes for each face of shape
                (N, out_dim, 4) where the last 4 numbers correspond to
                start and end coordinates - x1, y1, x2, y2.
            landms: The landmarks for each face of shape
                (N, out_dim, num_landmarks * 2) where the last dim 
                corresponds to landmark coordinates x1, y1, ... . By
                default, num_landmarks is 5.

        Returns:
            A tuple where the first element is a torch tensor of shape
            (num_faces, 4), the second element is a torch tensor of
            shape (num_faces, num_landmarks * 2) and the third element
            is a list of length num_faces. First and second elements
            correspond to bounding boxes and landmarks for each face
            across all samples and the third element provides an index
            for each bounding box/set of landmarks that identifies
            which sample that box/set (or that face) is extracted from
            (because each sample can have multiple faces).
        """
        # Init variables, identify masks to filter best faces
        cumsum, people_indices, sample_indices = 0, [], []
        masks = scores > self.vis_threshold

        # Flatten across batch filtered predictions, compute face areas
        scores, bboxes, landms = scores[masks], bboxes[masks], landms[masks]
        areas = (bboxes[:, 2]-bboxes[:, 0]+1) * (bboxes[:, 3]-bboxes[:, 1]+1)

        for i, num_valid in enumerate(masks.sum(dim=1)):
            # Extract all face preds for a single sample
            start, end, keep = cumsum, cumsum+num_valid, []
            bbox, area = bboxes[start:end], areas[start:end]
            scores_sorted = scores[start:end].argsort(descending=True)

            while scores_sorted.numel() > 0:
                # Append best face's index to keep
                keep.append(j := scores_sorted[0])
                
                # Find coordinates that at least bound the current face
                xy1 = torch.maximum(bbox[j, :2], bbox[scores_sorted[1:], :2])
                xy2 = torch.minimum(bbox[j, 2:], bbox[scores_sorted[1:], 2:])

                # Compute width and height for the current minimal face
                w = torch.maximum(torch.tensor(0.0), xy2[:, 0] - xy1[:, 0] + 1)
                h = torch.maximum(torch.tensor(0.0), xy2[:, 1] - xy1[:, 1] + 1)

                # Compute nms for identifying areas for the current face
                ovr = (a := w * h) / (area[j] + area[scores_sorted[1:]] - a)
                
                # Filter out current face, keep next best scores
                inds = torch.where(ovr <= self.nms_threshold)[0]
                scores_sorted = scores_sorted[inds + 1]
            
            # Update people and sample indices, increment cumsum
            people_indices.extend([cumsum + k for k in keep])
            sample_indices.extend([i] * len(keep))
            cumsum += num_valid
        
        # Slect the final landms and bboxes
        bboxes = bboxes[people_indices, :]
        landms = landms[people_indices, :]
        
        return landms, bboxes, sample_indices
    
    @torch.no_grad()
    def predict(self, images: torch.Tensor) -> tuple[torch.Tensor, list[int]]:
        # Retrieve the device the model is on
        device = next(self.parameters()).device

        # Convert images to appropriate input and predict landmarks
        x = images.flip(3).permute(0, 3, 1, 2).float().to(device)
        x -= torch.tensor([[[104]], [[117]], [[123]]], device=device)
        bboxes, scores, landms = self(x)

        # Create prior boxes and scale factors to decode bboxes & landms
        priors = PriorBox((x.size(2), x.size(3))).forward().to(device)
        scale_b = torch.tensor([x.size(3), x.size(2)] * 2, device=device)
        scale_l = torch.tensor([x.size(3), x.size(2)] * 5, device=device)

        # Decode the predictions and use them to parse landmarks
        scores = scores[..., 1]
        bboxes = self.decode_bboxes(bboxes, priors) * scale_b
        landms = self.decode_landms(landms, priors) * scale_l
        landms, bboxes, indices = self.parse_landmarks(scores, bboxes, landms)

        landms_new, indices_new = [], []
        cache = {"indices": [], "bboxes": [], "landms": []}

        for i in range(len(indices)):

            # Add everything to cache
            cache["indices"].append(indices[i])
            cache["bboxes"].append(bboxes[i])
            cache["landms"].append(landms[i])

            if i != len(indices) - 1 and cache["indices"][-1] == indices[i+1]:
                # No operations until cache for current idx is complete
                continue

            match self.strategy:
                case "all":
                    # Append all landmarks and indices
                    landms_new.extend(cache["landms"])
                    indices_new.extend(cache["indices"])
                case "best":
                    # Append the first set of landmarks
                    landms_new.append(cache["landms"][0])
                    indices_new.append(cache["indices"][0])
                case "largest":
                    # Compute bounding box areas
                    bbs = torch.stack(cache["bboxes"])
                    areas = (bbs[:, 2] - bbs[:, 0] + 1) *\
                            (bbs[:, 3] - bbs[:, 1] + 1)

                    # Append only the largest face landmarks and its idx
                    landms_new.append(cache["landms"][areas.argmax()])
                    indices_new.append(cache["indices"][0])
                case _:
                    raise ValueError(f"Unsupported startegy: {self.strategy}")
            
            # Clear cache (reinitialize empty lists)
            cache = {k: [] for k in cache.keys()}
        
        landms = torch.stack(landms_new)
        indices = indices_new

        return landms.view(len(landms), -1, 2), indices
        
        # indices, landmarks = [], []
        # keeps = []
        # cumsum = 0

        # for i in range(len(images)):
        #     inds = torch.where(scores[i] > self.vis_threshold)[0]
        #     bbs, lms, scs = bboxes[i][inds], landms[i][inds], scores[i][inds]
        #     keep = py_cpu_nms(torch.hstack((bbs, scs)), self.nms_threshold)

        #     keeps.extend([cumsum + k for k in keep])
        #     cumsum += len(inds)

        #     lms = lms[keep, :]
            
        #     lms[:, ::2] = (lms[:, ::2] - pad_tmp[i][2]) * sc_tmp[i]
        #     lms[:, 1::2] = (lms[:, 1::2] - pad_tmp[i][0]) * sc_tmp[i]

        #     if len(keep) == 0:
        #         continue

        #     if self.strategy == "largest":
        #         bbs = bbs[keep, :]
        #         idx = torch.argmax((bbs[:, 2] * bbs[:, 3]).squeeze())
        #         indices.append(i)
        #         landmarks.append(lms[idx].reshape(-1, 2))
        #     elif self.strategy == "all":
        #         indices.extend([i] * len(keep))
        #         landmarks.extend([*lms.reshape(len(lms), -1, 2)])
        #     else:
        #         raise ValueError(f"Unsupported strategy: '{self.strategy}'.")
        
        # print(keeps)
        
        # return landmarks, indices
    
    # @torch.no_grad()
    # def predict(
    #     self,
    #     images: list[np.ndarray],
    #     device: str | torch.device = "cpu",
    # ):
    #     # x = torch.stack([prep_img(image, padding, size) for image in images])
    #     x, transform_back = [], []

    #     for image in images:
    #         img, t = prep_img(image, self.padding_mode, self.size)
    #         x.append(img)
    #         transform_back.append(t)

    #     x = torch.stack(x)

    #     x = x.permute(0, 3, 1, 2).to(device)
    #     loc, conf, landms = self.to(device)(x)

    #     priors = PriorBox(image_size=(x.size(2), x.size(3))).forward().to(device)
    #     scale_b = torch.Tensor([x.size(3), x.size(2)] * 2, device=device)
    #     scale_l = torch.Tensor([x.size(3), x.size(2)] * 5, device=device)

    #     scores = conf[..., 1:2]
    #     boxes = decode(loc, priors, self.variance) * scale_b
    #     landms = decode_landm(landms, priors, self.variance) * scale_l

    #     indices, landmarks = [], []

    #     for i in range(len(images)):
    #         inds = torch.where(scores[i] > self.vis_threshold)[0]
    #         bbs, lms, scs = boxes[i][inds], landms[i][inds], scores[i][inds]
    #         keep = py_cpu_nms(torch.hstack((bbs, scs)), self.nms_threshold)

    #         lms = lms[keep, :]
            
    #         lms[:, ::2] = lms[:, ::2] * transform_back[i][0] - transform_back[i][2]
    #         lms[:, 1::2] = lms[:, 1::2] * transform_back[i][1] - transform_back[i][3]


    #         if len(keep) == 0:
    #             continue

    #         if self.strategy == "largest":
    #             bbs = bbs[keep, :]
    #             idx = torch.argmax((bbs[:, 2] * bbs[:, 3]).squeeze())
    #             indices.append(i)
    #             landmarks.append(lms[idx].reshape(-1, 2))
    #         elif self.strategy == "all":
    #             indices.extend([i] * len(keep))
    #             landmarks.extend([*lms.reshape(len(lms), -1, 2)])
    #         else:
    #             raise ValueError(f"Unsupported strategy: '{self.strategy}'.")
        
    #     return torch.stack(landmarks), indices
