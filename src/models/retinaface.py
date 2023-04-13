import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from math import ceil
from itertools import product
from torchvision.models._utils import IntermediateLayerGetter


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
        self.conv3X3 = self.conv_bn_no_relu(in_channel, out_channel//2, stride=1)

        self.conv5X5_1 = self.conv_bn(in_channel, out_channel//4, stride=1, leaky = leaky)
        self.conv5X5_2 = self.conv_bn_no_relu(out_channel//4, out_channel//4, stride=1)

        self.conv7X7_2 = self.conv_bn(out_channel//4, out_channel//4, stride=1, leaky = leaky)
        self.conv7x7_3 = self.conv_bn_no_relu(out_channel//4, out_channel//4, stride=1)
    
    def conv_bn(self, inp, oup, stride = 1, leaky = 0):
        return nn.Sequential(
            nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
            nn.BatchNorm2d(oup),
            nn.LeakyReLU(negative_slope=leaky, inplace=True)
        )

    def conv_bn_no_relu(self, inp, oup, stride):
        return nn.Sequential(
            nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
            nn.BatchNorm2d(oup),
        )

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
        self.output1 = self.conv_bn1X1(in_channels_list[0], out_channels, stride = 1, leaky = leaky)
        self.output2 = self.conv_bn1X1(in_channels_list[1], out_channels, stride = 1, leaky = leaky)
        self.output3 = self.conv_bn1X1(in_channels_list[2], out_channels, stride = 1, leaky = leaky)

        self.merge1 = self.conv_bn(out_channels, out_channels, leaky = leaky)
        self.merge2 = self.conv_bn(out_channels, out_channels, leaky = leaky)
    
    def conv_bn(self, inp, oup, stride = 1, leaky = 0):
        return nn.Sequential(
            nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
            nn.BatchNorm2d(oup),
            nn.LeakyReLU(negative_slope=leaky, inplace=True)
        )
    
    def conv_bn1X1(self, inp, oup, stride, leaky=0):
        return nn.Sequential(
            nn.Conv2d(inp, oup, 1, stride, padding=0, bias=False),
            nn.BatchNorm2d(oup),
            nn.LeakyReLU(negative_slope=leaky, inplace=True)
        )

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

class Head(nn.Module):
    def __init__(self, num_out: int, in_channels: int = 512):
        super().__init__()
        self.num_out = num_out
        self.conv1x1 = nn.Conv2d(in_channels, 2 * num_out, 1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1x1(x)
        out = x.permute(0, 2, 3, 1).contiguous()
        
        return out.view(out.size(0), -1, self.num_out)
    
    @classmethod
    def make(cls, num_out: int, in_channels: int = 512) -> nn.ModuleList:
        head = nn.ModuleList([cls(num_out, in_channels) for _ in range(3)])
        return head

class RetinaFace(nn.Module):
    def __init__(self, strategy: str = "all"):
        super().__init__()

        # Initialize attributes
        self.strategy = strategy
        self.vis_threshold = 0.6
        self.nms_threshold = 0.4
        self.variance = [0.1, 0.2]

        # Set up backbone and config
        backbone = models.resnet50()
        in_channels, out_channels = 256, 256
        in_channels_list = [in_channels * x for x in [2, 4, 8]]
        return_layers = {'layer2': 1, 'layer3': 2, 'layer4': 3}

        # backbone = MobileNetV1()
        # in_channels, out_channels = 32, 64
        # in_channels_list = [in_channels * x for x in [2, 4, 8]]
        # return_layers = {'stage1': 1, 'stage2': 2, 'stage3': 3}

        # Construct the backbone by retrieving intermediate layers
        self.body = IntermediateLayerGetter(backbone, return_layers)

        # Construct sub-layers to extract features for heads
        self.fpn = FPN(in_channels_list, out_channels)
        self.ssh1 = SSH(out_channels, out_channels)
        self.ssh2 = SSH(out_channels, out_channels)
        self.ssh3 = SSH(out_channels, out_channels)

        # Construct 3 heads - score, bboxes & landms
        self.ClassHead = Head.make(2, out_channels)
        self.BboxHead = Head.make(4, out_channels)
        self.LandmarkHead = Head.make(10, out_channels)

        # Load weights
        self.load()
    
    def remove_prefix(self, state_dict, prefix):
        ''' Old style model is stored with all names of parameters sharing common prefix 'module.' '''
        f = lambda x: x.split(prefix, 1)[-1] if x.startswith(prefix) else x
        return {f(key): value for key, value in state_dict.items()}

    def load(self):
        # TODO: load weights automatically from ~/.cache/torch/hub/checkpoints/
        weights_path="weights/RetinaFace-R50.pth"

        # Load weights from default path
        weights = torch.load(weights_path)

        # Update module names by removing "module." prefix if it exists
        remove_prefix = lambda x: x[7:] if x.startswith("module.") else x
        weights = {remove_prefix(k): v for k, v in weights.items()}
        
        # Load weights for this class
        self.load_state_dict(weights)
        self.eval()
        
        for param in self.parameters():
            # Disable gradient tracing
            param.requires_grad = False

    def forward(self, x: torch.Tensor
                ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Extract FPN + SSH features
        fpn = self.fpn(self.body(x))
        fts = [self.ssh1(fpn[0]), self.ssh2(fpn[1]), self.ssh3(fpn[2])]

        # Create head list and use each to process feature list
        hs = [self.ClassHead, self.BboxHead, self.LandmarkHead]
        pred = [torch.cat([h[i](f) for i, f in enumerate(fts)], 1) for h in hs]
        
        return F.softmax(pred[0], dim=-1), pred[1], pred[2]
    
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
        landms = torch.cat(
            (priors[..., :2] + pre[..., :2] * self.variance[0] * priors[..., 2:],
             priors[..., :2] + pre[..., 2:4] * self.variance[0] * priors[..., 2:],
             priors[..., :2] + pre[..., 4:6] * self.variance[0] * priors[..., 2:],
             priors[..., :2] + pre[..., 6:8] * self.variance[0] * priors[..., 2:],
             priors[..., :2] + pre[..., 8:10] * self.variance[0] * priors[..., 2:],
            ), dim=2)
        return landms
    
    def filter_preds(
        self,
        scores: torch.Tensor,
        bboxes: torch.Tensor,
        landms: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, list[int]]:
        """Filters predictions for identified faces for each sample
        
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
        device = images.device # next(self.parameters()).device

        # Convert images to appropriate input and predict landmarks
        x = images.flip(1) # .permute(0, 3, 1, 2).float().to(device)
        x -= torch.tensor([[[104]], [[117]], [[123]]], device=device)
        scores, bboxes, landms = self(x)

        # Create prior boxes and scale factors to decode bboxes & landms
        priors = PriorBox((x.size(2), x.size(3))).forward().to(device)
        scale_b = torch.tensor([x.size(3), x.size(2)] * 2, device=device)
        scale_l = torch.tensor([x.size(3), x.size(2)] * 5, device=device)

        # Decode the predictions and use them to parse landmarks
        scores = scores[..., 1]
        bboxes = self.decode_bboxes(bboxes, priors) * scale_b
        landms = self.decode_landms(landms, priors) * scale_l
        landms, bboxes, idx = self.filter_preds(scores, bboxes, landms)

        landmarks, indices = [], []
        cache = {"idx": [], "bboxes": [], "landms": []}

        for i in range(len(idx)):
            # Apend everything to cache
            cache["idx"].append(idx[i])
            cache["bboxes"].append(bboxes[i])
            cache["landms"].append(landms[i])

            if i != len(idx) - 1 and cache["idx"][-1] == idx[i + 1]:
                # No operations until cache for current idx is full
                continue

            match self.strategy:
                case "all":
                    # Append all landmarks and indices
                    landmarks.extend(cache["landms"])
                    indices.extend(cache["idx"])
                case "best":
                    # Append the first set of landmarks
                    landmarks.append(cache["landms"][0])
                    indices.append(cache["idx"][0])
                case "largest":
                    # Compute bounding box areas
                    bbs = torch.stack(cache["bboxes"])
                    areas = (bbs[:, 2] - bbs[:, 0] + 1) *\
                            (bbs[:, 3] - bbs[:, 1] + 1)

                    # Append only the largest face landmarks and its idx
                    landmarks.append(cache["landms"][areas.argmax()])
                    indices.append(cache["idx"][0])
                case _:
                    raise ValueError(f"Unsupported startegy: {self.strategy}")
            
            # Clear cache (reinitialize empty lists)
            cache = {k: [] for k in cache.keys()}
        
        # Stack landmarks across batch dim and reshape as coords
        landmarks = torch.stack(landmarks).view(-1, 5, 2).cpu()

        return landmarks, indices
