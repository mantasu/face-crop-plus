import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from ._layers import LoadMixin, ContextPath, FeatureFusionModule, BiSeNetOutput


class BiSeNet(nn.Module, LoadMixin):
    WEIGHTS_FILENAME = "bise_parser.pth"

    def __init__(self, attr_groups: dict[str, list[int]] | None = None, mask_groups: dict[str, list[int]] | None = None):
        super().__init__()
        self.attr_groups = attr_groups
        self.mask_groups = mask_groups
        self.att_join_by_and = True
        self.attr_threshold = 5
        self.batch_size = 8

        self.cp = ContextPath()
        self.ffm = FeatureFusionModule(256, 256)
        self.conv_out = BiSeNetOutput(256, 256, 19)

    def forward(self, x):
        feat_out = self.conv_out(self.ffm(*self.cp(x)))
        return F.interpolate(feat_out, x.size()[2:], None, "bilinear", True)
    
    def group_by_attributes(self, parse_preds: torch.Tensor, att_groups: dict[str, list[int]], offset: int) -> dict[str, list[int]]:        
        att_join = torch.all if self.att_join_by_and else torch.any
        
        for k, v in self.attr_groups.items():
            attr = torch.tensor(v, device=parse_preds.device).view(1, -1, 1, 1)
            is_attr = (parse_preds.unsqueeze(1) == attr.abs()).sum(dim=(2, 3))

            is_attr = att_join(torch.stack([
                is_attr[:, i] > self.attr_threshold if a > 0 else
                is_attr[:, i] <= self.attr_threshold
                for i, a in enumerate(v)
            ], dim=1), dim=1)

            inds = [i + offset for i in range(len(is_attr)) if is_attr[i]]
            att_groups[k].extend(inds)
        
        return att_groups

    def group_by_segmentation(self, parse_preds: torch.Tensor, seg_groups: dict[str, tuple[list[int], list[np.ndarray]]], offset: int) -> dict[str, tuple[list[int], list[np.ndarray]]]:
        for k, v in self.mask_groups.items():
            attr = torch.tensor(v, device=parse_preds.device).view(1, -1, 1, 1)
            mask = (parse_preds.unsqueeze(1) == attr).any(dim=1)

            inds = [i for i in range(len(mask)) if mask[i].sum() > 5]
            masks = mask[inds].mul(255).cpu().numpy().astype(np.uint8)

            seg_groups[k][0].extend([i + offset for i in inds])
            seg_groups[k][1].extend(masks.tolist())

        return seg_groups
    
    # def parse(self, images: torch.Tensor):        
        

    #     for sub_batch in torch.split(images, self.batch_size):
    #         out = self.par_model.predict(sub_batch.to(self.device))

    #         if self.attr_groups is not None:
    #             att_groups = self.group_by_attributes(out, att_groups, offset)
            
    #         if self.mask_groups is not None:
    #             seg_groups = self.group_by_segmentation(out, seg_groups, offset)
            
    #         offset += len(sub_batch)
        
    #     if seg_groups is not None:
    #         seg_groups = {k: v for k, v in seg_groups.items() if len(v[1]) > 0}
    #         for k, v in seg_groups.items():
    #             seg_groups[k] = (v[0], np.stack(v[1]))
        
    #     return att_groups, seg_groups
    
    def predict(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        att_groups, seg_groups, offset = None, None, 0

        if self.attr_groups is not None:
            att_groups = {k: [] for k in self.attr_groups.keys()}
        
        if self.mask_groups is not None:
            seg_groups = {k: ([], []) for k in self.mask_groups.keys()}
        
        # Convert mean, std and attr to tensors and reshape in advance
        mean = torch.tensor([0.485, 0.456, 0.406], device=images.device)
        std = torch.tensor([0.229, 0.224, 0.225], device=images.device)

        # Create proper input, infer, resize back output
        x = F.interpolate(images.div(255), (512, 512), mode="bilinear")
        

        for sub_x in torch.split(x, self.batch_size):
            # out = self.par_model.predict(sub_batch.to(self.device))
            out = self((sub_x - mean.view(1, 3, 1, 1)) / std.view(1, 3, 1, 1))
            out = F.interpolate(out, images.size()[2:], mode="nearest").argmax(1)

            if self.attr_groups is not None:
                att_groups = self.group_by_attributes(out, att_groups, offset)
            
            if self.mask_groups is not None:
                seg_groups = self.group_by_segmentation(out, seg_groups, offset)
            
            offset += len(sub_x)
        
        if seg_groups is not None:
            seg_groups = {k: v for k, v in seg_groups.items() if len(v[1]) > 0}
            for k, v in seg_groups.items():
                seg_groups[k] = (v[0], np.stack(v[1]))

        return att_groups, seg_groups
