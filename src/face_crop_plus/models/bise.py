import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from ._layers import LoadMixin, ContextPath, FeatureFusionModule, BiSeNetOutput


class BiSeNet(nn.Module, LoadMixin):
    """Face attribute parser.

    This class is capable of predicting scores for 19 attributes for 
    face images. After it identifies the closest attribute for each 
    pixel, it can also put the whole face image to a corresponding 
    attribute or mask group.

    The 19 attributes are as follows (attributes are indicated from a  
    person's face perspective, meaning, for instance, left eye is the
    eye on the right hand-side of the picture, however, sides are not 
    always accurate):

        * 0 - neutral
        * 1 - skin
        * 2 - left eyebrow
        * 3 - right eyebrow
        * 4 - left eye
        * 5 - right eye
        * 6 - eyeglasses
        * 7 - left ear
        * 8 - right ear
        * 9 - earing
        * 10 - nose
        * 11 - mouth
        * 12 - upper lip
        * 13 - lower lip
        * 14 - neck
        * 15 - necklace
        * 16 - clothes
        * 17 - hair
        * 18 - hat

    Some examples of grouping by attributes:

        * ``'glasses': [6]`` - this will put each face image that 
          contains pixels labeled as 6 to a category called 'glasses'.
        * ``'earings_and_necklace': [9, 15]`` - this will put each image 
          that contains pixels labeled as 9 and also contains pixels
          labeled as 15 to a category called 'earings_and_necklace'.
        * ``'no_accessories': [-6, -9, -15, -18]`` - this will put each 
          face image that does not contain pixels labeled as either 6, 
          9, 15, or 18 to a category called 'no_accessories'.
    
    Some examples of grouping by mask:

        * ``'nose': [10]`` - this will put each face image that contains 
          pixels labeled as 10 to a category called 'nose' and generate 
          a corresponding mask.  
        * ``'eyes_and_eyebrows': [2, 3, 4, 5]`` - this will put each 
          image that contains pixels labeled as either 2, 3, 4, or 5 (or 
          any combination of them) to a category called 
          'eyes_and_eyebrows' and generate a corresponding mask.
    
    This class also inherits ``load`` method from ``LoadMixin`` class. 
    The method takes a device on which to load the model and loads the 
    model with a default state dictionary loaded from 
    ``WEIGHTS_FILENAME`` file. It sets this model to eval mode and 
    disables gradients.
    
    For more information on how BiSeNet model works, see this repo:
    `Face Parsing PyTorch <https://github.com/zllrunning/face-parsing.PyTorch>`_. 
    Most of the code was taken from that repository.

    Note:
        Whenever an input shape is mentioned, N corresponds to batch 
        size, C corresponds to the number of channels, H - to input
        height, and W - to input width.

    Be default, this class initializes the following attributes which 
    can be changed after initialization of the class (but, typically, 
    should not be changed):
    
    Attributes:
        attr_join_by_and (bool): Whether to add a face image to 
            an attribute group if the face meets all the specified 
            attributes in a list (joined by and) of at least one of 
            the attributes (joined by or). Please read the definition 
            of  `attr_groups` to get a clearer picture. In most cases, 
            this should be set True - if the attributes in a group 
            list are negative, this will ensure the selected face will 
            match none of the specified attributes. Also, if you want 
            to join the attributes by or (any), then separate 
            single-attribute groups can be created and manually merged 
            into one. Defaults to True.
        attr_threshold (int): Threshold, based on which the 
            attribute is determined as present in the face image. For 
            instance, if the threshold is 5, then at least 6 pixels 
            must be labeled of the same kind of attribute for that 
            attribute to be considered present in the face image. 
            Defaults to 5.
        mask_threshold (int): Threshold, based on which the 
            mask is considered to be a proper mask. For instance, if 
            the threshold is 15, then face images for which the number 
            of pixels with the values corresponding to a specified 
            mask group (face attributes) is less than or equal to 15 
            will be ignored and image-mask pair for that mask category 
            will not be generated. Defaults to 15.
        mean (list[float]): The list of mean values for each 
            input channel. The pixel values should be shifted by those 
            quantities during inference since this normalization was 
            applied during training. Defaults to 
            [0.485, 0.456, 0.406].
        std (list[float]): The list of standard deviation values 
            for each input channel. The pixel values should be scaled 
            by those quantities during inference since this 
            normalization was applied during training. Defaults to 
            [0.229, 0.224, 0.225].
    """
    #: WEIGHTS_FILENAME (str): The constant specifying the name of 
    #: ``.pth`` file from which the weights for this model should be 
    #: loaded. Defaults to "bise_parser.pth".
    WEIGHTS_FILENAME = "bise_parser.pth"

    def __init__(
        self,
        attr_groups: dict[str, list[int]] | None = None,
        mask_groups: dict[str, list[int]] | None = None,
        max_batch_size: int = 8,
    ):
        """Initializes BiSeNet model.

        First it assigns the passed values as attributes. Then this 
        method initializes BiSeNet layers required for face parsing, 
        i.e., labeling face parts.

        Note:
            Check class definition for the possible face attribute 
            values and examples of groups. Also note that all the 
            specified variables here are mainly relevant only for 
            :meth:`predict`.

        Args:
            attr_groups: Dictionary specifying attribute groups, based 
                on which the face images should be grouped. Each key 
                represents an attribute group name, e.g., 'glasses', 
                'earings_and_necklace', 'no_accessories', and each value 
                represents attribute indices, e.g., `[6]`, `[9, 15]`, 
                `[-6, -9, -15, -18]`, each index mapping to some 
                attribute. Since this model labels face image pixels, if 
                there is enough pixels with the specified values in the 
                list, the whole face image will be put into that 
                attribute category. For negative values, it will be 
                checked that the labeled face image does not contain
                those (absolute) values. If it is None, then there will 
                be no grouping according to attributes. Defaults to
                None.
            mask_groups: Dictionary specifying mask groups, based on 
                which the face images and their masks should be grouped. 
                Each key represents a mask group name, e.g., 'nose', 
                'eyes_and_eyebrows', and each value represents attribute 
                indices, e.g., `[10]`, `[2, 3, 4, 5]`, each index
                mapping to some attribute. Since this model labels face 
                image pixels, a mask will be created with ones at pixels 
                that match the specified attributes and zeros elsewhere.
                Note that negative values would make no sense here and 
                having them would cause an error. If it is None, then 
                there will be no grouping according to mask groups. 
                Defaults to None.
            max_batch_size: The maximum batch size used when performing 
                inference. There may be a lot of faces, in a single 
                batch thus splitting to sub-batches for inference and 
                then merging back predictions is a way to deal with 
                memory errors. This is a convenience variable because 
                batch size typically corresponds to the number of images 
                for a single inference, but the input given in 
                :meth:`predict` might have a larger batch 
                size because it represents the number of faces, many of 
                which can come from just a single image. Defaults to 8.
        """
        super().__init__()
        
        # Initialize class attributes
        self.attr_groups = attr_groups
        self.mask_groups = mask_groups
        self.batch_size = max_batch_size
        self.attr_join_by_and = True
        self.attr_threshold = 5
        self.mask_threshold = 10
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]

        # Init model layers
        self.cp = ContextPath()
        self.ffm = FeatureFusionModule(256, 256)
        self.conv_out = BiSeNetOutput(256, 256, 19)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Performs forward pass.

        Takes an input batch and performs inference based on the modules 
        it has. The input is a batch of face images and the output is a 
        corresponding batch of pixel-wise attribute scores. 

        Args:
            x: The input tensor of shape (N, 3, H, W).

        Returns:
            An output tensor of shape (N, 19, H, W) where each channel 
            corresponds to a specific attribute and each value at 
            (H, W) is an unbounded confidence score.
        """
        # Generate final features from layers, upscale
        feat_out = self.conv_out(self.ffm(*self.cp(x)))
        return F.interpolate(feat_out, x.size()[2:], None, "bilinear", True)
    
    def group_by_attributes(
        self,
        parse_preds: torch.Tensor,
        attr_groups: dict[str, list[int]],
        offset: int,
    ) -> dict[str, list[int]]:
        """Groups parse predictions by face attributes.

        Takes parse predictions for each face where each pixel 
        corresponds to some attribute group (the integer value 
        indicates that group) and extends the groups in attribute 
        dictionary to include more samples that match the group.

        Args:
            parse_preds: Face parsing predictions of shape (N, H, W) 
                with integer values indicating pixel categories.
            attr_groups: The dictionary with keys corresponding to 
                attribute group names (they match ``self.attr_groups`` 
                keys) and values corresponding to indices that map face
                images from other batches of ``parse_preds`` to the
                corresponding group. This is the dictionary that is 
                extended and returned.
            offset: The offset to add to each index. Originally, the
                indices will correspond only to the face parsings in the 
                current ``parse_preds`` batch and the offset allows to 
                generalize the each index by offsetting it by the 
                previous number of processes face parsings, i.e., the 
                offset is the number of previous batches 
                (``parse_preds``) times the batch size.
        Returns:
            Similar to ``attr_groups``, it is the dictionary with the 
            same keys but values (which are lists of indices) may be 
            extended with additional indices.
        """
        # Specify function/criteria to join the attributes in a list
        att_join = torch.all if self.attr_join_by_and else torch.any
        
        for k, v in self.attr_groups.items():
            # Get the list of face attributes per group and count pixels
            attr = torch.tensor(v, device=parse_preds.device).view(1, -1, 1, 1)
            is_attr = (parse_preds.unsqueeze(1) == attr.abs()).sum(dim=(2, 3))

            # Compare each face against each attribute in a  group
            is_attr = att_join(torch.stack([
                is_attr[:, i] > self.attr_threshold if a > 0 else
                is_attr[:, i] <= self.attr_threshold
                for i, a in enumerate(v)
            ], dim=1), dim=1)

            # Add indices of those faces which match the group attribute
            inds = [i + offset for i in range(len(is_attr)) if is_attr[i]]
            attr_groups[k].extend(inds)
        
        return attr_groups

    def group_by_masks(
        self,
        parse_preds: torch.Tensor,
        mask_groups: dict[str, tuple[list[int], list[np.ndarray]]],
        offset: int,
    ) -> dict[str, tuple[list[int], list[np.ndarray]]]:
        """Groups parse predictions by face mask attributes.

        Takes parse predictions for each face where each pixel 
        corresponds to some parse/mask group (the integer value 
        indicates that group) and extends the groups in mask 
        dictionary to include more samples that match the group.

        Args:
            parse_preds: Face parsing predictions of shape (N, H, W) 
                with integer values indicating pixel categories.
            mask_groups: The dictionary with keys corresponding to 
                mask group names (they match ``self.mask_groups`` keys)
                and values corresponding to tuples where the first value 
                is a list of indices that map face images from other 
                batches of ``parse_preds`` to the corresponding group 
                and the second is a list of corresponding masks as numpy 
                arrays of shape (H, W) of type :attr:`numpy.uint8` with 
                255 at pixels that match the mask group specification 
                and 0 elsewhere. This is the dictionary that is extended 
                and returned.
            offset: The offset to add to each index. Originally, the
                indices will correspond only to the face parsings in the 
                current ``parse_preds`` batch and the offset allows to 
                generalize the each index by offsetting it by the 
                previous number of processes face parsings, i.e., the 
                offset is the number of previous batches 
                (``parse_preds``) times the batch size.

        Returns:
            Similar to ``mask_groups``, it is the dictionary with the 
            same keys but values (which are tuples of a list of indices 
            and a list of masks) may be extended with additional indices 
            and masks.
        """
        # Retrieve threshold (shorter name)
        threshold = self.mask_threshold

        for k, v in self.mask_groups.items():
            # Get the list of face attributes per group and check match
            attr = torch.tensor(v, device=parse_preds.device).view(1, -1, 1, 1)
            mask = (parse_preds.unsqueeze(1) == attr).any(dim=1)

            # Identify proper masks and convert them to numpy image type
            inds = [i for i in range(len(mask)) if mask[i].sum() > threshold]
            masks = mask[inds].mul(255).cpu().numpy().astype(np.uint8)

            # Extend the lists of indices and masks for k group
            mask_groups[k][0].extend([i + offset for i in inds])
            mask_groups[k][1].extend([*masks])

        return mask_groups
    
    @torch.no_grad()
    def predict(
        self,
        images: torch.Tensor | list[torch.Tensor],
    ) -> tuple[dict[str, list[int]] | None, 
               dict[str, tuple[list[int], list[np.ndarray]]] | None]:
        """Predicts attribute and mask groups for face images.

        This method takes a batch of face images groups them according 
        to the specifications in ``self.attr_groups`` and 
        ``self.mask_groups``. For more information on how it works, see 
        this class' specification :class:`BiSeNet`. It returns 2 
        groups maps - one for grouping face images to different
        attribute categories, e.g., 'with glasses', 'no accessories' and 
        the other for grouping images to different mask groups, e.g., 
        'nose', 'lips and mouth'.

        Args:
            images: Image batch of shape (N, 3, H, W) in RGB form with 
                float values from 0.0 to 255.0. It must be on the same 
                device as this model. A list of tensors can also be 
                provided, however, they all must have the same spatial 
                dimensions to be stack-able to a single batch.

        Returns:
            A tuple of 2 dictionaries (either can be None):

                1. ``attr_groups`` - each key represents attribute 
                   category and each value is a list of indices 
                   indicating which  samples from ``images`` batch 
                   belong to that category. It can be None if 
                   ``self.attr_groups`` is None.
                2. `mask_groups` - each key represents attribute (mask) 
                   category and each value is a tuple where the first 
                   element is a list of indices indicating which samples 
                   from ``images`` batch belong to that mask group and 
                   the second element is a corresponding batch of masks 
                   of shape (N, H, W) of type :attr:`numpy.uint8` with 
                   values of either 0 or 255. The masks are presented in 
                   that order as the indices indicate which face images 
                   to  take for that mask group. It can be None if 
                   ``self.mask_groups`` is None.

        """
        # Initialize groups as None, a helper offset
        attr_groups, mask_groups, offset = None, None, 0

        if self.attr_groups is not None:
            # Initialize an empty dictionary of attribute groups
            attr_groups = {k: [] for k in self.attr_groups.keys()}
        
        if self.mask_groups is not None:
            # Initialize an empty dictionary of mask groups
            mask_groups = {k: ([], []) for k in self.mask_groups.keys()}
        
        if isinstance(images, list):
            # Stack the list of tensors
            images = torch.stack(images)
        
        # Convert mean and std to tensors and reshape, resize images
        mean = torch.tensor(self.mean, device=images.device).view(1, 3, 1, 1)
        std = torch.tensor(self.std, device=images.device).view(1, 3, 1, 1)
        x = F.interpolate(images.div(255), (512, 512), mode="bilinear")

        for sub_x in torch.split(x, self.batch_size):
            # Inference and resize back
            o = self((sub_x - mean) / std)
            o = F.interpolate(o, images.size()[2:], mode="nearest").argmax(1)

            if self.attr_groups is not None:
                # Extend each attribute group based on predictions
                attr_groups = self.group_by_attributes(o, attr_groups, offset)
            
            if self.mask_groups is not None:
                # Extend each mask group based on predictions
                mask_groups = self.group_by_masks(o, mask_groups, offset)
            
            # Increment offset
            offset += len(sub_x)
        
        if attr_groups is not None:
            # Filter out groups for which the list of indices is empty
            attr_groups = {k: v for k, v in attr_groups.items() if len(v) > 0}
        
        if mask_groups is not None:
            # Filter out groups for which the list of indices is empty
            mask_groups = {
                k: (v[0], np.stack(v[1]))
                for k, v in mask_groups.items() if len(v[1]) > 0
            }

        return attr_groups, mask_groups
