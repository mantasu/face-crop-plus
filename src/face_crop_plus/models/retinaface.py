import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.models._utils as _utils
from ._layers import LoadMixin, PriorBox, SSH, FPN, Head


class RetinaFace(nn.Module, LoadMixin):
    """RetinaFace face detector and 5-point landmark predictor.

    This class is capable of predicting 5-point landmarks from a batch 
    of images and filter them based on strategy, e.g., "all landmarks in 
    the image", "a single set of landmarks per image of the largest
    face". For more information, see the main method of this class
    :meth:`predict`. For main attributes, see :meth:`__init__`.

    This class also inherits ``load`` method from ``LoadMixin`` class. 
    The method takes a device on which to load the model and loads the 
    model with a default state dictionary loaded from 
    ``WEIGHTS_FILENAME`` file. It sets this model to eval mode and 
    disables gradients.

    For more information on how RetinaFace model works, see this repo:
    `PyTorch Retina Face <https://github.com/biubug6/Pytorch_Retinaface>`_. 
    Most of the code was taken from that repository.

    Note:
        Whenever an input shape is mentioned, N corresponds to batch 
        size, C corresponds to the number of channels, H - to input
        height, and W - to input width. ``out_dim`` corresponds to the
        total guesses (the number of priors) the model made about each
        sample. Within those guesses, there typically exists at least 1 
        face but can be more. By default, it should be 43,008.
    
    Be default, this class initializes the following attributes which 
    can be changed after initialization of the class (but, typically, 
    should not be changed):

    Attributes:
        nms_threshold (float): The threshold, based on which 
            multiple bounding box or landmark predictions for the same 
            face are merged into one. Defaults to 0.4.
        variance (list[int]): The variance of the bounding boxes 
            used to undo the encoding of coordinates of raw  bounding 
            box and landmark predictions.
    """
    #: WEIGHTS_FILENAME (str): The constant specifying the name of 
    #: ``.pth`` file from which the weights for this model should be 
    #: loaded. Defaults to "retinaface_detector.pth".
    WEIGHTS_FILENAME = "retinaface_detector.pth"

    def __init__(self, strategy: str = "all", vis: float = 0.6):
        """Initializes RetinaFace model.            
        
        This method initializes ResNet-50 backbone and further 
        layers required for face detection and bbox/landm predictions.

        Args:
            strategy: The strategy used to retrieve the landmarks when
                :meth:`predict` is called. The available options are:

                    * "all" - landmarks for all faces per single image
                      (single batch entry) will be considered.
                    * "best" - landmarks for a single face with the
                      highest confidence score per image will be 
                      considered.
                    * "largest" - landmarks for a single largest face
                      per image will be considered.

                The most efficient option is 'best' and the least
                efficient is "largest". Defaults to "all".
            vis: The visual threshold, i.e., minimum confidence score,
                for a face to be considered an actual face. Lower
                scores will allow the detection of more faces per image
                but can result in non-actual faces, e.g., random
                surfaces somewhat representing faces. Higher scores will 
                prevent detecting faulty faces but may result in only a
                few faces detected, whereas there can be more, e.g., 
                higher will prevent the detection of blurry faces. 
                Defaults to 0.6.
        """
        super().__init__()

        # Initialize attributes
        self.strategy = strategy
        self.vis_threshold = vis
        self.nms_threshold = 0.4
        self.variance = [0.1, 0.2]

        # Set up backbone and config
        backbone = models.resnet50()
        in_channels, out_channels = 256, 256
        in_channels_list = [in_channels * x for x in [2, 4, 8]]
        return_layers = {'layer2': 1, 'layer3': 2, 'layer4': 3}

        # Construct the backbone by retrieving intermediate layers
        self.body = _utils.IntermediateLayerGetter(backbone, return_layers)

        # Construct sub-layers to extract features for heads
        self.fpn = FPN(in_channels_list, out_channels)
        self.ssh1 = SSH(out_channels, out_channels)
        self.ssh2 = SSH(out_channels, out_channels)
        self.ssh3 = SSH(out_channels, out_channels)

        # Construct 3 heads - score, bboxes & landms
        self.ClassHead = Head.make(2, out_channels)
        self.BboxHead = Head.make(4, out_channels)
        self.LandmarkHead = Head.make(10, out_channels)

    def forward(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Performs forward pass.

        Takes an input batch and performs inference based on the modules 
        it has. Returns an unfiltered tuple of scores, bounding boxes 
        and landmarks for all the possible detected faces. The 
        predictions are encoded to comfortably compute the loss during 
        training and thus should be decoded to coordinates.

        Args:
            x: The input tensor of shape (N, 3, H, W).

        Returns:
            A tuple of torch tensors where the first element is
            confidence scores for each prediction of shape
            (N, out_dim, 2) with values between 0 and 1 representing
            probabilities, the second element is bounding boxes of shape 
            (N, out_dim, 4) with unbounded values and the last element 
            is landmarks of shape (N, ``out_dim``, 10) with unbounded
            values.
        """
        # Extract FPN + SSH features
        fpn = self.fpn(self.body(x))
        fts = [self.ssh1(fpn[0]), self.ssh2(fpn[1]), self.ssh3(fpn[2])]

        # Create head list and use each to process feature list
        hs = [self.ClassHead, self.BboxHead, self.LandmarkHead]
        pred = [torch.cat([h[i](f) for i, f in enumerate(fts)], 1) for h in hs]
        
        return F.softmax(pred[0], dim=-1), pred[1], pred[2]
    
    def decode_bboxes(
        self,
        loc: torch.Tensor,
        priors: torch.Tensor,
    ) -> torch.Tensor:
        """Decodes bounding boxes from predictions.

        Takes the predicted bounding boxes (locations) and undoes the 
        encoding for offset regression used at training time.

        Args:
            loc: Bounding box (location) predictions for loc layers of
                shape (N, out_dim, 4). 
            priors: Prior boxes in center-offset form of shape
                (out_dim, 4).

        Returns:
            A tensor of shape (N, out_dim, 4) representing decoded
            bounding box predictions where the last dim can be
            interpreted as x1, y1, x2, y2 coordinates - the start and
            the end corners defining the face box.
        """
        # Concatenate priors
        boxes = torch.cat((
            priors[:, :2] + loc[..., :2] * self.variance[0] * priors[:, 2:],
            priors[:, 2:] * torch.exp(loc[..., 2:] * self.variance[1])
        ), 2)
        
        # Adjust values for proper xy coords
        boxes[..., :2] -= boxes[..., 2:] / 2
        boxes[..., 2:] += boxes[..., :2]

        return boxes

    def decode_landms(
        self,
        pre: torch.Tensor,
        priors: torch.Tensor,
    ) -> torch.Tensor:
        """Decodes landmarks from predictions.

        Takes the predicted landmarks (pre) and undoes the encoding for
        offset regression used at training time.

        Args:
            pre: Landmark predictions for loc layers of shape
                (N, out_dim, 10).
            priors: Prior boxes in center-offset form of shape
                (out_dim, 4).

        Returns:
            A tensor of shape (N, out_dim, 10) representing decoded
            landmark predictions where the last dim can be
            interpreted as x1, y1, ..., x10, y10 coordinates - one for 
            each of the 5 landmarks.
        """
        # Concatenate priors
        var = self.variance
        landms = torch.cat((
            priors[..., :2] + pre[..., :2] * var[0] * priors[..., 2:],
            priors[..., :2] + pre[..., 2:4] * var[0] * priors[..., 2:],
            priors[..., :2] + pre[..., 4:6] * var[0] * priors[..., 2:],
            priors[..., :2] + pre[..., 6:8] * var[0] * priors[..., 2:],
            priors[..., :2] + pre[..., 8:10] * var[0] * priors[..., 2:],
        ), dim=2)

        return landms
    
    def filter_preds(
        self,
        scores: torch.Tensor,
        bboxes: torch.Tensor,
        landms: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, list[int]]:
        """Filters predictions for identified faces for each sample.
        
        This method works as follows:

            1. First, it filters out bad predictions based on
               ``self.vis_threshold``.
            2. Then it gathers all the remaining predictions across the
               batch dimension, i.e., the batch dimension becomes not
               the number of samples but the number of filtered out
               predictions.
            3. It loops for each set of filtered predictions per sample
               sorting each set of confidence scores from best to worst.
            4. For each set of confidence scores, it identifies distinct 
               faces and keeps the record of which indices to keep. At 
               this stage it uses ``self.nms_threshold`` to remove the 
               duplicate face predictions.
            5. Finally, it applies the kept indices for each person
               (each face) to select corresponding bounding boxes and
               landmarks.

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
            (``num_faces``, 4), the second element is a torch tensor of
            shape (``num_faces``, ``num_landmarks`` * 2) and the third 
            element is a list of length ``num_faces``. First and second 
            elements correspond to bounding boxes and landmarks for each 
            face across all samples and the third element provides an 
            index for each bounding box/set of landmarks that identifies
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
        
        # Select the final landms and bboxes
        bboxes = bboxes[people_indices, :]
        landms = landms[people_indices, :]
        
        return landms, bboxes, sample_indices
    
    def take_by_strategy(
        self,
        landms: torch.Tensor,
        bboxes: torch.Tensor,
        idx: list[int],
    ) -> tuple[torch.Tensor, list[int]]:
        """Filters landmarks according to strategy.

        This method takes a batch of landmarks and bounding boxes (one
        for each face) filters only specific landmarks by a specific
        strategy. Here are the following cases of strategy:

            * "all" - effectively, nothing is done and simply the
              already passed values `landms` and `idx` are returned 
              without any changes.
            * "best" - the very first set of landmarks for each image 
              image is returned (the first set is the best set because
              the landmarks were sorted when duplicates were filtered
              out in :meth:`filter_preds`). This means
              the returned indices list is unique, e.g., goes from 
              ``[0, 0, 0, 1, 1, 2, 3, 3]`` to ``[0, 1, 2, 3]``.
            * "largest" - similar to 'best', except that this strategy
              requires performing additional computation to find out the 
              largest face based on the area of bounding boxes. Thus the 
              length of the `idx` list (which is equal to the number of 
              sets of landmarks) is the same as for 'best' strategy,
              except not the first (best) faces (actually, their
              landmarks) for each image but selected faces are returned.

        Note:
            Strategy "best" is most memory efficient, strategy "largest" 
            is least time efficient. Strategy "all" is as fast as "best" 
            but takes up more space.

        Args:
            landms: Landmarks batch of shape 
                (``num_faces``, ``num_landm`` * 2).
            bboxes: Bounding boxes batch of shape (``num_faces``, 4).
            idx: Indices where each index maps to an image from
                which some face prediction (landmarks and bounding box) 
                was retrieved. For instance if the 2nd element of idx is 
                1, that means that the 2nd element of ``landms`` and the
                2nd element of ``bboxes`` correspond to the 1st image. 
                This list is ascending, meaning the elements are
                grouped and increase, for example, the list may look
                like this: ``[0, 0, 1, 2, 3, 3, 3, 3, 4, 4, 5, 6, 6]``.

        Raises:
            ValueError: If the strategy is not supported.

        Returns:
            A tuple where the first element is torch tensor of shape 
            (``num_faces``, ``num_landm`` * 2) representing the selected 
            sets of landmarks and the second element is a list of 
            indices where each index maps a corresponding set of 
            landmarks (face) to an image identified by that index.
        """
        if len(idx) == 0:
            # If no predicted landmarks, return empty lists
            return torch.tensor([], device=landms.device), []
        
        # Init helper variables
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

        # Stack landmarks across batch dim
        landmarks = torch.stack(landmarks)
    
        return landmarks, indices
    
    @torch.no_grad()
    def predict(self, images: torch.Tensor) -> tuple[np.ndarray, list[int]]:
        """Predict the sets of landmarks from the image batch.

        This method takes a batch of images, detect all visible faces, 
        predicts bounding boxes and landmarks for each face and then 
        filters those faces according to a specific strategy - see
        :meth:`take_by_strategy` for more info. Finally, it returns 
        those selected sets of landmarks and corresponding indices that 
        map each set to a specific image where the face was originally 
        detected.

        The predicted sets of landmarks are 5-point coordinates (they  
        are specified from an observer's viewpoint, meaning that, for 
        instance, left eye is the eye on the left hand-side of the image 
        rather than the left eye from the person's to whom the eye 
        belongs perspective):

            1. **(x1, y1)** - coordinate of the left eye
            2. **(x2, y2)** - coordinate of the right eye
            3. **(x3, y3)** - coordinate of the nose tip
            4. **(x4, y4)** - coordinate of the left mouth corner
            5. **(x5, y5)** - coordinate of the right mouth corner

        The coordinates are with respect to the sizes of the images 
        (typically padded) provided as an input to this method.

        Args:
            images: Image batch of shape (N, 3, H, W) in RGB form with 
                float values from 0.0 to 255.0. It must be on the same 
                device as this model.

        Returns:
            A tuple where the first element is a numpy array of shape 
            (``num_faces``, 5, 2) representing the selected sets of 
            landmark coordinates and the second element is a list of
            corresponding indices mapping each face to an image it comes
            from.
        """
        # Convert images to appropriate input and perform inference
        x, offset = images[:, [2, 1, 0]], torch.tensor([104, 117, 123])
        scores, bboxes, landms = self(x - offset.view(3, 1, 1).to(x.device))

        # Create prior boxes and scale factors to decode bboxes & landms
        priors = PriorBox((x.size(2), x.size(3))).forward().to(x.device)
        scale_b = torch.tensor([x.size(3), x.size(2)] * 2, device=x.device)
        scale_l = torch.tensor([x.size(3), x.size(2)] * 5, device=x.device)

        # Decode the predictions
        scores = scores[..., 1]
        bboxes = self.decode_bboxes(bboxes, priors) * scale_b
        landms = self.decode_landms(landms, priors) * scale_l

        # Filter out bad predictions, then filter by strategy
        filtered = self.filter_preds(scores, bboxes, landms)
        landmarks, indices = self.take_by_strategy(*filtered)

        # Stack landmarks across batch dim and reshape as coords
        landmarks = landmarks.view(-1, 5, 2).cpu().numpy()

        return landmarks, indices
