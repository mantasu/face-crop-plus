import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from ._layers import LoadMixin, RRDB


class RRDBNet(nn.Module, LoadMixin):
    """Face quality enhancer.

    This model is capable of detecting which images have low-quality 
    faces, i.e., which images have small face areas compared to the 
    dimensions of the image and is able to enhance the quality of those 
    images. The images are up-scaled 4 times and then resized to their 
    original size - this results in less blurry faces.

    This class also inherits ``load`` method from ``LoadMixin`` class. 
    The method takes a device on which to load the model and loads the 
    model with a default state dictionary loaded from 
    ``WEIGHTS_FILENAME`` file. It sets this model to eval mode and 
    disables gradients.

    For more information on how RetinaFace model works, see this repo:
    `BSRGAN <https://github.com/cszn/BSRGAN>`_. Most of the code was 
    taken from that repository.
    
    Note:
        Whenever an input shape is mentioned, N corresponds to batch 
        size, C corresponds to the number of channels, H - to input
        height, and W - to input width.
    """
    #: WEIGHTS_FILENAME (str): The constant specifying the name of 
    #: ``.pth`` file from which the weights for this model should be 
    #: loaded. Defaults to "bsrgan_x4_enhancer.pth".
    WEIGHTS_FILENAME = "bsrgan_x4_enhancer.pth"

    def __init__(self, min_face_factor: float = 0.001):
        """Initializes RRDB (BSRGAN) model.

        Just assigns the minimum face threshold attribute and constructs 
        module layers for quality inference.

        Args:
            min_face_factor: The minimum average face factor, i.e., face 
                area relative to the image, below which the whole image 
                is enhanced. Defaults to 0.001.
        """
        super().__init__()
        # Init minimum face factor attribute
        self.min_face_factor = min_face_factor

        # Initialize first layers that produce features
        self.conv_first = nn.Conv2d(3, 64, 3, 1, 1)
        self.RRDB_trunk = nn.Sequential(*[RRDB(64, 32) for _ in range(23)])
        self.trunk_conv = nn.Conv2d(64, 64, 3, 1, 1)
        self.upconv1 = nn.Conv2d(64, 64, 3, 1, 1)
        self.upconv2 = nn.Conv2d(64, 64, 3, 1, 1)

        # Final layers that produce enhanced image
        self.HRconv = nn.Conv2d(64, 64, 3, 1, 1)
        self.conv_last = nn.Conv2d(64, 3, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Performs forward pass.

        Takes an input tensor which is a batch of images and produces 
        the same batch, except images are up-scaled 4 times. 

        Args:
            x: The input tensor of shape (N, 3, H, W).

        Returns:
            An output tensor of shape (N, 3, 4*H, 4*W).
        """
        # Perform inference, get features, upscale 2 times, get enhanced
        fea = (x := self.conv_first(x)) + self.trunk_conv(self.RRDB_trunk(x))
        fea = self.lrelu(self.upconv1(F.interpolate(fea, scale_factor=2)))
        fea = self.lrelu(self.upconv2(F.interpolate(fea, scale_factor=2)))

        return self.conv_last(self.lrelu(self.HRconv(fea)))

    @torch.no_grad()
    def predict(
        self,
        images: torch.Tensor | list[torch.Tensor],
        landmarks: np.ndarray | None,
        indices: list[int] | None,
    ) -> torch.Tensor:
        """Enhances the quality of images with low-quality faces.

        Takes a batch of images and sets of landmarks for each image and 
        enhances the quality of those images for which the average face 
        area factor is lower than ``self.min_face_factor``. The face 
        factor is computed by dividing the face area (computed by 
        multiplying the width and the height of the face, specified by 
        left-eye, right-eye, left-mouth, right-mouth landmark 
        coordinates) by the image area.

        Note:
            The images are enhanced one by one instead of as a batch 
            because the inference is very memory consuming and can 
            result in memory errors.

        Args:
            images: Image batch of shape (N, 3, H, W) in RGB form with 
                float values from 0.0 to 255.0. It must be on the same 
                device as this model. It can also be a list of tensors
                of different shapes.
            landmarks: Landmarks batch of shape (``num_faces``, 5, 2) 
                used to compute average face area for each image. If 
                None, then every image will be enhanced.
            indices: Indices list mapping each set of landmarks to a 
                specific image in ``images`` batch (multiple sets of 
                landmarks can come from the same image). If None, then 
                every image will be enhanced.

        Returns:
            The same image batch as ``images`` - the shape is 
            (N, 3, H, W) channels are in RGB and values range from 
            0.0 to 255.0. The only difference is that some of the images 
            are of much higher quality, i.e., less blurry.
        """
        for i in range(len(images)):
            if landmarks is None or indices is None:
                # Create a dummy face factor to ensure enhance
                face_factor = np.array([self.min_face_factor])
            else:
                # Select all face landmarks in the current i'th image
                landmarks_i = landmarks[[idx == i for idx in indices]]

                if len(landmarks_i) == 0:
                    # No landmarks found
                    continue

                # Compute relative face factor (area face takes up)
                [w, h] = (landmarks_i[:, 4] - landmarks_i[:, 0]).T
                face_factor = w * h / (images[0].shape[1] * images[0].shape[2])

            if face_factor.mean() <= self.min_face_factor:
                # Enhance ith img if factor below threshold
                image_x4 = self(images[i].unsqueeze(0).div(255))
                image_x1 = F.interpolate(image_x4, None, 0.25, "bicubic")
                images[i] = image_x1.clamp(0, 1).mul(255).round()[0]

        return images
