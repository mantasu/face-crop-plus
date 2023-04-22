import os
import cv2
import json
import torch
import numpy as np


STANDARD_LANDMARKS_5 = np.float32([
    [0.31556875000000000, 0.4615741071428571],
    [0.68262291666666670, 0.4615741071428571],
    [0.50026249999999990, 0.6405053571428571],
    [0.34947187500000004, 0.8246919642857142],
    [0.65343645833333330, 0.8246919642857142],
])

def parse_landmarks_file(
    file_path: str,
    **kwargs,
) -> tuple[np.ndarray, np.ndarray]:
    """Parses landmarks file.

    Reads the file containing landmark coordinates and corresponding 
    image file names and generates a numpy array of those names and a 
    corresponding numpy array of those sets of landmarks.

    The files are expected to be formatted as follows:

        * `.json`::

            {
                "image_1.jpg": [23, 45, 64, 47, ...],
                "image_2.jpg": [17, 32, 30, 29, ...],
                ...
            }

        * `.csv`::

            images,x1,y1,x2,y2,...
            image_1.jpg,23,45,64,47,...
            image_2.jpg,17,32,30,29,...
            ...

        * `.txt` and other::

            image_1.jpg 23 45 64 47 ...
            image_2.jpg 17 32 30 29 ...
            ...

    Note:
        The number of landmarks does not matter, all will be 
        transformed to shape (-1, 2), where -1 stands for the number of 
        facial points (landmark coordinates), e.g., 5, 68 etc.

    Args:
        file_path: The path to the landmarks file.
        **kwargs: Additional keyword arguments that go into 
            :func:`numpy.genfromtxt`. Please do not specify *dtype* and 
            *usecols* arguments as they are already specified.

    Returns:
        A tuple where the first element is the parsed landmarks batch as 
        a numpy array of shape (N, ``num_landm``, 2) of type 
        :attr:`numpy.float32` and  the second element is a corresponding 
        batch of image file names of shape (N,) of type 
        :class:`numpy.str_`.
    """
    if file_path.endswith(".json"):
        with open(file_path, 'r') as f:
            # Read and parse
            data = json.load(f)
            filenames = np.array(data.keys())
            landmarks = np.array(data.values())
    else:
        if file_path.endswith(".csv"):
            # Set default params for csv files
            kwargs.setdefault("delimiter", ',')
            kwargs.setdefault("skip_header", 1)

        # Use the first column for filenames, the rest for landmarks
        filenames = np.genfromtxt(file_path, usecols=0, dtype=str, **kwargs)
        landmarks = np.genfromtxt(file_path, dtype=np.float32, **kwargs)[:, 1:]

    return landmarks.reshape(len(landmarks), -1, 2), filenames

def get_landmark_slices_5(num_landmarks: int) -> list[slice]:
    """Gets the landmarks slices that show where the 5 landmarks are.

    Generates slices of which coordinates to select in a larger set of 
    landmarks (e.g., 12, 68, 106) to represent the coordinates of 
    5-points landmarks set. The slice of indices can be used to select 
    multiple coordinates and average them to a single point.

    Args:
        num_landmarks: The number of landmarks in the larger set.

    Raises:
        ValueError: If the number of landmarks in the larger set is 
            not supported.

    Returns:
        A list of slices where each slice indicates the indices of 
        coordinates to select from the larger set of landmarks to 
        represent a 5-point landmarks set.
    """
    match num_landmarks:
        case 5:
            indices = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)]
        case 12:
            indices = [(10, 11), (11, 12), (2, 3), (3, 4), (4, 5)]
        case 17:
            indices = [(2, 5), (7, 10), (10, 11), (13, 14), (16, 17)]
        case 21:
            indices = [(6, 9), (9, 12), (14, 15), (17, 18), (19, 20)]
        case 29:
            indices = [(4, 9), (13, 18), (19, 20), (22, 23), (27, 28)]
        case 49: # same as 51
            indices = [(19, 25), (25, 31), (13, 14), (31, 32), (37, 38)]
        case 68:
            indices = [(36, 42), (42, 48), (30, 31), (48, 49), (54, 55)]
        case 98:
            indices = [(60, 68), (68, 76), (54, 55), (76, 77), (82, 83)]
        case 106:
            indices = [(66, 75), (75, 84), (54, 55), (85, 86), (91, 92)]
        case _:
            raise ValueError(f"Invalid number of landmarks: {num_landmarks}")

    return [slice(*x) for x in indices]

def get_ldm_slices(
    num_tgt_landmarks: int,
    num_src_landmarks: int,
) -> list[slice]:
    """Generates a list of slices that form a reduced landmarks set.

    Takes the number of target landmarks and the number of source 
    landmarks and generates slices that show which coordinates to select 
    from a larger landmarks set (that should be averaged) to form a 
    reduced landmarks set that has the same number of landmarks as the 
    target landmarks set.

    Args:
        num_tgt_landmarks: The number of reduced landmarks to generate 
            slices for.
        num_src_landmarks: The number of actual landmarks that is larger 
            or equal to the number of target landmarks. Based on this 
            number, generated slices will contain different indices.

    Raises:
        ValueError: If the number of target landmarks is not supported.

    Returns:
        A list of slices where each slice indicates the indices of 
        coordinates to select from the larger set of landmarks to 
        represent a reduced (like target) landmarks set.
    """
    match num_tgt_landmarks:
        case 5:
            slices = get_landmark_slices_5(num_src_landmarks)
        case _:
            raise ValueError(f"The number of target (standard) landmarks is "
                             f"not supported {num_tgt_landmarks}")
    
    return slices

def as_numpy(
    img: torch.Tensor | np.ndarray | list[torch.Tensor] | list[np.ndarray],
) -> np.ndarray | list[np.ndarray]:
    """Converts batch of images to numpy type.

    Converts a batch of images to numpy type. UINT8 type and channel 
    dimension is last. If the batch of images is already of numpy type, 
    it is simply returned.

    Args:
        img: The image batch represented as a torch tensor of shape 
            (N, 3, H, W) or a list of torch tensors of different 
            spatial sizes.

    Returns:
        A batch of images represented as a numpy array of shape 
        (N, H, W, 3) of type :attr:`numpy.uint8` or a list of numpy 
        arrays of different spatial sizes.
    """
    if isinstance(img[0], np.ndarray):
        return img
    elif isinstance(img, list):
        img = [x.permute(1, 2, 0).cpu().numpy().astype(np.uint8) for x in img]
    else:
        img = img.permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)

    return img

def as_tensor(
    img: np.ndarray | torch.Tensor | list[np.ndarray] | list[torch.Tensor],
    device: str | torch.device = "cpu",
) -> torch.Tensor | list[torch.Tensor]:
    """Converts batch of images to torch tensor type.

    Converts a batch of images to torch tensor type. Float 32 type and 
    channel dimension is before spatial dimension. If the batch of 
    images is already of torch tensor type, it is simply returned.

    Args:
        img: The image batch represented as a numpy array of shape 
            (N, H, W, 3) or a list of numpy arrays of different 
            spatial sizes.
        device: The device on which to return the torch tensor.

    Returns:
        A batch of images represented as a torch tensor of shape 
        (N, 3, H, W) of type :attr:`torch.float32` or a list of torch 
        tensors of different spatial sizes.
    """
    if isinstance(img[0], torch.Tensor):
        return img
    elif isinstance(img, list):
        img = [torch.from_numpy(x).permute(2, 0, 1).float().to(device) for x in img]
    else:
        img = torch.from_numpy(img).permute(0, 3, 1, 2).float().to(device)
    
    return img

def create_batch_from_files(
    file_names: list[str],
    input_dir: str,
    size: int | tuple[int, int] = 512,
    padding_mode: str = "constant",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Creates image batch from a list of image paths.

    For every image path in the given list, the image is read, resized
    to not exceed either of the dimensions specified in ``size`` while
    keeping the same aspect ratio and the shorter dimension is padded to
    fully match the specified size. All the images are stacked and
    returned as a batch. Variables required to transform the images back
    to the original ones (padding and scale) are also returned as a
    batch.

    Example:
        If some loaded image dimension is (1280×720) and the desired
        output ``size`` is specified as *(512, 256)*, then the image is
        first be resized to *(455, 256)* and then the width is padded 
        from both sides. The final image size is *(512, 256)*.

    Args:
        file_names: The list of paths to images.
        padding_mode: The type of padding to apply to pad the shorter
            dimension. For the available options, see
            `OpenCV BorderTypes <https://docs.opencv.org/3.4/d2/de8/group__core__array.html#ga209f2f4869e304c82d07739337eae7c5>`_.
            It can be all lowercase. Defaults to "constant".
        size: The width and the height each image should be resized +
            padded to. I.e., the spacial dimensions of the batch. If
            a single number is specified then it is the same for width
            height. Defaults to 512.

    Returns:
        A tuple of stacked numpy arrays representing 3 batches - resized 
        + padded images of shape (N, H, W, 3) of type 
        :attr:`numpy.uint8` with values from 0 to 255, un-scale factors 
        of shape (N,) of type :attr:`numpy.float32`, and applied 
        paddings of shape (N, 4) of type :attr:`numpy.int64` with 
        values >= 0.
    """
    # Init lists, resize dims, border type
    images, unscales, paddings = [], [], []
    size = (size, size) if isinstance(size, int) else size
    border_type = getattr(cv2, f"BORDER_{padding_mode.upper()}")
    file_paths = [os.path.join(input_dir, file) for file in file_names]

    for path in file_paths:
        # Read the image from the given path, convert to RGB form
        image = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
    
        # Get width, height, padding & check interpolation
        (h, w), m = image.shape[:2], max(*image.shape[:2])
        interpolation = cv2.INTER_AREA if m > max(size) else cv2.INTER_CUBIC

        if (ratio_w := size[0] / w) < (ratio_h := size[1] / h):
            # Based on width 
            unscale = ratio_w
            (ww,hh) = size[0], int(h * ratio_w)
            padding = [(size[1] - hh) // 2, (size[1] - hh + 1) // 2, 0, 0]
        else:
            # Based on height
            unscale = ratio_h
            (ww,hh) = int(w * ratio_h), size[1]
            padding = [0, 0, (size[0] - ww) // 2, (size[0] - ww + 1) // 2]
    
        # Pad the lower dimension with specific border type, then resize
        image = cv2.resize(image, (ww, hh), interpolation=interpolation)
        image = cv2.copyMakeBorder(image, *padding, borderType=border_type)

        # Append to lists
        images.append(image)
        unscales.append(np.array(unscale))
        paddings.append(np.array(padding))

    return np.stack(images), np.stack(unscales), np.stack(paddings)
