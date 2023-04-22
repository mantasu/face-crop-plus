import os
import cv2
import tqdm
import torch
import numpy as np

from functools import partial
from collections import defaultdict
from multiprocessing.pool import ThreadPool

from .models import BiSeNet
from .models import RRDBNet
from .models import RetinaFace

from .utils import (
    STANDARD_LANDMARKS_5, 
    parse_landmarks_file, 
    get_ldm_slices, 
    create_batch_from_files, 
    as_numpy, 
    as_tensor,
)


class Cropper():
    """Face cropper class with bonus features.

    This class is capable of automatically aligning and center-cropping 
    faces, enhancing image quality and grouping the extracted faces 
    according to specified face attributes, as well as generating masks 
    for those attributes.
    
    Capabilities
    ------------

    This class has the following 3 main features:

        1. **Face cropping** - automatic face alignment and cropping 
           based on landmarks. The landmarks can either be predicted via 
           face detection model (see :class:`.RetinaFace`) or they 
           can be provided as txt, csv, json etc. file. It is possible 
           to control face factor in the extracted images and strategy 
           of extraction (e.g., largest face, all faces per image).
        2. **Face enhancement** - automatic quality enhancement of 
           images where the relative face area is small. For instance, 
           there may be images with many faces, but the quality of those 
           faces, if zoomed in, is low. Quality enhancement feature 
           allows to remove the blurriness. It can also enhance the 
           quality of every image, if desired (see 
           :class:`.RRDBNet`).
        3. **Face parsing** - automatic face attribute parsing and 
           grouping to sub-directories according selected attributes. 
           Attributes can indicate to group faces that contain specific 
           properties, e.g., "earrings and neckless", "glasses". They 
           can also indicate what properties the faces should not 
           include to form a group, e.g., "no accessories" group would 
           indicate to include faces without hats, glasses, earrings, 
           neckless etc. It is also possible to generate masks for 
           selected face attributes, e.g., "glasses", 
           "eyes and eyebrows". For more intuition on how grouping 
           works, see :class:`.BiSeNet` and 
           :meth:`save_groups`.

    The class is designed to perform all or some combination of the 
    functions in one go, however, each feature is independent of one 
    another and can work one by one. For example, it is possible to 
    first extract all the faces in some output directory, then apply 
    quality enhancement for every face to produce better quality faces 
    in another output directory and then apply face parsing to group 
    faces into different sub-folders according to some common attributes 
    in a final output directory.

    It is possible to configure the number of processing units and the 
    batch size for significant speedups., if the hardware allows.

    Examples
    --------

    Command line example
        >>> python face_crop_plus -i path/to/images -o path/to/out/dir

    Auto face cropping (with face factor) and quality enhancement:
        >>> cropper = Cropper(face_factor=0.7, enh_threshold=0.01)
        >>> cropper.process_dir(input_dir="path/to/images")

    Very fast cropping with already known landmarks (no enhancement):
        >>> cropper = Cropper(landmarks="path/to/landmarks.txt", 
                              num_processes=24,
                              enh_threshold=None)
        >>> cropper.process_dir(input_dir="path/to/images")

    Face cropping to attribute groups to custom output dir:
        >>> attr_groups = {"glasses": [6], "no_glasses_hats": [-6, -18]}
        >>> cropper = Cropper(attr_groups=attr_groups)
        >>> inp, out = "path/to/images", "path/to/parent/out/dir"
        >>> cropper.process_dir(input_dir=inp, output_dir=out)

    Face cropping and grouping by face attributes (+ generating masks):
        >>> groups = {"glasses": [6], "eyes_and_eyebrows": [2, 3, 4, 5]}
        >>> cropper = Cropper(output_format="png", mask_groups=groups)
        >>> cropper.process_dir("path/to/images")

    For grouping by face attributes, see documented face attribute 
    indices in :class:`.BiSeNet`.

    Class Attributes
    ----------------

    For how to initialize the class and to understand its functionality 
    better, please refer to class attributes initialized via 
    :meth:`__init__`. Here, further class attributes are 
    described automatically initialized via  :meth:`_init_models` and 
    :meth:`_init_landmarks_target`.

    Attributes:
        det_model (RetinaFace): Face detection model 
            (:class:`torch.nn.Module`) that is capable of detecting 
            faces and predicting landmarks used for face alignment. See 
            :class:`.RetinaFace`.
        enh_model (RRDBNet): Image quality enhancement model 
            (torch.nn.Module) that is capable of enhancing the quality 
            of images with faces. It can automatically detect which 
            faces to enhance based on average face area in the image, 
            compared to the whole image area. See :class:`.RRDBNet`.
        par_model (BiSeNet): Face parsing model (torch.nn.Module) that 
            is capable of classifying pixels according to specific face 
            attributes, e.g., "left_eye", "earring". It is able to group 
            faces to different groups and generate attribute masks. See 
            :class:`.BiSeNet`.
        landmarks_target (numpy.ndarray): Standard normalized landmarks 
            of shape  (``self.num_std_landmarks``, 2). These are scaled 
            by ``self.face_factor`` and used as ideal landmark 
            coordinates for the extracted faces. In other words, they 
            are reference landmarks used to estimate the transformation 
            of an image based on some actual set of face landmarks for 
            that image.
    """
    def __init__(
        self,
        output_size: int | tuple[int, int] | list[int] = 256,
        output_format: str | None = None,
        resize_size: int | tuple[int, int] | list[int] = 1024,
        face_factor: float = 0.65,
        strategy: str = "largest",
        padding: str = "reflect",
        allow_skew: bool = False,
        landmarks: str | tuple[np.ndarray, np.ndarray] | None = None,
        attr_groups: dict[str, list[int]] | None = None,
        mask_groups: dict[str, list[int]] | None = None,
        det_threshold: float | None = 0.6,
        enh_threshold: float | None = 0.001,
        batch_size: int = 8,
        num_processes: int = 1,
        device: str | torch.device = "cpu",
    ):
        """Initializes the cropper.

        Initializes class attributes. 

        Args:
            output_size: The output size (width, height) of cropped 
                image faces. If provided as a single number, the same 
                value is used for both width and height. Defaults to
                256.
            output_format: The output format of the saved face images. 
                For available options, see 
                `OpenCV imread <https://docs.opencv.org/4.x/d4/da8/group__imgcodecs.html#ga288b8b3da0892bd651fce07b3bbd3a56>`_. 
                If not specified, then the same image extension will not 
                be changed, i.e., face images will be of the same format 
                as the images from which they are extracted. Defaults to 
                None.
            resize_size: The interim size (width, height) each image 
                should be resized to before processing images. This is 
                used to resize images to a common size to allow to make 
                a batch. It should ideally be the mean width and height 
                of all the images to be processed (but can simply be a
                square). Images will be resized to to the specified size 
                while maintaining the aspect ratio (one of the 
                dimensions will always match either the specified width 
                or height). The shorter dimension would afterwards be 
                padded - for more information on how it works, see 
                :func:`.utils.create_batch_from_files`. Defaults to 
                1024.
            face_factor: The fraction of the face area relative to the 
                output image. Defaults to 0.65.
            strategy: The strategy to use to extract faces from each 
                image. The available options are:

                    * "all" - all faces will be extracted form each 
                      image.
                    * "best" - one face with the largest confidence 
                      score will be extracted from each image.
                    * "largest" - one face with the largest face area 
                      will be extracted from each image.

                For more info, see :meth:`.RetinaFace.__init__`. 
                Defaults to "largest".
            padding: The padding type (border mode) to apply when 
                cropping out faces. If faces are near edge, some part of 
                the resulting center-cropped face image may be blank, in 
                which case it can be padded with specific values. For 
                available options, see 
                `OpenCV BorderTypes <https://docs.opencv.org/3.4/d2/de8/group__core__array.html#ga209f2f4869e304c82d07739337eae7c5>`_. 
                If specified as "constant", the value of 0 will be used. 
                Defaults to "reflect".
            allow_skew: Whether to allow skewing when aligning the face 
                according to its landmarks. If True, then facial points 
                will be matched very closely to the ideal standard 
                landmark points (which is a set of reference points 
                created internally when preforming the transformation). 
                If all faces face forward, i.e., in portrait-like 
                manner, then this could be set to True which results in 
                minimal perspective changes. However, most of the time 
                this should be set to False to preserve the face 
                perspective. For more details, see 
                :meth:`.crop_align`. Defaults to False.
            landmarks: If landmarks are already known, they should be 
                specified via this variable. If specified, landmark 
                estimation will not be performed. There are 2 ways to 
                specify landmarks:

                    1. As a path to landmarks file, in which case str 
                       should be provided. The specified file should 
                       contain file (image) names and corresponding 
                       landmark coordinates. Duplicate file names are 
                       allowed (in case multiple faces are present in 
                       the same image). For instance, it could be 
                       .txt file where each row contains space-separated 
                       values: the first value is the file name and the 
                       other 136 values represent landmark coordinates 
                       in x1, y1, x2, y2, ... format. For more details 
                       about the possible file formats and how they are 
                       parsed, see 
                       :func:`~.utils.parse_landmarks_file`.
                    2. As a tuple of 2 numpy arrays. The first one is of 
                       shape (``num_faces``, ``num_landm``, 2) of type 
                       :attr:`numpy.float32` and represents the 
                       landmarks of every face that is going to be 
                       extracted from images. The second is a numpy 
                       array of shape (``num_faces``,) of type 
                       :class:`numpy.str_` where each value specifies a 
                       file name to which a corresponding set of 
                       landmarks belongs.

                If not specified, 5 landmark coordinates will be 
                estimated for each face automatically. Defaults to None.
            attr_groups: Attribute groups dictionary that specifies how
                to group the output face images according to some common
                attributes. The keys are names describing some common
                attribute, e.g., "glasses", "no_accessories" and the 
                values specify which attribute indices belong (or don't 
                belong, if negative) to that group, e.g., [6], 
                [-6, -9, -15]. For more information, see 
                :class:`.BiSeNet` and :meth:`save_groups`. 
                If not provided, output images will not be grouped by 
                attributes and no attribute sub-folders will be created
                in the desired output directory. Defaults to None.
            mask_groups: Mask groups dictionary that specifies how to 
                group the output face images according to some face
                attributes that make up a segmentation mask. The keys 
                are mask type names, e.g., "eyes", and the values 
                specify which attribute indices should be considered for 
                that mask, e.g., [4, 5]. For every group, not only face 
                images will be saved in a corresponding sub-directory, 
                but also black and white face attribute masks (white 
                pixels indicating the presence of a mask attribute). For 
                more details, see For more info, see 
                :class:`.BiSeNet` and :py:meth:`save_groups`.
                If not provided, no grouping is applied. Defaults to 
                None.
            det_threshold: The visual threshold, i.e., minimum 
                confidence score, for a detected face to be considered 
                an actual face. See :meth:`.RetinaFace.__init__` for 
                more details. If None, no face detection will be 
                performed. Defaults to 0.6.
            enh_threshold: Quality enhancement threshold that tells when 
                the image quality should be enhanced (it is an expensive 
                operation). It is the minimum average face factor, i.e., 
                face area relative to the image, below which the whole 
                image is enhanced. Defaults to 0.001.
            batch_size: The batch size. It is the maximum number of 
                images that can be processed by every processor at a 
                single time-step. Large values may result in memory 
                errors, especially, when GPU acceleration is used. 
                Increase this if less models (i.e., landmark detection, 
                quality enhancement, face parsing models) are used and 
                decrease otherwise. Defaults to 8.
            num_processes: The number of processes to launch to perform 
                image processing. Each process works in parallel on 
                multiple threads, significantly increasing the 
                performance speed. Increase if less prediction models 
                are used and increase otherwise. Defaults to 1.
            device: The device on which to perform the predictions, 
                i.e., landmark detection, quality enhancement and face 
                parsing. If landmarks are provided, no enhancement and 
                no parsing is desired, then this has no effect. Defaults
                to "cpu".
        """
        # Init specified attributes
        self.output_size = output_size
        self.output_format = output_format
        self.resize_size = resize_size
        self.face_factor = face_factor
        self.strategy = strategy
        self.padding = padding
        self.allow_skew = allow_skew
        self.landmarks = landmarks
        self.attr_groups = attr_groups
        self.mask_groups = mask_groups
        self.det_threshold = det_threshold
        self.enh_threshold = enh_threshold
        self.batch_size = batch_size
        self.num_processes = num_processes
        self.device = device

        # The only option for STD
        self.num_std_landmarks = 5

        # Modify attributes to have proper type
        if isinstance(self.output_size, int):
            self.output_size = (self.output_size, self.output_size)
        
        if len(self.output_size) == 1:
            self.output_size = (self.output_size[0], self.output_size[0])
        
        if isinstance(self.resize_size, int):
            self.resize_size = (self.resize_size, self.resize_size)
        
        if len(self.resize_size) == 1:
            self.resize_size = (self.resize_size[0], self.resize_size[0])

        if isinstance(self.device, str):
            self.device = torch.device(device)

        if isinstance(self.landmarks, str):
            self.landmarks = parse_landmarks_file(self.landmarks)

        # Further attributes
        self._init_models()
        self._init_landmarks_target()
    
    def _init_models(self):
        """Initializes detection, enhancement and parsing models.

        The method initializes 3 models:
            1. If ``self.det_threshold`` is provided and no landmarks 
               are known in advance, the detection model is initialized 
               to estimate 5-point landmark coordinates. For more info, 
               see :class:`.RetinaFace`.
            2. If ``self.enh_threshold`` is provided, the quality 
               enhancement model is initialized. For more info, see
               :class:`.RRDBNet`.
            3. If ``self.attr_groups`` or ``self.mask_groups`` is 
               provided, then face parsing model is initialized. For 
               more info, see :class:`.BiSeNet`.

        Note:
            This is a useful initializer function if multiprocessing is 
            used, in which case copies of all the models can be created 
            on separate cores.
        """
        # Init models as None
        self.det_model = None
        self.enh_model = None
        self.par_model = None

        if torch.cuda.is_available() and self.device.index is not None:
            # Helps to prevent CUDA memory errors
            torch.cuda.set_device(self.device.index)
            torch.cuda.empty_cache()

        if self.det_threshold is not None and self.landmarks is None:
            # If detection threshold is set, we will predict landmarks
            self.det_model = RetinaFace(self.strategy, self.det_threshold)
            self.det_model.load(device=self.device)
        
        if self.enh_threshold is not None:
            # If enhancement threshold is set, we might enhance quality
            self.enh_model = RRDBNet(self.enh_threshold)
            self.enh_model.load(device=self.device)
        
        if self.attr_groups is not None or self.mask_groups is not None:
            # If grouping by attributes or masks is set, use parse model
            args = (self.attr_groups, self.mask_groups, self.batch_size)
            self.par_model = BiSeNet(*args)
            self.par_model.load(device=self.device)
    
    def _init_landmarks_target(self):
        """Initializes target landmarks set.

        This method initializes a set of standard landmarks. Standard, 
        or target, landmarks refer to an average set of landmarks with 
        ideal normalized coordinates for each facial point. The source 
        facial points will be rotated, scaled and translated to match 
        the standard landmarks as close as possible.

        Both source (computed separately for each image) and target 
        landmarks must semantically match, e.g., the left eye coordinate 
        in target landmarks also corresponds to  the left eye coordinate 
        in source landmarks.

        There should be a standard landmarks set defined for a desired 
        number of landmarks. Each coordinate in that set is normalized, 
        i.e., x and y values are between 0 and 1. These values are then 
        scaled based on face factor and resized to match the desired 
        output size as defined by ``self.output_size``.

        Note:
            Currently, only 5 standard landmarks are supported.

        Raises:
            ValueError: If the number of standard landmarks is not 
                supported. The number of standard landmarks is 
                ``self.num_std_landmarks``.
        """
        match self.num_std_landmarks:
            case 5:
                # If the number of std landmarks is 5
                std_landmarks = STANDARD_LANDMARKS_5
            case _:
                # Otherwise the number of STD landmarks is not supported
                raise ValueError(f"Unsupported number of standard landmarks "
                                 f"for estimating alignment transform matrix: "
                                 f"{self.num_std_landmarks}.")
        
        # Apply appropriate scaling based on face factor and out size
        std_landmarks[:, 0] *= self.output_size[0] * self.face_factor
        std_landmarks[:, 1] *= self.output_size[1] * self.face_factor

        # Add an offset to standard landmarks to center the cropped face
        std_landmarks[:, 0] += (1 - self.face_factor) * self.output_size[0] / 2
        std_landmarks[:, 1] += (1 - self.face_factor) * self.output_size[1] / 2

        # Pass STD landmarks as target landms
        self.landmarks_target = std_landmarks

    def crop_align(
        self,
        images: np.ndarray | list[np.ndarray],
        padding: np.ndarray | None,
        indices: list[int],
        landmarks_source: np.ndarray,
    ) -> np.ndarray:
        """Aligns and center-crops faces based on the given landmarks.

        This method takes a batch of images (can be padded), and loops 
        through each image (represented as a numpy array) performing the 
        following actions:

            1. Removes the padding.
            2. Estimates affine transformation from source landmarks to 
               standard landmarks.
            3. Applies transformation to align and center-crop the face 
               based on the face factor.
            4. Returns a batch of face images represented as numpy 
               arrays of the same length ans the number of sets of 
               landmarks.

        Crucial role in this method plays ``self.landmarks_target`` 
        which is the standard set of landmarks used as a reference for 
        the source landmarks. Target and source landmark sets are used 
        to estimate transformations of images - each image to which a 
        set of landmarks (from source landmarks batch) belongs is 
        transformed such that the area covers the those landmarks as the 
        standard  (target) landmarks set (as ideally as possible). For 
        more details about target landmarks, check 
        :meth:`_init_landmarks_target`.

        Note:
            If ``self.allow_skew`` is set to True, then facial points 
            will also be skewed to match ``self.landmarks_target`` as 
            close as possible (resulting in, e.g., longer/flatter faces 
            than in the original images).

        Args:
            images: Image batch of shape (N, H, W, 3) of type 
                :attr:`numpy.uint8` (doesn't matter if RGB or BGR) where 
                each nth image is transformed to extract face(-s). 
                (H, W) should be ``self.resize_size``. It can also be a 
                list of :attr:`numpy.uint8` numpy arrays of different 
                shapes.
            padding: Padding of shape (N, 4) where the integer values 
                correspond to the number of pixels padded from each 
                side: top, bottom, left, right. Padding was originally 
                applied to each image, e.g., to make the image square, 
                so that all images could be stacked as a batch. 
                Therefore, it is needed here to remove the padding. If 
                specified as None, it is assumed that the images are 
                un-padded.
            indices: Indices list of length num_faces where each index 
                specifies which image is used to extract faces for each 
                set of landmarks in ``landmarks_source``.
            landmarks_source: Landmarks batch of shape 
                (num_faces, ``self.num_std_landmarks``, 2). These are 
                landmark sets of all the desired faces to extract from 
                the given batch of N images.

        Returns:
            A batch of aligned and center-cropped faces where the factor 
            of the area of a face relative to the whole face image area
            is ``self.face_factor``. The output is a numpy array of 
            shape (N, H, W) of type :attr:`numpy.uint8` (same channel 
            structure as for the input images). (H, W) is defined by 
            ``self.output_size``.
        """
        # Init list, border mode
        transformed_images = []
        border_mode = getattr(cv2, f"BORDER_{self.padding.upper()}")
        
        for landmarks_idx, image_idx in enumerate(indices):
            if self.allow_skew:
                # Perform full perspective transformation
                transform_function = cv2.estimateAffine2D
            else:
                # Preform only rotation, scaling and translation
                transform_function = cv2.estimateAffinePartial2D
            
            # Estimate transformation matrix to apply
            transform_matrix = transform_function(
                landmarks_source[landmarks_idx],
                self.landmarks_target,
                ransacReprojThreshold=np.inf,
            )[0]

            if transform_matrix is None:
                # Could not estimate
                continue

            # Retrieve current image
            image = images[image_idx]

            if padding is not None:
                # Crop out the un-padded area
                [t, b, l, r] = padding[image_idx]
                image = image[t:image.shape[0]-b, l:image.shape[1]-r]

            # Apply affine transformation to the image
            transformed_images.append(cv2.warpAffine(
                image,
                transform_matrix,
                self.output_size,
                borderMode=border_mode
            ))
        
        # Normally stacking would be applied unless the list is empty
        numpy_fn = np.stack if len(transformed_images) > 0 else np.array
        
        return numpy_fn(transformed_images)
    
    def save_group(
        self,
        faces: np.ndarray,
        file_names: list[str],
        output_dir: str,
    ):
        """Saves a group of images to output directory.

        Takes in a batch of faces or masks as well as corresponding file 
        names from where the faces were extracted and saves the 
        faces/masks to a specified output directory with the same names 
        as those image files (appends counter suffixes if multiple faces 
        come from the same file). If the batch of face images/masks is 
        empty, then the output directory is not created either.

        Args:
            faces: Face images (cropped and aligned) represented as a
                numpy array of shape (N, H, W, 3) with values of type
                :attr:`numpy.uint8` ranging from 0 to 255. It may also 
                be face mask of shape (N, H, W) with values of 255 where 
                some face attribute is present and 0 elsewhere.
            file_names: The list of filenames of length N. Each face 
                comes from a specific file whose name is also used to 
                save the extracted face. If ``self.strategy`` allows 
                multiple faces to be extracted from the same file, such 
                as "all", counters at the end of filenames are added.
            output_dir: The output directory to save ``faces``.
        """
        if len(faces) == 0:
            # Just return
            return
        
        # Create output directory, name counts
        os.makedirs(output_dir, exist_ok=True)
        file_name_counts = defaultdict(lambda: -1)

        for face, file_name in zip(faces, file_names):
            # Split each filename to base name, ext
            name, ext = os.path.splitext(file_name)

            if self.output_format is not None:
                # If specific img format given
                ext = '.' + self.output_format

            if self.strategy == "all":
                # Attach numbering to filenames
                file_name_counts[file_name] += 1
                name += f"_{file_name_counts[file_name]}"
            
            if face.ndim == 3:
                # If it's a colored img (not a mask), to BGR
                face = cv2.cvtColor(face, cv2.COLOR_RGB2BGR)

            # Make image path based on file format and save
            file_path = os.path.join(output_dir, name + ext)
            cv2.imwrite(file_path, face)
    
    def save_groups(
        self,
        faces: np.ndarray,
        file_names: np.ndarray,
        output_dir: str,
        attr_groups: dict[str, list[int]] | None,
        mask_groups: dict[str, tuple[list[int], np.ndarray]] | None,
    ):
        """Saves images (and masks) group-wise.

        This method takes a batch of face images of equal dimensions, a 
        batch of file names identifying which image each face comes 
        from, and, optionally, attribute and/or mask groups telling how 
        to split the face images (and masks) across different folders.
        This method then loops through all the groups and saves images 
        accordingly.

        Example 1:
            If neither ``attr_groups`` nor ``mask_groups`` are provided, 
            the face images will be saved according to this structure::

                ├── output_dir
                |    ├── face_image_0.jpg
                |    ├── face_image_1.png
                |    ...

        Example 2:
            If only ``attr_groups`` is provided (keys are names 
            describing common attributes across faces in that group and 
            they are also sub-directories of ``output_dir``), the 
            structure is as follows::

                ├── output_dir
                |    ├── attribute_group_1
                |    |    ├── face_image_0.jpg
                |    |    ├── face_image_1.png
                |    |    ...
                |    ├── attribute_group_2
                |    ...

        Example 3:
            If only ``mask_groups`` is provided (keys are names 
            describing the mask type and they are also sub-directories 
            of ``output_dir``), the structure is as follows::

                ├── output_dir
                |    ├── group_1
                |    |    ├── face_image_0.jpg
                |    |    ├── face_image_1.png
                |    |    ...
                |    ├── group_1_mask
                |    |    ├── face_image_0.jpg
                |    |    ├── face_image_1.png
                |    |    ...
                |    ├── group_2
                |    |    ...
                |    ├── group_2_mask
                |    |    ...
                |    ...

        Example 4:
            If both ``attr_groups`` and ``mask_groups`` are provided, 
            then all images and masks will first be grouped by 
            attributes and then by mask groups. The structure is then as 
            follows::

                ├── output_dir
                |    ├── attribute_group_1
                |    |    ├── group_1_mask
                |    |    |    ├── face_image_0.jpg
                |    |    |    ├── face_image_1.png
                |    |    |    ...
                |    |    ├── group_1_mask
                |    |    |    ├── face_image_0.jpg
                |    |    |    ├── face_image_1.png
                |    |    |    ...
                |    |    ├── group_2
                |    |    |    ...
                |    |    ├── group_2_mask
                |    |    |    ...
                |    |    ...
                |    |
                |    ├── attribute_group_2
                |    |    ...
                |    ...

        Args:
            faces: Face images (cropped and aligned) represented as a
                numpy array of shape (N, H, W, 3) with values of type
                :attr:`numpy.uint8` ranging from 0 to 255.
            file_names: File names of images from which the faces were 
                extracted. This value is a numpy array of shape (N,) 
                with values of type :class:`numpy.str_`. Each nth 
                face in ``faces`` maps to exactly one file nth name in
                this array, thus there may be duplicate file names
                (because different faces may come from the same file).
            output_dir: The output directory where the faces or folders 
                of faces will be saved to.
            attr_groups: Face groups by attributes. Each key represents 
                the group name (describes common attributes across
                faces) and each value is a list of indices identifying 
                faces (from `faces`) that should go to that group.
            mask_groups: Face groups by extracted masks. Each key
                represents group name (describes the mask type) and each 
                value is a tuple where the first element is a list of 
                indices identifying faces (from ``faces``) that should 
                go to that group and the second element is a batch of 
                masks corresponding to indexed faces represented as 
                numpy arrays of shape (N, H, W) with values of type 
                :attr:`numpy.uint8` and being either 0 (negative) or 255 
                (positive).
        """
        if attr_groups is None:
            # No-name group of idx mapping to all faces
            attr_groups = {'': list(range(len(faces)))}
        
        if mask_groups is None:
            # No-name group mapping to all faces, with no masks
            mask_groups = {'': (list(range(len(faces))), None)}

        for attr_name, attr_indices in attr_groups.items():
            for mask_name, (mask_indices, masks) in mask_groups.items():
                # Make mask group values that fall under attribute group
                group_idx = list(set(attr_indices) & set(mask_indices))
                group_dir = os.path.join(output_dir, attr_name, mask_name)

                # Retrieve group values & save
                face_group = [faces[idx] for idx in group_idx]
                file_name_group = file_names[group_idx]
                self.save_group(face_group, file_name_group, group_dir)

                if masks is not None:
                    # Save to masks dir
                    group_dir += "_mask"
                    masks = masks[[mask_indices.index(i) for i in group_idx]]
                    self.save_group(masks, file_name_group, group_dir)

    def process_batch(self, file_names: list[str], input_dir: str, output_dir: str):
        """Extracts faces from a batch of images and saves them.

        Takes file names, input directory, reads images and extracts 
        faces and saves them to the output directory. This method works 
        as follows:

            1. *Batch generation* - a batch of images form the given 
               file names is generated. Each images is padded and 
               resized to ``self.resize_size`` while keeping the same 
               aspect ratio.
            2. *Landmark detection* - detection model is used to predict 
               5 landmarks for each face in each image, unless the 
               landmarks were already initialized  or face alignment + 
               cropping is not needed.
            3. *Image enhancement* - some images are enhanced if the 
               faces compared with the image size are small. If 
               landmarks are None, i.e., if no alignment + cropping was 
               desired, all images are enhanced. Enhancement is not done 
               if ``self.enh_threshold`` is None.
            4. *Image grouping* - each face image is parsed, i.e., a map 
               of face attributes is generated. Based on those 
               attributes, each face image is put to a corresponding 
               group. There may also be mask groups, in which case masks 
               for each image in that group are also generated. Faces 
               are not parsed if ``self.attr_groups`` and 
               ``self.mask_groups`` are both None.
            5. *Image saving* - each face image (and a potential mask) 
               is saved according to the group structure (if there is 
               any).
        
        Note:
            If detection model is not used, then batch is just a list of 
            loaded images of different dimensions.

        Args:
            file_names: The list of image file names (not full paths). 
                All the images should be in the same directory.
            input_dir: Path to input directory with image files.
            output_dir: Path to output directory to save the extracted 
                face images.
        """
        if self.landmarks is None and self.det_model is None:
            # Initialize empty lists, landmarks and paddings as None
            images, indices, landmarks, paddings = [], [], None, None

            for i, file_name in enumerate(file_names):
                # Make path, load image, convert to RGB
                path = os.path.join(input_dir, file_name)
                image = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)

                # Update the lists
                indices.append(i)
                images.append(image)
        elif self.landmarks is not None:
            # Initialize empty image and index lists, set padding None
            images, img_indices, ldm_indices, paddings = [], [], [], None

            for i, file_name in enumerate(file_names):
                # Make path, load image, check indices
                path = os.path.join(input_dir, file_name)
                image = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
                indices_i = np.where(file_name == self.landmarks[1])[0]

                # Update the lists
                images.append(image)
                img_indices.extend([i] * len(indices_i))
                ldm_indices.extend(indices_i.tolist())
            
            # Update indices and select landmarks
            indices = img_indices
            landmarks = self.landmarks[0][ldm_indices]
        elif self.det_model is not None:
            # Create a batch of images (with faces) and their paddings
            b = create_batch_from_files(file_names, input_dir, self.resize_size)
            images, paddings = as_tensor(b[0], self.device), b[2]

            # If landmarks were not given, predict, undo padding
            landmarks, indices = self.det_model.predict(images)
            landmarks -= paddings[indices][:, None, [2, 0]]

        if landmarks is not None and len(landmarks) == 0:
            # Nothing to save
            return
            
        if landmarks is not None and landmarks.shape[1] != self.num_std_landmarks:
            # Compute the mean landmark coordinates from retrieved slices
            slices = get_ldm_slices(self.num_std_landmarks, landmarks.shape[1])            
            landmarks = np.stack([landmarks[:, s].mean(1) for s in slices], 1)

        if self.enh_model is not None:
            # Enhance some images
            images = as_tensor(images, self.device)
            images = self.enh_model.predict(images, landmarks, indices)

        # Convert to numpy images, initialize groups
        images, groups = as_numpy(images), (None, None)

        if landmarks is not None:    
            # Generate source, target landmarks, estimate & apply transform
            images = self.crop_align(images, paddings, indices, landmarks)

        if self.par_model is not None:
            # Predict attribute and mask groups if face parsing desired
            groups = self.par_model.predict(as_tensor(images, self.device))

        # Pick file names for each face, save faces
        file_names = np.array(file_names)[indices]
        self.save_groups(images, file_names, output_dir, *groups)
    
    def process_dir(self, input_dir: str, output_dir: str | None = None):
        """Processes images in the specified input directory.

        Splits all the file names in the input directory to batches 
        and processes batches on multiple cores. For every file name 
        batch, images are loaded, some are optionally enhanced, 
        landmarks are generated and used to optionally align and 
        center-crop faces, and grouping is optionally applied based on
        face attributes. For more details, check 
        :meth:`process_batch`.

        Note:
            There might be a few seconds delay before the actual 
            processing starts if there are a lot of files in the 
            directory - it takes some time to split all the file names 
            to batches.

        Args:
            input_dir: Path to input directory with image files.
            output_dir: Path to output directory to save the extracted 
                (and optionally grouped to sub-directories) face images. 
                If None, then the same path as for ``input_dir`` is used 
                and additionally "_faces" suffix is added to the name.
        """
        if output_dir is None:
            # Create a default output dir name
            output_dir = input_dir + "_faces"

        # Create batches of image file names in input dir
        files, bs = os.listdir(input_dir), self.batch_size
        file_batches = [files[i:i+bs] for i in range(0, len(files), bs)]

        if len(file_batches) == 0:
            # Empty
            return
        
        # Define worker function and its additional arguments
        kwargs = {"input_dir": input_dir, "output_dir": output_dir}
        worker = partial(self.process_batch, **kwargs)
        
        with ThreadPool(self.num_processes, self._init_models) as pool:
            # Create imap object and apply workers to it
            imap = pool.imap_unordered(worker, file_batches)
            list(tqdm.tqdm(imap, total=len(file_batches), desc="Processing"))
