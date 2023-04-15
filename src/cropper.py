import os
import cv2
import tqdm
import torch
import numpy as np
import torch.nn.functional as F

from collections import defaultdict
from multiprocessing import cpu_count
from multiprocessing.pool import ThreadPool

from models.bise import BiSeNet
from models.rrdb import RRDBNet
from models.retinaface import RetinaFace


from utils import parse_landmarks_file, get_landmark_indices_5, STANDARD_LANDMARKS_5, create_batch_from_img_path_list

class Cropper():
    # [eye_g, ear_r, neck_l, hat] == [6, 9, 15, 18]

    def __init__(
        self,
        output_size: int | tuple[int, int] = 256,
        output_format: str | None = None,
        resize_size: int | tuple[int, int] = 1024,
        face_factor: float = 0.65,
        strategy: str = "largest",
        padding: str = "reflect",
        is_partial: bool = True,
        landmarks: str | tuple[np.ndarray, np.ndarray] | None = None,
        attr_groups: dict[str, list[int]] | None = None,
        mask_groups: dict[str, list[int]] | None = None,
        det_threshold: float | None = 0.6,
        enh_threshold: float | None = 0.001,
        batch_size: int = 8,
        num_cpus: int = cpu_count(),
        device: str | torch.device = "cpu",
    ):
        # Init specified attributes
        self.output_size = output_size
        self.output_format = output_format
        self.resize_size = resize_size
        self.face_factor = face_factor
        self.strategy = strategy
        self.padding = padding
        self.is_partial = is_partial
        self.landmarks = landmarks
        self.attr_groups = attr_groups
        self.mask_groups = mask_groups
        self.det_threshold = det_threshold
        self.enh_threshold = enh_threshold
        self.batch_size = batch_size
        self.num_cpus = num_cpus
        self.device = device

        # Default parameters
        self.num_std_landmarks = 5
        self.att_join_by_and = True
        self.attr_threshold = 5

        if isinstance(self.output_size, int):
            self.output_size = (self.output_size, self.output_size)
        
        if isinstance(self.resize_size, int):
            self.resize_size = (self.resize_size, self.resize_size)

        if isinstance(self.device, str):
            self.device = torch.device(device)

        if isinstance(self.landmarks, str):
            self.landmarks = parse_landmarks_file(self.landmarks)

        self._init_models()
    
    def _init_models(self):
        # Init models as None
        self.det_model = None
        self.enh_model = None
        self.par_model = None

        if self.det_threshold is not None:
            # If detection threshold is set, we will predict landmarks
            self.det_model = RetinaFace(self.strategy, self.det_threshold)
            self.det_model.load(device=self.device)
        
        if self.enh_threshold is not None:
            # If enhancement threshold is set, we might enhance quality
            self.enh_model = RRDBNet(self.enh_threshold)
            self.enh_model.load(device=self.device)
        
        if self.attr_groups is not None or self.mask_groups is not None:
            # If grouping by attributes or masks is set, use parse model
            self.par_model = BiSeNet(self.attr_groups, self.mask_groups)
            self.par_model.load(device=self.device)
    
    def crop_align(
        self,
        images: np.ndarray,
        padding: np.ndarray,
        indices: list[int],
        landmarks_source: np.ndarray,
        landmarks_target: np.ndarray,
    ) -> np.ndarray:
        # Init list, border mode
        transformed_images = []
        border_mode = getattr(cv2, f"BORDER_{self.padding.upper()}")
        
        for landmarks_idx, image_idx in enumerate(indices):
            if self.is_partial:
                # Preform only rotation, scaling and translation
                transform_function = cv2.estimateAffinePartial2D
            else:
                # Perform full perspective transformation
                transform_function = cv2.estimateAffine2D
            
            # Estimate transformation matrix to apply
            transform_matrix = transform_function(
                landmarks_source[landmarks_idx],
                landmarks_target,
                ransacReprojThreshold=np.inf,
            )[0]

            if transform_matrix is None:
                # Could not estimate
                continue
            
            # Crop out the raw image area (without pdding)
            img, pad = images[image_idx], padding[image_idx]
            img = img[pad[0]:img.shape[0]-pad[1], pad[2]:img.shape[1]-pad[3]]

            # Apply affine transformation to the image
            transformed_images.append(cv2.warpAffine(
                img,
                transform_matrix,
                self.output_size,
                borderMode=border_mode
            ))
        
        return np.stack(transformed_images)
    
    def generate_source_and_target_landmarks(
        self,
        landmarks: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        # Num STD landmarks cannot exceed the actual number of landmarks
        num_std_landmarks = min(landmarks.shape[1], self.num_std_landmarks)

        match num_std_landmarks:
            case 5:
                # If the number of standard lms is 5
                std_landmarks = STANDARD_LANDMARKS_5
                slices = get_landmark_indices_5(landmarks.shape[1])
            case _:
                # Otherwise the number of STD landmarks is not supported
                raise ValueError(f"Unsupported number of standard landmarks "
                                 f"for estimating alignment transform matrix: "
                                 f"{num_std_landmarks}.")
        
        # Compute the mean landmark coordinates from retrieved slices
        landmarks = np.stack([landmarks[:, s].mean(1) for s in slices], axis=1)
        
        # Apply appropriate scaling based on face factor and out size
        std_landmarks[:, 0] *= self.output_size[0] * self.face_factor
        std_landmarks[:, 1] *= self.output_size[1] * self.face_factor

        # Add an offset to standard landmarks to center the cropped face
        std_landmarks[:, 0] += (1 - self.face_factor) * self.output_size[0] / 2
        std_landmarks[:, 1] += (1 - self.face_factor) * self.output_size[1] / 2

        return landmarks, std_landmarks
    
    def enhance_quality(self, images: torch.Tensor, landmarks: np.ndarray, indices: list[int]) -> torch.Tensor:
        if self.enh_model is None:
            return images
        
        indices = np.array(indices)

        for i in range(len(images)):
            # Select faces for curr sample
            faces = landmarks[indices == i]

            if len(faces) == 0:
                continue
            
            # Compute relative face factor
            [w, h] = (faces[:, 4] - faces[:, 0]).T
            face_factor = w * h / (images.shape[2] * images.shape[3])

            if face_factor.mean() < self.enh_threshold:
                # Enhance curr image if face factor below threshold
                images[i:i+1] = self.enh_model.predict(images[i:i+1])

        return images
    
    def save_group(
        self,
        faces: np.ndarray,
        file_names: list[str],
        output_dir: str,
    ):
        # Create output directory, name counts
        os.makedirs(output_dir, exist_ok=True)
        file_name_counts = defaultdict(lambda: -1)

        for face, file_name in zip(faces, file_names):
            # Split each filename to base name, ext
            name, ext = os.path.splitext(file_name)

            if self.output_format is not None:
                # If specific img format given
                ext = '.' + self.output_format

            if self.det_model.strategy == "all":
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
            If neither `attr_groups` nor `mask_groups` are provided, the 
            face images will be saved according to this structure:
            ```
            ├── output_dir
            |    ├── face_image_0.jpg
            |    ├── face_image_1.png
            |    ...
            ```

        Example 2:
            If only `attr_groups` is provided (keys are names describing 
            common attributes across faces in that group and they are
            also sub-directories of `output_dir`), the structure is as
            follows:
            ```
            ├── output_dir
            |    ├── attribute_group_1
            |    |    ├── face_image_0.jpg
            |    |    ├── face_image_1.png
            |    |    ...
            |    ├── attribute_group_2
            |    ...
            ```
        
        Example 3:
            If only `mask_groups` is provided (keys are names describing 
            the mask type and they are also sub-directories of
            `output_dir`), the structure is as follows:
            ```
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
            ```
        
        Example 4:
            If both `attr_groups` and `mask_groups` are provided, then 
            all images and masks will first be grouped by attributes and 
            then by mask groups. The structure is then as follows:
            ```
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
            ```

        Args:
            faces: Face images (cropped and aligned) represented as a
                numpy array of shape (N, H, W, 3) with values of type
                np.uint8 ranging from 0 to 255.
            file_names: File names of images from which the faces were 
                extracted from. This value is a numpy array of shape
                (N,) with values of type 'U' (numpy string type). Each
                nth face in `faces` maps to exactly one file nth name in
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
                indices identifying faces (from `faces`) that should go 
                to that group and the second element is a batch of masks
                corresponding to indexed faces represented as a numpy
                arrays of shape (N, H, W) with values of type np.uint8 
                and being either 0 (negative) or 255 (positive).
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
                face_group = faces[group_idx]
                file_name_group = file_names[group_idx]
                self.save_group(face_group, file_name_group, group_dir)

                if masks is not None:
                    # Save to masks dir
                    group_dir += "_mask"
                    masks = masks[[mask_indices.index(i) for i in group_idx]]
                    self.save_group(masks, file_name_group, group_dir)

    def process_batch(self, file_names: list[str], input_dir: str, output_dir: str):
        file_paths = [os.path.join(input_dir, file) for file in file_names]
        # images, paddings = load_images_as_batch(file_paths, self.mean_size)
        images, scales, paddings = create_batch_from_img_path_list(file_paths, size=self.resize_size)
        paddings = paddings.numpy()

        images = images.permute(0, 3, 1, 2).float().to(self.device)

        if self.det_model is not None:
            # If landmarks were not given, predict them
            landmarks, indices = self.det_model.predict(images)            
            landmarks = landmarks.numpy()
            landmarks -= paddings[indices][:, None, [2, 0]]
        elif self.landmarks is not None:
            # Generate indices for landmarks to take, then get landmarks
            indices = np.where(np.isin(self.landmarks[0], file_names))[0]
            landmarks = self.landmarks[1][indices]

        images = self.enhance_quality(images, landmarks, indices)
        images = images.permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)

        # Generate source, target landmarks, estimate & apply transform
        src, tgt = self.generate_source_and_target_landmarks(landmarks)
        faces = self.crop_align(images, paddings, indices, src, tgt)
        
        file_names = np.array(file_names)[indices]

        if self.par_model is None:
            groups = None, None
        else:
            x = torch.from_numpy(faces).permute(0, 3, 1, 2).float()
            groups = self.par_model.predict(x.to(self.device))

        self.save_groups(faces, file_names, output_dir, *groups)

    
    def process_dir(self, input_dir: str, output_dir: str | None = None):
        if output_dir is None:
            # Create a default output dir name
            output_dir = input_dir + "_aligned"
        
        # Create the actual output directory
        os.makedirs(output_dir, exist_ok=True)

        # Create batches of image file names in input dir
        files, bs = os.listdir(input_dir), self.batch_size
        file_batches = [files[i:i+bs] for i in range(0, len(files), bs)]

        for i, file_batch in enumerate(file_batches):
            print("Processing batch", i, f"[{len(file_batch)}]" )
            self.process_batch(file_batch, input_dir, output_dir)

        # with ThreadPool(processes=self.num_cpus, initializer=self._init_models) as pool:
        #     # Create imap object and apply workers to it
        #     args = (file_batches, input_dir, output_dir)
        #     imap = pool.imap_unordered(self.process_batch, args)
        #     list(tqdm(imap, total=len(file_batches)))


if __name__ == "__main__":
    attr_groups = {"glasses": [6], "earings": [9], "neckless": [15], "hat": [18], "no_accessuaries": [-6, -9, -15, -18]}
    mask_groups = {"eyes_and_eyebrows": [2, 3, 4, 5], "lip_and_mouth": [11, 12, 13], "nose": [10]}
    # attr_groups = None
    # mask_groups = None
    

    cropper = Cropper(strategy="all", resize_size=1024, device="cuda:0", face_factor=0.55, attr_groups=attr_groups, mask_groups=mask_groups)
    cropper.process_dir("ddemo2")