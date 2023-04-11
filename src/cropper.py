import os
import cv2
import tqdm
import torch
import numpy as np
import torchsr
import torchsr.models
import torch.nn.functional as F

from collections import defaultdict
from multiprocessing import cpu_count
from multiprocessing.pool import ThreadPool

from models.retinaface import RetinaFace

from utils import parse_landmarks_file, get_landmark_indices_5, STANDARD_LANDMARKS_5, create_batch_from_img_path_list

class Cropper():

    def __init__(
        self,
        face_factor: float = 0.65,
        padding: str = "reflect",
        strategy: str = "largest",
        output_size: tuple[int, int] = (256, 256),
        num_std_landmarks: int = 5,
        is_partial: bool = True,
        det_model_name: str = "retinaface",
        det_backbone: str = "mobilenet0.25",
        det_threshold: float = 0.6,
        det_resize_size: int = 512,
        sr_scale: int = 1,
        sr_model_name: str = "ninasr_b0",
        landmarks: str | tuple[np.ndarray, np.ndarray] | None = None,
        
        output_format: str | None = None,
        batch_size: int = 8,
        num_cpus: int = cpu_count(),
        device: str | torch.device = "cpu"
    ):
        self.face_factor = face_factor
        self.det_threshold = det_threshold
        self.det_resize_size = det_resize_size
        self.padding = padding
        self.output_size = output_size
        self.output_format = output_format
        self.strategy = strategy
        self.batch_size = batch_size
        self.num_cpus = num_cpus
        self.num_std_landmarks = num_std_landmarks
        self.is_partial = is_partial
        self.sr_model_name = sr_model_name
        self.det_model_name = det_model_name
        self.det_backbone = det_backbone

        self.sr_scale = sr_scale

        if isinstance(device, str):
            device = torch.device(device)

        if isinstance(landmarks, str):
            landmarks = parse_landmarks_file(landmarks)
        
        # self.det_model = det_model
        self.landmarks = landmarks
        self.device = device

        self._init_models()
    
    def _init_models(self):
        if self.det_model_name == "retinaface":
            det_model = RetinaFace(
                backbone_name=self.det_backbone,
                size=self.det_resize_size,
                vis_threshold=self.det_threshold,
                strategy=self.strategy,
            ).to(self.device).eval()
        else:
            raise ValueError(f"Unsupported model: {self.det_model_name}.")
        
        if self.sr_scale > 1:
            sr_model = getattr(torchsr.models, self.sr_model_name)
            sr_model = sr_model(self.sr_scale, True).to(self.device).eval()
        else:
            sr_model = None

        self.det_model = det_model
        self.sr_model = sr_model
    
    def estimated_align(
        self,
        images: list[np.ndarray],
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

            # Apply affine transformation to the image
            transformed_images.append(cv2.warpAffine(
                images[image_idx],
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
    
    @torch.no_grad()
    def enhance_quality(self, images: np.ndarray) -> np.ndarray:
        if self.sr_model is None:
            return images

        x = torch.from_numpy(images).to(self.device)
        x = self.sr_model(x.permute(0, 3, 1, 2).div(255))
        # x = F.interpolate(x, size=self.output_size[::-1], mode="area")
        x = x.permute(0, 2, 3, 1).multiply(255).cpu().numpy().astype(np.uint8)

        return x

    def process_batch(self, file_batch: list[str], input_dir: str, output_dir: str):
        # batch = [cv2.imread(os.path.join(input_dir, f), cv2.IMREAD_COLOR) for f in file_batch]

        file_paths = [os.path.join(input_dir, file) for file in file_batch]
        batch, unscales, paddings = create_batch_from_img_path_list(file_paths)

        if self.landmarks is None:
            # If landmarks were not given, predict them
            landmarks, indices = self.det_model.predict(batch, self.device)
            landmarks = landmarks.cpu().numpy()

            preds = self.det_model.predict(batch, unscales, paddings)
            landmarks, indices = landmarks.cpu().numpy()
        else:
            # Generate indices for landmarks to take, then get landmarks
            indices = np.where(np.isin(self.landmarks[0], file_batch))[0]
            landmarks = self.landmarks[1][indices]

        # Generate source, target landmarks, estimate & apply transform
        src, tgt = self.generate_source_and_target_landmarks(landmarks)
        batch_aligned = self.estimated_align(batch, indices, src, tgt)
        batch_enhanced = self.enhance_quality(batch_aligned)
        
        # Update to include only images with existing landmarks
        file_name_counts = defaultdict(lambda: -1)
        
        for file_idx, image in zip(indices, batch_enhanced):
            file_name = file_batch[file_idx]

            if self.strategy == "all":
                file_name_counts[file_name] += 1
                name, ext = os.path.splitext(file_name)
                file_name = f"{name}_{file_name_counts[file_name]}{ext}"
            
            file_path = os.path.join(output_dir, file_name)
            cv2.imwrite(file_path, image)
    
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
    cropper = Cropper(strategy="all", det_backbone="resnet50", sr_scale=1)
    cropper.process_dir("ddemo2")