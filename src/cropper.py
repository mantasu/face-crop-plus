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
        device: str | torch.device = "cpu",
        enh_threshold = 0.001
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

        self.enh_threshold = enh_threshold
        

        # [eye_g, ear_r, neck_l, hat] == [6, 9, 15, 18]

        self.att_threshold = 5
        self.att_join_by_and = True
        
        self.att_groups = {"glasses": [6], "earings_and_necklesses": [9, 15], "no_accessuaries": [-6, -9, -15, -18]}
        self.seg_groups = {"eys_and_eyebrows": [2, 3, 4, 5], "lip_and_mouth": [11, 12, 13], "nose": [10]}

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
            det_model = RetinaFace(strategy=self.strategy)
            det_model.to(self.device)
        else:
            raise ValueError(f"Unsupported model: {self.det_model_name}.")
        
        if self.enh_threshold is not None:
            enh_model = RRDBNet()
            enh_model.load(device=self.device)
        else:
            enh_model = None
        
        if self.groups is not None:
            par_model = BiSeNet()
            par_model.load(device=self.device)
        else:
            par_model = None

        self.det_model = det_model
        self.enh_model = enh_model
        self.par_model = par_model
    
    def align(
        self,
        # images: list[np.ndarray],
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
            w = faces[:, 4, 0] - faces[:, 0, 0]
            h = faces[:, 4, 1] - faces[:, 0, 1]
            face_factor = w * h / (images.shape[2] * images.shape[3])

            if face_factor.mean() < self.enh_threshold:
                # Enhance curr image if face factor below threshold
                images[i:i+1] = self.enh_model.predict(images[i:i+1])

        return images
    
    def group_by_attributes(self, parse_preds:  torch.Tensor):
        if self.att_groups is None:
            return None
        
        att_groups = {}

        att_join = torch.all if self.att_join_by_and else torch.any
        
        for k, v in self.att_groups:
            att_groups[k] = []
            attr = torch.tensor(v, device=self.device).view(1, -1, 1, 1)
            is_attr = (parse_preds.unsqueeze(1) == attr.abs()).sum(dim=(2, 3))

            is_attr = att_join(torch.stack([
                is_attr[:, i] > self.att_threshold if a > 0 else
                is_attr[:, i] <= self.att_threshold
                for i, a in enumerate(v)
            ], dim=1), dim=1)

            for i, pred in enumerate(parse_preds):
                if is_attr[i]:
                    att_groups[k].append((pred, i))
        
        return att_groups

    def group_by_segmentation(self, ):
        pass
    
    def parse_face_features(self, images: torch.Tensor):
        images = torch.from_numpy(images).permute(0, 3, 1, 2).float().to(self.device)
        print("Parsing", images.shape)
        batch = []
        t = 0

        att_groups, seg_groups = None, None

        if self.att_groups is not None:
            att_groups = {k: [] for k in self.att_groups.keys()}
        
        if self.seg_groups is not None:
            seg_groups = {k: [] for k in self.seg_groups.keys()}

        for sub_batch in torch.split(images, self.batch_size):
            out = self.par_model.predict(sub_batch.to(self.device))

            if self.att_groups is not None:
                for k, v in self.att_groups.items():
                    att = torch.tensor(v, device=self.device).view(1, -1, 1, 1)
                    is_att = (out.unsqueeze(1) == att.abs()).sum(dim=(2, 3))

            
            if self.seg_groups is not None:
                pass
            
            t += len(sub_batch)
        
        return images

    def process_batch(self, file_batch: list[str], input_dir: str, output_dir: str):
        file_paths = [os.path.join(input_dir, file) for file in file_batch]
        batch, scales, padding = create_batch_from_img_path_list(file_paths, size=self.det_resize_size)
        padding = padding.numpy()

        batch = batch.permute(0, 3, 1, 2).float().to(self.device)

        if self.landmarks is None:
            # If landmarks were not given, predict them
            landmarks, indices = self.det_model.predict(batch)            
            landmarks = landmarks.numpy()
            landmarks -= padding[indices][:, None, [2, 0]]
        else:
            # Generate indices for landmarks to take, then get landmarks
            indices = np.where(np.isin(self.landmarks[0], file_batch))[0]
            landmarks = self.landmarks[1][indices]

        batch = self.enhance_quality(batch, landmarks, indices)
        batch = batch.permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)

        # Generate source, target landmarks, estimate & apply transform
        src, tgt = self.generate_source_and_target_landmarks(landmarks)
        batch_aligned = self.align(batch, padding, indices, src, tgt)
        batch = self.parse_face_features(batch_aligned)

        batch_aligned = batch.permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)
        
        # Update to include only images with existing landmarks
        file_name_counts = defaultdict(lambda: -1)
        
        for file_idx, image in zip(indices, batch_aligned):
            file_name = file_batch[file_idx]

            if self.strategy == "all":
                file_name_counts[file_name] += 1
                name, ext = os.path.splitext(file_name)
                file_name = f"{name}_{file_name_counts[file_name]}{ext}"
            
            file_path = os.path.join(output_dir, file_name)
            cv2.imwrite(file_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    
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
    cropper = Cropper(strategy="largest", det_backbone="resnet50", det_resize_size=1024, device="cuda:0", face_factor=0.55)
    cropper.process_dir("ddemo2")