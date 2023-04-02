import os
import cv2
import tqdm
import torch
import numpy as np
import torchsr
import torchsr.models
import torchvision.transforms.functional as F

from nvidia.dali import pipeline_def
import nvidia.dali.fn as fn
import nvidia.dali.math as math
import nvidia.dali.types as types

from collections import defaultdict
from multiprocessing import cpu_count
from multiprocessing.pool import ThreadPool

from models.retinaface import RetinaFace

from utils import parse_landmarks_file, get_landmark_indices

class Cropper():
    def __init__(
        self,
        face_factor: float = 0.65,
        det_model: str = "retinaface",
        det_backbone: str = "mobile0.25",
        det_threshold: float = 0.6,
        det_resize_size: int = 512,
        padding: str = "reflect",
        strategy: str = "largest",
        sr_scale: int = 1,
        sr_model: str = "ninasr_b0",
        landmarks: str | tuple[np.ndarray, np.ndarray] | None = None,
        output_size: tuple[int, int] = (256, 256),
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

        if det_model == "retinaface":
            det_model = RetinaFace(backbone_name=det_backbone).eval()
        else:
            raise ValueError(f"Unsupported detection model: {det_model}.")

        if isinstance(device, str):
            device = torch.device(device)
        
        if sr_scale > 1:
            sr_model = getattr(torchsr.models, sr_model)
            sr_model = sr_model(sr_scale, True).to(device)
        else:
            sr_model = None

        if isinstance(landmarks, str):
            landmarks = parse_landmarks_file(landmarks)
        
        self.det_model = det_model
        self.landmarks = landmarks
        self.sr_model = sr_model
        self.device = device

    @pipeline_def
    def face_align_pipeline(self, batch_size: int, num_threads: int, device_id: int, out_w: int, out_h: int, face_factor: float = 0.8):
        images = fn.external_source(name="images")
        left_eye = fn.external_source(name="left_eye")
        right_eye = fn.external_source(name="right_eye")
        nose = fn.external_source(name="nose")

        # calculate the angle between the eyes
        dx = right_eye[0] - left_eye[0]
        dy = right_eye[1] - left_eye[1]
        angle = math.atan2(dy, dx)

        # calculate the scaling factor
        dist = math.sqrt(dx * dx + dy * dy)
        scale = (face_factor * out_w) / dist

        # build the transformation matrix
        transform = fn.transforms.rotation(angle=angle, center=nose)
        transform = fn.transforms.scale(transform, scale=scale)

        # apply the transformation
        aligned_images = fn.warp_affine(images.gpu(), matrix=transform, size=(out_h,out_w), border='reflect', interp_type=types.INTERP_LINEAR)
        aligned_images = fn.cast(aligned_images, dtype=types.UINT8)

        # crop around the nose
        crop_pos_x = nose[0] / out_w
        crop_pos_y = nose[1] / out_h
        aligned_images = fn.crop(aligned_images, crop_pos_x=crop_pos_x, crop_pos_y=crop_pos_y, crop_w=out_w, crop_h=out_h)

        return aligned_images
    

    @pipeline_def
    def align_faces(self, images, left_eye_coords, right_eye_coords, nose_coords, out_w, out_h, face_factor):
        # convert external sources to DALI tensors
        images = fn.external_source(source=images)
        left_eye_coords = fn.external_source(source=left_eye_coords)
        right_eye_coords = fn.external_source(source=right_eye_coords)
        nose_coords = fn.external_source(source=nose_coords)

        # calculate rotation angles
        # dx = right_eye_coords[:, 0] - left_eye_coords[:, 0]
        # dy = right_eye_coords[:, 1] - left_eye_coords[:, 1]
        # angle = math.atan2(dy, dx) * 180 / np.pi

        # calculate the angle between the eyes
        dx = right_eye_coords[0] - left_eye_coords[0]
        dy = right_eye_coords[1] - left_eye_coords[1]
        angle = math.atan2(dy, dx)

        # calculate the scaling factor
        dist = math.sqrt(dx * dx + dy * dy)
        scale = (face_factor * out_w) / dist

        # rotate images
        rotated_images = fn.transforms.rotation(images,
                                                angle=angle,
                                                center=nose_coords)

        # calculate crop coordinates
        eye_centers = (left_eye_coords + right_eye_coords) / 2
        face_sizes = face_factor * math.sqrt((dx ** 2) + (dy ** 2))
        x1 = eye_centers[:, 0] - face_sizes / 2
        y1 = eye_centers[:, 1] - face_sizes / 2
        x2 = x1 + face_sizes
        y2 = y1 + face_sizes

        # crop images
        cropped_images = fn.transforms.crop(rotated_images,
                                            crop_pos_x=x1,
                                            crop_pos_y=y1,
                                            crop=[x2-x1,y2-y1])

        # resize images
        aligned_images = fn.resize(cropped_images,
                                resize_x=out_w,
                                resize_y=out_h)

        return aligned_images
    

    @pipeline_def
    def align_crop_pipeline(self, file_names, root_dir, left_eye_coords, right_eye_coords, nose_coords, out_w, out_h, face_factor):
        left_eye_coords = fn.external_source(source=left_eye_coords)
        right_eye_coords = fn.external_source(source=right_eye_coords)
        nose_coords = fn.external_source(source=nose_coords)

        files, _ = fn.readers.file(file_root=root_dir, files=file_names)
        images = fn.decoders.image(files, device="mixed")
        shapes = fn.peek_image_shape(files)
        h, w = shapes[0], shapes[1]

        # calculate rotation angles
        # dx = right_eye_coords[:, 0] - left_eye_coords[:, 0]
        # dy = right_eye_coords[:, 1] - left_eye_coords[:, 1]
        # angles = math.atan2(dy, dx) * 180 / np.pi

        # calculate the angle between the eyes
        dx = right_eye_coords[0] - left_eye_coords[0]
        dy = right_eye_coords[1] - left_eye_coords[1]
        angle = math.atan2(dy, dx)

        # calculate the scaling factor
        face_sizes = face_factor * ((dx ** 2) + (dy ** 2))
        x1 = nose_coords[:, 0] - face_sizes * out_w
        y1 = nose_coords[:, 1] - face_sizes * out_h
        x2 = x1 + face_sizes * out_w
        y2 = y1 + face_sizes * out_h

        mt = fn.transforms.rotation(angle=angle, center=nose_coords)
        # mt = fn.transforms.crop(mt, from_start=[x1, y1], from_end=[x2, y2], to_start = [0,0], to_end=[out_w, out_h])
        images = fn.warp_affine(images, size=[out_w, out_h], matrix=mt, fill_value=0, inverse_map=False)

        return images
    

    
    def temp(self, images, landmarks):
        new_images = []

        for image, landmark in zip(images, landmarks):
            b = list(map(int, landmark))
            cv2.circle(image, (b[0], b[1]), 1, (0, 0, 255), 4)
            cv2.circle(image, (b[2], b[3]), 1, (0, 255, 255), 4)
            cv2.circle(image, (b[4], b[5]), 1, (255, 0, 255), 4)
            cv2.circle(image, (b[6], b[7]), 1, (0, 255, 0), 4)
            cv2.circle(image, (b[8], b[9]), 1, (255, 0, 0), 4)

            new_images.append(image)
        
        return new_images

    def process_batch(self, file_batch: list[str], input_dir: str, output_dir: str):
        batch = [cv2.imread(os.path.join(input_dir, f), cv2.IMREAD_COLOR) for f in file_batch]

        if self.landmarks is None:
            # If landmarks were not given, predict them
            landmarks, indices, new_batch = self.det_model.predict(
                images=batch,
                padding=self.padding,
                size=self.det_resize_size,
                vis_threshold=self.det_threshold,
                nms_threshold=0.4,
                strategy=self.strategy,
                variance=[0.1, 0.2],
                device=self.device,
            )
        else:
            # Generate indices for landmarks to take, then get landmarks
            indices = np.where(np.isin(self.landmarks[0], file_batch))[0]
            landmarks = self.landmarks[1][indices]


        landmarks = np.stack(landmarks).reshape(len(landmarks), -1, 2)
        indices_dict = get_landmark_indices(landmarks.shape[1])
        eyel = landmarks[:, indices_dict["left_eye"]].mean(axis=1)
        eyer = landmarks[:, indices_dict["right_eye"]].mean(axis=1)
        nose = landmarks[:, indices_dict["nose_tip"]]

        pipe = self.align_crop_pipeline(np.array(file_batch)[indices].tolist(), input_dir, eyel, eyer, nose, *self.output_size, self.face_factor, batch_size=len(indices), num_threads=1, device_id=0)
        pipe.build()
        batch_aligned = pipe.run()
        
        # Update to include only images with existing landmarks
        file_name_counts = defaultdict(lambda: -1)
        
        for file_idx, image in zip(indices, batch_aligned):
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

        # with ThreadPool(processes=self.num_cpus) as pool:
        #     # Create imap object and apply workers to it
        #     args = (file_batches, input_dir, output_dir)
        #     imap = pool.imap_unordered(self.process_batch, args)
        #     list(tqdm(imap, total=len(file_batches)))


if __name__ == "__main__":
    cropper = Cropper(strategy="all", padding="constant", det_resize_size=800)
    cropper.process_dir("ddemo2")