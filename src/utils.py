import json
import numpy as np


def parse_landmarks_file(file_path: str, **kwargs,
                         ) -> tuple[np.ndarray, np.ndarray]:
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
        filenames = np.genfromtxt(file_path, usecols=0, **kwargs)
        landmarks = np.genfromtxt(file_path, **kwargs)[:, 1:]
    
    return filenames, landmarks.reshape(len(landmarks), -1, 2)

def get_landmark_indices(num_landmarks: int) -> dict:
    landmark_indices = {}

    match num_landmarks:
        case 5:
            landmark_indices = {
                "nose_tip": 2,
                "left_eye": slice(0, 1),
                "right_eye": slice(1, 2)
            }
        case 24:
            landmark_indices = {
                "nose_tip": 11,
                "left_eye": slice(3, 5),
                "right_eye": slice(5, 7)
            }
        case 30:
            landmark_indices = {
                "nose_tip": 14,
                "left_eye": slice(4, 6),
                "right_eye": slice(6, 8)
            }
        case 39:
            landmark_indices = {
                "nose_tip": 30,
                "left_eye": slice(36, 42),
                "right_eye": slice(42, 48)
            }
        case 48:
            landmark_indices = {
                "nose_tip": 28,
                "left_eye": slice(22, 27),
                "right_eye": slice(27, 32)
            }
        case 60:
            landmark_indices = {
                "nose_tip": 28,
                "left_eye": slice(22, 27),
                "right_eye": slice(27, 32)
            }
        case 68:
            landmark_indices = {
                "nose_tip": 30,
                "left_eye": slice(36, 42),
                "right_eye": slice(42, 48)
            }
        case 106:
            landmark_indices = {
                "nose_tip": 55,
                "left_eye": slice(60, 68),
                "right_eye": slice(68, 76)
            }
        case 194:
            landmark_indices = {
                "nose_tip": 33,
                "left_eye": slice(37, 46),
                "right_eye": slice(46, 55)
            }
        case _:
            raise ValueError(f"Unsupported number of landmarks: {num_landmarks}")

    return landmark_indices