import argparse
from cropper import Cropper

def parse_args():
    pass

def main():
    attr_groups = {"glasses": [6], "earings": [9], "neckless": [15], "hat": [18], "no_accessuaries": [-6, -9, -15, -18]}
    mask_groups = {"eyes_and_eyebrows": [2, 4], "lip_and_mouth": [11, 12, 13], "nose": [10]}

    kwargs = {
        # "det_threshold": None,
        "landmarks": "landmark.txt",
        "enh_threshold": 0.01,
        "attr_groups": None,
        "mask_groups": None,
        "num_processes": 24,
        "batch_size": 8,
    }

    cropper = Cropper(strategy="best", resize_size=1024, device="cuda:0", **kwargs)
    cropper.process_dir("img_celeba")

if __name__ == "__main__":
    main()