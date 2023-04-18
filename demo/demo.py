from face_crop_plus import Cropper
from torch.cuda import is_available
from os.path import join, dirname, abspath

INPUT_DIR = join(dirname(abspath(__file__)), "input_images")
OUTPUT_DIR = None # Defaults to "path/to/input_images_faces"

# Set all to False if running on CPU (unless you can wait for a bit)
TEST_QUALITY_ENHANCEMENT = True
TEST_ATTR_GROUPING = True
TEST_MASK_GROUPING = False

if __name__ == "__main__":
    # Initialize as None
    enh_threshold = None
    attr_groups = None
    mask_groups = None

    if TEST_QUALITY_ENHANCEMENT:
        enh_threshold = 0.001

    if TEST_ATTR_GROUPING:
        attr_groups = {"hat": [18], "no_accessories": [-6, -9, -15, -18]}
    
    if TEST_MASK_GROUPING:
        mask_groups = {"nose": [10], "eyes_and_eyebrows": [2, 3, 4, 5]}

    # Initialize cropper
    cropper = Cropper(
        output_size=(256, 256),
        output_format="jpg",
        face_factor=0.7,
        strategy="all",
        device = "cuda:0" if is_available() else "cpu",
        enh_threshold=enh_threshold,
        attr_groups=attr_groups,
        mask_groups=mask_groups,
    )

    # Process images in the input dir and save face images to output dir
    cropper.process_dir(input_dir=INPUT_DIR, output_dir=OUTPUT_DIR)
