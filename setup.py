import setuptools

with open("README.md", "r", encoding = "utf-8") as f:
    long_description = f.read()

setuptools.setup(
    name = "face_crop_plus",
    version = "0.1.0",
    author = "mantasu (Mantas Birškus)",
    author_email = "<mantix7@gmail.com>",
    license = "MIT",
    description = "Automatic face aligner and cropper with quality enhancement",
    long_description = long_description,
    long_description_content_type = "text/markdown",
    url = "TODO: Package URL",
    project_urls = {
        "Bug Tracker": "https://github.com/mantasu/face-crop-plus/issues",
    },
    keywords = [
        "face",
        "python",
        "pytorch",
        "alignment",
        "cropping",
        "super resolution",
        "quality enhancement",
        "parsing",
        "grouping",
        "attributes",
        "mask",
        "segmentation",
    ],
    install_requires = [
        "tqdm",
        "opencv-python",
        "torch",
        "torchvision",
    ],
    # dependency_links = [
    #     "https://download.pytorch.org/whl/cu118",
    # ],
    classifiers = [
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    package_dir = {"": "src"},
    packages = setuptools.find_packages(where="src"),
    entry_points={
        "console_scripts": [
            "face_crop_plus=face_crop_plus.main:main"
        ]
    },
    python_requires = ">=3.10"
)