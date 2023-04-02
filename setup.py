import setuptools

with open("README.md", "r", encoding = "utf-8") as f:
    long_description = f.read()

setuptools.setup(
    name = "face-cropper-plus",
    version = "0.0.1",
    author = "mantasu (Mantas Birškus)",
    author_email = "<mantix7@gmail.com>",
    description = "Automatic face aligner and cropper with quality enhancement",
    long_description = long_description,
    long_description_content_type = "text/markdown",
    url = "TODO: Package URL",
    project_urls = {
        "Bug Tracker": "https://github.com/mantasu/face-cropper-plus/issues",
    },
    keywords = [
        "python",
        "face alignment",
        "super resolution",
        "center cropping",
    ],
    install_requires = [
        "nvidia-dali-cuda120",
        "torchsr",
    ],
    dependency_links = [
        "https://developer.download.nvidia.com/compute/redist",
    ],
    classifiers = [
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    package_dir = {"": "src"},
    packages = setuptools.find_packages(where="src"),
    python_requires = ">=3.10"
)