========
Examples
========

.. _usage-via-command-line:

Usage via Command Line
----------------------

The simplest way to run the program is to just specify the input directory with images:

.. code-block:: bash

    face-crop-plus --input-dir path/to/images

Most likely, you will want to change processing behavior, such as output size, face factor, and device:

.. code-block:: bash

    face-crop-plus -i path/to/images --output-size 200 300 --face-factor 0.75 -d cuda:0

You can also specify the command-line arguments via JSON config file and provide the path to it. Further command-line arguments would overwrite the values taken from the JSON file.

.. code-block:: bash

    face-crop-plus --config path/to/json --attr-groups '{"glasses": [6]}'

An example JSON config file is provided `here <https://github.com/mantasu/face-crop-plus/blob/main/demo/demo.json>`_. If you've cloned the repository, you can run from it:

.. code-block:: bash

    face-crop-plus --config demo/demo.json --device cuda:0 # overwrite device


For a full list of arguments, see :doc:`cli` or just type:

.. code-block:: bash

    face-crop-plus -h

.. note::

    You can use ``fcp`` as ``face-crop-plus`` alias , e.g., ``fcp -c config.json``

Usage via Python Script
-----------------------

To use the package in your own python script, simply import ``Cropper`` class, initialize it and run ``process_dir`` method:

.. code-block:: python

    from face_crop_plus import Cropper

    cropper = Cropper(face_factor=0.7, strategy="largest")
    cropper.process_dir(input_dir="path/to/images")

For a quick demo, you can experiment with `demo.py <https://github.com/mantasu/face-crop-plus/blob/main/demo/demo.py>`_ file:

.. code-block:: python

    git clone https://github.com/mantasu/face-crop-plus
    cd face-crop-plus/demo
    python demo.py

.. _pure-enhancement-parsing:

Pure Enhancement/Parsing
------------------------

If you already have aligned and center-cropped face images, you can perform quality enhancement and face parsing without re-cropping them. Here is an example of enhancing quality of every face and parsing them to (note that none of the parameters described in *Alignment and Cropping* section have any affect here):

.. code-block:: python

    from face_crop_plus import Cropper

    cropper = Cropper(
        det_threshold=None,
        enh_threshold=1, # enhance every image   
        attr_groups={"hats": [18], "no_hats": [-18]},
        mask_groups={"hats": [18], "ears": [7, 8, 9]},
        device="cuda:0",
    )

    cropper.crop(input_dir="path/to/images")

This would result in the following output directory structure:

.. code-block:: bash

    └── path/to/images_faces
        ├── hats
        |    ├── hats       # Images with hats
        |    ├── hats_mask  # Hat masks for images in upper dir
        |    ├── ears       # Images with hats and visible ears
        |    └── ears_mask  # Ears masks for images in upper dir
        |
        └── no_hats
            ├── ears       # Masks with no hats and visible ears
            └── ears_mask  # Ears masks for images in upper dir


To just enhance the quality of images (e.g., if you have blurry photos), you can run enhancement feature separately:

.. code-block:: bash

    face-crop-plus -i path/to/images -dt -1 -et 1 --device cuda:0


To just generate masks for images (e.g., as part of your research pipeline), you can run segmentation feature separately. This will only consider images for which the masks are actually present.

.. code-block:: bash

    face-crop-plus -i path/to/images -dt -1 -et -1 -mg '{"glasses": [6]}'


Please beware of the following:

    * While you can perform quality enhancement on images of different sizes (because, due to large amount of computations, images are processed one by one), you cannot perform face parsing (attribute-based grouping/segmentation) if images have different dimensions (though a possible work around is to set the batch size to 1).
    * It is not advised to perform quality enhancement after cropping the images since there is not enough information for the model on how to improve the quality. If you still need to enhance the quality after cropping, using larger image sizes, e.g., `512×512`, may help. Regardless whether you use it before or after cropping, do not use input images of spatial size over `2000×2000`, unless you have a powerful GPU.

Preprocessing CelebA
--------------------

Here is an example pipeline of how to pre-process `CelebA <https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html>`_ dataset. It is useful if you want to customize the cropped face properties, e.g., face factor, output size. It only takes a few minutes to pre-process the whole dataset using multiple processors and the provided landmarks:

1. Download the following files from *Google Drive*:

    * Download `img_celeba.7z <https://drive.google.com/drive/folders/0B7EVK8r0v71peklHb0pGdDl6R28?resourcekey=0-f5cwz-nTIQC3KsBn3wFn7A>`_ folder and put it under ``data/img_celeba.7z``
    * Download `nnotations.zip <https://drive.google.com/file/d/1xd-d1WRnbt3yJnwh5ORGZI3g-YS-fKM9/view>`_ file and put it under ``data/annotations.zip``

2. Unzip the data:

    >>> 7z x data/img_celeba.7z/img_celeba.7z.001 -o./data
    >>> unzip data/annotations.zip -d data

3. Create a script file, e.g., ``preprocess_celeba.py``, in the same directory:

    .. code-block:: python

        from face_crop_plus import Cropper
        from multiprocessing import cpu_count

        cropper = Cropper(
            output_size=256,
            face_factor=0.7,
            landmarks="data/landmark.txt",
            enh_threshold=None,
            num_processes=cpu_count(),
        )

        cropper.process_dir("data/img_celeba")

4. Run the script to pre-process the data:

    >>> python preprocess_celeba.py

5. Clean up the data dir (remove the original images and the annotations):

    >>> rm -r data/img_celeba.7z data/img_celeba
    >>> rm data/annotations.zip data/*.txt
