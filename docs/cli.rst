====================
Command Line Options
====================

These flags allow you to change the behavior of :doc:`face_crop_plus.cropper`. Check out how to use them in :ref:`usage-via-command-line`.

.. option:: -c <path/to/config.json>, --config <path/to/config.json>

    Path to JSON file with arguments. If other arguments are further specified via command line, they will overwrite the ones with the same name in the JSON file.

    Default: None

.. option:: -i <path/to/images>, --input_dir <path/to/images>

    Path to input directory with image files. Input path is required to be specified either via this option or in the config file.

    Default: None

.. option:: -o <path/to/faces>, --output-dir <path/to/faces>

    Path to output directory to save the extracted face images. If not specified, the same path is used as for input_dir, except '_faces' suffix is added the name.

    Default: None

.. option:: -cn, --clean-names
    
    Whether to rename the files to os-compatible before processing. For instance, this will rename '北亰.jpg' to 'Bei Jing.jpg', '<>a?bc.jpg.jpg' to 'abcjpg.jpg' etc. Useful because some path errors could occur while reading those images when processing. Note that this will create a temporary directory with renamed images; to rename the images in-place, use ``-ci``.

.. option:: -ci, --clean-names-inplace,
    
    Same functionality as ``--clean-names``, except that all the files are renamed in ``input_dir``. This is not advised, however, if the directory contains many images, copying them to a temporary directory may be inefficient, thus this option can just rename the files in-place. Note that specifying this, will override ``-cn`` option, regardless if it's specified of not.

.. option:: -s <width> <height>, --output-size <width> <height>

    The output size (width, height) of cropped image faces. If provided as a single number, the same value is used for both width and height.

    Default: 256

.. option:: -f <extension>, --output-format <extension>

    The output format of the saved face images, e.g., 'jpg', 'png'. If not specified, the same format as the image from which the face is extracted will be used.

    Default: None

.. option:: -r <width> <height>, --resize-size <width> <height>

    The interim size (width, height) each image should be resized to before processing them. If provided as a single number, the same value is used for both width and height.

    Default: 1024

.. option:: -ff <ratio>, --face-factor <ratio>

    The fraction of the desired face area relative to the output image.

    Default: 0.65

.. option:: -st <type>, --strategy <type>
    
    The strategy to use to extract faces from each image.

    Choices: all, best, largest

    Default: largest

.. option:: -p <type>, --padding <type>
    
    The padding type (border mode) to apply when cropping out faces near edges.

    Options: constant, replicate, reflect, wrap, reflect_101

    Default: constant

.. option::  -a, --allow-skew 
    
    Whether to allow skewing the faces to better match the the standard (average) face landmarks.

.. option::  -l <path/to/landmarks/file>, --landmarks <path/to/landmarks/file>

    Path to landmarks file if landmarks are already known and prediction is not needed. Common file types are json (``"image.jpg": [x1, y1, ...]``), csv (``image.jpg,x1,y1,...``; first line is header), txt and other (``image.jpg x1 y2 ...``).

    Default: None

.. option::  -ag <group_dict>, --attr-groups <group_dict>

    Attribute groups dictionary that specifies how to group the output face images according to some common attributes. Should be a JSON-parsable string dictionary of type `dict[str, list[int]]`, e.g., ``'{"glasses": [6]}'``.

    Default: None

.. option::  -mg <group_dict>, --mask-groups <group_dict>

    Mask groups dictionary that specifies how to group the output face images according to some face attributes that make up a segmentation mask. Should be a JSON-parsable string dictionary of type `dict[str, list[int]]`, e.g., ``'{"eyes": [4, 5]}'``.

    Default: None

.. option::  -dt <threshold>, --det-threshold <threshold>

    The visual threshold, i.e., minimum confidence score, for a detected face to be considered an actual face. If a negative value is provided, e.g., -1, landmark prediction is not performed.
    
    Default: 0.6

.. option::  -et <threshold>, --enh-threshold <threshold>

    Quality enhancement threshold that tells when the image quality should be enhanced. It is the minimum average face factor in the input image, below which the whole image is enhanced. It is advised to set this to a low number, like 0.001 - very high fractions might unnecessarily cause the image quality to be improved. If a negative value is provided, no enhancement is performed.

    Default: -1

.. option::  -b <batch_size>, --batch-size <batch_size>

    The batch size. It is the maximum number of images that can be processed by every processor at a single time-step.
    
    Default: 8

.. option::  -n <num_processes>, --num-processes <num_processes>

    The number of processes to launch to perform image processing. If landmarks are provided and no quality enhancement or attribute grouping is done, feel free to set this to the number of CPUs your machine has.

    Default: 1

.. option::  -d <device>, --device <device>

    The device on which to perform the predictions, i.e., landmark detection, quality enhancement and face parsing. If specified as 'auto', it will be checked if CUDA is available and thus used, otherwise CPU will be assigned. Defaults to 'auto'.
    
    Default: auto
