import sys
import json
import argparse
from typing import Any
from .cropper import Cropper

class ArgumentParserWithConfig(argparse.ArgumentParser):
    """An ArgumentParser that loads default values from a config file.

    This class extends the :class:`argparse.ArgumentParser` class to 
    load default values from a config file specified by a command-line 
    argument.
    """
    def __init__(
        self,
        *args,
        config_arg: str | list[str] = ["-c", "--config"],
        **kwargs
    ):
        """Initialize ArgumentParserWithConfig object.

        Args:
            *args: Additional arguments for initializing 
                :class:`argparse.ArgumentParser`.
            config_arg: The name (or the list of names) of the 
                command-line argument that specifies the path to the 
                JSON config file. Defaults to ["-c", "--config"].
            **kwargs: Additional keyword arguments for initializing 
                :class:`argparse.ArgumentParser`.
        """
        super().__init__(*args, **kwargs)
        self.config_arg = config_arg

        if isinstance(self.config_arg, str):
            # Convert string to list of strings
            self.config_arg = [self.config_arg]

        self.add_argument(
        *config_arg, type=str, 
        help=f"Path to JSON file with arguments. If other arguments are "
             f"further specified via command line, they will overwrite the "
             f"ones with the same name in the JSON file.")

    def parse_args(
        self, 
        args: Any | None = None, 
        namespace: argparse.Namespace | None = None
    ) -> argparse.Namespace:
        """Parse arguments and load default values from the config file.

        This method parses the command-line arguments and loads default 
        values from the config file specified by the ``config_arg`` 
        parameter. The default values are only loaded once.

        Args:
            args: The sequence of arguments to parse. If None, then the 
                sequence will be retrieved by reading the command-line 
                arguments. Defaults to None.
            namespace: The namespace object to extend with parsed 
                arguments. If None, then a new object will be created. 
                Defaults to None.

        Returns:
            A namespace object containing the parsed arguments.
        """
        # Either load the sequence as a list or read command-line
        args = sys.argv[1:] if args is None else list(args)
        
        if len(cfg := (set(self.config_arg)) & set(args)) > 0:
            # Pop the config key and the value in the arg list
            args.pop(index := args.index(next(iter(cfg))))
            config_path = args.pop(index)

            with open(config_path) as f:
                # Load new default val dict
                new_defaults = json.load(f)

            for key, val in new_defaults.items():
                for action in self._actions:
                    if key == action.dest \
                       and action.default is not argparse.SUPPRESS:
                        # Update default val
                        action.default = val
                        break

        for action in self._actions:
            if set(action.option_strings) == set(self.config_arg):
                # Remove config parse action
                self._remove_action(action)
                break

        # Parse the remaining arguments (no config)
        args = super().parse_args(args, namespace)

        return args

def parse_args() -> dict[str, Any]:
    """Parses command-line arguments.

    Defines the possible command-line arguments that must match the 
    acceptable arguments by :class:`.Cropper`. It parses those arguments 
    and converts them to a dictionary.

    Raises:
        ValueError: If ``input_dir`` is not specified.

    Returns:
        A dictionary where keys represent argument names and values 
        represent those argument values.
    """
    # Arg parser that can load default values
    parser = ArgumentParserWithConfig()

    # TODO: check list args, check transparent, check attr_groups
    parser.add_argument(
        "-i", "--input_dir", type=str, 
        help="Path to input directory with image files.")
    parser.add_argument(
        "-o", "--output-dir", type=str, 
        help=f"Path to output directory to save the extracted face images. If "
             f"not specified, the same path is used as for input_dir, except "
             f"'_faces' suffix is added the name.")
    parser.add_argument(
        "-s", "--output-size", type=int, nargs='+', default=[256, 256], 
        help=f"The output size (width, height) of cropped image faces. If "
             f"provided as a single number, the same value is used for both "
             f"width and height. Defaults to [256, 256].")
    parser.add_argument(
        "-f", "--output-format", type=str, 
        help=f"The output format of the saved face images, e.g., 'jpg', 'png'."
             f" If not specified, the same format as the image from which the "
             f"face is extracted will be used.")
    parser.add_argument(
        "-r", "--resize-size", type=int, nargs='+', default=[1024, 1024], 
        help=f"The interim size (width, height) each image should be resized "
             f"to before processing them. If provided as a single number, the "
             f"same value is used for both width and height. Defaults to "
             f"[1024, 1024].")
    parser.add_argument(
        "-ff", "--face-factor", type=float, default=0.65, 
        help=f"The fraction of the face area relative to the output image. "
             f"Defaults to 0.65.")
    parser.add_argument(
        "-st", "--strategy", type=str, default="largest",
        choices=["all", "best", "largest"],
        help=f"The strategy to use to extract faces from each image. Defaults "
             f"to 'largest'.")
    parser.add_argument(
        "-p", "--padding", type=str, default="reflect",
        choices=["constant", "replicate", "reflect", "wrap", "reflect_101"], 
        help=f"The padding type (border mode) to apply when cropping out faces"
             f" near edges. Defaults to 'reflect'.")
    parser.add_argument(
        "-a", "--allow-skew", action="store_true", 
        help=f"Whether to allow skewing the faces to better match the the "
             f"standard (average) face landmarks.")
    parser.add_argument(
        "-l", "--landmarks", type=str, 
        help=f"Path to landmarks file if landmarks are already known and "
             f"prediction is not needed. Common file types are json "
             f"(\"image.jpg\": [x1, y1, ...]), csv (image.jpg,x1,y1,...; "
             f"first line is header), txt and other (image.jpg x1 y2).")
    parser.add_argument(
        "-ag", "--attr-groups", type=json.loads,
        help=f"Attribute groups dictionary that specifies how to group the "
             f"output face images according to some common attributes. Should "
             f"be a JSON-parsable string dictionary of type "
             f"dict[str, list[int]], e.g., '{{\"glasses\": [6]}}'.")
    parser.add_argument(
        "-mg", "--mask-groups", type=json.loads, 
        help=f"Mask groups dictionary that specifies how to group the output "
             f"face images according to some face attributes that make up a "
             f"segmentation mask. Should be a JSON-parsable string dictionary "
             f"of type dict[str, list[int]], e.g., '{{\"eyes\": [4, 5]}}'.")
    parser.add_argument(
        "-dt", "--det-threshold", type=float, default=0.6, 
        help=f"The visual threshold, i.e., minimum confidence score, for a "
             f"detected face to be considered an actual face. If a negative "
             f"value is provided, e.g., -1, landmark prediction is not "
             f"performed. Defaults to 0.6.")
    parser.add_argument(
        "-et", "--enh-threshold", type=float, default=0.001, 
        help=f"Quality enhancement threshold that tells when the image quality"
             f"should be enhanced. It is the minimum average face factor in "
             f"the input image. Defaults to 0.001.")
    parser.add_argument(
        "-b", "--batch-size", type=int, default=8, 
        help=f"The batch size. It is the maximum number of images that can be "
             f"processed by every processor at a single time-step. Defaults "
             f"to 8.")
    parser.add_argument(
        "-n", "--num-processes", type=int, default=1, 
        help=f"The number of processes to launch to perform image processing. "
             f"If landmarks are provided and no quality enhancement or "
             f"attribute grouping is done, feel free to set this to the "
             f"number of CPUs your machine has. Defaults to 1.")
    parser.add_argument(
        "-d", "--device", type=str, default="cpu", 
        help=f"The device on which to perform the predictions, i.e., landmark "
             f"detection, quality enhancement and face  parsing. Defaults to "
             f"'cpu'.")

    # Parse arguments and convert to dict
    kwargs = vars(parser.parse_args())
    
    if kwargs["input_dir"] is None:
        raise ValueError("Input directory must be specified.")
    
    if kwargs["det_threshold"] is not None and kwargs["det_threshold"] < 0:
        # If negative, set it to None
        kwargs["det_threshold"] = None
    
    if kwargs["enh_threshold"] is not None and kwargs["enh_threshold"] < 0:
        # If negative, set it to None
        kwargs["enh_threshold"] = None
    
    return kwargs
    

def main():
    """Processes an input dir of images

    Creates a cropper object based on the provided command-line 
    arguments and processes a specified directory of images. There are 
    3 main features the cropper can do (either together or separately): 
    align and center-crop face images, enhance quality, group by 
    attributes. For more details, see :class:`.Cropper`.
    """
    # Parse arguments
    kwargs = parse_args()

    # Pop the input and output dirs
    input_dir = kwargs.pop("input_dir")
    output_dir = kwargs.pop("output_dir")

    # Init cropper and process
    cropper = Cropper(**kwargs)
    cropper.process_dir(input_dir, output_dir)

if __name__ == "__main__":
    main()