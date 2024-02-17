# Image Recognition Script

The Image Recognition Script is part of the Scriptools collection, designed to leverage machine learning for image processing tasks. This script utilizes state-of-the-art image captioning models to generate descriptive captions for a collection of images.

## Features

* Generate descriptive captions for images using advanced AI models.
* Supports multiple models, including Salesforce BLIP and Microsoft GIT for flexible image understanding.
* Outputs a CSV file cataloging each image with its generated caption.
* Easy integration into larger workflows for automated image processing and analysis.

## Target Audience

This script is ideal for:

* Developers and researchers working on machine learning and image processing projects.
* Content creators and media professionals seeking automated captioning for images.
* Educators and students exploring AI applications in image recognition.

## Installation

To install the necessary dependencies for the Image Recognition Script:

1. Navigate to the script's folder within the Scriptools repository.
2. Open a terminal in this folder.
3. Run the command:

```bash
pip install -r requirements.txt
```

Ensure Python is installed on your system before proceeding.

## Usage

To use the Image Recognition Script, follow these steps:

1. Prepare a folder containing the images you wish to process.
2. Open a terminal and navigate to the script's folder.
3. Execute the script with the following command:

```bash
python clip-picture-search.py <path_to_image_folder> -m <model_name>
```

* `<path_to_image_folder>`: The path to the folder containing your images.
* `<model_name>`: Specify `"blip"` for Salesforce BLIP or `"git"` for Microsoft GIT model. Default is `"blip"`.

Example:

```bash
python clip-picture-search.py ./images -m blip
```

This will process all images in the `./images` folder using the Salesforce BLIP model and output a CSV file with captions.

## Contributing

Contributions to enhance the Image Recognition Script or any part of the Scriptools project are welcome. Whether it's adding support for more models, improving efficiency, or fixing bugs, your input is valuable.

## License

The Image Recognition Script is licensed under the MIT License, consistent with the broader Scriptools project.

## Author

* Nasser Al-Hilal (@nalhilal on X.com)

For more information on Scriptools and its various scripts, visit the main project page.