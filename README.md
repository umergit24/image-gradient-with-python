# Image Gradient Processing with Gaussian Blur

This project implements image gradient processing with Gaussian blurring using OpenCV and NumPy. The program reads an input image, applies grayscale conversion, Gaussian blurring, and computes image gradients in the x and y directions. It then calculates the gradient magnitude and phase, which are visualized and saved as output images.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Outputs](#outputs)
- [Contributing](#contributing)
- [License](#license)

## Overview

Image gradient processing is a key technique in computer vision and image processing, often used for edge detection and feature extraction. The gradient magnitude represents the strength of the edges, while the phase (or direction) indicates the orientation of the edges. In this project, the Gaussian blur is applied before calculating gradients to reduce noise and produce smoother edge results.

## Features

- Grayscale conversion of the image.
- Gaussian blur for noise reduction.
- Calculation of image gradients in both the x and y directions.
- Gradient magnitude and phase computation.
- Visualization and saving of the results.

## Installation

To run this project, ensure you have the following dependencies installed:

```bash
pip install opencv-python-headless numpy
```

## Usage
To process an image, you need to provide an image file as input. Modify the following line in the process_image function to point to your image file:

python
Copy code
process_image('path_to_your_image.png')
Run the script:

```bash
python image_gradient.py
```
## Outputs
The script generates the following outputs:

Grayscale Image: Conversion of the input image to grayscale.
Blurred Image: Image after applying Gaussian blur to reduce noise.
Gradient Magnitude: Shows the strength of edges in the image.
Phase Image: Represents the direction of the edges.
These images are saved in the images/ directory.

## Example
If you provide an image, such as imageb.png,
[original image](https://github.com/umergit24/image-gradient-with-python/blob/main/images/image.png)
the program will output several images that highlight different aspects of the gradient processing:

gray.png - Grayscale version of the image.
blurred.png - Blurred version using Gaussian filtering.
magnitude.png - Edge strength.
phase.png - Edge direction.

## Contributing
Feel free to fork the repository, make your changes, and submit a pull request. Contributions are welcome!

## License
This project is open-source and available under the MIT License.



This file gives an overview of your project, including its purpose, installation steps, usage instructions, and expected outputs. Feel free to adjust details like the image path or additional features if necessary. &#8203;:contentReference[oaicite:0]{index=0}&#8203;
