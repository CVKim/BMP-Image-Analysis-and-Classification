# BMP-Image-Analysis-and-Classification

This repository contains a Python script `bmp_image_classifier.py` which is designed to analyze BMP images based on two features: the PNG size and the image gradient. After the analysis, it classifies the images using KMeans clustering.

## Functions

### `get_center_crop(img, percent=50)`

- **Purpose**: 
  - Crops the image to its center according to a given percentage.
- **Parameters**: 
  - `img`: An Image object.
  - `percent`: Desired percentage for the cropped image's size.
- **Returns**: 
  - A cropped Image object.

### `get_grid_threshold(img)`

- **Purpose**: 
  - Calculates the average gradient of the image which can be used as a feature for classification.
- **Parameters**: 
  - `img`: An Image object.
- **Returns**: 
  - Average gradient value of the image.

### `get_png_size(img)`

- **Purpose**: 
  - Computes the size of the image when saved as PNG.
- **Parameters**: 
  - `img`: An Image object.
- **Returns**: 
  - Size of the image in bytes.

### `compute_gradient(img_files, gradients, event)`

- **Purpose**: 
  - Computes the gradient value for a list of image files.
- **Parameters**: 
  - `img_files`: List of image file paths.
  - `gradients`: Empty list where the computed gradient values will be stored.
  - `event`: A threading event.
- **Returns**: 
  - None. The computed gradients are saved to the `gradients` list.

### `compute_png_size(img_files, png_sizes, event)`

- **Purpose**: 
  - Computes the PNG size for a list of image files.
- **Parameters**: 
  - `img_files`: List of image file paths.
  - `png_sizes`: Empty list where the computed PNG sizes will be stored.
  - `event`: A threading event.
- **Returns**: 
  - None. The computed sizes are saved to the `png_sizes` list.

You will be prompted to select an input directory containing the BMP images to be analyzed and classified.

## Dependencies

- PIL (from Pillow library)
- os, os.path
- glob
- numpy
- pandas
- pathlib
- tkinter
- sklearn
- tqdm
- shutil
- matplotlib
- seaborn
- io
- threading
- 
