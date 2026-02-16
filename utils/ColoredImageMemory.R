################################################################################
# File: ColoredImageMemory.R
# Purpose: Continuous-state image processing for Color Hopfield Networks.
# 
# Workflow: 
#   1. Load image via 'magick', resize, and convert to RGB numeric vectors (0-1).
#   2. Return a structured list containing R, G, B planes and image dimensions.
#   3. Reconstruct the 3D array from vectors using Transpose [t()] and 
#      Column-Major matrices to ensure the image is displayed upright.
################################################################################

library(magick)
library(grid)

# Convert hex strings to integers 0-255
hex_to_int = function(hex_matrix) {
  apply(hex_matrix, c(1,2), function(x) strtoi(x, 16L))
}

# Image processing function
image_to_vectors = function(jpeg, scale = "10%") {
  img = image_read(jpeg)
  img = image_orient(img) # Auto-rotates based on photo metadata
  img = image_convert(img, format = "bmp")
  # Resize to keep the network manageable
  resized_img = image_resize(img, scale)
  
  # Get rgb values for pixel in the image
  img_data = image_data(resized_img, channels = "rgb")
  
  # IMPORTANT: magick dimensions are [channels, width, height]
  w = dim(img_data)[2]
  h = dim(img_data)[3]
  
  # Scaling values to be from 0 to 1
  # Extract RGB channels as vectors
  # R's as.vector on a matrix slices by COLUMN
  red = as.vector(hex_to_int(img_data[1,,])) / 255
  green = as.vector(hex_to_int(img_data[2,,])) / 255
  blue = as.vector(hex_to_int(img_data[3,,])) / 255
  
  return(list(r = red, g = green, b = blue, width = w, height = h))
}

# Reconstruction function
vector_to_colored_image = function(data) {
  # This expands the drawing area to the absolute edges of the R window.
  par(mar = c(0, 0, 1, 0)) 

  # Reconstruct matrices: 
  # Use nrow = width because that's the first dimension of img_data[1,,]  
  matred = matrix(data$r, nrow = data$width, ncol = data$height, byrow = FALSE)
  matgreen = matrix(data$g, nrow = data$width, ncol = data$height, byrow = FALSE)
  matblue = matrix(data$b, nrow = data$width, ncol = data$height, byrow = FALSE)
  
  # In R display terms, this is [height, width, 3]
  image_array = array(0, dim = c(data$height, data$width, 3))

  # Use transpose t() to flip the width/height matrices 
  # so they align with the vertical/horizontal axes of the screen
  image_array[,,1] = t(matred)
  image_array[,,2] = t(matgreen)
  image_array[,,3] = t(matblue)
  
  # Draw
  rasterImage(image_array, 0, 0, 1, 1)
}





