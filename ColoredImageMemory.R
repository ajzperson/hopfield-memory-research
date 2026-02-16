# NOT FINISHED
#
#

source("continuoushopfield.R")

library(magick)
library(grid)

# so basically image_data returns hex code so this will convert to integer form
hex_to_int = function(hex_matrix) {
  apply(hex_matrix, c(1,2), function(x) strtoi(x, 16L))
}

# use "c9groupphoto.jpg"
image_to_vectors = function(jpeg) {
  img = image_read(jpeg)
  img = image_convert(img, format = "bmp")
  resized_img = image_resize(img, "10%")
  
  # gets rgb values for pixel in the image
  img_data = image_data(resized_img, channels = "rgb")
  
  height = dim(img_data)[2]
  width = dim(img_data)[3]
  
  # scaling values to be from 0 to 1
  red = as.vector(hex_to_int(img_data[1,,])) / 255
  green = as.vector(hex_to_int(img_data[2,,])) / 255
  blue = as.vector(hex_to_int(img_data[3,,])) / 255
  
  return(c(red, green, blue))
}

# img = image_read("c9groupphoto.jpg")
# img = image_convert(img, format = "bmp")
# resized_img = image_resize(img, "10%")
# 
# # gets rgb values for pixel in the image
# img_data = image_data(resized_img, channels = "rgb")
# 
# height = dim(img_data)[2]
# width = dim(img_data)[3]
# 
# # scaling values to be from 0 to 1
# red = as.vector(hex_to_int(img_data[1,,])) / 255
# green = as.vector(hex_to_int(img_data[2,,])) / 255
# blue = as.vector(hex_to_int(img_data[3,,])) / 255
# 
# vector_to_colored_image = function(vr, vg, vb) {
#   
#   matred = matrix(vr, nrow = height, ncol = width)
#   matgreen = matrix(vg, nrow = height, ncol = width)
#   matblue = matrix(vb, nrow = height, ncol = width)
#   
#   image_array = array(0, dim = c(height, width, 3))
#   image_array[,,1] = matred
#   image_array[,,2] = matgreen
#   image_array[,,3] = matblue
#   
#   grid.raster(image_array)
# }

# rred = (random_nodes(1, height*width) + 1) / 2
# rgreen = (random_nodes(1, height*width) + 1) / 2
# rblue = (random_nodes(1, height*width) + 1) / 2

fixed_state_red = red
fixed_state_green = green
fixed_state_blue = blue

simulate_until_fixed_continuous_animated = function(cur_network, fixed_states, max_steps = 100, stepSize = 0.1) {
  
  for (step in 1:max_steps) {
    newState = attention_update(cur_network, fixed_states)
    update = stepSize * (newState - cur_network)
    
    if (max(abs(update)) < 0.000001)
    {
      break
    }
    cur_network = cur_network + update
  }
  return(cur_network)
}


vector_to_colored_image(rred, rgreen, rblue)





