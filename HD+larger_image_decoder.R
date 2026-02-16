# Hopfield Network for Custom Images
library(ggplot2)
library(magick)

#===============================
#== CLASSIC HOPFIELD FUNCTIONS =
#===============================
random_nodes = function(x, y) {
  network_final = matrix(1:(x*y), nrow = x, ncol = y) * 0
  for(j in 1:x) {
    network = numeric(y)
    for(i in 1:y) {
      change = runif(1)
      network[i] = ifelse(change >= 0.5, 1, -1)
    }
    network_final[j,] = network
  }
  return(network_final)
}

random_weights = function (start_network) {
  num_nodes = length(start_network)
  weights = matrix(nrow = num_nodes, ncol = num_nodes)
  for (n in 1:num_nodes) {
    for (m in 1:num_nodes) {
      weights[n, m] = ifelse(n == m, 0, ifelse(runif(1) >= 0.5, 1, -1))
    }
  }
  return(weights)
}

fixed_weights = function(fixed_states) {
  num_nodes = length(fixed_states[1,])
  weights_total = matrix(0, nrow = num_nodes, ncol = num_nodes)
  for (j in 1:nrow(fixed_states)) {
    fixed_state = fixed_states[j,]
    weights = fixed_state %*% t(fixed_state)
    diag(weights) <- 0
    weights_total = weights_total + weights
  }
  return(weights_total)
}

threshold = function (x) ifelse(x >= 0, 1, -1)

update_node = function (cur_network, weights, node_index) {
  cur_network[node_index] = threshold(sum(cur_network * weights[node_index,]))
  return(cur_network)
}

net_energy = function(cur_network, weights) {
  total_energy = -sum(weights * (cur_network %*% t(cur_network)))
  return(total_energy)
}

scramble = function (cur_network) {
  return(ifelse(runif(length(cur_network)) > 0.5, 1, -1))
}

flip = function (cur_network, x) {
  idx = sample(1:length(cur_network), x)
  cur_network[idx] = -cur_network[idx]
  return(cur_network)
}

simulate_random = function(cur_network, weights, steps) {
  for (i in 1:steps) {
    update = sample(1:length(cur_network), 1)
    cur_network = update_node(cur_network, weights, update)
  }
  return(cur_network)
}

simulate_sequential = function(cur_network, weights, steps = 1) {
  for (s in 1:steps) {
    for (i in 1:length(cur_network)) {
      cur_network = update_node(cur_network, weights, i)
    }
  }
  return(cur_network)
}

simulate_until_fixed = function(cur_network, weights, max_steps = 1000) {
  for (step in 1:max_steps) {
    prev_network = cur_network
    cur_network = simulate_sequential(cur_network, weights)
    if (all(cur_network == prev_network)) break
  }
  return(cur_network)
}

learn = function(cur_network, weights, pattern) {
  newweight = outer(pattern, pattern)
  diag(newweight) <- 0
  weights = weights + newweight
  return(weights)
}

test_resilience = function(memory, weights, num_flips, steps=1000) {
  scrambled = flip(memory, num_flips)
  recovered = simulate_until_fixed(scrambled, weights, steps)
  return(all(recovered == memory))
}

#====================
#== ILLUSTRATIONS ===
#====================
pattern_to_vector = function(matrix_pattern) {
  return(as.vector(matrix_pattern))
}

train_weights_from_pattern = function(pattern) {
  weights = matrix(0, length(pattern), length(pattern))
  weights = learn(pattern, weights, pattern)
  return(weights)
}

scramble_pattern = function(pattern, frac = 0.2) {
  num_flips = round(length(pattern) * frac)
  return(flip(pattern, num_flips))
}

plot_state = function(state, step) {
  size = sqrt(length(state))
  mat = matrix(state, nrow=size, byrow=FALSE)
  df = expand.grid(x=1:size, y=1:size)
  df$val = as.vector(mat)
  
  p = ggplot(df, aes(x=x, y=y, fill=factor(val))) +
    geom_tile(color="grey") +
    scale_fill_manual(values=c("-1"="black", "1"="white")) +
    theme_void() +
    theme(legend.position="none") +
    ggtitle(paste("Hopfield Network Decoding - Step", step)) +
    coord_fixed()
  print(p)
}

decode_slowly = function(state, weights, steps = 1000, delay = 0.1) {
  N = length(state)
  for (step in 1:steps) {
    idx = ((step - 1) %% N) + 1
    state = update_node(state, weights, idx)
    plot_state(state, step)
    Sys.sleep(delay)
  }
  return(state)
}

hamming_distance = function(pattern1, pattern2) {
  if (length(pattern1) != length(pattern2)) stop("Patterns must be of same length.")
  return(sum(pattern1 != pattern2))
}

decode_with_hamming = function(state, weights, target, max_steps = 1000, delay = 0.1) {
  N = length(state)
  for (step in 1:max_steps) {
    idx = ((step - 1) %% N) + 1
    state = update_node(state, weights, idx)
    plot_state(state, step)
    hd = hamming_distance(state, target)
    cat("Step", step, "- Hamming Distance:", hd, "\n")
    if (hd == 0) {
      cat("Converged to memory!\n")
      break
    }
    Sys.sleep(delay)
  }
  return(state)
}

decode_track_hamming = function(state, weights, target, max_steps = 1000) {
  N = length(state)
  hamming_vals = numeric(max_steps)
  for (step in 1:max_steps) {
    idx = ((step - 1) %% N) + 1
    state = update_node(state, weights, idx)
    hamming_vals[step] = hamming_distance(state, target)
    if (hamming_vals[step] == 0) break
  }
  plot(1:step, hamming_vals[1:step], type = "l", col = "blue",
       xlab = "Step", ylab = "Hamming Distance", main = "Convergence Tracking")
  return(state)
}

#========================
#== IMAGE PREPROCESSING =
#========================
processImg = function(imgPath, maxSize = 25, thres = "50%") {
  img = image_read(imgPath)
  img = image_convert(img, colorspace = "gray")
  img = image_convert(img, "png")
  info = image_info(img)
  ogWidth = info$width
  ogHeight = info$height
  
  if (ogWidth < ogHeight) {
    img = image_scale(img, paste0("x", maxSize))
  } else {
    img = image_scale(img, as.character(maxSize))
  }
  
  img = image_threshold(img, type = "white", threshold = thres)
  img = image_threshold(img, type = "black", threshold = thres)
  
  info = image_info(img)
  width = info$width
  height = info$height
  
  pixels = as.raster(img)
  neurons = matrix(nrow = height, ncol = width)
  
  for (i in 1:height) {
    for (j in 1:width) {
      neurons[i,j] = ifelse(pixels[i,j] == "#ffffffff", 1, -1)
    }
  }
  
  neurons = t(apply(neurons, 2, rev))
  return(neurons)
}

#============================
#== USAGE EXAMPLE (CUSTOM) ==
#============================
# Load your image
img_matrix = processImg("downloads/images.png", maxSize = 25)

# Preview
image(img_matrix, col = gray.colors(2))

# Train the network
pattern = pattern_to_vector(img_matrix)
weights = train_weights_from_pattern(pattern)

# Scramble and decode
scrambled = scramble_pattern(pattern, frac = 0.3)
final_state = decode_with_hamming(scrambled, weights, pattern, max_steps = 1000, delay = 0.06)
