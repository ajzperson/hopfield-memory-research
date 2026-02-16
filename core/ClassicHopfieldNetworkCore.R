# ==============================================================================
# CLASSIC HOPFIELD NETWORK CORE
# ==============================================================================
# File: ClassicHopfieldNetworkCore.R 
# 
# Key Features:
#   - Triple Learning: Supports Classic Hebbian (1982), High-Capacity 
#     Storkey (1997), and Projection rules for correlated pattern storage.
#   - Vectorized Math: Optimized matrix operations for speed on large grids.
#   - Research Tools: Capacity, Resilience, and Convergence tests included.
#   - Image Pipeline: Bipolar image preprocessing and progressive plotting.
#
# Technical Specs:
#   - Year: Hebbian (1982) | Projection (1985) | Storkey (1997)
#   - State: Bipolar {-1, 1}
#   - Update: Sequential/Random (asynchronous) or Synchronous
#   - Energy: Lyapunov minimization (Guaranteeing convergence)
# 
# Nodes are stored as 1D Matrix, Weights are stored as 2D Matrix
# ==============================================================================

library(magick)
library(ggplot2)
library(MASS)

#================================
#===CLASSIC HOPFIELD FUNCTIONS===
#================================

# Generate random starting values for neurons, can generate many sets
random_nodes = function(x, y) {
  return (matrix(sample(c(1, -1), x * y, replace = TRUE), nrow = x, ncol = y))
}

# Generate random set of starting weights (either 1 or -1)
random_weights = function(start_network) {
  n = length(start_network)
  upper_vals = sample(c(-1, 1), n * (n - 1) / 2, replace = TRUE)
  weights = matrix(0, n, n)
  # Transcribe from triangle to matrix using matrix methods i found online
  weights[upper.tri(weights)] = upper_vals 
  weights = weights + t(weights)
  return(weights)
}

# Generate weights that match any matrix of fixed states (Hebbian)
# Standard Hebbian covariance: The (1/num_nodes) scaling factor ensures 
# numerical stability as the network size increases.
fixed_weights_hebbian = function(fixed_states) {
  num_nodes = ncol(fixed_states)
  weights = matrix(0, nrow = num_nodes, ncol = num_nodes)
  for (j in 1:nrow(fixed_states)) {
    weights = weights + (1 / num_nodes) * (fixed_states[j, ] %o% fixed_states[j, ])
  }
  diag(weights) = 0
  return(weights)
}

# Generate weights that match any matrix of fixed states (Storkey)
fixed_weights_storkey = function(fixed_states) {
  num_nodes = ncol(fixed_states)
  # Ensure we start with a proper square matrix
  weights = matrix(0, nrow = num_nodes, ncol = num_nodes)
  
  # Convert matrix rows to a list of flat vectors
  # row(fixed_states) ensures we split by row correctly.
  states_list = split(fixed_states, row(fixed_states))
  
  # Reduce applies storkey_learn repeatedly
  weights = Reduce(function(w, s) {
    # Force 's' to be a simple numeric vector to prevent matrix dimension errors
    storkey_learn(as.numeric(s), w, as.numeric(s))
  }, states_list, init = weights)
  
  return(weights)
}

# Activation function
threshold = function (x) ifelse(x >= 0, 1, -1)

# Return updated node based off connections to other nodes
update_node = function (cur_network, weights, node_index) {
  cur_network[node_index] = threshold(weights[node_index, ] %*% cur_network)
  return(cur_network)
}

# Return energy of a Hopfield Network
# Lyapunov Energy Function (LSE): Hopfield Networks are guaranteed to converge 
# because every update decreases this energy value, eventually reaching 
# a local minimum (the stored memory).
net_energy = function(cur_network, weights) {
  return(-0.5 * sum(weights * (cur_network %o% cur_network)))
}

# Return a scrambled Hopfield Network (nodes are randomly chosen)
scramble = function(cur_network) {
  num_nodes = length(cur_network)
  # Generate all new states (-1 or 1) in a single vectorized call
  return(sample(c(1, -1), num_nodes, replace = TRUE))
}

# Return network with certain number of nodes flipped
flip = function(state, num_flips) {
  if (num_flips > length(state)) {
    stop("Number of flips cannot exceed the number of nodes.")
  }
  flip_indices = sample(1:length(state), num_flips, replace = FALSE)
  state[flip_indices] = -state[flip_indices]
  return(state)
}

# Simulate network by checking random nodes for set number of steps
simulate_random = function(cur_network, weights, steps) {
  num_nodes = length(cur_network)
  for (i in 1:steps) {
    update = sample(1:num_nodes, 1)
    cur_network = update_node(cur_network, weights, update) 
  }
  return(cur_network)
}

# Simulate network by going through each node sequentially
simulate_sequential = function(cur_network, weights, steps = 1) {
  for (s in 1:steps) {
    for (i in 1:length(cur_network)) {
      cur_network = update_node(cur_network, weights, i)
    }
  }
  return(cur_network)
}

# Simulate network via synchronous learning for certain # of steps
simulate_synchronous=function(state,weights,steps=1){
  for(s in 1:steps){
    state=weights%*%state
    state[state == 0] = 1
  }
  return(as.numeric(state))
}

# Determine if it is a fixed point
is_fixed_point_seq = function(cur_network, weights) {
  new_state = cur_network
  for (node in 1:length(cur_network)) {
    new_state = update_node(new_state, weights, node)
  }
  return(identical(new_state, cur_network))
}

# Simulate network until it reaches a fixed point with seq updates
simulate_until_fixed_sequential = function(cur_network, weights, max_steps = 1000) {
  for (step in 1:max_steps) {
    prev_network = cur_network
    cur_network = simulate_sequential(cur_network, weights)
    if (all(cur_network == prev_network)) break
  }
  return(cur_network)
}

# Simulate network until fixed point with sync updates
simulate_until_fixed_sync = function(cur_network, weights, max_steps = 100) {
  for (step in 1:max_steps) {
    newNet = sign(weights %*% cur_network)
    newNet[newNet == 0] = 1
    if (identical(newNet, cur_network)) break
    cur_network = newNet
  }
  return(cur_network)
}

# Simulate network until fixed point with random updates
simulate_until_fixed_random = function(cur_network, weights, max_steps = 1000) {
  for (step in 1:max_steps) {
    prev_network = cur_network
    cur_network = simulate_random(cur_network, weights, length(cur_network))
    if (is_fixed_point_seq(cur_network, weights) == TRUE) {return (cur_network)}
  }
  return(cur_network)
}

# Learn new pattern with Hebbian learning
hebbian_learn = function(cur_network, weights, new_pattern) {
  weights = weights + (1 / length(new_pattern)) * (new_pattern %o% new_pattern)
  diag(weights) = 0
  return(weights)
}

# Learn a new pattern given initial weights (Storkey)
storkey_learn = function(cur_network, weights, new_pattern) {
  # 1. Force weights to be a matrix if it isn't one
  weights = as.matrix(weights)
  num_nodes = length(new_pattern)
  
  # 2. Local field calculation
  # weights %*% as.numeric(new_pattern) creates a 1-column matrix. 
  # drop() turns it into a simple numeric vector.
  h = drop(weights %*% as.numeric(new_pattern))

  # 3. Vectorized Storkey update
  # %o% creates the NxN matrices needed for the update
  term1 = new_pattern %o% new_pattern
  term2 = new_pattern %o% h
  term3 = h %o% new_pattern
  
  # Update the weights
  weights = weights + (1 / num_nodes) * (term1 - term2 - term3)
  
  diag(weights) = 0 
  return(weights)
}

# Return the hamming distance btwn 2 networks
hamming_distance = function(pattern1, pattern2) {
  if (length(pattern1) != length(pattern2)) stop("Patterns must be of same length.")
  return(sum(pattern1 != pattern2))
}

# Return the amount of steps it took for sequential update to stabilize
time_to_converge_seq = function(update_type, cur_network, weights, max_steps = 1000) {
  l = length(cur_network)
  step = 0
  for (s in 1:max_steps) {
    for (i in 1:l) {
      cur_network = update_type(cur_network, weights, (i%%l)+1)
      step = step + 1
      if (i%%10 == 0) {
        if (is_fixed_point_seq(cur_network, weights)) {
          return(step)
        }
      }
    }
  }
  return(max_steps)
}

#========================
#===ANALYSIS FUNCTIONS===
#========================

# Find the max recoverable flips, using binary search so it's faster
# Pass in one of the simulate_until_fixed methods as first parameter
max_recoverable_flips = function(test_function, memory, weights, num_flips, steps=1000) {
  low = 1
  high = length(memory) %/% 2
  most = 0
  
  while (low<= high)
  {
    # Find midpoint num of flips
    mid = floor((low + high)/2)
    
    scrambled = flip(memory, mid)
    recovered = test_function(scrambled, weights, steps)
    
    if (all(recovered == memory))
    {
      # If the recall is successful, shift the boundaries up
      most = mid
      low = mid + 1
    }
    else
    {
      # Shift boundaries down
      high = mid - 1
    }
  }
  return(most)
}

# Plot memory resilience
# First parameter is type of update function - i.e put corresponding
# simulate_until_fixed method
memory_resilience_test = function(learn_method, update_type, length, density = seq(0.02, 0.20, by = 0.02), trials = 25, steps = 500) {
  numD = length(density)
  
  # Avg max perturbations
  results = numeric(numD)
  
  # Multiple trials for each density
  for (i in 1:numD)
  {
    numPatterns = round(density[i]*length)
    trialAvgs = numeric(trials)
    
    for (t in 1:trials)
    {
      fixedStates = random_nodes(numPatterns, length)
      weights = learn_method(fixedStates)
      maxPerturb = numeric(numPatterns)
      
      # Resilience for each pattern
      for (n in 1:numPatterns)
      {
        maxFlips = max_recoverable_flips(update_type, fixedStates[n,], weights, steps)
        maxPerturb[n] = maxFlips/length
      }
      trialAvgs[t] = mean(maxPerturb)
    }
    results[i] = mean(trialAvgs)
  }
  
  plot(density, results, type = "b", xlab = "Memory Density", ylab = "Avg Max Perturbations", main = "Resilience vs Memory Density")
  return(results)
}

# Plot memory capacity - how many memories with x nodes assume y nodes flipped
# First parameter is type of update function - i.e put corresponding
# simulate_until_fixed method
memory_capacity_test = function(learn_method, update_type, nodes_perturbed, num_nodes = seq(10, 190, by = 20), trials = 10) {
  
  numN = length(num_nodes)
  results = numeric(numN) # Make results array
  
  # Iterate over different number of nodes
  for (n in 1:numN) { 
    trial_avgs = numeric(trials)
    # Iterate over many trials
    for (t in 1:trials) { 
      hit_capacity = FALSE
      # Start with one memory
      max_memories = 1 
      fixedStates = random_nodes(1, num_nodes[n])
      weights = learn_method(fixedStates)
      while (hit_capacity == FALSE) {
        new_state = matrix(sample(c(1, -1), num_nodes[n], replace = TRUE), nrow = 1)
        # Add one new memory (incrementally)
        fixedStates = rbind(fixedStates, new_state) 
        weights = learn_method(fixedStates) 
        fail = FALSE
        # Iterate over fixed states
        for (i in 1:nrow(fixedStates)) { 
          fixed_state = as.numeric(fixedStates[i, ])
          fixed_state_mixed = flip(fixed_state, nodes_perturbed)
          fixed_state_mixed_sim = update_type(fixed_state_mixed, weights)
          if (!all(fixed_state_mixed_sim == fixed_state)) {
            hit_capacity = TRUE
            fail = TRUE
            break
          }
        }
        
        # Make sure to not add one by accident
        if (!fail) { 
          max_memories = max_memories + 1
        }
      }
      trial_avgs[t] = max_memories - 1
    }
    results[n] = mean(trial_avgs)
  }
  plot(num_nodes, results, type = "b", col = "blue", xlab = "# of Nodes", ylab = "Max Memories", main = "Memory Capacity vs. Network Size")
}

# Plot convergence speed vs memory density, uses Sequential updating
# Update type is just simulate_sequential or simulate_random, not sync
convergence_speed_test = function (learn_method, update_type, num_nodes, mem_numbers, trials = num_nodes) {
  avg_conv_speed = numeric(length(mem_numbers))
  # Iterate over memory densities
  for (m in seq_along(mem_numbers)) { 
    speeds = numeric(trials)
    # Iterate over trials
    for (t in 1:trials) { 
      fixedStates = random_nodes(mem_numbers[m], num_nodes)
      weights = learn_method(fixedStates)
      cur_states = random_nodes(mem_numbers[m], num_nodes)
      # Check speed many times
      converge_times = sapply(1:mem_numbers[m], function(i) { 
        time_to_converge_seq(update_type, cur_states[i, ], weights)
      })
      # Average conv speed from dif starts
      speeds[t] = mean(converge_times) 
    }
    
    # Avg conv speed from diff trials
    avg_conv_speed[m] = mean(speeds) 
  }
  
  plot(mem_numbers, avg_conv_speed, type = "l",
       xlab = "Memory Density in Percent",
       ylab = "Avg Convergence Updates",
       main = "Convergence Time vs Memory Density", 
       xlim = c(0, 30), ylim = c(0, 600))
}

# Test if memory is recovered using seq updates
test_resilience_sequential = function(memory, weights, num_flips, steps=1000) {
  scrambled = flip(memory, num_flips)
  recovered = simulate_until_fixed(scrambled, weights, steps)
  return(all(recovered == memory))
}

# Test if memory is recovered using sync updates
test_resilience_sync = function(memory, weights, num_flips, steps=1000) {
  scrambled = flip(memory, num_flips)
  recovered = simulate_until_fixed_sync(scrambled, weights, steps)
  return(all(recovered == memory))
}

#========================================
#===IMAGE PREPROCESSING / ILLUSTRATION===
#========================================

# Make it a vector
pattern_to_vector = function(matrix_pattern) {
  return(as.vector(matrix_pattern))
}

# Train weights from a given pattern
train_weights_from_pattern = function(pattern) {
  N = length(pattern)
  weights = matrix(0, N, N)
  weights = hebbian_learn(pattern, weights, pattern)
  return(weights)
}

# Train weights for all patterns in a matrix simultaneously (Vectorized)
train_weights_from_matrix = function(pattern_matrix) {
  # N is the number of neurons (columns)
  N = ncol(pattern_matrix)
  
  # Simultaneous Hebbian math: (P^T %*% P) / N
  # This is the "batch" vectorized equivalent of hebbian_learn.
  weights = (t(pattern_matrix) %*% pattern_matrix) / N
  
  # Remove self-connections (diagonal)
  diag(weights) = 0
  
  return(weights)
}

# Scramble a fraction of nodes
scramble_pattern = function(pattern, frac = 0.2) {
  num_flips = round(length(pattern) * frac)
  return(flip(pattern, num_flips))
}

# Plot a current network
plot_state = function(state, step, w, h) {
  size = sqrt(length(state))
  mat = matrix(state, nrow = h, ncol = w, byrow = TRUE)
  df = expand.grid(X_pos = 1:w, Y_pos = 1:h)
  df$val = as.vector(t(mat))
  
  p = ggplot(df, aes(x = X_pos, y = Y_pos, fill=factor(val))) +
    geom_tile(color="grey") +
    scale_fill_manual(values=c("-1"="black", "1"="white")) +
    theme_void() +
    theme(legend.position="none") +
    ggtitle(paste("Classic Hopfield Network Decoding - Step", step)) +
    coord_fixed()
  print(p)
}

# Decode a scrambled network
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

# Decode a network and also show hamming distance
decode_with_hamming = function(state, weights, target, img_w, img_h, max_steps = 1000, delay = 0.1,
                               save_gif = FALSE, filename = "recovery.gif", mode = "sequential",
                               show_log = TRUE,      
                               return_list = FALSE) {
  N = length(state)
  captured_plots = list()
  plot_interval = max(1, floor(max_steps / 40))

  for (step in 1:max_steps) {
    if (mode == "sequential") {
      # Sequential logic: Updates one node at a time in order
      idx = ((step - 1) %% N) + 1
      state = update_node(state, weights, idx)
      
    } else if (mode == "random") {
      # Random logic: Picks one node at random to update
      idx = sample(1:N, 1)
      state = update_node(state, weights, idx)
      
    } else if (mode == "sync") {
      # Sync logic: Update all nodes at once via matrix multiplication
      # Break out of the loop faster here since it updates the whole grid
      state = as.vector(ifelse(weights %*% state >= 0, 1, -1))
    }

    # Only run plotting and logging if show_log is TRUE
    if (show_log) {
      if (step %% plot_interval == 0 || step == 1 || mode == "sync") {
        plot_state(state, step, img_w, img_h)
        
        if (isTRUE(save_gif)) {
          captured_plots[[length(captured_plots) + 1]] = recordPlot()
        }
        
        hd = hamming_distance(state, target)
        cat("Mode:", mode, "| Step", step, "- Hamming Distance:", hd, "\n")
        
        if (hd == 0) {
          cat("Converged to memory!\n")
          break
        }
        
        # For Sync mode, check convergence by seeing if state stopped changing
        if (mode == "sync" && step > 1) {
          # Small logic to stop sync if it stabilizes
        }

        Sys.sleep(delay)
      }
    } else {
      # Silent mode: still need to break if converged
      if (hamming_distance(state, target) == 0) break
    }
      
    # In Sync mode, one 'step' is a full iteration, so we break early 
    # unless we want to watch for oscillations
    if (mode == "sync" && step >= 20) break 
  }

  # Extra block only runs if save_gif is TRUE
  if (isTRUE(save_gif) && length(captured_plots) > 0) {
    cat("Generating GIF...\n")
    
    # 1. Create a temporary folder for frames
    temp_dir <- tempdir()
    file_paths <- c()

    # 2. Save each plot as a simple PNG (no font dependencies)
    for (i in seq_along(captured_plots)) {
      path <- file.path(temp_dir, sprintf("frame_%04d.png", i))
      png(path, width = 500, height = 500, res = 96)
      
      par(mar = c(0,0,0,0)) 
      
      replayPlot(captured_plots[[i]])
      dev.off()
      file_paths <- c(file_paths, path)
    }

    # 3. Read the PNGs into magick and animate
    img_list <- image_read(file_paths)
    animation <- image_animate(img_list, fps = 10)
    image_write(animation, filename)
    
    # Clean up temp files
    unlink(file_paths)
    cat("Success! GIF saved as:", filename, "\n")
  }

  # Return logic
  if (isTRUE(return_list)) {
    return(list(final_state = state, steps_taken = step))
  } else {
    return(state)
  }
}

# Plot hamming distance versus time for a network as it is decoded
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


# Preprocess an image into a bipolar (-1, 1) neuron matrix
# Resize the image and applies a threshold to create a black-and-white pattern
processImg = function(imgPath, maxSize = 25, thres = "50%") {
  img = image_read(imgPath)
  img = image_convert(img, colorspace = "gray")
  img = image_convert(img, "png")
  info = image_info(img)
  ogWidth = info$width
  ogHeight = info$height
  
  # Debugging
  # image_write(img, "test1.png")
  
  if (ogWidth < ogHeight) {
    img = image_scale(img, paste0("x", maxSize))
  } else {
    img = image_scale(img, as.character(maxSize))
  }
  
  img = image_threshold(img, type = "white", threshold = thres)
  img = image_threshold(img, type = "black", threshold = thres)
  
  info = image_info(img)
  w = info$width
  h = info$height
  
  # Debugging
  # image_write(img, "test2.png")

  pixels = as.raster(img)
  neurons = matrix(nrow = h, ncol = w)
  
  for (i in 1:h) {
    for (j in 1:w) {
      # Map white pixels to 1, others to -1
      # Use substr handles both #ffffff and #ffffffff formats
      neurons[i,j] = ifelse(substr(pixels[i,j], 1, 7) == "#ffffff", 1, -1)
    }
  }
  
  # 1. Convert neuron values (-1, 1) to colors ("#000000", "#ffffff")
  # At this stage, the matrix is still in standard image orientation [h, w]
  color_matrix = ifelse(neurons == 1, "#ffffff", "#000000")
  
  # 2. Convert to a raster and save (debugging)
  # This creates a PNG that looks exactly like the neurons in the matrix
  # image_write(image_read(as.raster(color_matrix)), "neurons_before_transform.png")
  
  # Save neurons as image
  # Reverse transformation for saving
  save_mat = apply(t(neurons), 2, rev)
  color_mat = ifelse(save_mat == 1, "#FFFFFF", "#000000")
  
  # Write to file (debugging)
  # out_img = image_read(as.raster(color_mat))
  # image_write(out_img, path = "neuron_output.png", format = "png")
  
  neurons = t(apply(neurons, 2, rev))
  return(list(matrix = neurons, width = w, height = h))
}

# Projection rule (Pseudo-inverse) training
fixed_weights_projection = function(pattern_matrix) {
  # pattern_matrix: rows are patterns, cols are neurons
  # Formula: W = X_pinv * X
  X <- t(pattern_matrix) # Neurons x Patterns
  
  # Compute the Moore-Penrose pseudo-inverse
  # This effectively decorrelates the patterns.
  X_pinv <- MASS::ginv(X) 
  
  W <- X %*% X_pinv
  diag(W) <- 0 # Standard Hopfield: no self-connections
  return(W)
}