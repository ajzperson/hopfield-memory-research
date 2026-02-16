# Classical Hopfield Network
# Includes all necessary methods as well as illustration methods (bottom)
# Nodes are stored as 1D Matrix, Weights are stored as 2D Matrix 
# IMPORTANT: Open this file in RSTudio, then do work in separate file
# Call source("thisfilespath.R") and it should carry over function definitions


#================================
#===CLASSIC HOPFIELD FUNCTIONS===
#================================


# Generate random starting values for neurons - can generate many sets
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

# Generate weights that match any matrix of fixed states (hebbian)
fixed_weights_hebbian = function(fixed_states) {
  num_nodes = ncol(fixed_states)
  weights = matrix(0, nrow = num_nodes, ncol = num_nodes)
  for (j in 1:nrow(fixed_states)) {
    weights = weights + (1 / num_nodes) * (fixed_states[j, ] %o% fixed_states[j, ])
  }
  diag(weights) = 0
  return(weights)
}


# Generate weights that match any matrix of fixed states (storkey)
#
# NOTE: lots of nested for loops so likely can be optimized?
fixed_weights_storkey = function(fixed_states) {
  num_nodes = length(fixed_states[1,])
  weights = matrix(0, nrow = num_nodes, ncol = num_nodes)
  for (j in 1:nrow(fixed_states)) {
    fixed_state = fixed_states[j,]
    weights = storkey_learn(fixed_state, weights, fixed_state)
  }
  return(weights)
}


# Activation Function
threshold = function (x) {
  if (x >= 0) {
    return (1)
  }
  else {
    return (-1)
  }
}

# Returns updated node based of connections to other nodes
update_node = function (cur_network, weights, node_index) {
  new_state = cur_network
  new_state[node_index] = threshold(sum(cur_network * weights[node_index,]))
  return(new_state)
  
}

# Returns energy of a hopfield network
net_energy = function(cur_network, weights) {
  return(-0.5 * sum(weights * (cur_network %o% cur_network)))
}

# Returns a scrambled Hopfield network (nodes are randomly chosen)
scramble = function (cur_network) {
  num_nodes = length(cur_network)
  for (i in 1:num_nodes) {
    change = runif(1)
    if (change >= 0.5) {cur_network[i] = 1}
    else {cur_network[i] = -1}
  }
  return(cur_network)
}

# Returns network with certain number of nodes flipped
flip = function(state, num_flips) {
  if (num_flips > length(state)) {
    stop("Your flipping more nodes then exists")
  }
  flip_indices = sample(1:length(state), num_flips, replace = FALSE)
  state[flip_indices] = -state[flip_indices]
  return(state)
}

# Simulates network by checking random nodes for set number of steps
simulate_random = function(cur_network, weights, steps) {
  num_nodes = length(cur_network)
  for (i in 1:steps) {
    update = sample(1:num_nodes, 1)
    cur_network = update_node(cur_network, weights, update) 
  }
  return(cur_network)
}

# Simulates network by going through each node sequentially
simulate_sequential = function(cur_network, weights, node) {
  cur_network = update_node(cur_network, weights, node)
  return(cur_network)
}

# Simulates network via synchronous learning for certain # of steps
simulate_synchronous=function(state,weights,steps=1){
  for(s in 1:steps){
    state=weights%*%state
    state[state == 0] = 1
  }
  return(as.numeric(state))
}

# Determines if it is a fixed point
is_fixed_point_seq = function(cur_network, weights) {
  new_state = cur_network
  for (node in 1:length(cur_network)) {
    new_state = update_node(new_state, weights, node)
  }
  return(identical(new_state, cur_network))
}


# Simulates network until it reaches a fixed point with seq updates
simulate_until_fixed_sequential = function(cur_network, weights, max_steps = 1000) {
  l = length(cur_network)
  for (step in 1:max_steps) {
    for (i in 1:l) {
      cur_network = simulate_sequential(cur_network, weights, i)
    }
    if (is_fixed_point_seq(cur_network, weights)) {
      return(cur_network)
    }
  }
  return(cur_network)
}

# Simulates network until fixed point with sync updates
simulate_until_fixed_sync = function(cur_network, weights, max_steps = 100) {
  for (step in 1:max_steps) {
    newNet = sign(weights %*% cur_network)
    if (identical(newNet, cur_network))
    {
      break
    }
    cur_network = newNet
  }
  return(cur_network)
}

# Simulates network until fixed point with random updates
simulate_until_fixed_random = function(cur_network, weights, max_steps = 1000) {
  for (step in 1:max_steps) {
    prev_network = cur_network
    cur_network = simulate_random(cur_network, weights, length(cur_network))
    if (is_fixed_point_seq(cur_network, weights) == TRUE) {return (cur_network)}
  }
  return(cur_network)
}

# Learns new pattern with hebbian learning
hebbian_learn = function(cur_network, weights, new_pattern) {
  weights = weights + (1 / length(new_pattern)) * (new_pattern %o% new_pattern)
  diag(weights) = 0
  return(weights)
}

# Learns a new pattern given initial weights (storkey)
storkey_learn = function(cur_network, weights, new_pattern) {
  num_nodes = length(cur_network)
  diag(weights) = 0
  h = weights %*% new_pattern

  for (i in 1:num_nodes) {
    for (j in 1:num_nodes) {
      if (i != j) {
        x = (1 / num_nodes) * (
          new_pattern[i] * new_pattern[j] -
            new_pattern[i] * h[j] -
            new_pattern[j] * h[i]
        )
        weights[i, j] = weights[i, j] + x
      }
    }
  }
  return(weights)
}


# Returns the hamming distance between two networks
hamming_distance = function(pattern1, pattern2) {
  if (length(pattern1) != length(pattern2)) stop("Patterns must be of same length.")
  return(sum(pattern1 != pattern2))
}

# Returns the amount of steps it took for sequential update to stabilize
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
    # find midpoint num of flips
    mid = floor((low + high)/2)
    
    scrambled = flip(memory, mid)
    recovered = test_function(scrambled, weights, steps)
    
    if (all(recovered == memory))
    {
      # if the recall is successful, shift the boundaries up
      most = mid
      low = mid + 1
    }
    else
    {
      # shift boundaries down
      high = mid - 1
    }
  }
  return(most)
}

# Plots memory resilience
# First parameter is type of update function - i.e put corresponding
# simulate_until_fixed method
memory_resilience_test = function(learn_method, update_type, length, density = seq(0.02, 0.20, by = 0.02), trials = 25, steps = 500) {
  numD = length(density)
  
  # avg max perturbations
  results = numeric(numD)
  
  # multiple trials for each density
  for (i in 1:numD)
  {
    numPatterns = round(density[i]*length)
    trialAvgs = numeric(trials)
    
    for (t in 1:trials)
    {
      fixedStates = random_nodes(numPatterns, length)
      weights = learn_method(fixedStates)
      maxPerturb = numeric(numPatterns)
      
      # resilience for each pattern
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

# Plots memory capacity - how many memories with x nodes assumg y nodes flipped
# First parameter is type of update function - i.e put corresponding
# simulate_until_fixed method
memory_capacity_test = function(learn_method, update_type, nodes_perturbed, num_nodes = seq(10, 190, by = 20), trials = 10) {
  
  numN = length(num_nodes)
  results = numeric(numN) # Make results Array
  
  for (n in 1:numN) { # Iterate over different number of nodes
    trial_avgs = numeric(trials)
    for (t in 1:trials) { # Iterate over many trials
      hit_capacity = FALSE
      max_memories = 1 # Start with one memory
      fixedStates = random_nodes(1, num_nodes[n])
      weights = learn_method(fixedStates)
      while (hit_capacity == FALSE) {
        new_state = matrix(sample(c(1, -1), num_nodes[n], replace = TRUE), nrow = 1)
        fixedStates = rbind(fixedStates, new_state) # Add one new memory (incrementally)
        weights = learn_method(fixedStates) 
        fail = FALSE
        for (i in 1:nrow(fixedStates)) { # Iterate over fixed states
          fixed_state = as.numeric(fixedStates[i, ])
          fixed_state_mixed = flip(fixed_state, nodes_perturbed)
          fixed_state_mixed_sim = update_type(fixed_state_mixed, weights)
          if (!all(fixed_state_mixed_sim == fixed_state)) {
            hit_capacity = TRUE
            fail = TRUE
            break
          }
        }
        
        if (!fail) { # Make sure to not add one by accident
          max_memories = max_memories + 1
        }
      }
      trial_avgs[t] = max_memories - 1 # Because you stop at the one you fail at
    }
    results[n] = mean(trial_avgs)
  }
  plot(num_nodes, results, type = "b", col = "blue", xlab = "# of Nodes", ylab = "Max Memories", main = "Memory Capacity vs. Network Size")
}

# Plots convergence speed vs memory density, uses sequential updating
# Update type is just simulate_sequential or simulate_random, not sync
convergence_speed_test = function (learn_method, update_type, num_nodes, mem_numbers, trials = num_nodes) {
  avg_conv_speed = numeric(length(mem_numbers))
  for (m in seq_along(mem_numbers)) { # Iterates over memory densities
    speeds = numeric(trials)
    for (t in 1:trials) { # Iterates over trials
      fixedStates = random_nodes(mem_numbers[m], num_nodes)
      weights = learn_method(fixedStates)
      cur_states = random_nodes(mem_numbers[m], num_nodes)
      converge_times = sapply(1:mem_numbers[m], function(i) { # Checks speed many times
        time_to_converge_seq(update_type, cur_states[i, ], weights)
      })
      speeds[t] = mean(converge_times) # Averages conv speed from dif starts
    }
    
    avg_conv_speed[m] = mean(speeds) # Avg conv speed from dif trials
  }
  
  plot(mem_numbers, avg_conv_speed, type = "l",
       xlab = "Memory Density in Percent",
       ylab = "Avg Convergence Updates",
       main = "Convergence Time vs Memory Density", 
       xlim = c(0, 30), ylim = c(0, 600))
}

# Tests if memory is recovered using seq updates
test_resilience_sequential = function(memory, weights, num_flips, steps=1000) {
  scrambled = flip(memory, num_flips)
  recovered = simulate_until_fixed_sequential(scrambled, weights, steps)
  return(all(recovered == memory))
}


# Tests if memory is recovered using sync updates
test_resilience_sync = function(memory, weights, num_flips, steps=1000) {
  scrambled = flip(memory, num_flips)
  recovered = simulate_until_fixed_sync(scrambled, weights, steps)
  return(all(recovered == memory))
}

#========================================
#===IMAGE PREPROCESSING / ILLUSTRATION===
#========================================


library(ggplot2)


# Makes it a vector
pattern_to_vector = function(matrix_pattern) {
  return(as.vector(matrix_pattern))
}

# Trains weights from a given pattern
train_weights_from_pattern = function(pattern) {
  N = length(pattern)
  weights = matrix(0, N, N)
  weights = hebbian_learn(pattern, weights, pattern)
  return(weights)
}

# Scrambles a fraction of nodes
scramble_pattern = function(pattern, frac = 0.2) {
  num_flips = round(length(pattern) * frac)
  return(flip(pattern, num_flips))
}

# Plots a current network
plot_state = function(state, step) {
  mat = matrix(state, nrow=10, byrow=FALSE)
  df = expand.grid(x=1:10, y=1:10)
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

# Decodes a scrambled network
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

# Decodes a network and also shows hamming distance
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

# Plots hamming distance versus time for a network as it is decoded
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

processImg = function(imgPath, maxSize = 25, thres = "50%")
{
  img = image_read(imgPath)
  
  img = image_convert(img, colorspace = "gray")
  img = image_convert(img, "png")
  info = image_info(img)
  ogWidth = info$width
  ogHeight = info$height
  
  # scale image
  if (ogWidth < ogHeight)
  {
    # just only scale to 50 for now
    img = image_scale(img, "x50")
  }
  else
  {
    img = image_scale(img, "50")
  }
  img = image_threshold(img, type = "white", threshold = "50%")
  img = image_threshold(img, type = "black", threshold = "50%")
  
  info = image_info(img)
  width = info$width
  height = info$height
  
  # get pixel colors
  pixels = as.raster(img)
  
  # turn the pixel colors into neuron matrix of -1 for black and 1 for white
  neurons = matrix(nrow = height, ncol = width)
  for (i in 1:height)
  {
    for (j in 1:width)
    {
      if (pixels[i,j] == "#ffffffff")
      {
        neurons[i,j] = 1
      }
      else
      {
        neurons[i,j] = -1
      }
    }
  }
  
  # rotate this because idk why it is sideways
  neurons = t(apply(neurons, 2, rev))
  
  # return the matrix so you still have the dimensions
  return(neurons)
  
  # to make it a 1D vector use as.vector(inputttttttt)
}