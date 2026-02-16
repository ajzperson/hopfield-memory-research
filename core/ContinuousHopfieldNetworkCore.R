# ==============================================================================
# CONTINUOUS HOPFIELD NETWORK CORE: LEGACY (1984) & MODERN (2020)
# ==============================================================================
# This engine implements:
#
# 1. LEGACY CONTINUOUS (Hopfield, 1984):
#    - Uses a fixed Weight Matrix (W) derived from Hebbian Learning.
#    - State updates via Sigmoid/Tanh activation functions.
#
# 2. MODERN CONTINUOUS / MHN (Ramsauer et al., 2020):
#    - Utilizes a Log-Sum-Exp energy function and an update rule equivalent 
#      to the Self-Attention mechanism found in Transformers (e.g., GPT-4).
# 
# KEY FEATURES:
# 1. Exponential Storage Capacity: Stores exponentially more memories than classic models.
# 2. Continuous State Space: Neurons take real values in [-1, 1].
# 3. Numerical Stability: Uses LSE tricks to prevent overflow during softmax.
# ==============================================================================

# Generate x*y node values from -1 to 1
initialize_state = function(x, y) {
  matrix(runif(x*y, min = -1, max = 1), x, y)
}

# Modern Hopfield / attention update (normalized)
# Dividing by the norms ensures Beta acts as a true temperature control
# regardless of the length of the image vectors.
attention_update = function(cur_network, fixed_states, beta = 20) {
  # 1. Similarity (Normalized to range [-1, 1])
  dot_prod = fixed_states %*% cur_network
  norms = sqrt(rowSums(fixed_states^2)) * sqrt(sum(cur_network^2))
  s = as.vector(dot_prod / (norms + 1e-9))
  
  # 2. Stable Softmax
  s_adj = s - max(s)
  weights = exp(s_adj * beta) / sum(exp(s_adj * beta))
  
  # 3. Retrieval
  return(as.vector(weights %*% fixed_states))
}

# Find net energy
# Modern energy function: Unlike the classic quadratic energy, this 
# uses the LSE term to create much sharper wells around memories. 
# This is what allows the continuous model to store exponentially 
# more patterns than the classic model.
net_energy_continuous = function(beta, storedPatternsX, cur_network) {
  N = ncol(storedPatternsX)
  M = maxEuclideanNorm(storedPatternsX)
  
  er = -lse(beta, as.vector(storedPatternsX %*% cur_network))
  er = er + 0.5 * sum(cur_network^2)
  er = er + (1/beta) * log(N) + 0.5 * M * M
  return(er)
}

# Calculate the log-sum-exp
# Log-Sum-Exp (LSE): This is the smooth approximation of the 'max' function.
# As beta -> infinity, this model behaves exactly like the Classic 
# Hopfield Network but with much higher storage capacity.
lse = function(beta, z) {
  # Use the LSE for numerical stability
  # This prevents exp(large_number) from becoming Inf.
  m = max(beta * z)
  return(m + (1/beta) * log(sum(exp(beta * z - m))))
}

# Find the max Euclidean norm in the stored patterns
maxEuclideanNorm = function(X) {
  return(max(sqrt(colSums(X^2))))
}

# Add Gaussian noise to nodes (because you can't "flip" them anymore)
gaussianNoise = function(cur_network, num_flips) {
  if (num_flips > length(cur_network)) {
    stop("You're flipping more nodes than exists")
  }
  flipped_indices = sample(length(cur_network), num_flips)
  noise = rnorm(num_flips, mean = 0, sd = 1)
  cur_network[flipped_indices] = cur_network[flipped_indices] + 0.5 * noise

  # Clipping: Continuous neurons represent activation levels, usually 
  # bounded between -1 and 1. This ensures noise doesn't push the 
  # network into physically impossible states.
  cur_network = pmax(pmin(cur_network, 1), -1)
  return(cur_network)
}

# Simulates Modern Continuous Hopfield Network
simulate_continuous = function (cur_network, fixed_states, steps = 100, stepSize = 0.1) {
  for (s in 1:steps) {
    # Update the nodes
    newState = attention_update(cur_network, fixed_states)
    
    cur_network = cur_network + stepSize * (newState - cur_network)
  }
  return (cur_network)
}

# Simulates Modern Continuous Hopfield Network until fixed point
simulate_until_fixed_continuous = function(cur_network, fixed_states, max_steps = 100, stepSize = 0.1, beta = 2) {
  for (step in 1:max_steps) {
    newState = attention_update(cur_network, fixed_states, beta)
    update = stepSize * (newState - cur_network)
    
    # Break if the change is negligible (fixed point reached)
    if (max(abs(update)) < 0.000001) {
      break
    }
    cur_network = cur_network + update
  }
  return(cur_network)
}

# Get state and converge steps
simulate_with_steps = function(cur_network, fixed_states, max_steps = 100, stepSize = 0.1, beta = 2) {
  for (step in 1:max_steps) {
    newState = attention_update(cur_network, fixed_states, beta)
    update = stepSize * (newState - cur_network)
    
    if (max(abs(update)) < 0.000001)
    {
      return(list(state = cur_network + update, steps = step))
    }
    cur_network = cur_network + update
  }
  return(list(state = cur_network, steps = max_steps))
}

# Return the hamming distance between two networks
hamming_distance = function(pattern1, pattern2) {
  if (length(pattern1) != length(pattern2)) stop("Patterns must be of same length.")
  return(sum(pattern1 != pattern2))
}

# Measure the number of iterations required for the state to reach equilibrium
# Convergence is defined as the point where node updates fall below a fixed tolerance.
time_to_converge_continuous = function(state, fixedStates, maxSteps = 1000) {
  prevState = state
  for (m in 1:maxSteps) {
    newState = attention_update(prevState, fixedStates)
    if (all(abs(newState - prevState) < 0.00005)) {
      return (m)
    }
    prevState = newState
  }
  return(maxSteps)
}

#========================
#===ANALYSIS FUNCTIONS===
#========================

library(parallel)

# Evaluate the network's ability to recover memories under varying levels of Gaussian noise
# This test determines the maximum 'perturbation threshold' before the 
# continuous attractor fails to converge to the original stored pattern.
memory_resilience_test_continuous = function(num_nodes, density = 0.1, trials = 5, steps = 1000, beta = 2, similarity = 0) {
  numPatterns = max(1, round(density * num_nodes))
  
  trialAvgs = sapply(1:trials, function(t) {
    # Generate correlated patterns based on similarity
    base = initialize_state(1, num_nodes)
    fixedStates = matrix(nrow = numPatterns, ncol = num_nodes)
    fixedStates[1,] = base
    if (numPatterns > 1) {
      for (j in 2:numPatterns) {
        fixedStates[j,] = gaussianNoise(base, round(similarity * num_nodes))
      }
    }
    
    maxPerturb = numeric(numPatterns)
    
    for (n in 1:numPatterns) {
      # Test recovery across a range of noise
      # Find the threshold where it stops working
      max_f = 0
      for (f in 1:num_nodes) {
        noisy = gaussianNoise(fixedStates[n,], f)
        # Pass the beta parameter here
        recovered = simulate_until_fixed_continuous(noisy, fixedStates, steps, beta = beta)
        
        # Check if it still identifies the correct original pattern
        if (!is.na(recovered[1]) && which.max(fixedStates %*% as.vector(recovered)) == n) {
          max_f = f
        # Found the limit for this pattern
        } else {
          break 
        }
      }
      maxPerturb[n] = max_f / num_nodes
    }
    return(mean(maxPerturb))
  })
  
  return(mean(trialAvgs))
}

# Measure the storage capacity limit of the continuous network
# Incremental memories are added until the network can no longer uniquely 
# recover the original patterns from a noisy starting state, 
# indicating the formation of spurious states.
memory_capacity_test_continuous = function(nodes_perturbed, num_nodes = seq(1, 10, by = 1), trials = 50, beta = 2, similarity = 0) {
  numN = length(num_nodes)
  results = numeric(numN)
  
  for(n in 1:numN) {
    trials_avgs = numeric(trials)
    
    for(t in 1:trials) {
      hit_capacity = FALSE
      max_memories = 1
      # Establish the base pattern
      base_pattern = initialize_state(1, num_nodes[n])
      fixed_states = base_pattern
      
      while (hit_capacity == FALSE) {
        fail = FALSE
        for(i in 1:nrow(fixed_states)) {
          fixed_state = as.numeric(fixed_states[i,])
          fixed_state_mixed = gaussianNoise(fixed_state, nodes_perturbed)
          # Pass the beta parameter here
          fixed_state_sim = simulate_until_fixed_continuous(fixed_state_mixed, fixed_states, beta = beta)
          
          # Check for divergence or failure to recover
          if (any(is.na(fixed_state_sim)) || sum(abs(fixed_state_sim - fixed_state)) > (0.15 * length(fixed_state))) {
            hit_capacity = TRUE
            fail = TRUE
            break
          }
        }
        if (!fail) {
          # Limit capacity to prevent infinite loops if network is too stable
          if (max_memories > 50) break 
          
          # Generate new state based on similarity to the base_pattern
          new_state = gaussianNoise(base_pattern, round(similarity * num_nodes[n]))
          fixed_states = rbind(fixed_states, new_state)
          max_memories = max_memories + 1
        }
      }
      trials_avgs[t] = max_memories - 1
    }
    results[n] = mean(trials_avgs)
  }
  # Return the results so the calling function can use them
  return(results)
}

# Plot convergence speed (num updates) vs memory density for Modern Continuous Hopfield Networks
# Updates are technically synchronous.
convergence_speed_test_continuous = function (num_nodes, mem_numbers, trials = num_nodes) {
  avg_conv_speed = numeric(length(mem_numbers))
  for (m in seq_along(mem_numbers)) { # Iterates over memory densities
    speeds = numeric(trials)
    for (t in 1:trials) { # Iterates over trials
      fixedStates = initialize_state(mem_numbers[m], num_nodes)
      cur_states = initialize_state(mem_numbers[m], num_nodes)
      converge_times = sapply(1:mem_numbers[m], function(i) { # Checks speed many times
        time_to_converge_continuous(cur_states[i,], fixedStates, 1000)
      })
      speeds[t] = mean(converge_times) # Averages conv speed from dif starts
    }
    
    avg_conv_speed[m] = mean(speeds) # Avg conv speed from dif trials
  }
  
  plot(mem_numbers, avg_conv_speed, type = "l",
       xlab = "Memory Density in Percent",
       ylab = "Avg Convergence Updates",
       main = "Convergence Time vs Memory Density")
}

# beta (softmax temperature) vs convergence speed, accuracy, energy
# Low beta has a slower, smoother convergence.
# High beta has faster, more unstable convergence.
softmax_temp_test = function(betaVals, numTrials, numPatterns, numNodes, noisyFraction)
{
  # betaVals is a seq
  # Contain convergence time, accuracy, and final energy for each beta
  betaLen = length(betaVals)
  
  results = data.frame(beta = betaVals, avgSteps = numeric(betaLen), avgAccuracy = numeric(betaLen), avgEnergy = numeric(betaLen))
  
  for (b in 1:betaLen)
  {
    beta = betaVals[b]
    totalSteps = 0
    correctRecalls = 0
    totalEnergy = 0
    
    for (t in 1:numTrials)
    {
      fixedStates = initialize_state(numPatterns, numNodes)
      original = fixedStates[1,]
      noisy = gaussianNoise(original, noisyFraction * numNodes)
      
      # Run network until it converges
      result = simulate_with_steps(noisy, fixedStates, 1000, stepSize = 0.1, beta)
      final = result$state
      steps = result$steps
      
      totalSteps = totalSteps + steps
      
      # Measure accuracy
      recoveredIndex = which.max(fixedStates %*% as.vector(final))
      if (recoveredIndex == 1)
      {
        correctRecalls = correctRecalls + 1
      }
        
      # Energy
      er = net_energy_continuous(beta, fixedStates, final)
      totalEnergy = totalEnergy + er
    }
    
    results$avgSteps[b] = totalSteps/numTrials
    results$avgAccuracy[b] = correctRecalls/numTrials
    results$avgEnergy[b] = totalEnergy/numTrials
  }
  
  par(mfrow = c(1,3))
  
  plot(results$beta, results$avgSteps, type = "b", col = "blue", xlab = "Beta", ylab = "Avg Steps", main = "Convergence speed")
  plot(results$beta, results$avgAccuracy, type = "b", col = "blue", xlab = "Beta", ylab = "Avg Accuracy", main = "Accuracy")
  plot(results$beta, results$avgEnergy, type = "b", col = "blue", xlab = "Beta", ylab = "Avg Energy", main = "Energy")
  return(results)
}

# Legacy Continuous train [0, 1]
train_continuous_weights = function(target_vector) {
  # Center the data by subtracting 0.5. 
  # This makes 0 (black) become -0.5 and 1 (white) become +0.5.
  # Without this, the weight matrix 'explodes' with positive values.
  centered = target_vector - 0.5
  
  # Hebbian Learning: outer product of the vector with itself
  # weight_matrix[i,j] = centered[i] * centered[j]
  N = length(centered)
  weights = tcrossprod(centered) 
  
  # Remove self-connections (diagonal) to prevent the network from 
  # just "memorizing" its current state without logic
  diag(weights) = 0
  
  # Normalize weights by N to keep values stable
  return(weights / N)
}

# Legacy continuous decoding (Reconstruction)
decode_continuous = function(state, weights, max_iters = 50, learning_rate = 0.5) {
  current_state = state
  
  for (i in 1:max_iters) {
    # The brain update: centered_state * weights
    # Subtract 0.5 before the math and add it back after.
    input_signal = (current_state - 0.5) %*% weights
    
    # Update Rule: Linear integration with hard clamping
    # This gradually moves the state toward the target, using clamping 
    # to keep values bounded in the valid [0, 1] color space. 
    new_state = current_state + (learning_rate * input_signal)
    
    # "Clamp" the values so colors don't become very bright or dark
    current_state = pmin(pmax(new_state, 0), 1)
    
    # Progress report
    if (i %% 10 == 0) cat("Continuous Convergence iteration:", i, "\n")
  }
  
  return(as.vector(current_state))
}