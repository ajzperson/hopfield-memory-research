# generate x*y node values from -1 to 1
initialize_state = function(x, y) {
  matrix(runif(x*y, min = -1, max = 1), x, y)
}

attention_update = function(cur_network, fixed_states, beta = 20) {
  # store dot products between current state and fixed states
  num_patterns = nrow(fixed_states)
  s = fixed_states %*% cur_network
  
  # weighted soft-max values
  softmax_vals = numeric(length(cur_network))
  exp_s = exp(s * beta)
  weights = exp_s / (sum(exp_s))
  softmax_vals = as.vector(t(weights) %*% fixed_states)
  
  return(softmax_vals)
}

# find net energy
net_energy_continuous = function(beta, storedPatternsX, cur_network) {
  N = ncol(storedPatternsX)
  M = maxEuclideanNorm(storedPatternsX)
  
  er = -lse(beta, as.vector(storedPatternsX %*% cur_network))
  er = er + 0.5 * sum(cur_network^2)
  er = er + (1/beta) * log(N) + 0.5 * M * M
  return(er)
}

# calculate the log-sum-exp
lse = function(beta, z) {
  sum = 0
  for (i in 1:length(z)) {
    sum = sum + exp(beta * z[i])
  }
  return((1/beta) * log(sum))
}

# find the max Euclidean norm in the stored patterns
maxEuclideanNorm = function(X) {
  norms = apply(X, 2, function(col) sqrt(sum(col^2)))
  return(max(norms))
}

# adds gaussian noise to nodes (because you can't "flip" them anymore)
gaussianNoise = function(cur_network, num_flips) {
  if (num_flips > length(cur_network)) {
    stop("You're flipping more nodes than exists")
  }
  flipped_indices = sample(length(cur_network), num_flips)
  noise = rnorm(num_flips, mean = 0, sd = 1)
  cur_network[flipped_indices] = cur_network[flipped_indices] + 0.5 * noise
  return(cur_network)
}

# Simulates continuous hopfield network
simulate_continuous = function (cur_network, fixed_states, steps = 100, stepSize = 0.1) {
  for (s in 1:steps) {
    # update the nodes
    newState = attentionUpdate(cur_network, fixed_states)
    
    cur_network = cur_network + stepSize * (newState - cur_network)
  }
  return (cur_network)
}

# Simulates continuous hopfield network until fixed point
simulate_until_fixed_continuous = function(cur_network, fixed_states, max_steps = 100, stepSize = 0.1, beta = 2) {
  for (step in 1:max_steps) {
    newState = attention_update(cur_network, fixed_states, beta)
    update = stepSize * (newState - cur_network)
    
    if (max(abs(update)) < 0.000001)
    {
      break
    }
    cur_network = cur_network + update
  }
  return(cur_network)
}

# get state and converge steps
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

# Returns the hamming distance between two networks
hamming_distance = function(pattern1, pattern2) {
  if (length(pattern1) != length(pattern2)) stop("Patterns must be of same length.")
  return(sum(pattern1 != pattern2))
}

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

memory_resilience_test_continuous = function(num_nodes, density = seq(0.02, 0.50, by = 0.02), trials = 50, steps = 1000) {
  numN = length(density)
  results = numeric(numN) # Make results Array
  
  for (d in 1:numN) {
    numPatterns = max(1, round(density[d] * num_nodes))
    trialAvgs = unlist(mclapply(1:trials, function(t) {
      fixedStates = initialize_state(numPatterns, num_nodes)
      maxPerturb = numeric(numPatterns)
      cat("numN is", d)
      for (n in 1:numPatterns) {
        print(n)
        noisys = lapply(1:num_nodes, function(f) gaussianNoise(fixedStates[n,], f))
        
        # test recovery of each one
        recovereds = lapply(noisys, function(inp) simulate_until_fixed_continuous(inp, fixedStates, steps))
        
        # check if the index of the "closest" is the same as the index of the pattern
        similarities = sapply(recovereds, function(c) which.max(fixedStates %*% as.vector(c)) == n)
        
        maxFlips = 0
        if (any(similarities)) {
          maxFlips = max(which(similarities))
        }
        maxPerturb[n] = maxFlips/num_nodes
      }
      mean(maxPerturb)
    }))
    results[d] = mean(trialAvgs)
  }
  xvals = density * num_nodes
  yvals = results * num_nodes
  plot(xvals, yvals, type = "b", xlab = "# memories", ylab = "Avg Max Gaussian Perturbations", main = "Resilience vs Memory Density")
  return(results)
}

memory_capacity_test_continuous = function(nodes_perturbed, num_nodes = seq(1, 10, by = 1), trials = 50) {
  numN = length(num_nodes)
  results = numeric(numN)
  
  for(n in 1:numN) {
    trials_avgs = numeric(trials)
    
    for(t in 1:trials) {
      hit_capacity = FALSE
      max_memories = 1
      fixed_states = initialize_state(1, num_nodes[n])
      
      while (hit_capacity == FALSE) {
        fail = FALSE
        
        for(i in 1:nrow(fixed_states)) {
          fixed_state = as.numeric(fixed_states[i,])
          fixed_state_mixed = gaussianNoise(fixed_state, nodes_perturbed)
          fixed_state_sim = simulate_until_fixed_continuous(fixed_state_mixed, fixed_states)
          
          if (sum(abs(fixed_state_sim - fixed_state)) > (0.001 * length(fixed_state))) {
            hit_capacity = TRUE
            fail = TRUE
            break
          }
        }
        if (!fail) {
          new_state = initialize_state(1, num_nodes[n])
          fixed_states = rbind(fixed_states, new_state)
          max_memories = max_memories + 1
        }
      }
      trials_avgs[t] = max_memories - 1
    }
    results[n] = mean(trials_avgs)
  }
  plot(num_nodes, results, type = "b", col = "blue", xlab = "# of Nodes", ylab = "Max Memories", main = "Memory Capacity vs. Network Size")
}

# Plots convergence speed (num updates) vs memory density for continuous hopfield networks
# updates are technically synchronous
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
# low beta has a slower, smoother convergence
# high beta has faster, more unstable convergence
softmax_temp_test = function(betaVals, numTrials, numPatterns, numNodes, noisyFraction)
{
  # betaVals is a seq
  
  # contains convergence time, accuracy, and final energy for each beta
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
      
      # runs network until it converges
      result = simulate_with_steps(noisy, fixedStates, 1000, stepSize = 0.1, beta)
      final = result$state
      steps = result$steps
      
      totalSteps = totalSteps + steps
      
      # measure accuracy
      recoveredIndex = which.max(fixedStates %*% as.vector(final))
      if (recoveredIndex == 1)
      {
        correctRecalls = correctRecalls + 1
      }
        
      # energy
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