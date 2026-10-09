# ==============================================================================
# HOPFIELD NETWORK: CORRELATION & CROSSTALK BENCHMARKING SUITE
# ==============================================================================
# Description:
#   This script evaluates the performance limits of Classic vs. 
#   Modern Continuous (Attention) Hopfield Networks when subjected to 
#   highly correlated pattern sets.
#
# Key Benchmarks:
#   1. Resilience Test: Measures how much noise (perturbation) a network can 
#      tolerate before it fails to distinguish between similar stored memories.
#   2. Capacity Test: Determines the maximum number of patterns a network can 
#      store as similarity increases before spurious states dominate.
#
# Technical Context:
#   - Classic Networks struggle with similarity due to Hebbian interference.
#   - Continuous Networks utilize Softmax-based Attention (beta) to 'sharpen' 
#     energy basins, allowing for exponentially higher storage of similar data.
# ==============================================================================

source('core/ClassicHopfieldNetworkCore.R')
source('core/ContinuousHopfieldNetworkCore.R')

correlationClassicNetwork = function(correlationLevels = seq(0, 100, by = 5), 
                                     x = 10, y = 10, patternCount = 13, trials = 50, 
                                     steps = 1000, update_fn = simulate_until_fixed_random, 
                                     learn_fn = fixed_weights_storkey,
                                     stability_threshold = 0.05)
{
  numLvls = length(correlationLevels)
  resNums = numeric(numLvls)
  capNums = numeric(numLvls)
  leng = x * y
  
  # Calculate how many bits to flip based on network size
  # Ensure 'Stability' check is consistent across different grid sizes
  stability_noise = max(1, round(stability_threshold * leng))

  for (i in 1:numLvls)
  {
    cat("Testing Similarity level: ",(correlationLevels[i]),"%\n")
    percentChange = correlationLevels[i]/100
    
    trialRes = numeric(trials)
    trialCap = numeric(trials)
    
    for (t in 1:trials)
    {
      base = pattern_to_vector(random_nodes(x, y))
      patternMatrix = matrix(nrow = patternCount, ncol = leng)
      patternMatrix[1,] = base
      
      # Asymmetric flip logic (breaks symmetry + fixes non-integer flips)
      flip_fraction = 1 - (percentChange ^ 2)
      numFlips = round(flip_fraction * leng)
      numFlips = max(0, min(numFlips, leng - 1))
      
      for (j in 2:patternCount)
      {
        patternMatrix[j,] = flip(base, numFlips)
      }
      
      # 1. Resilience test
      # Measure the max noise tolerance for the current pattern set
      weights = learn_fn(patternMatrix)
      maxFlips = numeric(patternCount)
      for (p in 1:patternCount)
      {
        maxFlips[p] = max_recoverable_flips(update_fn, patternMatrix[p,], weights, steps)
      }
      trialRes[t] = mean(maxFlips)/leng
      
      # 2. Capacity test
      # Incremental test to see how many similar patterns can be stored
      cap = 1
      current_pattern = patternMatrix[1, , drop = FALSE]
      # 2. Initial weights for just the first pattern
      weights = (t(current_pattern) %*% current_pattern) / leng
      diag(weights) = 0 # No self connection
      
      while (cap < patternCount) {
        # 3. Get the next pattern
        next_pat = patternMatrix[cap + 1, , drop = FALSE]
        
        # 4. add it to existing weights
        weights = weights + (t(next_pat) %*% next_pat) / leng
        diag(weights) = 0 
        
        # 5. stability check (only check what's stored so far)
        fail = FALSE
        for (k in 1:(cap + 1)) {
          testSt = flip(patternMatrix[k,], stability_noise)
          recovered = update_fn(testSt, weights, steps)
          
          if (!all(recovered == patternMatrix[k,])) {
            fail = TRUE
            break
          }
        }
        if (fail) break
        cap = cap + 1
      }
      trialCap[t] = cap
    }
    resNums[i] = mean(trialRes)
    capNums[i] = mean(trialCap)
  }

  # Setup layout for plotting
  par(mfrow = c(1, 2), mar = c(4, 4, 2, 1), oma = c(0, 0, 0, 0))

  # Resilience plot 
  plot(correlationLevels, resNums*100, 
    type = "b", col = "blue", ylim = c(0, 50),
    xlab = "", ylab = "",
    main = "Classic Hopfield Resilience Test", 
    xlim = c(0, 100), 
    cex.main = 0.75, 
    axes = FALSE)
  box()
  axis(1, at = seq(0, 100, by = 20), labels = c("0", "20", "40", "60", "80", "100"))
  axis(2, at = seq(0, 50, by = 10), las = 1)
  title(xlab = "Pattern Similarity (%)", line = 2.5, cex.lab = 0.85)
  title(ylab = "Noise Tolerance (%)", line = 2.5, cex.lab = 0.85)

  # Capacity plot
  plot(correlationLevels, capNums, 
       type = "b", col = "red", ylim = c(0, patternCount + 2),
       xlab = "", ylab = "", 
       main = "Classic Hopfield Capacity Test",
       xlim = c(0, 100),
       cex.main = 0.8, 
       axes = FALSE)
  box()
  axis(1, at = seq(0, 100, by = 20), labels = c("0", "20", "40", "60", "80", "100"))
  axis(2, at = seq(0, 16, by = 2), labels = seq(0, 16, by = 2), las = 1)
  title(xlab = "Pattern Similarity (%)", line = 2.5, cex.lab = 0.85)
  title(ylab = "Max Memories", line = 2.5, cex.lab = 0.85)

  return(list(resilience = resNums, capacity = capNums))
}

correlationContinuousNetwork = function(correlationLevels = seq(0, 100, by = 5), x = 5, y = 5, 
                                        patternCount = 5, trials = 50, steps = 1000, beta = 20) {
  numLvls = length(correlationLevels)
  resNums = numeric(numLvls)
  capNums = numeric(numLvls)
  leng = x * y
  
  # Calculate a fixed density based on the desired patternCount for the resilience test
  fixed_resilience_density = patternCount / leng
  
  for (i in 1:numLvls) {
    cat("Testing Level:", correlationLevels[i], "%\n")
    
    # current_ratio represents the similarity percentage
    current_ratio = correlationLevels[i] / 100
    
    # 1. Resilience Test: Fixed patterns density, variable similarity
    resNums[i] = mean(
      replicate(trials, {
        memory_resilience_test_continuous(leng, density = fixed_resilience_density, 
                                          trials = 1L, steps = steps, beta = beta, # 1L = explicit integer
                                          similarity = current_ratio)
      })
    )
    
    # 2. Capacity Test: Fixed 15% noise floor, variable pattern similarity
    capNums[i] = mean(
      replicate(trials, {
        memory_capacity_test_continuous(nodes_perturbed = round(0.15 * leng), 
                                        num_nodes = leng, trials = 1L, beta = beta, # 1L = explicit integer
                                        similarity = current_ratio)
      })
    )
  }
  
  # Match classic plot margins (mar + oma) to eliminate top space
  par(mfrow = c(1, 2), mar = c(4, 4, 2, 1), oma = c(0, 0, 0, 0)) 

  # Modern Continuous Hopfield resilience test
  # Cap noise tolerance at 100% (pmin()) to avoid invalid values
  plot(correlationLevels, pmin(resNums*100, 100), 
      type = "b", col = "blue",
      xlab = "", ylab = "",
      main = "Modern Continuous Hopfield Resilience Test", 
      xlim = c(0, 100), cex.main = 0.6, axes = FALSE)
  box()
  axis(1, at = seq(0, 100, by = 20), labels = c("0", "20", "40", "60", "80", "100"))
  axis(2, at = seq(0, 50, by = 10), las = 1)
  title(xlab = "Pattern Similarity (%)", line = 2.2, cex.lab = 0.85)
  title(ylab = "Noise Tolerance (%)", line = 2.2, cex.lab = 0.85)

  # Modern Continuous Hopfield capacity test
  plot(correlationLevels, capNums, 
      type = "b", col = "red",
      xlab = "", ylab = "", 
      main = "Modern Continuous Hopfield Capacity Test",
      cex.main = 0.6, axes = FALSE)
  box()
  axis(1, at = seq(0, 100, by = 20), labels = c("0", "20", "40", "60", "80", "100"))
  axis(2, at = seq(0, 50, by = 10), las = 1)
  title(xlab = "Pattern Similarity (%)", line = 2.2, cex.lab = 0.85)
  title(ylab = "Max Memories", line = 2.2, cex.lab = 0.85)

  return(list(resilience = resNums, capacity = capNums))
}