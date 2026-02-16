source('Documents/cosmosR/ClassicHopfieldNetworkV4.R')
source('Documents/cosmosR/ContinuousHopfieldNetwork.R')


correlationClassicNetwork = function(correlationLevels = seq(0, 100, by = 5), x = 5, y = 5, patternCount = 5, trials = 50, steps = 1000, update_fn = simulate_until_fixed_sync, learn_fn = fixed_weights_hebbian)
{
  numLvls = length(correlationLevels)
  resNums = numeric(numLvls)
  capNums = numeric(numLvls)
  leng = x * y
  
  for (i in 1:numLvls)
  {
    cat("testing correlation ",(correlationLevels[i]),"%\n")
    percentChange = correlationLevels[i]/100
    
    trialRes = numeric(trials)
    trialCap = numeric(trials)
    
    for (t in 1:trials)
    {
      base = pattern_to_vector(random_nodes(x, y))
      patternMatrix = matrix(nrow = patternCount, ncol = leng)
      patternMatrix[1,] = base
      
      for (j in 2:patternCount)
      {
        patternMatrix[j,] = flip(base, percentChange*leng)
      }
      
      # resilience tests
      weights = learn_fn(patternMatrix)
      maxFlips = numeric(patternCount)
      for (p in 1:patternCount)
      {
        maxFlips[p] = max_recoverable_flips(update_fn, patternMatrix[p,], weights, steps)
      }
      trialRes[t] = mean(maxFlips)/leng
      
      # capacity test
      cap = 1
      fixedStates = patternMatrix[1, , drop = FALSE]
      weights = learn_fn(fixedStates)
      
      while (cap < patternCount)
      {
        newSt = patternMatrix[cap+1,]
        fixedStates = rbind(fixedStates, newSt)
        weights = learn_fn(fixedStates)
        
        fail = FALSE
        for (k in 1:nrow(fixedStates))
        {
          testSt = flip(fixedStates[k,], 2)
          recovered = update_fn(testSt, weights)
          if (!all(sign(recovered) == fixedStates[k,]))
          {
            fail = TRUE
            break
          }
        }
        if (fail)
        {
          break
        }
        cap = cap + 1
      }
      trialCap[t] = cap
    }
    resNums[i] = mean(trialRes)
    capNums[i] = mean(trialCap)
  }
  
  par(mfrow = c(1,2))
  plot(correlationLevels, resNums*100, type = "b", col = "blue", xlab = "Similarity (%)", ylab = "# Perturbations (%)", main = "Resilience Test", xlim = c(0, 100))
  plot(correlationLevels, capNums, type = "b", col = "red", xlab = "Similarity (%)", ylab = "# Memories", main = "Capacity Test")
  
  return("lkdsjflksjdklfdjs")
}

correlationContinuousNetwork = function(correlationLevels = seq(0, 100, by = 5), x = 5, y = 5, patternCount = 5, trials = 50, steps = 1000, beta = 2) {
  numLvls = length(correlationLevels)
  resNums = numeric(numLvls)
  capNums = numeric(numLvls)
  leng = x * y
  
  for (i in 1:numLvls) {
    cat("Testing noise level (%)", correlationLevels[i], "%\n")
    
    noise_fraction = correlationLevels[i] / 100
    
    # Run resilience test for this noise level
    resNums[i] = mean(
      replicate(trials, {
        memory_resilience_test_continuous(leng, density = noise_fraction, trials = 1, steps = steps)
      })
    )
    
    # Run capacity test for this noise level
    capNums[i] = mean(
      replicate(trials, {
        memory_capacity_test_continuous(nodes_perturbed = noise_fraction * leng, num_nodes = leng, trials = 1)
      })
    )
  }
  
  par(mfrow = c(1,2))
  plot(correlationLevels, resNums * 100, type = "b", col = "blue",
       xlab = "Noise Level (%)", ylab = "Avg Max Noise (%)", main = "Resilience Test", xlim = c(0, 100))
  plot(correlationLevels, capNums, type = "b", col = "red",
       xlab = "Noise Level (%)", ylab = "Max Memories", main = "Capacity Test")
  
  return(list(resilience = resNums, capacity = capNums))
}