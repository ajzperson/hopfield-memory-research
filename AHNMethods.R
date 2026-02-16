# ==============================================================================
# Asymmetric Hopfield Network (AHN) - Sequence Memory Research
# ==============================================================================
# SUMMARY:
# This module implements an AHN designed for temporal sequence recall. 
# 
# Usage - source("AHNMethods.R")
# then call sequence_capacity_test(20), sequence_capacity_test(200), etc.
# 
# TODO - ADD NOISE TOLERANCE
# ==============================================================================

# Generate random set of starting weights (either 1 or -1) but not symmetric
random_weights_asymmetric=function(start_network){
  num_nodes=length(start_network)
  random_matrix=matrix(sample(c(1,-1),num_nodes*num_nodes,replace=TRUE),nrow=num_nodes,ncol=num_nodes)
  return(random_matrix)
}

# Network learns the sequence of fixed states, not any particular one.
fixed_weights_asymmetric_sequence = function(patterns) {
  num_patterns = nrow(patterns)
  num_nodes = ncol(patterns)
  w = matrix(0, nrow = num_nodes, ncol = num_nodes)
  
  for (i in 1:(num_patterns - 1)) {
    # Transition weight: Pattern i pushes toward Pattern i+1
    w = w + (patterns[i+1, ] %*% t(patterns[i, ]))
  }
  # Keep diagonal 0 to prevent pattern locking
  diag(w) = 0
  return(w)
}

# Generate a sequence of nodes with 90% correlation
random_nodes_sequence=function(L,num_nodes,correlation=0) {
  patterns=matrix(0,nrow=L,ncol=num_nodes)
  patterns[1,]=sample(c(1,-1),num_nodes,replace=TRUE)
  for(i in 2:L){
    patterns[i,]=patterns[i-1,]
    flip_idx=sample(1:num_nodes,size=round((1-correlation)*num_nodes))
    patterns[i,flip_idx]=-patterns[i,flip_idx]
  }
  return(patterns)
}

# Determine if two states are the same
states_match = function(state1, state2) {
  all(state1 == state2)
}

# Simulate asynchronous update for asymmetric weights
simulate_asynchronous = function(state, weights) {
  num_nodes = length(state)
  update_order = sample(1:num_nodes)  # random update order
  for (i in update_order) {
    net_i = sum(weights[i, ] * state)
    state[i] = ifelse(net_i >= 0, 1, -1)
  }
  return(as.numeric(state))
}

# Determine whether AHN goes through entire sequence of memories, starting
# at the first one
# Takes in 2D matrix of memories
# Check if AHN goes through all memories in order
check_sequence_order = function(weights, memories, max_steps = 100) {
  cur_state = memories[1, ]
  L = nrow(memories)
  target_mem = 2
  
  for (step in 1:max_steps) {
    # Sync update
    # Multiply weights by current state and take the sign
    net_input = weights %*% cur_state
    cur_state = as.numeric(ifelse(net_input >= 0, 1, -1))
    
    # Check distance to the next memory in the sequence
    dist_to_target = sum(cur_state != memories[target_mem, ]) / length(cur_state)
    
    # If we are close (20% error margin), we've successfully transitioned.
    if (dist_to_target <= 0.20) {
      target_mem = target_mem + 1
      if (target_mem > L) return(TRUE) 
    }
  }
  return(FALSE)
}

# Sequence memory capacity test
# Plot percent of networks that succeed vs length
# First iterate over lengths, then iterate over trials
sequence_capacity_test=function(num_nodes,max_length=25,trials=50){
  lengths=seq(3,max_length,by=3)
  success_rate=numeric(length(lengths))
  trial_success=numeric(trials)
  for(i in seq_along(lengths)){
    L=lengths[i]
    for(t in 1:trials){
      patterns = random_nodes_sequence(L, num_nodes, correlation=0.2)
      weights=fixed_weights_asymmetric_sequence(patterns)
      trial_success[t]=check_sequence_order(weights,patterns)
    }
    success_rate[i]=mean(trial_success)
  }

  plot(lengths, success_rate, 
       type = "l", 
       xlab = "Sequence Length", 
       ylab = "Success Rate", 
       main = paste("Sequence Memory Capacity (Number of Neurons = ", num_nodes, ")", sep = ""),
       cex.main = 0.9)
  grid()
  return(success_rate)
}
