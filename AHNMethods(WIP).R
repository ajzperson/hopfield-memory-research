# Asymmetric Hopfield Network
# Contains all necessary methods to use AHN

# Perhaps add noise tolerance?


# Generate random set of starting weights (either 1 or -1) but not symmetric
random_weights_asymmetric=function(start_network){
  num_nodes=length(start_network)
  random_matrix=matrix(sample(c(1,-1),num_nodes*num_nodes,replace=TRUE),nrow=num_nodes,ncol=num_nodes)
  return(random_matrix)
}

# Network learns the sequence of fixed states, not any particular one
fixed_weights_asymmetric_sequence=function(patterns){
  num_patterns=nrow(patterns)
  num_nodes=ncol(patterns)
  w=(t(patterns[1:(num_patterns-1),,drop=FALSE]) %*% patterns[2:num_patterns,,drop=FALSE]) / num_nodes
  diag(w)=0
  w = w / max(abs(w)) 
  return (w)
}

# Generates a sequence of nodes with 90% correlation
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

# Determines if two states are the same
states_match = function(state1, state2) {
  all(state1 == state2)
}

# Simulates asynchronous update for asymmetric weights
simulate_asynchronous = function(state, weights) {
  num_nodes = length(state)
  update_order = sample(1:num_nodes)  # random update order
  for (i in update_order) {
    net_i = sum(weights[i, ] * state)
    state[i] = ifelse(net_i >= 0, 1, -1)
  }
  return(as.numeric(state))
}

# Determines whether AHN goes through entire sequence of memories, starting
# at the first one. Takes in 2D matrix of memories.
# Check if AHN goes through all memories in order
check_sequence_order = function(weights, memories, max_steps = 100) {
  max_steps = 3*nrow(memories)*ncol(memories)
  cur_state = memories[1, ]
  L = nrow(memories)
  cur_mem = 1
  
  for (step in 1:max_steps) {
    cur_state = simulate_asynchronous(cur_state, weights)
    
    # Check if we have transitioned close to the next memory
    if (sum(cur_state != memories[cur_mem, ]) <= 0.05 * length(cur_state)) {
      cur_mem = cur_mem + 1
      if (cur_mem > L) {
        return(TRUE)  # completed sequence
      }
    }
  }
  return(FALSE)
}



# Sequence Memory Capacity Test
# Plot percent of networks that succeed vs length
# First iterate over lengths, then iterate over trials
sequence_capacity_test=function(num_nodes,max_length=25,trials=50){
  lengths=seq(3,max_length,by=3)
  success_rate=numeric(length(lengths))
  trial_success=numeric(trials)
  for(i in seq_along(lengths)){
    L=lengths[i]
    for(t in 1:trials){
      patterns = random_nodes_sequence(L, num_nodes, correlation=0.8)
      weights=fixed_weights_asymmetric_sequence(patterns)
      trial_success[t]=check_sequence_order(weights,patterns)
    }
    success_rate[i]=mean(trial_success)
  }
  plot(lengths,success_rate,type="l",xlab="Sequence Length",ylab="Success Rate",main="Sequence Memory Capacity")
  return(success_rate)
}

sequence_capacity_test(50)