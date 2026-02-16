# ==============================================================================
# CLASSIC HOPFIELD NETWORK BENCHMARK: HEBBIAN VS. STORKEY LEARNING
# (Hebbian vs. Storkey, Sequential)
# ==============================================================================
# File: ClassicHopfieldHebbianStorkey.R
# Core Logic: Classic Hopfield Network
#
# Description:
#   Demonstrate superior storage and interference resistance of Storkey.
#
# Workflow:
#   1. Source core Hopfield logic and dependencies.
#   2. Initialize a 400-neuron grid (20x20) for bipolar state storage.
#   3. Generate a training set of 8 highly correlated (overlapping) patterns.
#   4. Compute weight matrices using both Hebbian and Storkey learning rules.
#   5. Inject 40% noise, execute Sequential (asynchronous) updates.
#   7. Compare recovery fidelity and Hamming distances to identify spurious states.
#
# Expected Outputs: 
#   - Hebbian Fail: Simple summation leads to 'Spurious States' (attractor blurring).
#   - Storkey High-Fidelity: Local Field (h) subtraction decorrelates memories,
#     maintaining the target structure even when the basin is slightly distorted.
# 
# TODO - PARAMETERIZATION
# ==============================================================================

# 1. Load libraries and Classic Hopfield engine
library(gridExtra)
library(grid)

source("core/ClassicHopfieldNetworkCore.R")

# 2. Setup: 20x20 grid (400 neurons)
grid_dim <- 20
n_nodes <- grid_dim^2

# Pattern generation
p1_mat <- matrix(-1, grid_dim, grid_dim)
for(i in 1:grid_dim) { 
  p1_mat[i, i] <- 1; p1_mat[i, (grid_dim-i+1)] <- 1 
}
p1_vec <- as.vector(p1_mat) 

train_list <- list(p1_vec)
for(k in 1:7) {
  tmp <- matrix(-1, grid_dim, grid_dim)
  tmp[ (5+k):(15-k), 8:12 ] <- 1
  tmp[ 8:12, (5+k):(15-k) ] <- 1
  train_list[[k+1]] <- as.vector(tmp)
}
training_set <- do.call(rbind, train_list)

# 3. Training
w_hebb <- fixed_weights_hebbian(training_set)
w_storkey <- fixed_weights_storkey(training_set)

# 4. Scramble w/ 40% Noise 
set.seed(123) 
scrambled_x <- flip(p1_vec, round(n_nodes * 0.40))

# 5. Recovery
rec_hebb <- simulate_until_fixed_sequential(scrambled_x, w_hebb)
rec_storkey <- simulate_until_fixed_sequential(scrambled_x, w_storkey)

# 6. Visualization
render = function(vec, title) {
  df = expand.grid(X = 1:grid_dim, Y = 1:grid_dim)
  df$val = as.vector(t(matrix(vec, nrow=grid_dim, byrow=T)))
  ggplot(df, aes(x=X, y=Y, fill=factor(val))) +
    geom_tile() + scale_fill_manual(values=c("-1"="black", "1"="white")) +
    theme_void() + theme(legend.position="none", plot.title=element_text(hjust=0.5, face="bold", size=9)) +
    ggtitle(title) + coord_fixed()
}

# Header
header_text <- paste(
  "Classic Hopfield Network Capacity Analysis: Hebbian vs. Storkey, Sequential",
  "\nExperiment: 400 Neurons | 8 Correlated Patterns | 40% Noise",
  "\nHebbian: High Crosstalk forces convergence to a Spurious State.",
  "\nStorkey: Local Field cancellation enables High-Fidelity pattern recovery.",
  sep = ""
)

grid.arrange(
  render(p1_vec, "Target Pattern"),
  render(scrambled_x, "40% Noisy Input"),
  render(rec_hebb, "Hebbian (Spurious State)"),
  render(rec_storkey, "Storkey (High-Fidelity)"),
  ncol = 2,
  top = textGrob(header_text, gp = gpar(fontsize = 10.5, font = 2, lineheight = 1.1))
)

# 7. Metrics
cat("Hebbian Hamming Distance:", hamming_distance(rec_hebb, p1_vec), "\n")
cat("Storkey Hamming Distance:", hamming_distance(rec_storkey, p1_vec), "\n")