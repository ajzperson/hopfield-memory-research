# ==============================================================================
# File: HopfieldHeatMap.R
# Purpose: Capacity stress test and 3D visualization
#
# Description:
#   This script serves as an experimental harness for a Classic Hopfield Network.
#   It measures the network's storage limits by testing its ability to recover 
#   original memories from noisy inputs across various network sizes and 
#   connection densities.
#
# Workflow:
#   - Learning: Uses the Hebbian rule (sum of outer products) to set weights.
#   - Retrieval: Employs sequential (asynchronous) updates until convergence.
#   - Evaluation: Determines the "breaking point" where the network can no 
#     longer attract a 3-bit perturbed state back to its original memory.
#
# TODO - PARAMETERIZATION
# ==============================================================================

# Dependencies:
#   - ClassicHopfieldNetworkCore.R: Required for core network functions.
#   - Libraries: plotly, htmlwidgets.
source("core/ClassicHopfieldNetworkCore.R")

library(plotly)
library(htmlwidgets)

# Parameters:
#   nodes_perturbed: 3 (The noise threshold for the basin of attraction).
#   trials: 5 (Averaged to find the mean capacity per configuration).
# TODO - robust input for the parameters
trials <- 20
nodes_perturbed <- 3
num_nodes <- seq(10, 200, by = 10)
density <- seq(0.02, 0.20, by = 0.02)

# Create result matrix
z = matrix(0, nrow = length(density), ncol = length(num_nodes))

for (i in seq_along(density)) {
  for (j in seq_along(num_nodes)) {
    d = density[i]
    n = num_nodes[j]
    num_patterns = round(d * n)
    trial_avgs = numeric(trials)
    
    for (t in 1:trials) {
      max_memories = 1
      hit_capacity = FALSE
      
      while (!hit_capacity) {
        fixedStates = random_nodes(max_memories, n)
        weights = fixed_weights_hebbian(fixedStates)
        
        for (p in 1:max_memories) {
          original = as.numeric(fixedStates[p, ])
          mixed = flip(original, nodes_perturbed)
          decoded = simulate_until_fixed_sequential(mixed, weights)
          if (!all(decoded == original)) {
            hit_capacity = TRUE
            break
          }
        }
        
        if (!hit_capacity) max_memories = max_memories + 1
      }
      trial_avgs[t] = max_memories - 1
    }
    z[i, j] = mean(trial_avgs)
  }
}

# Return grid for 3D plotting
res <- list(x = num_nodes, y = density, z = z)

# 3D Plot using base R
# persp(res$x, res$y, t(res$z),
#       theta = 40, phi = 180,
#       expand = 0.6,
#       col = "lightblue",
#       ltheta = 120,
#       shade = 0.75,
#       ticktype = "detailed",
#       xlab = "# of Nodes",
#       ylab = "Memory Density",
#       zlab = "Memory Capacity",
#       main = "3D Heatmap: Memory Capacity")

p <- plot_ly(x = res$x, y = res$y, z = res$z) %>%
  add_surface(colorscale = "Viridis") %>%
  layout(
    title = "Memory Capacity Surface Plot",
    scene = list(
      xaxis = list(title = "# of Nodes"),
      yaxis = list(title = "Memory Density"),
      zaxis = list(title = "Memory Capacity")
    )
  )

saveWidget(as_widget(p), "ClassicH_Capacity_3D.html", selfcontained = TRUE)
browseURL("ClassicH_Capacity_3D.html")
