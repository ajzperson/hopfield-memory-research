# Required parameters that i added everything else is same
trials <- 5
nodes_perturbed <- 3
num_nodes <- seq(10, 100, by = 10)
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
persp(res$x, res$y, res$z,
      theta = 40, phi = 180,
      expand = 0.6,
      col = "lightblue",
      ltheta = 120,
      shade = 0.75,
      ticktype = "detailed",
      xlab = "# of Nodes",
      ylab = "Memory Density",
      zlab = "Memory Capacity",
      main = "3D Heatmap: Memory Capacity")

library(plotly)
library(htmlwidgets)

# # Your original interactive plot (example)
# p <- plot_ly(z = ~volcano) %>% add_surface()
# 
# # Save and open in your default browser
# saveWidget(as_widget(p), "plotly_surface.html", selfcontained = TRUE)
# browseURL("plotly_surface.html")

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

saveWidget(as_widget(p), "hopfield_surface.html", selfcontained = TRUE)
browseURL("hopfield_surface.html")
