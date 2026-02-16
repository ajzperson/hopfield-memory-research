# ==============================================================================
# MODERN CONTINUOUS HOPFIELD NETWORK: COLOR NOISE RESILIENCE GALLERY
# ==============================================================================
# File: ContinuousHopfieldGallery.R
# Core Logic: Modern Continuous Hopfield Network (Attention-based)
#
# Description:
#   This script evaluates the retrieval capabilities of the Continuous Hopfield 
#   Network (MHN) using color imagery. It tests how varying the Beta parameter 
#   (Softmax temperature) affects recovery under different Gaussian noise levels.
# ==============================================================================

source("utils/ColoredImageMemory.R")
source("core/ContinuousHopfieldNetworkCore.R")

generate_mhn_gallery <- function(noise_1 = 0.1, 
                                 noise_2 = 0.5, 
                                 noise_3 = 0.9, 
                                 beta_1 = 50, 
                                 beta_2 = 15, 
                                 beta_3 = 1.5, 
                                 num_distractors = 15) {
  
  # File selector
  img_path = tryCatch({
    cat("Waiting for file selection...\n")
    file.choose()
  }, error = function(e) {
    # Fallback if dialog is cancelled
    default_img = "assets/AZ-Koala.jpg"
    cat("Selection cancelled. Falling back to default:", default_img, "\n")
    return(default_img)
  })

  img_data = image_to_vectors(img_path, scale = "20%")
  target_vec = c(img_data$r, img_data$g, img_data$b)
  
  # 2. Memory setup (target + distractors)
  distractor_matrix = matrix(runif(num_distractors * length(target_vec)), nrow = num_distractors)
  memories = rbind(target_vec, distractor_matrix) 

  # 3. Generate noise function
  add_noise = function(v, sd_val) {
    pmin(pmax(v + rnorm(length(v), 0, sd_val), 0), 1)
  }
  
  noise_v1 = add_noise(target_vec, noise_1)
  noise_v2 = add_noise(target_vec, noise_2)
  noise_v3 = add_noise(target_vec, noise_3)

  # 4. Modern recovery (Calls attention_update inside core)
  rec_1 = simulate_until_fixed_continuous(noise_v1, memories, beta = beta_1)
  rec_2 = simulate_until_fixed_continuous(noise_v2, memories, beta = beta_2)
  rec_3 = simulate_until_fixed_continuous(noise_v3, memories, beta = beta_3)

  # 5. Energy analysis (Explicitly utilizing net_energy_continuous and lse)
  e1 = net_energy_continuous(beta_1, memories, rec_1)
  e2 = net_energy_continuous(beta_2, memories, rec_2)
  e3 = net_energy_continuous(beta_3, memories, rec_3)

  # 6. Helper for plotting
  prep_item = function(vec) {
    N = length(img_data$r)
    list(r = vec[1:N], g = vec[(N+1):(2*N)], b = vec[(2*N+1):(3*N)])
  }

  library(grid)
  grid.newpage()
  pushViewport(viewport(layout = grid.layout(2, 4)))

  grid_plot_mhn = function(vec, row, col, label, energy_val = NULL) {
    vec = pmax(vec, 0)
    pushViewport(viewport(layout.pos.row = row, layout.pos.col = col))
    item = prep_item(vec)
    img_array = array(0, dim = c(img_data$height, img_data$width, 3))
    img_array[,,1] = matrix(item$r, nrow = img_data$height, byrow = TRUE)
    img_array[,,2] = matrix(item$g, nrow = img_data$height, byrow = TRUE)
    img_array[,,3] = matrix(item$b, nrow = img_data$height, byrow = TRUE)
    
    grid.raster(img_array)
    
    pushViewport(viewport(y = unit(1, "npc") - unit(0.5, "lines"), height = unit(1.0, "lines")))
    grid.rect(gp = gpar(fill = rgb(0, 0, 0, 0.5), col = NA))
    display_text = if(!is.null(energy_val)) {
      paste0("Beta: ", sub("B: ", "", label), " (E: ", round(energy_val, 1), ")")
    } else {
      label
    }
    grid.text(display_text, gp = gpar(col="white", fontface="bold", cex=0.7))
    popViewport(); popViewport()
  }

  # --- Draw gallery ---
  grid_plot_mhn(target_vec, 1, 1, "Original")
  grid_plot_mhn(noise_v1,   1, 2, paste0(noise_1*100, "% Noise"))
  grid_plot_mhn(noise_v2,   1, 3, paste0(noise_2*100, "% Noise"))
  grid_plot_mhn(noise_v3,   1, 4, paste0(noise_3*100, "% Noise"))

  # Info panel
  pushViewport(viewport(layout.pos.row = 2, layout.pos.col = 1))
  total_patterns = num_distractors + 1
  grid.text(paste("Modern Continuous\n Hopfield\n", 
                  total_patterns, "Patterns\n",
                  "(1 Image +", num_distractors, "Distractors)"), 
                  gp = gpar(cex=0.7, fontface="italic"))
  popViewport()

  grid_plot_mhn(rec_1, 2, 2, paste0("B: ", beta_1), energy_val = e1)
  grid_plot_mhn(rec_2, 2, 3, paste0("B: ", beta_2), energy_val = e2)
  grid_plot_mhn(rec_3, 2, 4, paste0("B: ", beta_3), energy_val = e3)
}