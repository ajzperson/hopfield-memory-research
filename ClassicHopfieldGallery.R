# ==============================================================================
# CLASSIC HOPFIELD NETWORK: NOISE RESILIENCE GALLERY (Multi-Rule, Multi-Mode)
# ==============================================================================
# File: ClassicHopfieldGallery.R
# Core Logic: Classic Hopfield Network
#
# Description:
#   This script evaluates the robustness of the network by testing its ability 
#   to reconstruct a stored pattern under varying levels of discrete 
#   bit-flip noise (e.g., 10%, 30%, and 50%). It supports Hebbian or Storkey 
#   learning rules and Sequential, Random, or Synchronous update.
#
# Workflow:
#   1. Trains the weight matrix using the selected rule on a single image 
#      plus a specified number of random distractor patterns.
#   2. Creates corrupted versions of the target image at three noise levels.
#   3. Executes the chosen recovery mode (Sequential, Random, or Sync) until 
#      a fixed-point stable state is reached.
#   4. Generates a 2x4 comparison grid of Inputs vs. Recovered Outputs.
#
# Expected Outputs:
#   - R Plot: A 2x4 grid display. 
#       * Top Row: The original image followed by three noisy versions.
#       * Bottom Row: Info panel followed by the network's three "recovered" results.
# ==============================================================================

# Load core logic
source("core/ClassicHopfieldNetworkCore.R")

generate_classic_hopfield_gallery <- function(learning_rule = "Hebbian", 
                                              recovery_mode = "Sequential",
                                              noise_1 = 0.48, 
                                              noise_2 = 0.49, 
                                              noise_3 = 0.5, 
                                              num_distractors = 15) {
  # Internal validation function
  validate_noise = function(n, label, default_val) {
    if (n < 0 || n > 1) {
      cat("Warning: ", label, " must be between 0 and 1. Falling back to default: ", default_val, "\n")
      return(default_val) 
    }
    return(n)
  }
  
  # Apply validation
  nf1 = validate_noise(noise_1, "noise_frac1", 0.1)
  nf2 = validate_noise(noise_2, "noise_frac2", 0.3)
  nf3 = validate_noise(noise_3, "noise_frac3", 0.5)
  
  if (num_distractors < 0) {
    cat("Warning: num_distractors must be >= 0. Falling back to default: 15\n")
    num_distractors = 15
  }
  
  total_patterns = num_distractors + 1

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

  # Load and process image
  img_data = processImg(img_path, maxSize = 100)
  img_matrix = img_data$matrix 

  # Capture exact dimensions from the processed matrix
  height = nrow(img_matrix)
  width = ncol(img_matrix)
  pattern = pattern_to_vector(img_matrix)
  num_pixels = length(pattern)

  # Generate distractors and combine into a matrix
  distractor_matrix = random_nodes(num_distractors, num_pixels)
  all_patterns = rbind(pattern, distractor_matrix)
  
  # Train the network
  if (tolower(learning_rule) == "storkey") {
    weights = fixed_weights_storkey(all_patterns)
  } else {
    weights = train_weights_from_matrix(all_patterns)
  }

  # 4. Generate test cases (noise)
  noise_v1 = scramble_pattern(pattern, frac = nf1)
  noise_v2 = scramble_pattern(pattern, frac = nf2)
  noise_v3 = scramble_pattern(pattern, frac = nf3)

  mode_lc = tolower(recovery_mode)
  if (mode_lc == "sync") {
    rec_func = simulate_until_fixed_sync
  } else if (mode_lc == "random") {
    rec_func = simulate_until_fixed_random
  } else {
    rec_func = simulate_until_fixed_sequential
  }
  rec_10 = rec_func(noise_v1, weights)
  rec_30 = rec_func(noise_v2, weights)
  rec_50 = rec_func(noise_v3, weights)

  # 6. Synchronized display helper
  show = function(v) {
    m = matrix(v, nrow = height, ncol = width, byrow = FALSE)
    return(m)
  }

  # 7. Final gallery
  if(!is.null(dev.list())) dev.off() 
  par(mfrow = c(2, 4), mar = c(1, 1, 3, 1))

  # --- Row 1: Inputs ---
  image(img_matrix, col = c("black", "white"), main = "Original", axes = FALSE)
  image(show(noise_v1), col = c("black", "white"), main = paste0(nf1*100, "% Noise"), axes = FALSE)
  image(show(noise_v2), col = c("black", "white"), main = paste0(nf2*100, "% Noise"), axes = FALSE)
  image(show(noise_v3), col = c("black", "white"), main = paste0(nf3*100, "% Noise"), axes = FALSE)

  # --- Row 2: Outputs ---
  plot.new() 
  text(0.5, 0.8, paste(tools::toTitleCase(tolower(learning_rule)), "Rule"), cex = 1, font = 2)
  text(0.5, 0.6, paste(tools::toTitleCase(tolower(recovery_mode)), "Recovery"), cex = 1, font = 2)
  text(0.5, 0.4, paste(total_patterns, "Patterns"), cex = 1.0)
  text(0.5, 0.2, paste("(1 Image +", num_distractors, "Distractors)"), cex = 0.8)

  image(show(rec_10), col = c("black", "white"), main = paste0("Rec ", nf1*100, "%"), axes = FALSE)
  image(show(rec_30), col = c("black", "white"), main = paste0("Rec ", nf2*100, "%"), axes = FALSE)
  image(show(rec_50), col = c("black", "white"), main = paste0("Rec ", nf3*100, "%"), axes = FALSE)
}