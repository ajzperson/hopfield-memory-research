# ==============================================================================
# SCRIPT: ClassicHopfieldHomoDigitRec.R (Storkey, Random)
# DESCRIPTION: 
#   Homogeneous Digit Recognition Test. This script evaluates the stability of 
#   the Hopfield Network when trained on multiple variations of the SAME digit 
#   class (e.g., five different versions of a '3'). It tests if the network 
#   can maintain distinct sub-attractors within a single category or if 
#   they collapse into a single "average" representation.
# 
# Discrete Hopfield Network (Bipolar -1, 1)
# Storkey Rule (Iterative decorrelation)
# Random (asynchronous)
# ==============================================================================

source("core/ClassicHopfieldNetworkCore.R")

run_digit_test = function(digit_to_test, num_patterns = 5, frac_scramble = 0.35) {
  
  folder_path <- paste0("digits/", digit_to_test, "/")
  
  # 1. Load homogeneous patterns (versions of the same digit)
  all_files <- list.files(folder_path, pattern = "\\.png$", full.names = TRUE)
  
  if(length(all_files) < num_patterns) {
    stop("Not enough images in folder for digit ", digit_to_test, ". Found: ", length(all_files))
  }
  
  selected_files <- sample(all_files, num_patterns)
  cat("\n--- Loading", num_patterns, "homogeneous patterns for digit:", digit_to_test, "---\n")
  
  pattern_list <- list()
  pattern_matrix <- NULL
  
  for (i in 1:num_patterns) {
    # Using maxSize = 100 for high-resolution neuron mapping
    img_data <- processImg(selected_files[i], maxSize = 100)
    vec <- pattern_to_vector(img_data$matrix)
    
    # Ensure strict bipolar format for discrete state transitions
    vec <- ifelse(vec <= 0, -1, 1)
    
    pattern_list[[i]] <- vec
    pattern_matrix <- rbind(pattern_matrix, vec)
  }
  
  height <- img_data$height
  width  <- img_data$width
  num_pixels <- length(pattern_list[[1]])
  
  # 2. Train (Storkey rule)
  # Storkey is used here to minimize overlap interference between similar shapes
  cat("Training Storkey weights on homogeneous set...\n")
  weights_storkey <- fixed_weights_storkey(pattern_matrix)
  
  # 3. Prepare test case
  target_pattern <- pattern_list[[1]] 
  scrambled <- scramble_pattern(target_pattern, frac = frac_scramble)
  scrambled_mat <- matrix(scrambled, nrow = height, ncol = width)
  
  # 4. Execute recovery
  # Random mode is used to ensure the network settles into a local energy minimum.
  final_state = decode_with_hamming(
    state = scrambled, 
    weights = weights_storkey, 
    target = target_pattern, 
    img_w = width, 
    img_h = height, 
    mode = "random", 
    save_gif = TRUE, 
    filename = paste0("recovery_homo_digit_", digit_to_test, ".gif"),
    max_steps = num_pixels * 10, 
    delay = 0 
  )
  
  # 5. Final visualization (Original -> Scrambled -> Recovered)
  if (!is.null(dev.list())) dev.off() 
  par(mfrow = c(1, 3), mar = c(2, 2, 2, 1), oma = c(0, 0, 3, 0), pty = "s")
  
  # Plot target
  image(matrix(target_pattern, nrow = height), col = c("black", "white"), ann = FALSE)
  title("Original")
  
  # Plot scrambled input
  image(scrambled_mat, col = c("black", "white"), ann = FALSE)
  title(paste0("Scrambled (", frac_scramble*100, "%)"))
  
  # Plot final recovered state
  final_mat <- matrix(final_state, nrow = height, ncol = width)
  image(final_mat, col = c("black", "white"), ann = FALSE)
  title("Recovered")
  
  mtext(paste("Classic Hopfield Homo-Digit Recognition (Storkey + Random): Digit", digit_to_test, "(", num_patterns, "Samples )"), 
        outer = TRUE, line = 0.5, cex = 0.8, font = 2)
  
  cat("Process complete for homogeneous digit", digit_to_test, "\n")
  return(invisible(final_state)) 
}