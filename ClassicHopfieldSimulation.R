# ==============================================================================
# CLASSIC HOPFIELD NETWORK: SIMULATION RUNNER (Multi-Rule, Multi-Mode)
# ==============================================================================
# File: ClassicHopfieldSimulation.R
# Core Logic: Classic Hopfield Network
#
# Description:
#   Execute a full pattern recovery pipeline supporting both Hebbian and Storkey
#   learning rules, and Sequential, Random, or Synchronous update dynamics.
#
# Workflow:
#   1. Source core Hopfield logic and dependencies.
#   2. Preprocess a target image into a bipolar {-1, 1} neural grid.
#   3. Calculate the weight matrix (Hebbian or Storkey) using the target 
#      pattern and a specified number of random distractors.
#   4. Corrupt the pattern with noise (scrambling).
#   5. Recover the pattern using the chosen mode (Sequential, Random, or Sync).
#   6. Export a GIF for asynchronous modes (Sequential/Random).
#
# Expected Outputs:
#   - R Plot: Initial preview of target pattern, followed by the scrambled 
#             version, then the recovered version.
#   - R Console: Real-time Hamming Distance tracking and convergence status.
#   - File: Dynamic GIF (e.g., 'recovery_classic_storkey_random.gif') showing 
#           the neural evolution (skipped for 'Sync' mode).
# ==============================================================================

source("core/ClassicHopfieldNetworkCore.R")

simulate_classic_hopfield <- function(learning_rule = "Hebbian", recovery_mode = "Sequential", noise_frac = 0.35, num_distractors = 15) {
  # parameter sanitization and input validation
  # Ensure learning_rule is valid; otherwise, reset to Hebbian
  if (!tolower(learning_rule) %in% c("hebbian", "storkey")) {
    cat("Warning: learning_rule must be 'Hebbian' or 'Storkey'. Falling back to default: Hebbian\n")
    learning_rule = "Hebbian"
  }
  # Ensure recovery_mode is valid; otherwise, reset to Sequential
  if (!tolower(recovery_mode) %in% c("sequential", "random", "sync")) {
    cat("Warning: recovery_mode must be 'Sequential', 'Random', or 'Sync'.\n")
    recovery_mode = "Sequential"
  }
  # Ensure noise_frac is between 0 and 1; otherwise, reset to 0.35
  if (noise_frac < 0 || noise_frac > 1) {
    cat("Warning: noise_frac must be between 0 and 1. Falling back to default: 0.35\n")
    noise_frac = 0.35
  }
  # Ensure num_distractors >= 0; otherwise, reset to 15
  if (num_distractors < 0) {
  cat("Warning: num_distractors must be >= 0. Falling back to default: 15\n")
  num_distractors = 15
  }
  # --------------------

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

  # Load the image
  img_data = processImg(img_path, maxSize = 100)
  # Extract the matrix to train 
  # img_matrix is "Original"
  img_matrix = img_data$matrix
  pattern = pattern_to_vector(img_matrix)
  num_pixels = length(pattern)

  # --- DISTRACTORS ---
  # Use the num_distractors variable passed into the function
  distractor_matrix = random_nodes(num_distractors, num_pixels)
  all_patterns = rbind(pattern, distractor_matrix)
  
  # Count total patterns for the title
  total_patterns = nrow(all_patterns)

  # Train the network
  if (tolower(learning_rule) == "storkey") {
    weights = fixed_weights_storkey(all_patterns)
  } else {
    weights = train_weights_from_matrix(all_patterns)
  }

  # Scramble 
  # scrambled_mat is "Scrambled"
  scrambled = scramble_pattern(pattern, frac = noise_frac)
  height <- nrow(img_matrix)
  width  <- ncol(img_matrix)
  scrambled_mat <- matrix(scrambled, nrow = height, ncol = width)

  #  Decode
  current_delay = ifelse(tolower(learning_rule) == "hebbian", 0.06, 0)
  dynamic_filename = paste0("recovery_classic_", tolower(learning_rule), "_", tolower(recovery_mode), ".gif")
  should_save_gif = tolower(recovery_mode) %in% c("sequential", "random")
  final_state = decode_with_hamming(
    scrambled, 
    weights, 
    pattern, 
    img_w = img_data$width, 
    img_h = img_data$height, 
    mode = tolower(recovery_mode),
    max_steps = num_pixels*10, 
    delay = current_delay,
    save_gif = should_save_gif, 
    filename = dynamic_filename
  )

  # Display "Original", "Scrambled", "Recovered"
  if(!is.null(dev.list())) dev.off()
  par(mfrow = c(1, 3), mar = c(2, 2, 2, 1), pty = "s")
  # Original
  image(img_matrix, col = c("black", "white"), xaxt = "s", yaxt = "s", ann = FALSE)
  title("Original")

  # "Scrambled"
  image(scrambled_mat, col = c("black", "white"), xaxt = "s", yaxt = "s", ann = FALSE)
  title(paste0("Scrambled ", noise_frac * 100, "%"))

  # Final State ("Recovered")
  final_mat <- matrix(final_state, nrow = height, ncol = width)
  image(final_mat, col = c("black", "white"), xaxt = "s", yaxt = "s", ann = FALSE, frame.plot = FALSE)
  title("Recovered")

  # Add the main overall title
  mtext(paste0("Classic Hopfield: ", tools::toTitleCase(tolower(learning_rule)), " Rule & ", tools::toTitleCase(tolower(recovery_mode)), " Recovery\n", 
             total_patterns, " Patterns: 1 Image + ", num_distractors, " Distractors"), 
      outer = TRUE, line = -2.5, cex = 0.8, font = 2)
}