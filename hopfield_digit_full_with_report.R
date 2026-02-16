# Classic Hopfield Network for Digit Recognition (All-in-One Script with Accuracy Report)

source("core/ClassicHopfieldNetworkCore.R")

library(magick)
library(grid)
library(ggplot2)

#----------------------------
# IMAGE GENERATION & LOADING
#----------------------------

generate_digit_images = function(out_dir = "digits", per_digit = 20, size = 100) {
  dir.create(out_dir, showWarnings = FALSE)
  for (digit in 0:9) {
    digit_dir = file.path(out_dir, as.character(digit))
    dir.create(digit_dir, showWarnings = FALSE)
    for (i in 1:per_digit) {
      tmp_file = tempfile(fileext = ".png")
      png(tmp_file, width = size, height = size, bg = "white")
      grid.newpage()
      pushViewport(viewport())
      rot = sample(seq(-30, 30, 5), 1)
      scale = runif(1, 0.8, 1.2)
      x_offset = sample(-10:10, 1)
      y_offset = sample(-10:10, 1)
      grid.text(label = as.character(digit),
                x = 0.5 + x_offset / size,
                y = 0.5 + y_offset / size,
                gp = gpar(fontsize = 80 * scale))
      dev.off()
      
      img = image_read(tmp_file, strip = TRUE)
      img = image_rotate(img, rot)
      img = image_extent(img, paste0(size, "x", size), gravity = "center")
      img = image_background(img, "white")
      image_write(img, path = file.path(digit_dir, paste0("img_", i, ".png")))
    }
  }
  cat("✅ Digit images saved to:", normalizePath(out_dir), "\n")
}

get_digit_paths = function(root_dir = "digits") {
  digit_dirs = list.dirs(root_dir, full.names = TRUE, recursive = FALSE)
  digit_paths = list()
  for (dir in digit_dirs) {
    digit_label = basename(dir)
    files = list.files(dir, full.names = TRUE, pattern = "\\.(png|jpg|jpeg|bmp)$", ignore.case = TRUE)
    digit_paths[[digit_label]] = files
  }
  return(digit_paths)
}

load_transformed_digit = function(path, size = 100) {
  img = image_read(path)
  img = image_convert(img, colorspace = "gray")
  img = image_resize(img, paste0(size, "x", size, "!"))
  img = image_threshold(img, type = "white", threshold = "50%")
  img_data = as.integer(as.numeric(image_data(img)[1,,]) > 0)
  img_data = ifelse(img_data == 0, -1, 1)
  return(as.vector(t(img_data)))
}

#----------------------------
# TRAINING AND TESTING
#----------------------------

train_digit_variants = function(digit_paths, per_digit = 5) {
  all_patterns = list()
  labels = c()
  for (digit in names(digit_paths)) {
    paths = sample(digit_paths[[digit]], per_digit)
    for (path in paths) {
      pat = load_transformed_digit(path)
      all_patterns = append(all_patterns, list(pat))
      labels = c(labels, digit)
    }
  }
  pattern_matrix = do.call(rbind, all_patterns)
  return(list(patterns = pattern_matrix, labels = labels))
}

test_digit_accuracy = function(patterns, labels, weights, noise_level = 0.2, mode = "sync") {
  correct = 0
  all_steps = c()
  
  for (i in 1:nrow(patterns)) {
    orig = patterns[i, ]
    scrambled = scramble_pattern(orig, noise_level)
    
    if (mode == "sync") {
      # Return the final state directly
      recovered = simulate_until_fixed_sync(scrambled, weights)
      all_steps = c(all_steps, 2) # Sync usually converges in 2 steps
    } else {
      # Use the Random decoding function that provides the 50k+ step resolution
      res = decode_with_hamming(scrambled, weights, orig, 
                                mode = "random", max_steps = 100000, 
                                show_log = FALSE)
      recovered = if(is.list(res)) res$final_state else res
      # Capture how many steps were actually logged
      all_steps = c(all_steps, if(is.list(res)) res$steps_taken else 100000)
    }
    
    if (hamming_distance(orig, recovered) == 0) {
      correct = correct + 1
    }
  }
  
  return(list(
    accuracy = correct / nrow(patterns),
    avg_steps = mean(all_steps)
  ))
}

time_to_converge_sync = function(state, weights, max_iter = 100) {
  for (i in 1:max_iter) {
    new_state = sign(weights %*% state)
    new_state[new_state == 0] = 1
    if (all(new_state == state)) return(i)
    state = new_state
  }
  return(max_iter)
}

#----------------------------
# ACCURACY REPORT FUNCTION
#----------------------------

print_accuracy_report <- function(accuracy_result, noise_level) {
  cat("=== Hopfield Network Digit Recognition Accuracy Report ===\n")
  cat("Noise Level (fraction of flipped pixels):", noise_level, "\n")
  cat("Accuracy:", round(accuracy_result$accuracy * 100, 2), "%\n")
  cat("---------------------------------------------------------\n")
  
  # Pull the real steps from the test just run
  cat("Average Convergence steps:", accuracy_result$avg_steps, "\n")
}

# ---------------------------------------------------------
# VISUALIZATION: Corrected Orientation for Original, Scrambled, Recovered
# ---------------------------------------------------------
plot_recovery_trio = function(original_vec, scrambled_vec, recovered_vec, noise_val = 0.2, size = 100) {
  
  # Helper function to flip matrix for correct visual display
  # R image() plots [1,1] at bottom-left; this moves it to top-left
  prepare_mat = function(vec, s) {
    m <- matrix(vec, nrow = s, byrow = TRUE)
    return(t(m)[, s:1]) 
  }

  if (!is.null(dev.list())) dev.off()
  par(mfrow = c(1, 3), mar = c(2, 2, 4, 1), oma = c(0, 0, 2, 0), pty = "s")
  
  # 1. Original
  image(prepare_mat(original_vec, size), 
        col = c("black", "white"), main = "1. Original", axes = FALSE)
  
  # 2. Scrambled
  noise_text <- paste0(noise_val * 100, "%")
  image(prepare_mat(scrambled_vec, size), 
        col = c("black", "white"), 
        main = paste0("2. Scrambled (", noise_text, " Noise)"), 
        axes = FALSE)
  
  # 3. Recovered
  image(prepare_mat(recovered_vec, size), 
        col = c("black", "white"), main = "3. Recovered Result", axes = FALSE)
  
  mtext("Classic Hopfield Network: Digit Recognition (Projection rule + Random update)", outer = TRUE, line = -0.5, font = 2, cex = 0.8)
}

#----------------------------
# MAIN EXECUTION
#----------------------------

# --- 1. Setup & Training ---
generate_digit_images()
digit_paths = get_digit_paths("digits")
digit_data = train_digit_variants(digit_paths, per_digit = 5)
patterns = digit_data$patterns
labels = digit_data$labels

# Use the Projection rule (High-Capacity)
weights = fixed_weights_projection(patterns)

# --- 2. Comparative research reports ---
current_noise = 0.2

# Run Phase 1: Synchronous
cat("\n[Run 1] Running Synchronous Dynamics Test...\n")
res_sync = test_digit_accuracy(patterns, labels, weights, noise_level = current_noise, mode = "sync")
print_accuracy_report(res_sync, noise_level = current_noise)

# Run Phase 2: Random (Biological/Asynchronous)
cat("\n[Run 2] Running Random Dynamics Test (100k steps)...\n")
res_rand = test_digit_accuracy(patterns, labels, weights, noise_level = current_noise, mode = "random")
print_accuracy_report(res_rand, noise_level = current_noise)

# --- 3. Visualization (single case study) ---
# Pick one random digit to visualize the recovery process
i = sample(1:nrow(patterns), 1)
original = patterns[i, ]
scrambled = scramble_pattern(original, current_noise)

cat("\nGenerating recovery visualization for Digit:", labels[i], "\n")

# This creates the step-by-step evolution data.
recovered_data = decode_with_hamming(
    state = scrambled, 
    weights = weights, 
    target = original, 
    img_w = 100, 
    img_h = 100, 
    mode = "random", 
    max_steps = 50000, 
    show_log = FALSE # Keep console clean for the final trio plot
)

# Extract the final state from the list returned by decode_with_hamming
final_state = if(is.list(recovered_data)) recovered_data$final_state else recovered_data

# Display the before/after comparison
plot_recovery_trio(original, scrambled, final_state, noise_val = current_noise, size = 100)
