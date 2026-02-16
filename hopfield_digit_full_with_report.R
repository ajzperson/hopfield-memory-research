# Hopfield Network for Digit Recognition (All-in-One Script with Accuracy Report)

library(magick)
library(grid)
library(ggplot2)

#----------------------------
# IMAGE GENERATION & LOADING
#----------------------------

generate_digit_images = function(output_dir = "digits", per_digit = 30, size = 100) {
  dir.create(output_dir, showWarnings = FALSE)
  for (digit in 0:9) {
    digit_dir = file.path(output_dir, as.character(digit))
    dir.create(digit_dir, showWarnings = FALSE)
    for (i in 1:per_digit) {
      img = image_blank(size, size, color = "white")
      img = image_annotate(img, text = as.character(digit), size = size * 0.8, color = "black",
                           gravity = "center", font = "Arial")
      img = image_rotate(img, runif(1, -20, 20))
      img = image_resize(img, paste0(sample(seq(0.9, 1.1, by = 0.01), 1) * size))
      img = image_extent(img, paste0(size, "x", size), gravity = "center")
      x_shift = sample(-10:10, 1)
      y_shift = sample(-10:10, 1)
      img = image_roll(img, paste0(x_shift, "x", y_shift))
      image_write(img, path = file.path(digit_dir, paste0("img_", i, ".png")))
    }
  }
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
# HOPFIELD NETWORK FUNCTIONS
#----------------------------

storkey_learn = function(patterns) {
  n = ncol(patterns)
  weights = matrix(0, n, n)
  for (p in 1:nrow(patterns)) {
    x = patterns[p, ]
    h = weights %*% x
    outer_x = outer(x, x)
    outer_h = outer(x, h) + outer(h, x)
    weights = weights + (outer_x - outer_h) / n
  }
  diag(weights) = 0
  return(weights)
}

simulate_until_fixed_sync = function(state, weights, max_iter = 100) {
  for (i in 1:max_iter) {
    new_state = sign(weights %*% state)
    new_state[new_state == 0] = 1
    if (all(new_state == state)) break
    state = new_state
  }
  return(state)
}

hamming_distance = function(a, b) {
  sum(a != b)
}

scramble_pattern = function(pattern, noise_level = 0.2) {
  n = length(pattern)
  flip_n = round(n * noise_level)
  idx = sample(1:n, flip_n)
  scrambled = pattern
  scrambled[idx] = -scrambled[idx]
  return(scrambled)
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

test_digit_accuracy = function(patterns, labels, weights, noise_level = 0.2) {
  correct = 0
  for (i in 1:nrow(patterns)) {
    orig = patterns[i, ]
    scrambled = scramble_pattern(orig, noise_level)
    recovered = simulate_until_fixed_sync(scrambled, weights)
    if (hamming_distance(orig, recovered) == 0) {
      correct = correct + 1
    }
  }
  accuracy = correct / nrow(patterns)
  return(list(accuracy = accuracy))
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
# VISUALIZATION (Optional)
#----------------------------

plot_state = function(state, size = 100) {
  matrix_state = matrix(state, nrow = size, byrow = TRUE)
  df = expand.grid(x = 1:size, y = 1:size)
  df$val = as.factor(as.vector(matrix_state))
  ggplot(df, aes(x, y, fill = val)) + geom_tile() + scale_fill_manual(values = c("-1" = "black", "1" = "white")) +
    theme_void() + theme(legend.position = "none") + coord_fixed()
}

decode_with_hamming = function(scrambled, weights, target) {
  steps = list()
  dists = c()
  state = scrambled
  for (i in 1:20) {
    steps[[i]] = state
    dists[i] = hamming_distance(state, target)
    new_state = sign(weights %*% state)
    new_state[new_state == 0] = 1
    if (all(new_state == state)) break
    state = new_state
  }
  par(mfrow = c(2, ceiling(length(steps)/2)))
  for (s in steps) {
    image(matrix(ifelse(s == 1, 1, 0), nrow = 100), col = gray.colors(2), axes = FALSE)
  }
  plot(dists, type = "b", main = "Hamming Distance", xlab = "Iteration", ylab = "Distance")
}

#----------------------------
# ACCURACY REPORT FUNCTION
#----------------------------

print_accuracy_report <- function(accuracy_result, noise_level) {
  cat("=== Hopfield Network Digit Recognition Accuracy Report ===\n")
  cat("Noise Level (fraction of flipped pixels):", noise_level, "\n")
  cat("Accuracy:", round(accuracy_result$accuracy * 100, 2), "%\n")
  cat("---------------------------------------------------------\n")
}

#----------------------------
# MAIN EXECUTION
#----------------------------

generate_digit_images()
digit_paths = get_digit_paths("digits")
digit_data = train_digit_variants(digit_paths, per_digit = 5)
patterns = digit_data$patterns
labels = digit_data$labels
weights = storkey_learn(patterns)

accuracy_result = test_digit_accuracy(patterns, labels, weights, noise_level = 0.2)
print_accuracy_report(accuracy_result, noise_level = 0.2)

i = sample(1:nrow(patterns), 1)
original = patterns[i, ]
scrambled = scramble_pattern(original, 0.2)
steps = time_to_converge_sync(scrambled, weights)
cat("Convergence steps:", steps, "\n")

decode_with_hamming(scrambled, weights, original)
source("hopfield_digit_full_with_report.R")
