#==============================
#====== LIBRARIES =============
#==============================
library(magick)
library(grid)
library(ggplot2)

#==============================
#====== DIGIT GENERATOR =======
#==============================
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

      img = image_read(tmp_file)
      img = image_rotate(img, rot)
      img = image_extent(img, paste0(size, "x", size), gravity = "center")
      img = image_background(img, "white")
      image_write(img, path = file.path(digit_dir, paste0("img_", i, ".png")))
    }
  }
  cat("✅ Digit images saved to:", normalizePath(out_dir), "\n")
}

#==============================
#====== PATH LOADER ===========
#==============================
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

#==============================
#====== IMAGE TO VECTOR =======
#==============================
load_transformed_digit = function(path, size = 100) {
  img = image_read(path)
  img = image_convert(img, colorspace = "gray")
  img = image_scale(img, paste0(size))
  img = image_threshold(img, type = "white", threshold = "50%")
  img = image_threshold(img, type = "black", threshold = "50%")
  pixels = as.raster(img)
  neurons = matrix(nrow = size, ncol = size)
  for (i in 1:size) {
    for (j in 1:size) {
      neurons[i, j] = ifelse(pixels[i, j] == "#FFFFFFFF", 1, -1)
    }
  }
  return(as.vector(t(neurons)))
}

#==============================
#====== HOPFIELD CORE =========
#==============================
flip = function(state, num_flips) {
  flip_indices = sample(1:length(state), num_flips, replace = FALSE)
  state[flip_indices] = -state[flip_indices]
  return(state)
}

storkey_learn = function(cur_network, weights, new_pattern) {
  num_nodes = length(cur_network)
  diag(weights) = 0
  h = weights %*% new_pattern
  for (i in 1:num_nodes) {
    for (j in 1:num_nodes) {
      if (i != j) {
        x = (1 / num_nodes) * (
          new_pattern[i] * new_pattern[j] -
            new_pattern[i] * h[j] -
            new_pattern[j] * h[i]
        )
        weights[i, j] = weights[i, j] + x
      }
    }
  }
  return(weights)
}

fixed_weights_storkey = function(fixed_states) {
  num_nodes = length(fixed_states[1,])
  weights = matrix(0, nrow = num_nodes, ncol = num_nodes)
  for (j in 1:nrow(fixed_states)) {
    weights = storkey_learn(fixed_states[j,], weights, fixed_states[j,])
  }
  return(weights)
}

simulate_until_fixed_sync = function(cur_network, weights, max_steps = 100) {
  for (step in 1:max_steps) {
    newNet = sign(weights %*% cur_network)
    newNet[newNet == 0] = 1
    if (identical(newNet, cur_network)) break
    cur_network = newNet
  }
  return(cur_network)
}

hamming_distance = function(p1, p2) {
  return(sum(p1 != p2))
}

#==============================
#====== TRAIN & TEST ==========
#==============================
train_digit_variants = function(digit_paths, variants_per_digit = 5, size = 100) {
  memories = list()
  labels = c()
  for (digit in 0:9) {
    paths = digit_paths[[as.character(digit)]]
    for (v in 1:variants_per_digit) {
      path = sample(paths, 1)
      img_vec = load_transformed_digit(path, size)
      memories[[length(memories) + 1]] = img_vec
      labels = c(labels, digit)
    }
  }
  memory_matrix = do.call(rbind, memories)
  return(list(patterns = memory_matrix, labels = labels))
}

scramble_pattern = function(pattern, frac = 0.2) {
  num_flips = round(length(pattern) * frac)
  return(flip(pattern, num_flips))
}

test_digit_accuracy = function(patterns, labels, weights, noise_level = 0.2) {
  correct = 0
  total = length(labels)
  for (i in 1:total) {
    original = patterns[i, ]
    noisy = scramble_pattern(original, noise_level)
    recovered = simulate_until_fixed_sync(noisy, weights)
    distances = apply(patterns, 1, function(p) hamming_distance(p, recovered))
    closest = which.min(distances)
    if (labels[closest] == labels[i]) correct = correct + 1
  }
  acc = correct / total
  cat("✅ Accuracy:", round(acc * 100, 2), "%\n")
  return(acc)
}

time_to_converge_sync = function(initial, weights, max_steps = 1000) {
  for (i in 1:max_steps) {
    updated = sign(weights %*% initial)
    updated[updated == 0] = 1
    if (all(updated == initial)) return(i)
    initial = updated
  }
  return(max_steps)
}

#==============================
#====== VISUALIZATION =========
#==============================
plot_state = function(state, step, size = 100) {
  mat = matrix(state, nrow = size, byrow = FALSE)
  df = expand.grid(x = 1:size, y = 1:size)
  df$val = as.vector(mat)
  p = ggplot(df, aes(x = x, y = y, fill = factor(val))) +
    geom_tile(color = "gray") +
    scale_fill_manual(values = c("-1" = "black", "1" = "white")) +
    theme_void() + theme(legend.position = "none") +
    ggtitle(paste("Step", step)) +
    coord_fixed()
  print(p)
}

decode_with_hamming = function(state, weights, target, max_steps = 1000, delay = 0.05) {
  N = length(state)
  for (step in 1:max_steps) {
    idx = ((step - 1) %% N) + 1
    state[idx] = sign(sum(weights[idx, ] * state))
    state[state == 0] = 1
    plot_state(state, step)
    hd = hamming_distance(state, target)
    cat("Step", step, "- Hamming:", hd, "\n")
    if (hd == 0) {
      cat("✅ Converged to memory!\n")
      break
    }
    Sys.sleep(delay)
  }
  return(state)
}

#==============================
#====== RUN IT ALL! ===========
#==============================

generate_digit_images("digits", per_digit = 30, size = 100)
digit_paths = get_digit_paths("digits")
digit_data = train_digit_variants(digit_paths, variants_per_digit = 10, size = 100)
weights = fixed_weights_storkey(digit_data$patterns)
test_digit_accuracy(digit_data$patterns, digit_data$labels, weights, noise_level = 0.2)
times = sapply(1:10, function(i) {
  scrambled = scramble_pattern(digit_data$patterns[i, ], 0.3)
  time_to_converge_sync(scrambled, weights)
})
cat("🕒 Convergence Times:", times, "\n")
scrambled = scramble_pattern(digit_data$patterns[1, ], 0.4)
decode_with_hamming(scrambled, weights, digit_data$patterns[1, ])
