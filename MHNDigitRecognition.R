# Trains an MHN on digit recognition - uses MNIST handwritted digits, random
# sampling. The model works much better than CHN, since it is both smaller and
# uses much less data. 

# Comment out the data output that you don't want (at the bottom)

# ---------------
# Generating Data
# ---------------

library(keras)

# Load MNIST
mnist_data <- dataset_mnist()
x_train <- mnist_data$train$x
y_train <- mnist_data$train$y

# Parameters
test_per_digit <- 100
image_size <- 28 * 28

# Runs MHN given differnt per_digit values
run_mhn_experiment <- function(per_digit) {
  training_patterns <- list()
  labels <- c()
  
  # Creates all the fixed states
  for (d in 0:9) {
    indices <- which(y_train == d)
    random_index <- sample(indices, per_digit)
    imgs <- x_train[random_index,,]
    normalized_imgs <- (imgs / 255) * 2 - 1
    img_vectors <- t(apply(normalized_imgs, 1, function(m) as.vector(t(m))))
    training_patterns <- append(training_patterns, list(img_vectors))
    labels <- c(labels, rep(d, per_digit))
  }
  fixed_states <- do.call(rbind, training_patterns)
  
  # Checks whether it worked
  corresponding_digit <- function(final_state, fixed_states, per_digit) {
    distances <- apply(fixed_states, 1, function(row) sum((row - final_state)^2))
    i <- which.min(distances)
    return(trunc((i-1) %/% per_digit))
  }
  
  # Tests accuracy, checks test_per_digit times
  overall_accuracy <- numeric(10)
  for (d in 0:9) {
    indices <- which(y_train == d)
    random_index <- sample(indices, test_per_digit)
    imgs <- x_train[random_index,,]
    normalized_imgs <- (imgs / 255) * 2 - 1
    img_vectors <- t(apply(normalized_imgs, 1, function(m) as.vector(t(m))))
    
    correct_pred <- 0
    for (i in 1:test_per_digit) {
      cur_state <- img_vectors[i,]
      cur_state <- simulate_until_fixed_continuous(cur_state, fixed_states)
      if (corresponding_digit(cur_state, fixed_states, per_digit) == d) {
        correct_pred <- correct_pred + 1
      }
    }
    overall_accuracy[d+1] <- correct_pred / test_per_digit
  }
  
  # Returns end data
  return(overall_accuracy)
}

# -------------
# Bar Plot Data
# -------------

# Runs experiments for all training sizes
overall_accuracy = run_mhn_experiment(c(25))

# Single accuracy vector (from your last run)
barplot(overall_accuracy,
        names.arg = 0:9,
        col = "skyblue",
        main = "MHN Digit Recognition Accuracy",
        xlab = "Digit",
        ylab = "Accuracy",
        ylim = c(0,1),
        border = "black")
print (mean(overall_accuracy))

# ------------------------------
# Accuracy vs Training Data Size
# ------------------------------

# Stores mean accuracy for each digit
training_sizes = seq(10, 100, by = 10)

mean_accuracy = sapply(training_sizes, function(p) {
  acc <- run_mhn_experiment(p)      # accuracy vector (length 10)
  mean(acc)                         # average across digits
})

# Plot Accuracy vs Training Size
plot(training_sizes, mean_accuracy,
     type = "b", pch = 16, col = "blue", lwd = 2,
     main = "MHN Accuracy vs Training Images per Digit",
     xlab = "Training Images per Digit",
     ylab = "Average Accuracy",
     ylim = c(0,1))
