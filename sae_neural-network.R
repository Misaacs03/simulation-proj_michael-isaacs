# Install and load required packages
required_packages <- c("tidyverse", "gganimate", "gifski", "viridis")
missing_packages <- required_packages[!(required_packages %in% installed.packages()[,"Package"])]
if(length(missing_packages) > 0) {
  install.packages(missing_packages)
}
invisible(lapply(required_packages, library, character.only = TRUE))

# Set random seed for reproducibility
set.seed(123)

# Define network architecture
input_size <- 4
hidden_size <- 6
output_size <- 1

#Generate samples
n_samples <- 500

velocity <- runif(n_samples, 10, 30)

size <- runif(n_samples, 0.5, 3.0)

starting_distance <- runif(n_samples, 50, 150)

binocular_disparity <- exp(-starting_distance / 80) + rnorm(n_samples, 0, 0.05)
binocular_disparity <- pmax(0, pmin(1, binocular_disparity))

X <- cbind(velocity, size, binocular_disparity, starting_distance)

true_tta <- starting_distance / velocity

min_tta <- min(true_tta)
max_tta <- max(true_tta)
y <- matrix((true_tta - min_tta) / (max_tta - min_tta), ncol = 1)


# Initialize weights and biases
init_weights <- function(in_size, out_size) {
  matrix(rnorm(in_size * out_size), nrow = in_size, ncol = out_size)
}

W1 <- init_weights(input_size, hidden_size)
b1 <- matrix(0, nrow = 1, ncol = hidden_size)
W2 <- init_weights(hidden_size, output_size)
b2 <- matrix(0, nrow = 1, ncol = output_size)

# Activation function (sigmoid) and its derivative
sigmoid <- function(x) 1 / (1 + exp(-x))
sigmoid_derivative <- function(x) x * (1 - x)

# Forward propagation
forward_prop <- function(X, W1, b1, W2, b2) {
  z1 <- X %*% W1 + rep(b1, each = nrow(X))
  a1 <- sigmoid(z1)
  z2 <- a1 %*% W2 + rep(b2, each = nrow(a1))
  a2 <- sigmoid(z2)
  list(a1 = a1, a2 = a2)
}

# Backpropagation
backprop <- function(X, y, a1, a2, W1, W2) {
  m <- nrow(X)
  dz2 <- a2 - y
  dW2 <- t(a1) %*% dz2 / m
  db2 <- colMeans(dz2)
  dz1 <- (dz2 %*% t(W2)) * sigmoid_derivative(a1)
  dW1 <- t(X) %*% dz1 / m
  db1 <- colMeans(dz1)
  list(dW1 = dW1, db1 = db1, dW2 = dW2, db2 = db2)
}

# Training function
train <- function(X, y, W1, b1, W2, b2, learning_rate, epochs) {
  history <- data.frame()
  for (epoch in 1:epochs) {
    # Forward propagation
    output <- forward_prop(X, W1, b1, W2, b2)
    a1 <- output$a1
    a2 <- output$a2
    
    # Backpropagation 
    grads <- backprop(X, y, a1, a2, W1, W2)
    
    # Update weights and biases
    W1 <- W1 - learning_rate * grads$dW1
    b1 <- b1 - learning_rate * grads$db1
    W2 <- W2 - learning_rate * grads$dW2
    b2 <- b2 - learning_rate * grads$db2
    
    # Calculate loss
    loss <- mean(-(y * log(a2) + (1 - y) * log(1 - a2)))
    
    # Store history for visualization
    history <- rbind(history, data.frame(epoch = epoch, loss = loss, 
                                         W1 = c(W1), W2 = c(W2)))
  }
  list(W1 = W1, b1 = b1, W2 = W2, b2 = b2, history = history)
}

# Train the network
result <- train(X, y, W1, b1, W2, b2, learning_rate = 0.1, epochs = 900)

# Prepare data for visualization
vis_data <- result$history %>%
  pivot_longer(cols = starts_with("W"), names_to = "weight", values_to = "value") %>%
  mutate(layer = ifelse(str_detect(weight, "W1"), "Hidden", "Output"))

# Create the animation
p <- ggplot(vis_data, aes(x = epoch, y = value, color = weight, group = weight)) +
  geom_line() +
  facet_wrap(~layer, scales = "free_y", ncol = 1) +
  scale_color_viridis_d() +
  labs(title = "SAE Neural Network Training",
       subtitle = "Epoch: {frame_along}",
       x = "Epoch", y = "Weight Value") +
  theme_minimal() +
  transition_reveal(epoch)

# Render and save the animation
animate(p, nframes = 200, fps = 30, width = 800, height = 600, renderer = gifski_renderer())
anim_save("size_arrival_perceptual_learning.gif")

cat("Animation completed. Check your working directory for size_arrival_perceptual_learning.gif\n")