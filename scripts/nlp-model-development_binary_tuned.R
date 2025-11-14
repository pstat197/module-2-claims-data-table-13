###############################################################
## FINAL TRAINING SCRIPT — Compatible with Prediction Script ##
###############################################################

library(tidyverse)
library(tidymodels)
library(keras)
library(tensorflow)

source("scripts/preprocessing.R")

#------------------------------------------------------------
# 1. Load & preprocess labeled data
#------------------------------------------------------------
load("data/claims-raw.RData")

claims_clean <- claims_raw %>%
  parse_data() %>%
  mutate(bclass_num = as.numeric(bclass) - 1L)

set.seed(110122)
partitions <- initial_split(claims_clean, prop = 0.8, strata = bclass)

train_df <- training(partitions)
val_df   <- testing(partitions)

train_text   <- train_df %>% pull(text_clean)
train_labels <- train_df %>% pull(bclass_num)

val_text   <- val_df %>% pull(text_clean)
val_labels <- val_df %>% pull(bclass_num)


#------------------------------------------------------------
# 2. Shared Vectorizers (all models accept string input)
#------------------------------------------------------------

### TF-IDF vectorizer for Model A
vectorizer_tfidf <- layer_text_vectorization(
  split = "whitespace",
  standardize = NULL,
  output_mode = "tf_idf"
)
vectorizer_tfidf %>% adapt(train_text)

### Integer sequence vectorizer for B & C
max_tokens <- 20000
seq_len    <- 300

vectorizer_int <- layer_text_vectorization(
  split = "whitespace",
  standardize = NULL,
  output_mode = "int",
  max_tokens = max_tokens,
  output_sequence_length = seq_len
)
vectorizer_int %>% adapt(train_text)


#------------------------------------------------------------
# 3. MODEL DEFINITIONS (all sequential → compatible)
#------------------------------------------------------------

#------------------------------------------------------------
# MODEL A — Original TF-IDF baseline
#------------------------------------------------------------
build_model_A <- function() {
  keras_model_sequential() %>%
    vectorizer_tfidf() %>%
    layer_dropout(0.2) %>%
    layer_dense(25, activation = "relu") %>%
    layer_dropout(0.2) %>%
    layer_dense(1, activation = "sigmoid")
}

#------------------------------------------------------------
# MODEL B — Improved Dense NN with embedding
#------------------------------------------------------------
build_model_B <- function() {
  keras_model_sequential() %>%
    vectorizer_int() %>%
    layer_embedding(input_dim = max_tokens,
                    output_dim = 128) %>%
    layer_global_average_pooling_1d() %>%
    layer_dense(128, activation = "relu") %>%
    layer_dropout(0.5) %>%
    layer_dense(64, activation = "relu") %>%
    layer_dense(1, activation = "sigmoid")
}

#------------------------------------------------------------
# MODEL C — TextCNN v2 (modified to be sequential)
#------------------------------------------------------------
build_model_C <- function() {
  
  keras_model_sequential() %>%
    vectorizer_int() %>%
    layer_embedding(input_dim = max_tokens, output_dim = 128) %>%
    layer_spatial_dropout_1d(0.2) %>%
    
    # Parallel convolution paths merged into sequential model:
    # Conv 1
    layer_conv_1d(filters = 128, kernel_size = 3, activation = "relu") %>%
    layer_global_max_pooling_1d() %>%
    
    # Dense head
    layer_dense(128, activation = "relu") %>%
    layer_dropout(0.4) %>%
    layer_dense(1, activation = "sigmoid")
}

#------------------------------------------------------------
# 4. Compile models
#------------------------------------------------------------

models <- list(
  A = build_model_A(),
  B = build_model_B(),
  C = build_model_C()
)

compile_settings <- list(
  A = list(lr = 0.001),
  B = list(lr = 0.0007),
  C = list(lr = 0.0005)
)

for (i in names(models)) {
  models[[i]] %>% compile(
    loss = "binary_crossentropy",
    optimizer = optimizer_adam(compile_settings[[i]]$lr),
    metrics = "binary_accuracy"
  )
}

#------------------------------------------------------------
# 5. Train each model
#------------------------------------------------------------

results <- list()

for (i in names(models)) {
  
  cat("\n============================\n")
  cat("Training Model", i, "...\n")
  cat("============================\n")
  
  hist <- models[[i]] %>% fit(
    x = train_text,
    y = train_labels,
    validation_data = list(val_text, val_labels),
    epochs = 25,
    batch_size = 32,
    callbacks = list(
      callback_early_stopping(
        monitor = "val_binary_accuracy",
        patience = 4,
        restore_best_weights = TRUE
      ),
      callback_reduce_lr_on_plateau(
        monitor = "val_loss",
        patience = 2,
        factor = 0.5,
        min_lr = 1e-5
      )
    ),
    verbose = 1
  )
  
  best_val <- max(hist$metrics$val_binary_accuracy)
  
  results[[i]] <- list(
    name = i,
    model = models[[i]],
    best_val = best_val
  )
  
  cat("Best validation accuracy for Model", i, ":", best_val, "\n")
}

#------------------------------------------------------------
# 6. Select & Save Best Model (Prediction-Compatible)
#------------------------------------------------------------

best_name <- names(which.max(sapply(results, \(x) x$best_val)))
best_model <- results[[best_name]]$model
best_acc   <- results[[best_name]]$best_val

cat("\n====================================\n")
cat("BEST MODEL:", best_name, "\n")
cat("Validation accuracy:", best_acc, "\n")
cat("====================================\n\n")

dir.create("results/tuned_binary_nn_model", showWarnings = FALSE, recursive = TRUE)

save_model_tf(best_model, "results/tuned_binary_nn_model")
saveRDS(best_acc, "results/binary_best_accuracy.rds")

cat("Saved best model to results/tuned_binary_nn_model\n")
