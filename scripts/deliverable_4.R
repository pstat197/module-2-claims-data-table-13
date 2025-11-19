# Deliverable 4
# This script loads the saved binary and multiclass models, generates 
# predictions on the test data, and saves the final results file

library(tidyverse)
library(keras)
library(tensorflow)

# Source preprocessing to get the 'parse_data' function
source("scripts/preprocessing.R")

# Load data
load("data/claims-test.RData", verbose = TRUE)
load("data/claims-raw.RData", verbose = TRUE)

# Preprocess the test data
test_clean <- claims_test %>%
  parse_data() 

test_text <- test_clean$text_clean

# Generate binary predictions
binary_model <- load_model_tf("results/tuned_binary_nn_model")
b_probs <- predict(binary_model, test_text) %>% as.numeric()
b_levels <- levels(claims_raw$bclass)
b_preds_indices <- ifelse(b_probs > 0.5, 2, 1)
b_preds_labels  <- b_levels[b_preds_indices]

# Generate multiclass predictions
multiclass_model <- load_model_tf("results/multiclass_model")
m_probs <- predict(multiclass_model, test_text)
m_preds_indices <- apply(m_probs, 1, which.max)
m_levels <- levels(claims_raw$mclass)
m_preds_labels <- m_levels[m_preds_indices]

# Save predictions
pred_df <- test_clean %>%
  select(.id) %>%
  mutate(
    bclass.pred = factor(b_preds_labels, levels = b_levels),
    mclass.pred = factor(m_preds_labels, levels = m_levels)
  )

save(pred_df, file = "results/preds-group13.RData")