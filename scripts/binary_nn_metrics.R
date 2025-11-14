# ===============================================================
# validation_metrics.R
# Compute Sensitivity, Specificity, Accuracy for Best NN Model
# ===============================================================

library(tidyverse)
library(tidymodels)
library(keras)
library(tensorflow)

source("scripts/preprocessing.R")

# ---------------------------------------------------------------
# 1. Load data and reproduce the same split as training
# ---------------------------------------------------------------
load("data/claims-raw.RData")

claims_clean <- claims_raw %>%
  parse_data() %>%
  mutate(bclass_num = as.numeric(bclass) - 1L)

set.seed(110122)
partitions <- initial_split(claims_clean, prop = 0.8, strata = bclass)

val_df <- testing(partitions)

val_text   <- val_df$text_clean
val_labels <- val_df$bclass_num

# ---------------------------------------------------------------
# 2. Load the saved best model
# ---------------------------------------------------------------
model <- load_model_tf("results/tuned_binary_nn_model")

# ---------------------------------------------------------------
# 3. Generate predictions
# ---------------------------------------------------------------
pred_probs <- predict(model, val_text) %>% as.numeric()
pred_class <- ifelse(pred_probs > 0.5, 1, 0)

# ---------------------------------------------------------------
# 4. Confusion matrix values
# ---------------------------------------------------------------
TP <- sum(pred_class == 1 & val_labels == 1)
TN <- sum(pred_class == 0 & val_labels == 0)
FP <- sum(pred_class == 1 & val_labels == 0)
FN <- sum(pred_class == 0 & val_labels == 1)

# ---------------------------------------------------------------
# 5. Compute metrics
# ---------------------------------------------------------------
accuracy     <- (TP + TN) / (TP + TN + FP + FN)
sensitivity  <- TP / (TP + FN)
specificity  <- TN / (TN + FP)

# ---------------------------------------------------------------
# 6. Make output table (nice for QMD)
# ---------------------------------------------------------------
metrics_tbl <- tibble(
  Metric = c("Accuracy", "Sensitivity", "Specificity"),
  Value  = c(accuracy, sensitivity, specificity)
)

print(metrics_tbl)
