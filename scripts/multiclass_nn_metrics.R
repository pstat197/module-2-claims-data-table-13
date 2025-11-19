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
  mutate(mclass_num = as.numeric(mclass) - 1L)

#View(claims_clean)
set.seed(110122)
partitions <- initial_split(claims_clean, prop = 0.8)

val_df <- testing(partitions)

val_text   <- val_df$text_clean
val_labels <- val_df$mclass_num

# ---------------------------------------------------------------
# 2. Load the saved best model
# ---------------------------------------------------------------
model <- load_model_tf("results/multiclass_model")

# ---------------------------------------------------------------
# 3. Generate predictions
# ---------------------------------------------------------------
pred_classes <- model %>% predict(val_text) %>% 
  apply(1, which.max) - 1

# ---------------------------------------------------------------
# 4. Confusion matrix values

library(caret) # for validation
# ---------------------------------------------------------------
cM <- confusionMatrix(
  factor(pred_classes),
  factor(val_labels)
)[2]
cM
# ---------------------------------------------------------------
# 5. Compute metrics
# ---------------------------------------------------------------
cT <- confusionMatrix(
  factor(pred_classes),
  factor(val_labels)
)[4]
cT$byClass[1,1]
# ---------------------------------------------------------------
# 6. Make output table (nice for QMD)
# ---------------------------------------------------------------
metrics_tbl <- tibble(
  metric = c("Accuracy", "Sensitivity", "Specificity"),
  noRelevantContent = c(cT$byClass[1,11], cT$byClass[1,1], cT$byClass[1,2]),
  physicalActivity = c(cT$byClass[2,11], cT$byClass[2,1], cT$byClass[2,2]),
  possibleFatality = c(cT$byClass[3,11], cT$byClass[3,1], cT$byClass[3,2]),
  potentialUnlawfulActivity = c(cT$byClass[4,11], cT$byClass[4,1], cT$byClass[4,2]),
  otherClaimContent = c(cT$byClass[5,11], cT$byClass[5,1], cT$byClass[5,2])
)

print(metrics_tbl)
