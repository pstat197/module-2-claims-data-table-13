## PREPROCESSING
#################

# can comment entire section out if no changes to preprocessing.R
source('scripts/preprocessing.R')

# load raw data
load('data/claims-raw.RData')

# preprocess (will take a minute or two)
claims_clean <- claims_raw %>%
  parse_data()

# export
save(claims_clean, file = 'data/claims-clean-example.RData')

## MODEL TRAINING (NN)
######################
library(tidyverse)
library(tidymodels)
library(keras)
library(tensorflow)

# load cleaned data
load('data/claims-clean-example.RData')

# partition
set.seed(110122)
partitions <- claims_clean %>%
  initial_split(prop = 0.8)

train_text <- training(partitions) %>%
  pull(text_clean)
train_labels <- training(partitions) %>%
  pull(mclass) %>%
  as.numeric() - 1
max_words <- 10000
max_len <- 270
# create a preprocessing layer
preprocess_layer <- layer_text_vectorization(
  standardize = 'lower_and_strip_punctuation',
  split = 'whitespace',
  ngrams = NULL,
  max_tokens = max_words,
  output_mode = 'int',
  output_sequence_length = max_len
)

preprocess_layer %>% adapt(train_text)
# does preprocess also go here? when I save?
# define NN architecture
model <- keras_model_sequential() %>%
  preprocess_layer() %>%
  layer_embedding(input_dim = max_words, 
                  output_dim = 32, 
                  input_length = max_len, mask_zero = TRUE) %>%
  bidirectional(
    layer_lstm(units = 32, dropout = 0.3, recurrent_dropout = 0.3)
  ) %>%
  layer_dense(units = 5, activation = "softmax")

model %>% compile(
  optimizer = "adam",
  loss = "sparse_categorical_crossentropy",
  metrics = "accuracy"
)


# train
history <- model %>%
  fit(train_text, 
      train_labels,
      validation_split = 0.3,
      epochs = 10)

## CHECK TEST SET ACCURACY HERE

# save the entire model as a SavedModel
save_model_tf(model, "results/multiclass_model")
