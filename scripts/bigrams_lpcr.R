# Preliminary task 2

library(tidyverse)
library(tidytext)
library(rsample)
library(tm)

source('scripts/preprocessing.R')
load('data/claims-clean.RData', verbose = TRUE)

# Split data
set.seed(111825)
split <- initial_split(clean_df, prop = 0.7, strata = bclass)
train_data <- training(split)
test_data <- testing(split)

# Function for tokenizing the dataframe and running PCA
get_pcs <- function(data, type = "words", num_pcs = 50) {
  tokens <- data %>%
    select(.id, text_clean)
  if (type == "words") {
    tokens <- tokens %>%
      unnest_tokens(output = term, input = text_clean, token = "words") %>%
      filter(!term %in% stop_words$word)
  } else {
    tokens <- tokens %>%
      unnest_tokens(output = term, input = text_clean, token = "ngrams", n = 2) %>%
      separate(term, c("word1", "word2"), sep = " ") %>%
      filter(!word1 %in% stop_words$word, !word2 %in% stop_words$word) %>%
      unite(term, word1, word2, sep = " ")
  }
  term_counts <- tokens %>% count(term) %>% filter(n > 5)
  tokens <- tokens %>% filter(term %in% term_counts$term)
  
  tfidf <- tokens %>%
    count(.id, term) %>%
    bind_tf_idf(term, .id, n) %>%
    cast_dtm(.id, term, tf_idf)
  
  # PCA
  pc_model <- prcomp(as.matrix(tfidf), scale. = TRUE, center = TRUE, rank. = num_pcs)
  pcs_df <- as.data.frame(pc_model$x) %>%
    mutate(.id = rownames(as.matrix(tfidf))) %>%
    select(.id, everything())
  
  return(list(model = pc_model, pcs = pcs_df, terms = colnames(tfidf)))
}

# Build LPCR model
word_pca_out <- get_pcs(train_data, type = "words", num_pcs = 50)
train_word_pcs <- word_pca_out$pcs

# Merge with labels
model1_df <- train_data %>%
  inner_join(train_word_pcs, by = ".id") %>%
  mutate(bclass = as.factor(bclass))

# Fit LPCR model to the word-tokenized data (model 1)
model1 <- glm(bclass ~ ., 
              data = model1_df %>% select(-.id, -text_clean, -text_tmp), 
              family = "binomial",
              control = list(maxit = 100))
train_log_odds <- predict(model1, type = "link")

# Process bigrams for model 2
bigram_pca_out <- get_pcs(train_data, type = "bigrams", num_pcs = 20)
train_bigram_pcs <- bigram_pca_out$pcs

# Create a dataframe for the model1's log odds
log_odds_df <- model1_df %>%
  select(.id, bclass) %>%
  mutate(word_log_odds = train_log_odds)

# Combine log_odds_df and bigram-tokenized data for model 2
model2_df <- log_odds_df %>%
  inner_join(train_bigram_pcs, by = ".id")

# Rename bigram columns
colnames(model2_df)[grep("PC", colnames(model2_df))] <- paste0("Bigram_", colnames(model2_df)[grep("PC", colnames(model2_df))])

# Fit model 2
model2 <- glm(bclass ~ ., 
              data = model2_df %>% select(-.id), 
              family = "binomial",
              control = list(maxit = 100))

# Evaluation
# Function for getting accuracy
calc_acc <- function(model, df, type="response") {
  preds <- predict(model, newdata = df, type = type)
  pred_class <- ifelse(preds > 0, "1", "0")
  if(type == "response") pred_class <- ifelse(preds > 0.5, "1", "0")
  actual <- ifelse(df$bclass == "1", "1", "0")
  mean(pred_class == actual)
}

acc1 <- calc_acc(model1, model1_df, type="link")
acc2 <- calc_acc(model2, model2_df, type="response")

print(paste("Model 1 (Words Only) Accuracy:", acc1))
print(paste("Model 2 (Words + Bigrams) Accuracy:", acc2))

# Check significance of bigram terms
summary(model2)

if (acc2 > acc1) {
  print(paste("Model 2 (Words + Bigrams) has a higher accuracy (", acc2, " vs ", acc1, ")."))
} else {
  print(paste("Model 2 (Words + Bigrams) did not improve accuracy (", acc2, " vs ", acc1, ")."))
}