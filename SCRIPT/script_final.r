#########################################################
# LIBRARIES
#########################################################

library(tidyverse)
library(caret)
library(lubridate)

#########################################################
# LOAD DATA (edx_setup.R should create 'edx')
#########################################################

RootDir    <- "c:/Users/aaric/projects/HARVARD DATA SCIENCE CAPSTONE _ FINALIZED VERSION"
setupDir   <- file.path(RootDir, "SETUP")
ResultsDir <- file.path(RootDir, "RESULTS")
predictor_file <- file.path(ResultsDir, "predict_ratings.rds")

####
if (file.exists(predictor_file)){
predict_ratings <- readRDS(predictor_file)

# New data (must have userId, movieId, timestamp, genres)
new_ratings <- data.frame(
  userId    = c(1, 2),
  movieId   = c(260, 1196),
  timestamp = c(974595468, 974595468),
  genres    = c("Action|Adventure", "Comedy|Romance")
)

# Get predictions (one line)
preds <- predict_ratings(new_ratings)

# 
}

new_ratings <- data.frame(
  userId    = c(1, 2,3),
  movieId   = c(260, 1196,1196),
  timestamp = c(974595468, 974595468,974595478),
  genres    = c("Action|Adventure", "Comedy|Romance","Comedy|Romance")
)

# Get predictions (one line)
preds <- round(predict_ratings(new_ratings),1)
new_ratings$predicted_rating <- preds
# View result
View(new_ratings)

stopifnot("Training results already exist. Remove or rename the file first." = !file.exists(predictor_file))


source(file.path(setupDir, "edx_setup.R"))   # provides 'edx' data frame

#########################################################
# FUNCTION: RMSE
#########################################################

RMSE <- function(true, pred) {
  sqrt(mean((true - pred)^2))
}

#########################################################
# FUNCTION: IMPROVEMENT BY PERCENTAGE
#########################################################

IBP <- function(base, final) {
  round(((base - final) * 100) / base, 2)
}


#########################################################
# Define the predictor creator (closure)
#########################################################
make_predictor <- function(mu, movie_bias, user_bias, 
                           year_bias, month_bias, week_bias, 
                           genre_bias) {
  
  force(mu); force(movie_bias); force(user_bias)
  force(year_bias); force(month_bias); force(week_bias); force(genre_bias)
  
  function(new_data) {
    # Input validation
    required_cols <- c("userId", "movieId", "timestamp", "genres")
    missing <- setdiff(required_cols, names(new_data))
    if (length(missing) > 0) 
      stop("Missing columns: ", paste(missing, collapse = ", "))
    
    if (!inherits(new_data$timestamp, "POSIXct")) {
      new_data$timestamp <- as.POSIXct(new_data$timestamp, origin = "1970-01-01", tz = "UTC")
    }
    
    pred_df <- new_data %>%
      mutate(row_id    = row_number(),
             year_bin  = floor_date(timestamp, "year"),
             month_bin = floor_date(timestamp, "month"),
             week_bin  = floor_date(timestamp, "week")) %>%
      mutate(pred = mu)
    
    # Movie bias
    pred_df <- pred_df %>%
      left_join(movie_bias, by = "movieId") %>%
      mutate(pred = pred + coalesce(b_i, 0)) %>% select(-b_i)
    
    # User bias
    pred_df <- pred_df %>%
      left_join(user_bias, by = "userId") %>%
      mutate(pred = pred + coalesce(b_u, 0)) %>% select(-b_u)
    
    # Time biases
    pred_df <- pred_df %>%
      left_join(year_bias, by = "year_bin") %>%
      mutate(pred = pred + coalesce(b_y, 0)) %>% select(-b_y) %>%
      left_join(month_bias, by = "month_bin") %>%
      mutate(pred = pred + coalesce(b_m, 0)) %>% select(-b_m) %>%
      left_join(week_bias, by = "week_bin") %>%
      mutate(pred = pred + coalesce(b_w, 0)) %>% select(-b_w)
    
    # Genre bias (weighted)
    genre_long <- pred_df %>%
      select(row_id, genres) %>%
      separate_rows(genres, sep = "\\|") %>%
      group_by(row_id) %>%
      mutate(genre_weight = 1 / n()) %>% ungroup()
    
    genre_contrib <- genre_long %>%
      left_join(genre_bias, by = c("genres" = "genre_single")) %>%
      mutate(genre_effect = coalesce(b_g, 0) * genre_weight) %>%
      group_by(row_id) %>%
      summarise(b_g_total = sum(genre_effect), .groups = "drop")
    
    pred_df <- pred_df %>%
      left_join(genre_contrib, by = "row_id") %>%
      mutate(pred = pred + coalesce(b_g_total, 0))
    
    return(pred_df$pred)
  }
}



#########################################################
# INITIAL TRAIN / TEST SPLIT (from edx)
#########################################################

set.seed(1)

test_index <- createDataPartition(edx$rating, p = 0.1, list = FALSE)
train_set  <- edx[-test_index, ]
test_set   <- edx[test_index, ]

#########################################################
# FEATURE ENGINEERING (done once, right after split)
#########################################################

# floor_date used (not round_date) to avoid assigning ratings to future periods
add_features <- function(df) {
  df %>%
    mutate(
      year_bin  = floor_date(as_datetime(timestamp), unit = "year"),
      month_bin = floor_date(as_datetime(timestamp), unit = "month"),
      week_bin  = floor_date(as_datetime(timestamp), unit = "week"),
      genres_key = sapply(strsplit(as.character(genres), "\\|"),
                          function(g) paste(sort(g), collapse = "|")),
      n_genres  = lengths(strsplit(as.character(genres), "\\|")),
      weight    = 1 / n_genres
    )
}

train_set <- add_features(train_set) %>% mutate(row_id = row_number())
test_set  <- add_features(test_set)  %>% mutate(row_id = row_number())

# Pre-unnest once — reused throughout
train_unnested <- train_set %>%
  mutate(genre_single = strsplit(as.character(genres), "\\|")) %>%
  tidyr::unnest(genre_single)

test_unnested <- test_set %>%
  mutate(genre_single = strsplit(as.character(genres), "\\|")) %>%
  tidyr::unnest(genre_single)

#########################################################
# BASELINE RMSE
#########################################################

mu            <- mean(train_set$rating)
rmse_baseline <- RMSE(test_set$rating, mu)
cat("Baseline RMSE (global mean):", rmse_baseline, "\n")

#########################################################
# HELPER: compute genre contribution (weighted sum per row_id)
#########################################################

compute_genre_contrib <- function(df_unnested, genre_bias_tbl) {
  df_unnested %>%
    left_join(genre_bias_tbl, by = "genre_single") %>%
    mutate(b_g = ifelse(is.na(b_g), 0, b_g)) %>%
    group_by(row_id) %>%
    summarise(b_g = sum(b_g * weight), .groups = "drop")
}

#########################################################
# HELPER: build full predictions for a dataset
#########################################################

build_predictions <- function(df, df_unnested, mu_val,
                              movie_b, user_b, year_b,
                              month_b, week_b, genre_b) {
  genre_contrib <- compute_genre_contrib(df_unnested, genre_b)
  
  df %>%
    left_join(movie_b,       by = "movieId") %>%
    left_join(user_b,        by = "userId") %>%
    left_join(year_b,        by = "year_bin") %>%
    left_join(month_b,       by = "month_bin") %>%
    left_join(week_b,        by = "week_bin") %>%
    left_join(genre_contrib, by = "row_id") %>%
    mutate(
      b_i  = ifelse(is.na(b_i),  0, b_i),
      b_u  = ifelse(is.na(b_u),  0, b_u),
      b_y  = ifelse(is.na(b_y),  0, b_y),
      b_m  = ifelse(is.na(b_m),  0, b_m),
      b_w  = ifelse(is.na(b_w),  0, b_w),
      b_g  = ifelse(is.na(b_g),  0, b_g),
      pred = mu_val + b_i + b_u + b_y + b_m + b_w + b_g
    )
}

#########################################################
# LAMBDA TUNING (80/20 validation split within train_set)
#########################################################

set.seed(123)

tune_index <- createDataPartition(train_set$rating, p = 0.8, list = FALSE)
tune_train <- train_set[tune_index, ]  %>% mutate(row_id = row_number())
tune_val   <- train_set[-tune_index, ] %>% mutate(row_id = row_number())

tune_train_unnested <- tune_train %>%
  mutate(genre_single = strsplit(as.character(genres), "\\|")) %>%
  tidyr::unnest(genre_single)

tune_val_unnested <- tune_val %>%
  mutate(genre_single = strsplit(as.character(genres), "\\|")) %>%
  tidyr::unnest(genre_single)

mu_tune  <- mean(tune_train$rating)
lambdas  <- seq(1, 10, by = 0.5)
val_rmse <- numeric(length(lambdas))

cat("\n=========================================\n")
cat("TUNING LAMBDA (validation set)\n")
cat("=========================================\n")

for (i in seq_along(lambdas)) {
  lambda <- lambdas[i]
  cat("Testing lambda =", lambda, "\n")
  
  movie_b <- tune_train %>%
    group_by(movieId) %>%
    summarise(b_i = sum(rating - mu_tune) / (n() + lambda), .groups = "drop")
  
  user_b <- tune_train %>%
    left_join(movie_b, by = "movieId") %>%
    mutate(b_i = ifelse(is.na(b_i), 0, b_i)) %>%
    group_by(userId) %>%
    summarise(b_u = sum(rating - mu_tune - b_i) / (n() + lambda), .groups = "drop")
  
  year_b <- tune_train %>%
    left_join(movie_b, by = "movieId") %>%
    left_join(user_b,  by = "userId") %>%
    mutate(b_i = ifelse(is.na(b_i), 0, b_i),
           b_u = ifelse(is.na(b_u), 0, b_u)) %>%
    group_by(year_bin) %>%
    summarise(b_y = sum(rating - mu_tune - b_i - b_u) / (n() + lambda), .groups = "drop")
  
  month_b <- tune_train %>%
    left_join(movie_b, by = "movieId") %>%
    left_join(user_b,  by = "userId") %>%
    left_join(year_b,  by = "year_bin") %>%
    mutate(b_i = ifelse(is.na(b_i), 0, b_i),
           b_u = ifelse(is.na(b_u), 0, b_u),
           b_y = ifelse(is.na(b_y), 0, b_y)) %>%
    group_by(month_bin) %>%
    summarise(b_m = sum(rating - mu_tune - b_i - b_u - b_y) / (n() + lambda), .groups = "drop")
  
  week_b <- tune_train %>%
    left_join(movie_b,  by = "movieId") %>%
    left_join(user_b,   by = "userId") %>%
    left_join(year_b,   by = "year_bin") %>%
    left_join(month_b,  by = "month_bin") %>%
    mutate(b_i = ifelse(is.na(b_i), 0, b_i),
           b_u = ifelse(is.na(b_u), 0, b_u),
           b_y = ifelse(is.na(b_y), 0, b_y),
           b_m = ifelse(is.na(b_m), 0, b_m)) %>%
    group_by(week_bin) %>%
    summarise(b_w = sum(rating - mu_tune - b_i - b_u - b_y - b_m) / (n() + lambda), .groups = "drop")
  
  genre_b <- tune_train_unnested %>%
    left_join(movie_b,  by = "movieId") %>%
    left_join(user_b,   by = "userId") %>%
    left_join(year_b,   by = "year_bin") %>%
    left_join(month_b,  by = "month_bin") %>%
    left_join(week_b,   by = "week_bin") %>%
    mutate(b_i = ifelse(is.na(b_i), 0, b_i),
           b_u = ifelse(is.na(b_u), 0, b_u),
           b_y = ifelse(is.na(b_y), 0, b_y),
           b_m = ifelse(is.na(b_m), 0, b_m),
           b_w = ifelse(is.na(b_w), 0, b_w)) %>%
    group_by(genre_single) %>%
    summarise(
      b_g = sum(weight * (rating - mu_tune - b_i - b_u - b_y - b_m - b_w)) /
        (sum(weight) + lambda),
      .groups = "drop"
    )
  
  val_pred    <- build_predictions(tune_val, tune_val_unnested, mu_tune,
                                   movie_b, user_b, year_b, month_b, week_b, genre_b)
  val_rmse[i] <- RMSE(val_pred$rating, val_pred$pred)
  cat("  Validation RMSE:", val_rmse[i], "\n")
}

best_lambda   <- lambdas[which.min(val_rmse)]
best_val_rmse <- min(val_rmse)

cat("\n=========================================\n")
cat("BEST LAMBDA:", best_lambda, "\n")
cat("Validation RMSE:", best_val_rmse, "\n")
cat("=========================================\n")



#########################################################
# FUNCTION: Iterative Bias Model (Coordinate Descent)
#########################################################

train_iterative_bias <- function(
    data,
    data_unnested,
    mu,
    lambda   = 2,
    max_iter = 30,
    tol      = 1e-5,
    verbose  = TRUE
) {
  
  # Initialise all biases at zero
  movie_bias <- data %>% distinct(movieId)   %>% mutate(b_i = 0)
  user_bias  <- data %>% distinct(userId)    %>% mutate(b_u = 0)
  year_bias  <- data %>% distinct(year_bin)  %>% mutate(b_y = 0)
  month_bias <- data %>% distinct(month_bin) %>% mutate(b_m = 0)
  week_bias  <- data %>% distinct(week_bin)  %>% mutate(b_w = 0)
  genre_bias <- data_unnested %>% distinct(genre_single) %>% mutate(b_g = 0)
  
  prev_rmse <- Inf
  
  for (iter in 1:max_iter) {
    
    # Attach all current biases to form base table
    base <- data %>%
      left_join(movie_bias, by = "movieId") %>%
      left_join(user_bias,  by = "userId") %>%
      left_join(year_bias,  by = "year_bin") %>%
      left_join(month_bias, by = "month_bin") %>%
      left_join(week_bias,  by = "week_bin")
    
    genre_contrib <- data_unnested %>%
      left_join(genre_bias, by = "genre_single") %>%
      mutate(b_g = ifelse(is.na(b_g), 0, b_g)) %>%
      group_by(row_id) %>%
      summarise(b_g = sum(b_g * weight), .groups = "drop")
    
    base <- base %>%
      left_join(genre_contrib, by = "row_id") %>%
      mutate(across(c(b_i, b_u, b_y, b_m, b_w, b_g), ~ifelse(is.na(.), 0, .)))
    
    # ---------- Movie ----------
    movie_bias <- base %>%
      group_by(movieId) %>%
      summarise(b_i = sum(rating - mu - b_u - b_y - b_m - b_w - b_g) / (n() + lambda),
                .groups = "drop")
    base <- base %>% select(-b_i) %>%
      left_join(movie_bias, by = "movieId") %>%
      mutate(b_i = ifelse(is.na(b_i), 0, b_i))
    
    # ---------- User ----------
    user_bias <- base %>%
      group_by(userId) %>%
      summarise(b_u = sum(rating - mu - b_i - b_y - b_m - b_w - b_g) / (n() + lambda),
                .groups = "drop")
    base <- base %>% select(-b_u) %>%
      left_join(user_bias, by = "userId") %>%
      mutate(b_u = ifelse(is.na(b_u), 0, b_u))
    
    # ---------- Year ----------
    year_bias <- base %>%
      group_by(year_bin) %>%
      summarise(b_y = sum(rating - mu - b_i - b_u - b_m - b_w - b_g) / (n() + lambda),
                .groups = "drop")
    base <- base %>% select(-b_y) %>%
      left_join(year_bias, by = "year_bin") %>%
      mutate(b_y = ifelse(is.na(b_y), 0, b_y))
    
    # ---------- Month ----------
    month_bias <- base %>%
      group_by(month_bin) %>%
      summarise(b_m = sum(rating - mu - b_i - b_u - b_y - b_w - b_g) / (n() + lambda),
                .groups = "drop")
    base <- base %>% select(-b_m) %>%
      left_join(month_bias, by = "month_bin") %>%
      mutate(b_m = ifelse(is.na(b_m), 0, b_m))
    
    # ---------- Week ----------
    week_bias <- base %>%
      group_by(week_bin) %>%
      summarise(b_w = sum(rating - mu - b_i - b_u - b_y - b_m - b_g) / (n() + lambda),
                .groups = "drop")
    base <- base %>% select(-b_w) %>%
      left_join(week_bias, by = "week_bin") %>%
      mutate(b_w = ifelse(is.na(b_w), 0, b_w))
    
    # ---------- Genre ----------
    genre_bias <- data_unnested %>%
      left_join(base %>% select(row_id, b_i, b_u, b_y, b_m, b_w), by = "row_id") %>%
      mutate(across(c(b_i, b_u, b_y, b_m, b_w), ~ifelse(is.na(.), 0, .))) %>%
      group_by(genre_single) %>%
      summarise(
        b_g = sum(weight * (rating - mu - b_i - b_u - b_y - b_m - b_w)) /
          (sum(weight) + lambda),
        .groups = "drop"
      )
    
    genre_contrib_new <- data_unnested %>%
      left_join(genre_bias, by = "genre_single") %>%
      mutate(b_g = ifelse(is.na(b_g), 0, b_g)) %>%
      group_by(row_id) %>%
      summarise(b_g = sum(b_g * weight), .groups = "drop")
    
    base <- base %>% select(-b_g) %>%
      left_join(genre_contrib_new, by = "row_id") %>%
      mutate(b_g = ifelse(is.na(b_g), 0, b_g))
    
    # ---------- Evaluate ----------
    base  <- base %>% mutate(pred = mu + b_i + b_u + b_y + b_m + b_w + b_g)
    rmse  <- RMSE(base$rating, base$pred)
    if (verbose) cat("Iteration:", iter, "| Training RMSE:", rmse, "\n")
    
    if (abs(prev_rmse - rmse) < tol) {
      if (verbose) cat("Converged at iteration:", iter, "\n")
      break
    }
    prev_rmse <- rmse
  }
  
  list(
    movie_bias = movie_bias,
    user_bias  = user_bias,
    year_bias  = year_bias,
    month_bias = month_bias,
    week_bias  = week_bias,
    genre_bias = genre_bias,
    rmse       = rmse
  )
}

#########################################################
# TRAIN FINAL MODEL (single coordinate descent run)
#########################################################

cat("\n=========================================\n")
cat("TRAINING FINAL MODEL\n")
cat("=========================================\n")

result <- train_iterative_bias(
  data          = train_set,
  data_unnested = train_unnested,
  mu            = mu,
  lambda        = best_lambda,
  max_iter      = 30,
  tol           = 1e-5,
  verbose       = TRUE
)

movie_bias_final <- result$movie_bias
user_bias_final  <- result$user_bias
year_bias_final  <- result$year_bias
month_bias_final <- result$month_bias
week_bias_final  <- result$week_bias
genre_bias_final <- result$genre_bias

#########################################################
# STEP-BY-STEP RMSE (accumulating effects on test set)
#########################################################

cat("\n=========================================\n")
cat("STEP-BY-STEP RMSE\n")
cat("=========================================\n")

# Step 1: baseline
pred_step          <- rep(mu, nrow(test_set))
rmse_step_baseline <- RMSE(test_set$rating, pred_step)
cat("Baseline              :", rmse_step_baseline, "\n")

# Step 2: + movie
pred_step       <- pred_step +
  (test_set %>% left_join(movie_bias_final, by = "movieId") %>%
     mutate(b_i = ifelse(is.na(b_i), 0, b_i)) %>% pull(b_i))
rmse_step_movie <- RMSE(test_set$rating, pred_step)
cat("+ Movie bias          :", rmse_step_movie, "\n")

# Step 3: + user
pred_step      <- pred_step +
  (test_set %>% left_join(user_bias_final, by = "userId") %>%
     mutate(b_u = ifelse(is.na(b_u), 0, b_u)) %>% pull(b_u))
rmse_step_user <- RMSE(test_set$rating, pred_step)
cat("+ User bias           :", rmse_step_user, "\n")

# Step 4: + year
pred_step      <- pred_step +
  (test_set %>% left_join(year_bias_final, by = "year_bin") %>%
     mutate(b_y = ifelse(is.na(b_y), 0, b_y)) %>% pull(b_y))
rmse_step_year <- RMSE(test_set$rating, pred_step)
cat("+ Year bias           :", rmse_step_year, "\n")

# Step 5: + month
pred_step       <- pred_step +
  (test_set %>% left_join(month_bias_final, by = "month_bin") %>%
     mutate(b_m = ifelse(is.na(b_m), 0, b_m)) %>% pull(b_m))
rmse_step_month <- RMSE(test_set$rating, pred_step)
cat("+ Month bias          :", rmse_step_month, "\n")

# Step 6: + week
pred_step      <- pred_step +
  (test_set %>% left_join(week_bias_final, by = "week_bin") %>%
     mutate(b_w = ifelse(is.na(b_w), 0, b_w)) %>% pull(b_w))
rmse_step_week <- RMSE(test_set$rating, pred_step)
cat("+ Week bias           :", rmse_step_week, "\n")

# Step 7: + genre
genre_contrib_test <- compute_genre_contrib(test_unnested, genre_bias_final)
pred_step          <- pred_step +
  (test_set %>% left_join(genre_contrib_test, by = "row_id") %>%
     mutate(b_g = ifelse(is.na(b_g), 0, b_g)) %>% pull(b_g))
rmse_step_genre <- RMSE(test_set$rating, pred_step)
cat("+ Genre bias          :", rmse_step_genre, "\n")

#########################################################
# EVALUATE FULL MODEL ON TEST SET
#########################################################

test_pred  <- build_predictions(
  test_set, test_unnested, mu,
  movie_bias_final, user_bias_final,
  year_bias_final,  month_bias_final,
  week_bias_final,  genre_bias_final
)

rmse_final <- RMSE(test_pred$rating, test_pred$pred)

#########################################################
# SAVE RESULTS
#########################################################

save(
  train_set, test_set, mu, rmse_baseline,
  best_lambda, best_val_rmse, lambdas, val_rmse,
  test_pred, rmse_final,
  movie_bias_final, user_bias_final, genre_bias_final,
  year_bias_final,  month_bias_final, week_bias_final,
  rmse_step_baseline, rmse_step_movie, rmse_step_user,
  rmse_step_year, rmse_step_month, rmse_step_week, rmse_step_genre,
  edx,
  file = file.path(ResultsDir, "movielens_results.RData")
)

#########################################################
# make predictor 
#########################################################

predict_ratings <- make_predictor(
  mu         = mu,
  movie_bias = movie_bias_final,
  user_bias  = user_bias_final,
  year_bias  = year_bias_final,
  month_bias = month_bias_final,
  week_bias  = week_bias_final,
  genre_bias = genre_bias_final
)

# Save for future use
saveRDS(predict_ratings, file = predictor_file)
message("Training complete. Predictor saved to ", predictor_file)


#########################################################
# FINAL RESULTS
#########################################################

cat("\n=========================================\n")
cat("FINAL RESULTS\n")
cat("=========================================\n")
cat("Best lambda (after tuning)  :", best_lambda,                          "\n")
cat("Baseline RMSE               :", rmse_baseline,                        "\n")
cat("Final Test RMSE             :", rmse_final,                           "\n")
cat("Improvement over baseline   :", IBP(rmse_baseline, rmse_final), "%\n")