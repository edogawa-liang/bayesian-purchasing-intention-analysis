# -----------------------------------------------------------
# Bayesian Logistic Regression (No LASSO)
# Online Shoppers Intention Dataset
# -----------------------------------------------------------

library(rstan)
library(loo)
library(rstanarm)
options(mc.cores = parallel::detectCores())
rstan_options(auto_write = TRUE)

# -----------------------------------------------------------
# 1. Load and preprocess data
# -----------------------------------------------------------
data <- read.csv("data/online_shoppers_intention.csv")

# Factorize categorical variables
cat_vars <- c("Month", "OperatingSystems", "Browser",
              "Region", "TrafficType", "VisitorType",
              "Weekend", "Revenue")
data[cat_vars] <- lapply(data[cat_vars], as.factor)

# Combine small TrafficType levels as "Other" 
traffic_counts <- table(data$TrafficType) 
rare_types <- names(traffic_counts[traffic_counts < 100]) 
levels(data$TrafficType)[levels(data$TrafficType) %in% rare_types] <- "Other" 
# Re-factor to drop unused levels 
data$TrafficType <- droplevels(data$TrafficType)

# Standardize numeric variables
num_cols <- setdiff(names(data), cat_vars)
data[num_cols] <- lapply(data[num_cols], as.numeric)
data[num_cols] <- scale(data[num_cols])

# Binary outcome variable
y <- as.numeric(data$Revenue) - 1

# -----------------------------------------------------------
# 2. Stan model code
# -----------------------------------------------------------

# --- Pooled model ---
stan_code_pool <- "
data {
  int<lower=0> N;
  int<lower=0> K;
  matrix[N, K] X;
  int<lower=0,upper=1> y[N];
}
parameters {
  real alpha;
  vector[K] beta;
}
model {
  alpha ~ normal(0, 5);
  beta ~ normal(0, 2);
  y ~ bernoulli_logit(alpha + X * beta);
}
generated quantities {
  vector[N] log_lik;
  for (n in 1:N)
    log_lik[n] = bernoulli_logit_lpmf(y[n] | alpha + dot_product(X[n], beta));
}
"

# --- Hierarchical model ---
stan_code_hier <- "
data {
  int<lower=0> N;
  int<lower=0> K;
  int<lower=1> J;
  int<lower=1,upper=J> group_id[N];
  matrix[N, K] X;
  int<lower=0,upper=1> y[N];
}
parameters {
  real alpha;
  vector[K] beta;
  vector[J] alpha_group;
  real<lower=0> sigma_group;
}
model {
  alpha ~ normal(0, 5);
  beta ~ normal(0, 2);
  sigma_group ~ exponential(1);
  alpha_group ~ normal(0, sigma_group);
  y ~ bernoulli_logit(alpha + X * beta + alpha_group[group_id]);
}
generated quantities {
  vector[N] log_lik;
  for (n in 1:N) {
    real eta = alpha + dot_product(X[n], beta) + alpha_group[group_id[n]];
    log_lik[n] = bernoulli_logit_lpmf(y[n] | eta);
  }
}
"

# --- Hierarchical model (2 group)---
stan_code_hier_2 <- "
data {
  int<lower=0> N;           
  int<lower=0> K;           
  int<lower=1> J1;          
  int<lower=1> J2;          
  int<lower=1,upper=J1> group1_id[N];  
  int<lower=1,upper=J2> group2_id[N];  
  matrix[N, K] X;           // predictor matrix
  int<lower=0,upper=1> y[N]; // binary outcome
}
parameters {
  real alpha;
  vector[K] beta;
  vector[J1] alpha_group1;
  vector[J2] alpha_group2;
  real<lower=0> sigma_group1;
  real<lower=0> sigma_group2;
}
model {
  alpha ~ normal(0, 5);
  beta ~ normal(0, 2);
  sigma_group1 ~ exponential(1);
  sigma_group2 ~ exponential(1);
  alpha_group1 ~ normal(0, sigma_group1);
  alpha_group2 ~ normal(0, sigma_group2);
  y ~ bernoulli_logit(alpha + X * beta +
                      alpha_group1[group1_id] +
                      alpha_group2[group2_id]);
}
generated quantities {
  vector[N] log_lik;
  for (n in 1:N) {
    real eta = alpha + dot_product(X[n], beta) +
               alpha_group1[group1_id[n]] +
               alpha_group2[group2_id[n]];
    log_lik[n] = bernoulli_logit_lpmf(y[n] | eta);
  }
}
"

stan_code_hier_2_lasso <- "
data {
  int<lower=0> N;                 
  int<lower=0> K;                
  int<lower=1> J1;                
  int<lower=1> J2;               
  int<lower=1,upper=J1> group1_id[N]; 
  int<lower=1,upper=J2> group2_id[N]; 
  matrix[N, K] X;                
  int<lower=0,upper=1> y[N];     
}

parameters {
  real alpha;
  vector[K] beta;
  vector[J1] alpha_group1;
  vector[J2] alpha_group2;
  real<lower=0> sigma_group1;
  real<lower=0> sigma_group2;
  real<lower=0> tau;              // global shrinkage (for Bayesian LASSO)
}

model {
  // Priors
  alpha ~ normal(0, 5);
  tau ~ exponential(1);                   // global shrinkage
  beta ~ double_exponential(0, tau);      // Laplace prior (LASSO)
  
  sigma_group1 ~ exponential(1);
  sigma_group2 ~ exponential(1);
  alpha_group1 ~ normal(0, sigma_group1);
  alpha_group2 ~ normal(0, sigma_group2);

  // Likelihood
  for (n in 1:N) {
    real eta = alpha + dot_product(X[n], beta)
               + alpha_group1[group1_id[n]]
               + alpha_group2[group2_id[n]];
    y[n] ~ bernoulli_logit(eta);
  }
}

generated quantities {
  vector[N] log_lik;
  for (n in 1:N) {
    real eta = alpha + dot_product(X[n], beta)
               + alpha_group1[group1_id[n]]
               + alpha_group2[group2_id[n]];
    log_lik[n] = bernoulli_logit_lpmf(y[n] | eta);
  }
}
"


# output stan code
writeLines(stan_code_pool, "stan_code/pool_logistic.stan")
writeLines(stan_code_hier, "stan_code/hier_logistic.stan")
writeLines(stan_code_hier_2, "stan_code/hier_2group_logistic.stan")
writeLines(stan_code_hier_2_lasso, "stan_code/hier_2group_logistic_lasso.stan")


# -----------------------------------------------------------
# 3. Fit model function
# -----------------------------------------------------------

# pool, Hierarchical (1 group)
fit_model <- function(model_name, hier_var = NULL) {
  if (is.null(hier_var)) {
    # Pooled model
    X <- model.matrix(Revenue ~ . - 1, data = data)
    stan_data <- list(N = nrow(X), K = ncol(X), X = X, y = y)
    stan_file <- "stan_code/pool_logistic.stan"
  } else {
    # Hierarchical model
    X <- model.matrix(as.formula(paste("Revenue ~ . -", hier_var)), data = data)[, -1]
    group <- as.integer(data[[hier_var]])
    stan_data <- list(N = nrow(X), K = ncol(X), J = length(unique(group)),
                      group_id = group, X = X, y = y)
    stan_file <- "stan_code/hier_logistic.stan"
  }
  
  cat("\n-----------------------------------------------------------\n")
  cat("Fitting:", model_name, "\n")
  cat("-----------------------------------------------------------\n")
  
  fit <- stan(file = stan_file, data = stan_data,
              chains = 4, iter = 2000, warmup = 1000, seed = 123)
  
  return(list(
    name = model_name,
    hier_var = hier_var,
    fit = fit,
    X = X,
    stan_data = stan_data
  ))
}

# Hierarchical 2 group
fit_model_two_groups <- function(model_name, hier_vars, use_lasso = FALSE) {
  stopifnot(length(hier_vars) == 2)
  
  X <- model.matrix(as.formula(paste("Revenue ~ . -", paste(hier_vars, collapse=" - "))), data = data)[, -1]
  
  g1 <- as.integer(data[[hier_vars[1]]])
  g2 <- as.integer(data[[hier_vars[2]]])
  
  stan_data <- list(
    N = nrow(X), K = ncol(X),
    J1 = length(unique(g1)), J2 = length(unique(g2)),
    group1_id = g1, group2_id = g2,
    X = X, y = y
  )
  
  stan_file <- if (use_lasso) {
    "stan_code/hier_2group_logistic_lasso.stan"
  } else {
    "stan_code/hier_2group_logistic.stan"
  }
  
  fit <- stan(file = stan_file, data = stan_data,
              chains = 4, iter = 2000, warmup = 1000, seed = 123)
  
  return(list(name = model_name, hier_vars = hier_vars, fit = fit))
}


# -----------------------------------------------------------
# 4. Evaluate model function
# -----------------------------------------------------------

evaluate_model <- function(model_obj) {
  fit <- model_obj$fit
  
  # log-likelihood
  log_lik <- extract_log_lik(fit, "log_lik")
  loo_res <- loo(log_lik)
  waic_res <- waic(log_lik)
  
  cat("\n===== Model summary:", model_obj$name, "=====\n")
  
  param_names <- names(rstan::extract(fit))
  if ("sigma_group1" %in% param_names && "sigma_group2" %in% param_names) {
    # 2 group Hierarchical
    print(fit, pars = c("alpha", "beta", "sigma_group1", "sigma_group2"), probs = c(0.025, 0.5, 0.975))
  } else if ("sigma_group" %in% param_names) {
    # 1 group Hierarchical
    print(fit, pars = c("alpha", "beta", "sigma_group"), probs = c(0.025, 0.5, 0.975))
  } else {
    # pooled model
    print(fit, pars = c("alpha", "beta"), probs = c(0.025, 0.5, 0.975))
  }
  
  # --- Baseline Probability---
  alpha_summary <- summary(fit, pars = "alpha")$summary
  baseline_prob <- plogis(alpha_summary[,"50%"])
  cat(sprintf("\nBaseline purchase probability ≈ %.2f%%\n", baseline_prob * 100))
  
  # --- Predictive evaluation ---
  cat(sprintf("\n--- Predictive performance ---\n"))
  cat(sprintf("elpd_loo = %.1f (SE = %.1f)\n", 
              loo_res$estimates["elpd_loo","Estimate"], 
              loo_res$estimates["elpd_loo","SE"]))
  cat(sprintf("p_loo = %.1f | looic = %.1f | n_bad_k = %d\n",
              loo_res$estimates["p_loo","Estimate"],
              loo_res$estimates["looic","Estimate"],
              sum(loo_res$diagnostics$pareto_k > 0.7)))
  
  cat(sprintf("elpd_waic = %.1f | p_waic = %.1f | waic = %.1f\n",
              waic_res$estimates["elpd_waic","Estimate"],
              waic_res$estimates["p_waic","Estimate"],
              waic_res$estimates["waic","Estimate"]))
  
  return(list(loo = loo_res, waic = waic_res))
}


# -----------------------------------------------------------
# 3. Run all models
# -----------------------------------------------------------
fit_pool    <- fit_model("Pooled logistic regression")
fit_month   <- fit_model("Hierarchical logistic regression - Month", "Month")
fit_region  <- fit_model("Hierarchical logistic regression - Region", "Region")
fit_traffic <- fit_model("Hierarchical logistic regression - TrafficType", "TrafficType")
fit_region_traffic <- fit_model_two_groups("Hierarchical logistic regression - Region + TrafficType",
  hier_vars = c("Region", "TrafficType")
)
fit_region_traffic_lasso <- fit_model_two_groups("Hierarchical logistic regression - Region + TrafficType",
                                           hier_vars = c("Region", "TrafficType"), 
                                           use_lasso = TRUE)

# -----------------------------------------------------------
# 4. Evaluate all models
# -----------------------------------------------------------
eval_pool    <- evaluate_model(fit_pool)
eval_month   <- evaluate_model(fit_month)
eval_region  <- evaluate_model(fit_region)
eval_traffic <- evaluate_model(fit_traffic)
eval_region_traffic <- evaluate_model(fit_region_traffic)
eval_region_traffic_lasso <- evaluate_model(fit_region_traffic_lasso)

# Compare
loo_compare(eval_pool$loo, eval_month$loo, eval_region$loo, eval_traffic$loo, eval_region_traffic$loo, eval_region_traffic_lasso$loo)

eval_pool
eval_month
eval_region
eval_traffic
eval_region_traffic
eval_region_traffic_lasso


# -----------------------------------------------------------
# 5. Estimate
# -----------------------------------------------------------
fit_best <- fit_region_traffic_lasso$fit
param_summary <- summary(fit_best, probs = c(0.025, 0.5, 0.975))$summary
param_summary <- as.data.frame(param_summary)
param_summary$Parameter <- rownames(param_summary)
colnames(param_summary)

param_clean <- param_summary[!grepl("^log_lik", param_summary$Parameter), ]
appendix_tbl <- param_clean[, c("Parameter", "mean", "se_mean", "sd", "2.5%", "97.5%", "Rhat")]

round_by_mcse <- function(x, mcse) {
  if (is.na(mcse) || mcse == 0) return(signif(x, 3))   # fallback if MCSE missing
    digits <- floor(log10(abs(mcse)))
    round(x, digits = -digits - 1)
}

# Apply rounding rule to all main numeric columns
appendix_tbl$mean   <- mapply(round_by_mcse, appendix_tbl$mean, appendix_tbl$se_mean)
appendix_tbl$sd     <- mapply(round_by_mcse, appendix_tbl$sd, appendix_tbl$se_mean)
appendix_tbl$`2.5%` <- mapply(round_by_mcse, appendix_tbl$`2.5%`, appendix_tbl$se_mean)
appendix_tbl$`97.5%`<- mapply(round_by_mcse, appendix_tbl$`97.5%`, appendix_tbl$se_mean)

# Round Rhat to two decimals as required by reporting guidelines
appendix_tbl$Rhat <- round(appendix_tbl$Rhat, 2)

# Print and export the final appendix table
cat("\n===== Appendix Table: Posterior parameter summary (MCSE-based rounding, no log_lik) =====\n")
print(appendix_tbl, row.names = FALSE)
write.csv(appendix_tbl, "appendix/appendix_posterior_summary.csv", row.names = FALSE)


# 6. Check
X <- model.matrix(as.formula(paste("Revenue ~ . -", paste(hier_vars = c("Region", "TrafficType"), collapse=" - "))), data = data)[, -1]
colnames(X)

# Traceplots (check convergence)
traceplot(fit_region_traffic_lasso$fit, 
          pars = c("alpha", "sigma_group1", "sigma_group2", "tau"), 
          inc_warmup = FALSE)


