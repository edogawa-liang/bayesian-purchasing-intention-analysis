
data {
  int<lower=0> N;                 // number of observations
  int<lower=0> K;                 // number of predictors
  int<lower=1> J1;                // number of groups for hierarchy 1
  int<lower=1> J2;                // number of groups for hierarchy 2
  int<lower=1,upper=J1> group1_id[N];  // group 1 ID for each observation
  int<lower=1,upper=J2> group2_id[N];  // group 2 ID for each observation
  matrix[N, K] X;                 // predictor matrix
  int<lower=0,upper=1> y[N];      // binary outcome
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

