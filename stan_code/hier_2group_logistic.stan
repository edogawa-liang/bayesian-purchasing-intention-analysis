
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

