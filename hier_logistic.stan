
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

