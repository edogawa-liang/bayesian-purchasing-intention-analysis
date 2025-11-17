
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

