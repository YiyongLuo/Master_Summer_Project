data {
    int<lower=0> K; // output dim
    int<lower=0> n2; // hidden layer dim
    int<lower=0> m1; // latent dim
    int<lower=0> N; // number of observation
    real x[N]; // known observations
    matrix[m1, n2] W1; // weight for latent layer to hidden layer in decoder
    matrix[1, n2] b1; // bias for latent layer to hidden layer in decoder
    matrix[n2, K] W3; // weight for hidden layer to output layer in decoder
    matrix[1, K] b3; // bias for hidden layer to output layer in decoder
    vector[K] mu1; // mean of theta_mu_star
    matrix[K,K] cov1; // covariance of theta_mu_star
    vector[K] mu2; // mean of theta_sd_star
    matrix[K,K] cov2; // covariance of theta_sd_star
}

parameters {
    vector[m1] Z;
    ordered[K] theta_mu_star;
    vector<lower=0>[K] theta_sd_star;
}

transformed parameters {
    matrix[1,m1] Z_;
    simplex[K] pi;
    Z_=to_matrix(Z,1,m1);
    pi=softmax(to_vector((tanh(Z_*W1+b1))*W3+b3));         
}

model {
    vector[K] log_pi = log(pi);
    vector[K] log_theta_sd_star = log(theta_sd_star);
    for (m in 1:m1)
        Z[m] ~ normal(0,1);
    theta_mu_star ~ multi_normal(mu1,cov1);
    log_theta_sd_star ~ multi_normal(mu2,cov2);
    for (n in 1:N){
        vector[K] lps = log_pi;
        for (k in 1:K)
            lps[k] += normal_lpdf(x[n] | theta_mu_star[k], theta_sd_star[k]);
        target += log_sum_exp(lps);
    }
}
