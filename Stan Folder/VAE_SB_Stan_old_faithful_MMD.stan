data {
    int<lower=0> K; // output dim
    int<lower=0> n1; // hidden layer 1 dim
    int<lower=0> n2; // hidden layer1 2 dim
    int<lower=0> m1; // latent dim
    int<lower=0> N; // number of observation
    matrix[4,N] x; // known observations
    matrix[m1, n1] W1; // weight for latent layer to hidden layer 1 in decoder
    matrix[1, n1] b1; // bias for latent layer to hidden layer 2
    matrix[n1, n2] W2; // weight for latent layer to hidden layer in decoder
    matrix[1, n2] b2; // bias for hidden layer 1 to hidden layer 2 in decoder
    matrix[n2, K] W3; // weight for hidden layer 2 to output layer in decoder
    matrix[1, K] b3; // bias for hidden layer 2 to output layer in decoder
    vector[K] mu1; // mean of theta_mu_star
    matrix[K,K] cov1; // covariance of theta_mu_star
    vector[K] mu2; // mean of theta_sd_star
    matrix[K,K] cov2; // covariance of theta_sd_star
}

parameters {
    vector<lower=0,upper=1>[m1] Z;
    ordered[K] theta_mu_star_1;
    vector[K] theta_mu_star_2;
    ordered[K] theta_mu_star_3;
    ordered[K] theta_mu_star_4;
    vector<lower=0>[K] theta_sd_star_1;
    vector<lower=0>[K] theta_sd_star_2;
    vector<lower=0>[K] theta_sd_star_3;
    vector<lower=0>[K] theta_sd_star_4;
}

transformed parameters {
    matrix[1,m1] Z_;
    matrix[4,K] theta_mu_star;
    matrix[4,K] theta_sd_star;
    simplex[K] pi;
    Z_=to_matrix(Z,1,m1);
    pi=softmax(to_vector(tanh(tanh(Z_*W1+b1)*W2+b2)*W3+b3));   
    theta_mu_star=append_row(append_row(to_row_vector(theta_mu_star_1), to_row_vector(theta_mu_star_2)),append_row(to_row_vector(theta_mu_star_3), to_row_vector(theta_mu_star_4)));
    theta_sd_star=append_row(append_row(to_row_vector(theta_sd_star_1), to_row_vector(theta_sd_star_2)),append_row(to_row_vector(theta_sd_star_3), to_row_vector(theta_sd_star_4)));
}

model {
    vector[K] log_pi = log(pi);
    vector[K] log_theta_sd_star_1 = log(theta_sd_star_1);
    vector[K] log_theta_sd_star_2 = log(theta_sd_star_2);
    vector[K] log_theta_sd_star_3 = log(theta_sd_star_3);
    vector[K] log_theta_sd_star_4 = log(theta_sd_star_4);
    for (m in 1:m1)
        Z[m] ~ beta(1,1);
    theta_mu_star_1 ~ multi_normal(mu1,cov1);
    theta_mu_star_2 ~ multi_normal(mu1,cov1);
    theta_mu_star_3 ~ multi_normal(mu1,cov1);
    theta_mu_star_4 ~ multi_normal(mu1,cov1);
    log_theta_sd_star_1 ~ multi_normal(mu2,cov2);
    log_theta_sd_star_2 ~ multi_normal(mu2,cov2);
    log_theta_sd_star_3 ~ multi_normal(mu2,cov2);
    log_theta_sd_star_4 ~ multi_normal(mu2,cov2);
    for (n in 1:N){
        vector[K] lps = log_pi;
        for (k in 1:K)
            lps[k] += multi_normal_lpdf(x[:,n] | theta_mu_star[:,k],diag_matrix(theta_sd_star[:,k]));
        target += log_sum_exp(lps);
    }
}
