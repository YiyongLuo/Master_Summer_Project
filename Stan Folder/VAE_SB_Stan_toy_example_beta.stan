data {
    int<lower=0> K; // output dim
    int<lower=0> n1; // hidden layer2 dim
    int<lower=0> n2; // hiddeb layer1 dim
    int<lower=0> m1; // latent dim
    int<lower=0> N; // number of observation
    real x[N]; // known observations
    matrix[m1, n1] W1; // weight for latent layer to hidden layer 1 in decoder
    matrix[1, n1] b1; // bias for latent layer to hidden layer 1 in decoder
    matrix[n1, n2] W2; // weight for hidden layer 1 to hidden layer 2 in decoder
    matrix[1, n2] b2; // bias for hidden layer 1 to hidden layer 2 in decoder
    matrix[n2, K] W3; // weight for hidden layer 2 to output layer in decoder
    matrix[1, K] b3; // bias for hidden layer 2 to output layer in decoder
    vector[K] mu1; // mean of theta_mu_star
    matrix[K,K] cov1; // covariance of theta_mu_star
}

parameters {
    vector<lower=0,upper=1>[m1] Z;
    ordered[K] theta_mu_star;
}

transformed parameters {
    matrix[1,m1] Z_;
    simplex[K] pi;
    Z_=to_matrix(Z,1,m1);
    pi=softmax(to_vector(tanh(tanh(Z_*W1+b1)*W2+b2)*W3+b3));         
}

model {
    vector[K] log_pi = log(pi);
    for (m in 1:m1)
        Z[m] ~ beta(1,0.5);
    theta_mu_star ~ multi_normal(mu1,cov1);
    for (n in 1:N){
        vector[K] lps = log_pi;
        for (k in 1:K)
            lps[k] += normal_lpdf(x[n] | theta_mu_star[k], 0.7);
        target += log_sum_exp(lps);
    }
}
