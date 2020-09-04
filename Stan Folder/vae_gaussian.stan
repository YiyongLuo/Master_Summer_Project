data {
      int<lower=0> N; // output dim
      int<lower=0> n1; // hidden layer1 dim
      int<lower=0> n2; // hidden layer2 dim
      int<lower=0> m1; // latent dim
      vector[N] y; // known observations
      matrix[m1, n2] W1; // weight for input layer to hidden layer 1 in decoder
      matrix[1, n2] b1; // bias for input layer to hidden layer 1 in decoder
      matrix[n2, n1] W2; // weight for hidden layer 1 to hidden layer 2
                            in decoder
      matrix[1, n1] b2; // bias for hidden layer 1 to hidden layer 2
                             in decoder
      matrix[n1, N] W3; // weight for hidden layer 2 to output layer in decoder
      matrix[1, N] b3; // bias for hidden layer 2 to output layer in decoder
      matrix[m1,m1] cov; // prior covariance matrix of latent variable
      vector[m1] mu; // prior mean of latent variable
}

parameters {
      vector[m1] Z;
      real<lower=0> sigma2;
}

transformed parameters {
      matrix[1,m1] Z1;
      vector[N] output;
      Z1=to_matrix(Z,1,m1);
      output=to_vector((tanh(tanh(Z1*W1+b1)*W2+b2))*W3+b3);
}

model {
      Z ~ multi_normal(mu, cov);
      sigma2 ~ uniform(0,10);
      y ~ normal(output, sigma2);
}

generated quantities {
      vector[N] y2;
      for (i in 1:N)
            y2[i] = normal_rng(output[i],sigma2);
}
