#include <stdio.h>
#include <omp.h>
#include <math.h>
#include <vector>
#include <iostream>
#include <algorithm>
#include <random>
#include <cstdlib>

#pragma omp declare reduction(vsum : std::vector<double> : std::transform(omp_out.begin(), omp_out.end(), omp_in.begin(), omp_out.begin(), std::plus<double>())) initializer(omp_priv = decltype(omp_orig)(omp_orig.size()))
 
void init_random(std::vector<double> &arr, std::uniform_real_distribution<double>& range, std::mt19937 gen) {
    //#pragma omp parallel for
    for (int i = 0; i < arr.size(); i++) {
        arr[i] = range(gen);
    }
}


double MSE(const std::vector<std::vector<double>> &X, const std::vector<double> &weights, 
           const double &bias, const std::vector<double> &y_true) {
    double error = 0.0;
    #pragma omp parallel for reduction(+:error)
    for (int i = 0; i < X.size(); i++) {
        double y_diff = -y_true[i];
        for (int j = 0; j < weights.size(); j++) {
            y_diff += X[i][j]*weights[j];
        }
        y_diff += bias;
        error += y_diff*y_diff;
    }
    return error;
}


int main() {
    const int N = 40000; // n rows
    const int K = 40; // n features
    std::mt19937 gen (0);//; std::random_device{}());

    std::uniform_real_distribution<double> uniform(-K, K);
    std::normal_distribution<double> normal(0, 0.5);

    std::vector<double> w_true(K);
    init_random(w_true, uniform, gen);
    double b_true = uniform(gen);
    
    std::vector<double> w_pred(K);
    init_random(w_pred, uniform, gen);
    double b_pred = uniform(gen);

    std::vector<std::vector<double>> X;
    // Fill in the X matrix
    for (int i = 0; i < N; i++) {
        std::vector<double> row(K);
        init_random(row, uniform, gen);
        X.push_back(row);
    }
    std::vector<double> y_true(N, 0);
    #pragma omp parallel for reduction(vsum:y_true)
    for (int i=0; i < N; i++) {
        for (int j = 0; j < K; j++) {
            y_true[i] += X[i][j] * w_true[j];
        }
        y_true[i] += b_true + normal(gen);
    }
    std::cout << "Minimal possible error = " << MSE(X, w_true, b_true, y_true) << std::endl;
    
    int n_epochs = 50;
    double lr = 0.00003;
    std::vector<double> new_weights = w_pred;
    std::vector<double> errors(N);
    double new_bias;
    for (int ep = 0; ep < n_epochs; ep++) {
        std::fill(errors.begin(), errors.end(), 0.0);
        #pragma omp parallel for reduction(vsum:errors)
        for(int i = 0; i < N; i++) {
            for (int j = 0; j < K; j++) {
                errors[i] += w_pred[j]*X[i][j];
            }
            errors[i] += b_pred - y_true[i];
        }
        #pragma omp parallel for reduction(vsum:new_weights) reduction(+:new_bias)
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < K; j++) {
                new_weights[j] -= lr * X[i][j] * errors[i] / N;
                if (i == 0) {
                    new_weights[j] -= lr * w_pred[j];
                }
            }
            new_bias -= lr * errors[i] / N;
        }
        w_pred = new_weights;
        b_pred = new_bias;
        if (ep % 10 != 0) {
            continue;
        }
        for (int i = 0; i < K; i++) {
            std::cout << w_pred[i] - w_true[i] << " ";
        }
        std::cout << std::endl;
        std::cout << "MSE = " << MSE(X, w_pred, b_pred, y_true) << std::endl;
    }

    return 0;
}
