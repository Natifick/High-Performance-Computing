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
    for (int i = 0; i < arr.size(); i++) {
        arr[i] = range(gen);
    }
}

float get_error(std::vector<std::vector<double>> &A, std::vector<double> &x, std::vector<double> &y) {
    double error = 0;
    #pragma omp parallel for //collapse(2)
    for (int i = 0; i < A.size(); i++) {
        double prediction = 0.0;
        for (int j = 0; j < A[i].size(); j++) {
            prediction += A[i][j] * x[j];
        }
        error += (prediction - y[i])*(prediction - y[i]);
    }
    return error / A.size();
}

int main() {
    const int N=10000;
    std::mt19937 gen (std::random_device{}());
  
    std::uniform_real_distribution<double> small_range(0, 1);
    std::uniform_real_distribution<double> large_range(0, N); 
    // I don't use multithreading here, since the point of the speed-up is not in here

    // Create random matrix NxM
    std::vector<std::vector<double>> A;
    for (int i = 0; i < N; i++) {
        std::vector<double> row(N);
        init_random(row, small_range, gen);
        A.push_back(row);
    }
    // To ensure diagonal dominance
    for (int i = 0; i < N; i++) {
        A[i][i] = large_range(gen);
    }

    // Init random needed output N-sized vector
    std::vector<double> y(N);
    init_random(y, large_range, gen);

    // Init random prediction of shape M
    std::vector<double> x_pred(N);
    init_random(x_pred, small_range, gen);

    std::cout << get_error(A, x_pred, y);
    
    int n_epochs = 10;
    double start = omp_get_wtime();
    // --- The Jakobi algorithm ---
    for (int ep = 0; ep < n_epochs; ep++) {
        std::vector<double> new_x(N,0);
        #pragma omp parallel for reduction(vsum:new_x) num_threads(12)
        for (int i = 0; i < N; i++) {
        //#pragma omp for
            for (int j = 0; j < N; j++) {
                if (i != j) {
                    new_x[i] += A[i][j] * x_pred[j];
                }
            }
            new_x[i] = (y[i] - new_x[i]) / A[i][i];
        }
        x_pred = new_x;
        std::cout << get_error(A, x_pred, y) << std::endl;
    }
    double end = omp_get_wtime();
    std::cout << "Time = " << end-start << std::endl;

    return 0;
}
