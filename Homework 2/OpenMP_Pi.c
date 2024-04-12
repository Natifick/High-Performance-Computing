#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

int main()
{
    const size_t N = 100000000;
    double step;

    double x, pi, sum = 0.;

    step = 1. / (double)N;
    
    double start, end;
    start = omp_get_wtime();

    #pragma omp parallel for reduction(+:sum) firstprivate(step) private(x)
    for (int i = 0; i < N; ++i)
    {
        x = (i + 0.5) * step;
	x = 4.0 / (1.0 + x * x);
        sum += x;
    }

    pi = step * sum;
    end = omp_get_wtime();

    printf("pi = %.16f\nTime = %.5f\n", pi, end-start);
    
    // ---- Monte-Carlo ----
    unsigned int seed = 420;
    double y;
    start = omp_get_wtime();
    sum = 0;
    #pragma omp parallel for firstprivate(seed) private(x, y) reduction(+:sum)
    for (int i = 0; i < N; i++) {
	seed = seed * i; //omp_get_thread_num();
	x = (double)rand_r(&seed) / RAND_MAX;
	y = (double)rand_r(&seed) / RAND_MAX;
        if (x*x + y*y < 1){
	    sum += 1;
	}
	else {
	    sum -= 1;
	}
    }
    end = omp_get_wtime();
    printf("Monte-Carlo pi = %.16f\nTime = %.5f\n", (double)sum*4/N, end-start);
    return 0;
}
