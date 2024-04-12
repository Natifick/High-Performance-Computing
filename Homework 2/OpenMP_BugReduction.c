#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

// Run with
// g++ -o out.o OpenMP_BugReduction.c -fopenmp && ./out.o

float dotprod(float * a, float * b, float &sum, size_t N)
{
    int i, tid;
    //float sum;

    tid = omp_get_thread_num();

    #pragma omp for
    for (i = 0; i < N; ++i)
    {
        sum += a[i] * b[i];
        printf("tid = %d i = %d\n", tid, i);
    }

    return sum;
}

int main (int argc, char *argv[])
{
    const size_t N = 100;
    int i;
    float sum;
    float a[N], b[N];


    for (i = 0; i < N; ++i)
    {
        a[i] = b[i] = (double)i;
    }

    sum = 0.0;

    #pragma omp parallel reduction(+:sum)
    dotprod(a, b, sum, N);

    printf("Sum = %f\n",sum);

    return 0;
}
