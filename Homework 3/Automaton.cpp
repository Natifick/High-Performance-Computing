#include <iostream>
#include <mpi.h>
#include <random>
#include <vector>

// Grid size, will be splitted among processes
const int N = 1000000;

bool make_move(std::vector<int>& my_part, int* permutation_table) {
    std::vector<int> new_arr(my_part.size());
    int sum;
    for (int i = 1; i < my_part.size()-1; i++) {
        sum = my_part[i-1] + my_part[i]*2 + my_part[i+1]*4;
        new_arr[i] = permutation_table[sum];
    }
    // We track wether we did or did not change anything
    bool result = false;
    // Pass the values back to original array
    for (int i = 1; i < my_part.size()-1; i++) {
        if (my_part[i] != new_arr[i]) {
            result = true;
        }
        my_part[i] = new_arr[i];
    }
    return result;
}


int main(int argc, char *argv[]) {
    
    int prank;
    int psize;
    MPI_Status status;

    MPI_Init(&argc, &argv);

    MPI_Comm_size(MPI_COMM_WORLD, &psize);
    MPI_Comm_rank(MPI_COMM_WORLD, &prank);
    
    std::mt19937 rng(11*prank+3);
    // All values are either 0 or 1
    std::uniform_int_distribution<int> dist(0,1);
    // Create the rules for this game by setting permuation table
    int *permutation_table = (int*)malloc(sizeof(int)*8);
    if (prank == 0) {
        for (int i = 0; i < 8; i++) {
            permutation_table[i] = dist(rng);
        }
        for (int i = 1; i < psize; i++) {
            MPI_Ssend(permutation_table, 8, MPI_INT, i, 0, MPI_COMM_WORLD);
        }
    }
    else {
        MPI_Recv(permutation_table, 8, MPI_INT, 0, 0, MPI_COMM_WORLD, &status);
    }
    double t1, t2;
    t1 = MPI_Wtime();
    // Init for this particular process
    // I believe it's better than distributing single array to all processes
    int subsize;
    if (N % psize == 0) {
        subsize = N / psize;
    }
    else {
        // Last rank takes the leftover
        if (prank != psize-1) {
            subsize = (N / psize) + 1;
        }
        else {
            subsize = N % ((int)(N / psize) + 1);
        }
    }
    // From left and right we have "ghost points"
    std::vector<int> subgrid(subsize+2);
    for(int i=0; i<subgrid.size(); i++) {
        subgrid[i] = dist(rng);
    }
    
    // ----- The algorithm part ----
    int n_epochs = 100;
    int local_result = 0;
    int global_result = 0;
    for (int ep = 0; ep < n_epochs; ep++) {
        make_move(subgrid, permutation_table);
        MPI_Sendrecv(subgrid.data(), 1, MPI_INT, 
                     prank!=0?prank-1:psize-1, 1,
                     subgrid.data()+subsize+1, 1, MPI_INT,
                     prank!=psize-1?prank+1:0, 1,
                     MPI_COMM_WORLD, &status);
        MPI_Sendrecv(subgrid.data()+subsize+1, 1, MPI_INT,
                     prank!=psize-1?prank+1:0, 2,
                     subgrid.data(), 1, MPI_INT,
                     prank!=0?prank-1:psize-1, 2,
                     MPI_COMM_WORLD, &status);
    }
    // Time to aggregate back
    int *full_field = (int*)malloc(sizeof(int)*N);
    MPI_Gather(subgrid.data(), subsize, MPI_INT, 
               full_field, subsize, MPI_INT, 0, MPI_COMM_WORLD);
    t2 = MPI_Wtime();
    if (prank == 0) {
        printf("%f\n", t2 - t1);
    }

    MPI_Finalize();
    return 0;
}
