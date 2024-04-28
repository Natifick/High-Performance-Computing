#include <iostream>
#include <mpi.h>
#include <random>
#include <vector>

// Grid size, will be splitted among processes
const int N = 1000000;

void make_move(std::vector<int>& my_part, int* permutation_table) {
    std::vector<int> new_arr(my_part.size());
    int sum;
    for (int i = 0; i < my_part.size(); i++) {
        sum = my_part[i-1] + my_part[i]*2 + my_part[i+1]*4;
        new_arr[i] = permutation_table[sum];
    }
    // Pass the values back to original array
    for (int i = 0; i < my_part.size(); i++) {
        my_part[i] = new_arr[i];
    }
}


int main(int argc, char *argv[]) {
    
    int prank;
    int psize;
    MPI_Status status;

    MPI_Init(&argc, &argv);

    MPI_Comm_size(MPI_COMM_WORLD, &psize);
    MPI_Comm_rank(MPI_COMM_WORLD, &prank);
    
    std::mt19937 rng(42*prank);
    // All values are either 0 or 1
    std::uniform_int_distribution<std::mt19937::result_type> dist(0,1);
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
    // Init for this particular process
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
    std::cout << subsize << std::endl;
    // From left and right we have "ghost points"
    std::vector<int> subgrid(subsize+2);
    for(int i=0; i<subgrid.size(); i++) {
        subgrid[i] = dist(rng);
    }
    /*std::vector<int> full_space(N);
    if (prank == 0) {
        for (int i = 0; i < N; i++) {
            full_space[i] = dist(rng);
            std::cout << full_space[i] << " ";
        }
        std::cout << std::endl;
    }
    int local_width = N / psize;
    std::vector<int> local_space(local_width);
    MPI_Scatter(&full_space, local_width, MPI_INT, &local_space, local_width, MPI_INT, 0, MPI_COMM_WORLD);
    
    int* my_part = &full_space[width];
    int* ghost_left = &full_space[0];
    int* ghost_right = &full_space[width+1];
    int* inner_left = &full_space[]
    */
    // ----- The algorithm part ----
    int n_epochs = 100;
    int local_result = 0;
    int global_result = 0;
    for (int ep = 0; ep < n_epochs; ep++) {
        make_move(subgrid, permutation_table);
        MPI_Sendrecv(&subgrid[0], 1, MPI_INT, 
                     prank!=0?prank-1:psize-1, 1,
                     &subgrid[subsize-1], 1, MPI_INT,
                     prank!=psize-1?prank+1:0, 1,
                     MPI_COMM_WORLD, &status);
        MPI_Sendrecv(&subgrid[subsize-1], 1, MPI_INT,
                     prank!=psize-1?prank+1:0, 2,
                     &subgrid[0], 1, MPI_INT,
                     prank!=0?prank-1:psize-1, 2,
                     MPI_COMM_WORLD, &status);
        std::cout << prank << " ";
    }
    /*if (prank == 1) {
        for (int i = 0; i < local_width; i++) {
            std::cout << local_space[i] << " ";
        }
    }*/
    

    MPI_Finalize();
    return 0;
}
