#include <mpi.h>
#include <iostream>
#include <vector>
#include <random> // for std::mt19937

static const size_t N = 24;

void pretty_print(size_t N, int* array){
    for(size_t i = 0; i < N; ++i){
        std::cout << array[i] << "\t";
    }
    std::cout << std::endl;
}


int main(int argc, char** argv) {

    int prank;
    int psize;
    MPI_Status status;

    MPI_Init(&argc, &argv);

    MPI_Comm_size(MPI_COMM_WORLD, &psize);
    MPI_Comm_rank(MPI_COMM_WORLD, &prank);

    std::mt19937 mt (42 * prank);
    std::uniform_int_distribution<int> dis(0, psize-2);
    
    // Remember the names that we passed
    // Initially there's only first process in the list
    std::vector<int> passed_names(N, 0);
    // Initialy first process holds the ball
    int ball_holder = 0;
    int current_iteration = 1;
    
    while (current_iteration <= N) {
        if (ball_holder == prank) {
            int new_member = dis(mt);
            new_member += int (new_member >= prank);
            MPI_Ssend(&current_iteration, 1, MPI_INT, new_member, 0, MPI_COMM_WORLD);
            MPI_Ssend(passed_names.data(), current_iteration, MPI_INT, new_member, 1, MPI_COMM_WORLD);
            ball_holder = new_member;
        }
        else {
            MPI_Recv(&current_iteration, 1, MPI_INT, MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, &status);
            //ball_holder = prank;
            if (current_iteration > N) {
                continue;
            }
            MPI_Recv(passed_names.data(), current_iteration, MPI_INT, MPI_ANY_SOURCE, 1, MPI_COMM_WORLD, &status);
            passed_names[current_iteration-1] = prank;
            current_iteration += 1;
            ball_holder = prank;
        }
    }
    if (prank == ball_holder) {
        pretty_print(N, passed_names.data());
        for (int i = 0; i < psize; i++) {
            if (i == prank) {
                continue;
            }
            MPI_Ssend(&current_iteration, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
            std::cout << "Sent to " << i << std::endl;
        }
    }
    std::cout << "Done";
    MPI_Finalize();
    return 0;
}
