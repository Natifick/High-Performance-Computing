
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <mpi.h>

/* Note: the MPI datatype corresponding to "signed char" is MPI_CHAR */
typedef signed char cell_t;

/* number of ghost cells on each side; this program assumes HALO ==
   1. */
const int HALO = 1;

void step( const cell_t *cur, cell_t *next, int ext_n )
{
    int i;
    const int LEFT = HALO;
    const int RIGHT = ext_n - HALO - 1;
    for (i = LEFT; i <= RIGHT; i++) {
        const cell_t east = cur[i-1];
        const cell_t center = cur[i];
        const cell_t west = cur[i+1];
        next[i] = ( (east && !center && !west) ||
                    (!east && !center && west) ||
                    (!east && center && !west) ||
                    (!east && center && west) );
    }
}

/**
 * Initialize the domain; all cells are 0, with the exception of a
 * single cell in the middle of the domain. `ext_n` is the width of the
 * domain PLUS the ghost cells.
 */
void init_domain( cell_t *cur, int ext_n )
{
    int i;
    for (i=0; i<ext_n; i++) {
        cur[i] = 0;
    }
    cur[ext_n/2] = 1;
}

/**
 * Dump the current state of the automaton to PBM file `out`. `ext_n`
 * is the true width of the domain PLUS the ghost cells.
 */
void dump_state( FILE *out, const cell_t *cur, int ext_n )
{
    int i;
    const int LEFT = HALO;
    const int RIGHT = ext_n - HALO - 1;

    for (i=LEFT; i<=RIGHT; i++) {
        fprintf(out, "%d ", cur[i]);
    }
    fprintf(out, "\n");
}

int main( int argc, char* argv[] )
{
    const char *outname = "rule30_parallel.pbm";
    FILE *out = NULL;
    int width = 1024, nsteps = 1024, s;
    /* `cur` is the memory buffer containint `width` elements; this is
       the full state of the CA. */
    cell_t *cur = NULL, *tmp;
    int my_rank, comm_sz;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);

    double t1, t2;

    t1 = MPI_Wtime();

    if ( 0 == my_rank && argc > 3 ) {
        fprintf(stderr, "Usage: %s [width [nsteps]]\n", argv[0]);
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    if ( argc > 1 ) {
        width = atoi(argv[1]);
    }

    if ( argc > 2 ) {
        nsteps = atoi(argv[2]);
    }

    if ( (0 == my_rank) && (width % comm_sz) ) {
        printf("The image width (%d) must be a multiple of comm_sz (%d)\n", width, comm_sz);
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    /* `ext_width` is the width PLUS the halo on both sides. The halo
       is required by the serial version only; the parallel version
       would work fine with a (full) domain of length `width`, but
       would still require the halo in the local partitions. */
    const int ext_width = width + 2*HALO;

    /* The master creates the output file */
    if ( 0 == my_rank ) {
        out = fopen(outname, "w");
        if ( !out ) {
            fprintf(stderr, "FATAL: Cannot create %s\n", outname);
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        }
        fprintf(out, "P1\n");
        fprintf(out, "# Produced by mpi-rule30\n");
        fprintf(out, "%d %d\n", width, nsteps);

        /* Initialize the domain

           NOTE: the parallel version does not need ghost cells in the
           cur[] array, but only in the local_cur[] blocks that are
           stored within each MPI process. For simplicity we keep the
           ghost cells in cur[]; after getting a working version,
           modify your program to remove them. */
        cur = (cell_t*)malloc( ext_width * sizeof(*cur) ); assert(cur != NULL);
        init_domain(cur, ext_width);
    }

    /* compute the rank of the next and previous process on the
       chain. These will be used to exchange the boundary */
    const int rank_next = (my_rank + 1) % comm_sz;
    const int rank_prev = (my_rank - 1 + comm_sz) % comm_sz;

    /* compute the size of each local domain; this should be set to
       `width / comm_sz + 2*HALO`, since it must include the ghost
       cells */
    const int local_width = width / comm_sz;
    const int local_ext_width = local_width + 2*HALO;

    /* `local_cur` and `local_next` are the local domains, handled by
       each MPI process. They both have `local_ext_width` elements each */
    cell_t *local_cur = (cell_t*)malloc(local_ext_width * sizeof(*local_cur)); assert(local_cur != NULL);
    cell_t *local_next = (cell_t*)malloc(local_ext_width * sizeof(*local_next)); assert(local_next != NULL);

    const int LEFT_GHOST = 0;
    const int LEFT = LEFT_GHOST + HALO;

    /* The master distributes the domain cur[] to the other MPI
       processes. Each process receives `width/comm_sz` elements of
       type MPI_INT. Note: the parallel version does not require ghost
       cells in cur[], so it would be possible to allocate exactly
       `width` elements in cur[]. */
    const int LOCAL_LEFT_GHOST = 0;
    const int LOCAL_LEFT = LOCAL_LEFT_GHOST + HALO;
    const int LOCAL_RIGHT = local_ext_width - 1 - HALO;
    const int LOCAL_RIGHT_GHOST = LOCAL_RIGHT + HALO;

    MPI_Scatter( &cur[LEFT],            /* sendbuf      */
                 local_width,           /* sendcount    */
                 MPI_CHAR,              /* datatype     */
                 &local_cur[LOCAL_LEFT],/* recvbuf      */
                 local_width,           /* recvcount    */
                 MPI_CHAR,              /* datatype     */
                 0,                     /* root         */
                 MPI_COMM_WORLD
                 );

    for (s=0; s<nsteps; s++) {
        printf("%d", my_rank);
        /* This is OK; the master dumps the current state of the automaton */
        if ( 0 == my_rank ) {
            /* Dump the current state to the output image */
            dump_state(out, cur, ext_width);
        }

        /* Send right boundary to right neighbor; receive left
           boundary from left neighbor (X=halo)

                 _________          _________
                /         V        /         V
           ...---+-+     +-+--------+-+     +-+---...
                 |X|     |X|        |X|     |X|
           ...---+-+     +-+--------+-+     +-+---...
                           local_cur

        */
        MPI_Sendrecv( &local_cur[LOCAL_RIGHT], /* sendbuf */
                      HALO,             /* sendcount    */
                      MPI_CHAR,         /* datatype     */
                      rank_next,        /* dest         */
                      0,                /* sendtag      */
                      &local_cur[LOCAL_LEFT_GHOST],/* recvbuf      */
                      HALO,             /* recvcount    */
                      MPI_CHAR,         /* datatype     */
                      rank_prev,        /* source       */
                      0,                /* recvtag      */
                      MPI_COMM_WORLD,
                      MPI_STATUS_IGNORE
                      );

        /* send left boundary to left neighbor; receive right boundary
           from right neighbor

                   _________          _________
                  V         \        V         \
           ...---+-+     +-+--------+-+     +-+---...
                 |X|     |X|        |X|     |X|
           ...---+-+     +-+--------+-+     +-+---...
                           local_cur

        */
        MPI_Sendrecv( &local_cur[LOCAL_LEFT], /* sendbuf      */
                      HALO,             /* sendcount    */
                      MPI_CHAR,         /* datatype     */
                      rank_prev,        /* dest         */
                      0,                /* sendtag      */
                      &local_cur[LOCAL_RIGHT_GHOST], /* recvbuf */
                      HALO,             /* recvcount    */
                      MPI_CHAR,         /* datatype     */
                      rank_next,        /* source       */
                      0,                /* recvtag      */
                      MPI_COMM_WORLD,
                      MPI_STATUS_IGNORE
                      );

        step( local_cur, local_next, local_ext_width );

        /* Gather the updated local domains at the root; it is
           possible to gather the result at cur[] instead than next[];
           actually, in the parallel version, next[] is not needed at
           all. */
        MPI_Gather( &local_next[LOCAL_LEFT],/* sendbuf      */
                    local_width,        /* sendcount    */
                    MPI_CHAR,           /* datatype     */
                    &cur[LEFT],         /* recvbuf      */
                    local_width,        /* recvcount    */
                    MPI_CHAR,           /* datatype     */
                    0,                  /* root         */
                    MPI_COMM_WORLD
                    );

        /* swap current and next domain */
        tmp = local_cur;
        local_cur = local_next;
        local_next = tmp;
    }

    t2 = MPI_Wtime();

    /* All done, free memory */
    free(local_cur);
    free(local_next);
    free(cur);

    if ( 0 == my_rank ) {
        fclose(out);
        printf("elapsed time = %f\n", t2 - t1);
    }

    MPI_Finalize();

    return EXIT_SUCCESS;
}
