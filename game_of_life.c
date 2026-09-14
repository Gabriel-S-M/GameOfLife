#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#include <omp.h>

#define DEFAULT_ROWS         30
#define DEFAULT_COLS         60
#define DEFAULT_GENERATIONS  100
#define DEFAULT_DELAY_MS     80
#define INITIAL_ALIVE_PROB   0.30

static int  *alloc_grid(int rows, int cols);
static void  free_grid(int *grid);
static void  init_random(int *grid, int rows, int cols, double prob);
static void  init_glider(int *grid, int rows, int cols);
static int   count_live_neighbors(const int *grid, int rows, int cols, int row, int col);
static void  next_generation(const int *current, int *next, int rows, int cols);
static void  print_grid(const int *grid, int rows, int cols, int generation,
                         int alive_count, double elapsed_ms);
static long  count_alive(const int *grid, int rows, int cols);
static void  sleep_ms(int ms);

static inline int idx(int cols, int row, int col) {
    return row * cols + col;
}

int main(int argc, char *argv[]) {
    int rows        = DEFAULT_ROWS;
    int cols        = DEFAULT_COLS;
    int generations = DEFAULT_GENERATIONS;
    int num_threads = omp_get_max_threads();

    if (argc >= 3) { rows = atoi(argv[1]); cols = atoi(argv[2]); }
    if (argc >= 4) { generations = atoi(argv[3]); }
    if (argc >= 5) { num_threads = atoi(argv[4]); }

    if (rows <= 0 || cols <= 0 || generations <= 0 || num_threads <= 0) {
        fprintf(stderr, "Invalid parameters.\n");
        fprintf(stderr, "Usage: %s [rows] [cols] [generations] [num_threads]\n", argv[0]);
        return EXIT_FAILURE;
    }

    omp_set_num_threads(num_threads);

    int *current_grid = alloc_grid(rows, cols);
    int *next_grid     = alloc_grid(rows, cols);

    srand((unsigned int) time(NULL));

    init_random(current_grid, rows, cols, INITIAL_ALIVE_PROB);

    printf("Conway's Game of Life - parallel version (OpenMP)\n");
    printf("Grid: %d x %d | Generations: %d | Threads: %d\n\n",
           rows, cols, generations, num_threads);

    double start_time = omp_get_wtime();

    for (int g = 0; g < generations; g++) {
        next_generation(current_grid, next_grid, rows, cols);

        int *tmp = current_grid;
        current_grid = next_grid;
        next_grid = tmp;

        long alive = count_alive(current_grid, rows, cols);
        double elapsed_ms = (omp_get_wtime() - start_time) * 1000.0;

        print_grid(current_grid, rows, cols, g + 1, (int) alive, elapsed_ms);
        sleep_ms(DEFAULT_DELAY_MS);
    }

    double total_time = omp_get_wtime() - start_time;
    printf("\nSimulation finished.\n");
    printf("Total time: %.3f s (%d generations, %dx%d grid, %d thread(s))\n",
           total_time, generations, rows, cols, num_threads);

    free_grid(current_grid);
    free_grid(next_grid);

    return EXIT_SUCCESS;
}

static int *alloc_grid(int rows, int cols) {
    int *grid = (int *) calloc((size_t) rows * (size_t) cols, sizeof(int));
    if (grid == NULL) {
        fprintf(stderr, "Error allocating memory for the grid.\n");
        exit(EXIT_FAILURE);
    }
    return grid;
}

static void free_grid(int *grid) {
    free(grid);
}

static void init_random(int *grid, int rows, int cols, double prob) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            double r = (double) rand() / (double) RAND_MAX;
            grid[idx(cols, i, j)] = (r < prob) ? 1 : 0;
        }
    }
}

static void init_glider(int *grid, int rows, int cols) {
    memset(grid, 0, (size_t) rows * (size_t) cols * sizeof(int));
    if (rows < 5 || cols < 5) return;

    int base_row = 1, base_col = 1;
    int cells[5][2] = {
        {0, 1}, {1, 2}, {2, 0}, {2, 1}, {2, 2}
    };
    for (int k = 0; k < 5; k++) {
        int r = (base_row + cells[k][0]) % rows;
        int c = (base_col + cells[k][1]) % cols;
        grid[idx(cols, r, c)] = 1;
    }
}

static int count_live_neighbors(const int *grid, int rows, int cols, int row, int col) {
    int total = 0;

    for (int dr = -1; dr <= 1; dr++) {
        for (int dc = -1; dc <= 1; dc++) {
            if (dr == 0 && dc == 0) continue;

            int nr = (row + dr + rows) % rows;
            int nc = (col + dc + cols) % cols;

            total += grid[idx(cols, nr, nc)];
        }
    }
    return total;
}

static void next_generation(const int *current, int *next, int rows, int cols) {
    #pragma omp parallel for collapse(2) schedule(static)
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            int alive = current[idx(cols, i, j)];
            int neighbors = count_live_neighbors(current, rows, cols, i, j);
            int new_state;

            if (alive) {
                if (neighbors < 2 || neighbors > 3) {
                    new_state = 0;
                } else {
                    new_state = 1;
                }
            } else {
                new_state = (neighbors == 3) ? 1 : 0;
            }

            next[idx(cols, i, j)] = new_state;
        }
    }
}

static long count_alive(const int *grid, int rows, int cols) {
    long total = 0;
    #pragma omp parallel for collapse(2) reduction(+:total) schedule(static)
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            total += grid[idx(cols, i, j)];
        }
    }
    return total;
}

static void print_grid(const int *grid, int rows, int cols, int generation,
                        int alive_count, double elapsed_ms) {
    printf("\033[H\033[J");

    printf("Generation: %d | Alive cells: %d | Elapsed time: %.1f ms\n\n",
           generation, alive_count, elapsed_ms);

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            putchar(grid[idx(cols, i, j)] ? '#' : '.');
        }
        putchar('\n');
    }
    fflush(stdout);
}

static void sleep_ms(int ms) {
    struct timespec ts;
    ts.tv_sec  = ms / 1000;
    ts.tv_nsec = (ms % 1000) * 1000000L;
    nanosleep(&ts, NULL);
}
