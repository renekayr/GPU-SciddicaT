#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "util.hpp"

// ----------------------------------------------------------------------------
// I/O parameters used to index argv[]
// ----------------------------------------------------------------------------
#define HEADER_PATH_ID 1
#define DEM_PATH_ID 2
#define SOURCE_PATH_ID 3
#define OUTPUT_PATH_ID 4
#define STEPS_ID 5

// ----------------------------------------------------------------------------
// Simulation parameters
// ----------------------------------------------------------------------------
#define P_R 0.5
#define P_EPSILON 0.001
#define ADJACENT_CELLS 4
#define STRLEN 256

// ----------------------------------------------------------------------------
// Tiled Halo Cell algorithm parameters
// ----------------------------------------------------------------------------
  // Von Neumann Neighborhood
#define MASK_WIDTH 3
  // Tile size can be dynamically calculated by using tile_width = 1 - mask_width - (pow(max_shared_memory, 2) / pow(sizeof(datatype), 2))
  // max_shared_memory can be queried from the CUDA API at runtime
  // This formula is derived by solving the following equation for for tile_width:
  // max_shared_memory = (mask_width + tile_width - 1)^2 * sizeof(datatype)
  // Else, an arbitrary or estimated amount that does not surpass the GPU's capacity is chosen
#define TILE_WIDTH 6
#define TILED_BLOCK_WIDTH (TILE_WIDTH + MASK_WIDTH - 1)
#define TILED_BUFFER_SIZE (TILED_BLOCK_WIDTH * TILED_BLOCK_WIDTH)

// ----------------------------------------------------------------------------
// Read/Write access macros linearizing single/multy layer buffer 2D indices
// ----------------------------------------------------------------------------
#define SET(M, columns, i, j, value) ((M)[(((i) * (columns)) + (j))] = (value))
#define GET(M, columns, i, j) (M[(((i) * (columns)) + (j))])
#define BUF_SET(M, rows, columns, n, i, j, value) ( (M)[( ((n)*(rows)*(columns)) + ((i)*(columns)) + (j) )] = (value) )
#define BUF_GET(M, rows, columns, n, i, j) ( M[( ((n)*(rows)*(columns)) + ((i)*(columns)) + (j) )] )

// ----------------------------------------------------------------------------
// I/O functions
// ----------------------------------------------------------------------------
void readHeaderInfo(char* path, int &nrows, int &ncols, double &nodata)
{
  FILE* f;
  
  if ((f = fopen(path, "r")) == 0) {
    printf("%s configuration header file not found\n", path);
    exit(0);
  }

  //Read the header
  char str[STRLEN];
  fscanf(f,"%s",&str); fscanf(f,"%s",&str); ncols = atoi(str);      //ncols
  fscanf(f,"%s",&str); fscanf(f,"%s",&str); nrows = atoi(str);      //nrows
  fscanf(f,"%s",&str); fscanf(f,"%s",&str);                         //xllcorner
  fscanf(f,"%s",&str); fscanf(f,"%s",&str);                         //yllcorner
  fscanf(f,"%s",&str); fscanf(f,"%s",&str);                         //cellsize
  fscanf(f,"%s",&str); fscanf(f,"%s",&str); nodata = atof(str);     //NODATA_value 
}

bool loadGrid2D(double *M, int rows, int columns, char *path)
{
  FILE *f = fopen(path, "r");

  if (!f) {
    printf("%s grid file not found\n", path);
    exit(0);
  }

  char str[STRLEN];
  for (int i = 0; i < rows; ++i)
    for (int j = 0; j < columns; ++j)
    {
      fscanf(f, "%s", str);
      SET(M, columns, i, j, atof(str));
    }

  fclose(f);

  return true;
}

bool saveGrid2Dr(double *M, int rows, int columns, char *path)
{
  FILE *f;
  f = fopen(path, "w");

  if (!f)
    return false;

  char str[STRLEN];
  for (int i = 0; i < rows; ++i)
  {
    for (int j = 0; j < columns; ++j)
    {
      sprintf(str, "%f ", GET(M, columns, i, j));
      fprintf(f, "%s ", str);
    }
    fprintf(f, "\n");
  }

  fclose(f);

  return true;
}

double* addLayer2D(int rows, int columns)
{
  double *tmp;
  checkError(cudaMallocManaged(&tmp, sizeof(double) * rows * columns), __LINE__, "error allocating memory");

  if (!tmp)
    return NULL;
  return tmp;
}

// ----------------------------------------------------------------------------
// init kernel, called once before the simulation loop
// ----------------------------------------------------------------------------
__global__ void sciddicaTSimulationInitKernel(int r, int c, double *Sz, double *Sh)
{
  int col_idx = threadIdx.x + blockDim.x * blockIdx.x;
  int row_idx = threadIdx.y + blockDim.y * blockIdx.y;
  int col_stride = blockDim.x * gridDim.x;
  int row_stride = blockDim.y * gridDim.y;

  double z, h;

  for (int row = row_idx + 1; row < r - 1; row += row_stride) {
    for (int col = col_idx + 1; col < c - 1; col += col_stride) {
      h = GET(Sh, c, row, col);

      if (h > 0.0) {
        z = GET(Sz, c, row, col);
        SET(Sz, c, row, col, z - h);
      }
    }
  }
}

// ----------------------------------------------------------------------------
// computing kernels, aka elementary processes in the XCA terminology
// ----------------------------------------------------------------------------
__global__ void sciddicaTResetFlowsKernel(int r, int c, double nodata, double* Sf)
{
  int col_idx = threadIdx.x + blockDim.x * blockIdx.x;
  int row_idx = threadIdx.y + blockDim.y * blockIdx.y;
  int col_stride = blockDim.x * gridDim.x;
  int row_stride = blockDim.y * gridDim.y;

  for (int row = row_idx + 1; row < r - 1; row += row_stride) {
    for (int col = col_idx + 1; col < c - 1; col += col_stride) {
      BUF_SET(Sf, r, c, 0, row, col, 0.0);
      BUF_SET(Sf, r, c, 1, row, col, 0.0);
      BUF_SET(Sf, r, c, 2, row, col, 0.0);
      BUF_SET(Sf, r, c, 3, row, col, 0.0);
    }
  }
}

__global__ void sciddicaTFlowsComputationKernel(int r, int c, double nodata, int* Xi, int* Xj, double *Sz, double *Sh, double *Sf, double p_r, double p_epsilon)
{
  int col_idx = threadIdx.x + blockDim.x * blockIdx.x;
  int row_idx = threadIdx.y + blockDim.y * blockIdx.y;
  int col_stride = blockDim.x * gridDim.x;
  int row_stride = blockDim.y * gridDim.y;

  bool eliminated_cells[5] = {false, false, false, false, false};
  bool again;
  int cells_count;
  double average;
  double m;
  double u[5];
  int n;
  double z, h;

  for (int row = row_idx + 1; row < r - 1; row += row_stride) {
    for (int col = col_idx + 1; col < c - 1; col += col_stride) {
      m = GET(Sh, c, row, col) - p_epsilon;
      u[0] = GET(Sz, c, row, col) + p_epsilon;
      z = GET(Sz, c, row + Xi[1], col + Xj[1]);
      h = GET(Sh, c, row + Xi[1], col + Xj[1]);
      u[1] = z + h;
      z = GET(Sz, c, row + Xi[2], col + Xj[2]);
      h = GET(Sh, c, row + Xi[2], col + Xj[2]);
      u[2] = z + h;
      z = GET(Sz, c, row + Xi[3], col + Xj[3]);
      h = GET(Sh, c, row + Xi[3], col + Xj[3]);
      u[3] = z + h;
      z = GET(Sz, c, row + Xi[4], col + Xj[4]);
      h = GET(Sh, c, row + Xi[4], col + Xj[4]);
      u[4] = z + h;

      do
      {
        again = false;
        average = m;
        cells_count = 0;

        for (n = 0; n < 5; ++n)
          if (!eliminated_cells[n])
          {
            average += u[n];
            ++cells_count;
          }

        if (cells_count != 0)
          average /= cells_count;

        for (n = 0; n < 5; ++n)
          if ((average <= u[n]) && (!eliminated_cells[n]))
          {
            eliminated_cells[n] = true;
            again = true;
          }
      } while (again);

      if (!eliminated_cells[1]) BUF_SET(Sf, r, c, 0, row, col, (average - u[1]) * p_r);
      if (!eliminated_cells[2]) BUF_SET(Sf, r, c, 1, row, col, (average - u[2]) * p_r);
      if (!eliminated_cells[3]) BUF_SET(Sf, r, c, 2, row, col, (average - u[3]) * p_r);
      if (!eliminated_cells[4]) BUF_SET(Sf, r, c, 3, row, col, (average - u[4]) * p_r);
    }
  }
}

__global__ void sciddicaTFlowsComputationHaloKernel(int r, int c, double nodata, int* Xi, int* Xj, double *Sz, double *Sh, double *Sf, double p_r, double p_epsilon)
{
  int col_idx = 1 + threadIdx.x + TILE_WIDTH * blockIdx.x;
  int row_idx = 1 + threadIdx.y + TILE_WIDTH * blockIdx.y;
  long col_halo = col_idx - MASK_WIDTH/2;
  long row_halo = row_idx - MASK_WIDTH/2;

  bool eliminated_cells[5] = {false, false, false, false, false};
  bool again;
  int cells_count;
  double average;
  double m;
  double u[5];
  int n;
  double z, h;

  __shared__ double Sz_ds[TILED_BUFFER_SIZE];
  __shared__ double Sh_ds[TILED_BUFFER_SIZE];

  // Phase 1: All block threads copy values into the block's shared memory
  if((col_halo >= 1) && (col_halo < c - 1) && (row_halo >= 1) && (row_halo < r - 1)) {  // TODO introduce proper indexing
    Sz_ds[threadIdx.x + threadIdx.y * blockDim.x] = GET(Sz, c, row_halo, col_halo);  // threadIdx.x + threadIdx.y * blockDim.x == threadIdx.x + threadIdx.y * TILED_BLOCK_WIDTH
    Sh_ds[threadIdx.x + threadIdx.y * blockDim.x] = GET(Sh, c, row_halo, col_halo);
  }
  else {  // populate ghost cells (outside of domain) with neutral elements w.r.t. operations performed on them
    Sz_ds[threadIdx.x + threadIdx.y * blockDim.x] = nodata;
    Sh_ds[threadIdx.x + threadIdx.y * blockDim.x] = nodata;
  }
  __syncthreads();

  // phase 2: Tile threads compute outputs
  if(threadIdx.x < TILE_WIDTH && threadIdx.y < TILE_WIDTH) {
    m = GET(Sh_ds, blockDim.x, threadIdx.y, threadIdx.x) - p_epsilon;
    u[0] = GET(Sz_ds, blockDim.x, threadIdx.y, threadIdx.x) + p_epsilon;
    z = GET(Sz_ds, blockDim.x, threadIdx.y + Xi[1], threadIdx.x + Xj[1]);
    h = GET(Sh_ds, blockDim.x, threadIdx.y + Xi[1], threadIdx.x + Xj[1]);
    u[1] = z + h;
    z = GET(Sz_ds, blockDim.x, threadIdx.y + Xi[2], threadIdx.x + Xj[2]);
    h = GET(Sh_ds, blockDim.x, threadIdx.y + Xi[2], threadIdx.x + Xj[2]);
    u[2] = z + h;
    z = GET(Sz_ds, blockDim.x, threadIdx.y + Xi[3], threadIdx.x + Xj[3]);
    h = GET(Sh_ds, blockDim.x, threadIdx.y + Xi[3], threadIdx.x + Xj[3]);
    u[3] = z + h;
    z = GET(Sz_ds, blockDim.x, threadIdx.y+ Xi[4], threadIdx.x + Xj[4]);
    h = GET(Sh_ds, blockDim.x, threadIdx.y+ Xi[4], threadIdx.x + Xj[4]);
    u[4] = z + h;

    do
    {
      again = false;
      average = m;
      cells_count = 0;

      for (n = 0; n < 5; ++n)
        if (!eliminated_cells[n])
        {
          average += u[n];
          ++cells_count;
        }

      if (cells_count != 0)
        average /= cells_count;

      for (n = 0; n < 5; ++n)
        if ((average <= u[n]) && (!eliminated_cells[n]))
        {
          eliminated_cells[n] = true;
          again = true;
        }
    } while (again);

    if (!eliminated_cells[1]) BUF_SET(Sf, r, c, 0, row_idx, col_idx, (average - u[1]) * p_r);
    if (!eliminated_cells[2]) BUF_SET(Sf, r, c, 1, row_idx, col_idx, (average - u[2]) * p_r);
    if (!eliminated_cells[3]) BUF_SET(Sf, r, c, 2, row_idx, col_idx, (average - u[3]) * p_r);
    if (!eliminated_cells[4]) BUF_SET(Sf, r, c, 3, row_idx, col_idx, (average - u[4]) * p_r);
  }
}

__global__ void sciddicaTWidthUpdateKernel(int r, int c, double nodata, int* Xi, int* Xj, double *Sz, double *Sh, double *Sf)
{
  int col_idx = threadIdx.x + blockDim.x * blockIdx.x;
  int row_idx = threadIdx.y + blockDim.y * blockIdx.y;
  int col_stride = blockDim.x * gridDim.x;
  int row_stride = blockDim.y * gridDim.y;

  double h_next;

  for (int row = row_idx + 1; row < r - 1; row += row_stride) {
    for (int col = col_idx + 1; col < c - 1; col += col_stride) {
      h_next = GET(Sh, c, row, col);
      h_next += BUF_GET(Sf, r, c, 3, row+Xi[1], col+Xj[1]) - BUF_GET(Sf, r, c, 0, row, col);
      h_next += BUF_GET(Sf, r, c, 2, row+Xi[2], col+Xj[2]) - BUF_GET(Sf, r, c, 1, row, col);
      h_next += BUF_GET(Sf, r, c, 1, row+Xi[3], col+Xj[3]) - BUF_GET(Sf, r, c, 2, row, col);
      h_next += BUF_GET(Sf, r, c, 0, row+Xi[4], col+Xj[4]) - BUF_GET(Sf, r, c, 3, row, col);

      SET(Sh, c, row, col, h_next);
    }
  }
}

__global__ void sciddicaTWidthUpdateHaloKernel(int r, int c, double nodata, int* Xi, int* Xj, double *Sz, double *Sh, double *Sf)
{
  int col_idx = 1 + threadIdx.x + TILE_WIDTH * blockIdx.x;
  int row_idx = 1 + threadIdx.y + TILE_WIDTH * blockIdx.y;
  long col_halo = col_idx - MASK_WIDTH/2;
  long row_halo = row_idx - MASK_WIDTH/2;

  double h_next;
  
  __shared__ double Sf_ds[TILED_BUFFER_SIZE * ADJACENT_CELLS];

  // Phase 1: All block threads copy values into shared memory
  if((col_halo >= 1) && (col_halo < c - 1) && (row_halo >= 1) && (row_halo < r - 1)) {  // TODO introduce proper indexing
    Sf_ds[threadIdx.x + threadIdx.y * blockDim.x] = GET(Sf, c, row_halo, col_halo);  // threadIdx.x + threadIdx.y * blockDim.x == threadIdx.x + threadIdx.y * TILED_BLOCK_WIDTH
  }
  else {  // populate ghost cells (outside of domain) with neutral elements w.r.t. operations performed on them
    Sf_ds[threadIdx.x + threadIdx.y * blockDim.x] = nodata;
  }
  __syncthreads();

  // phase 2: tile threads compute outputs
  if(threadIdx.x < TILE_WIDTH && threadIdx.y < TILE_WIDTH) {
    h_next = GET(Sh, c, row_idx, col_idx);
    h_next += BUF_GET(Sf_ds, blockDim.y, blockDim.x, 3, threadIdx.y + Xi[1], threadIdx.x + Xj[1]) - BUF_GET(Sf_ds, blockDim.y, blockDim.x, 0, threadIdx.y, threadIdx.x);
    h_next += BUF_GET(Sf_ds, blockDim.y, blockDim.x, 2, threadIdx.y + Xi[2], threadIdx.x + Xj[2]) - BUF_GET(Sf_ds, blockDim.y, blockDim.x, 1, threadIdx.y, threadIdx.x);
    h_next += BUF_GET(Sf_ds, blockDim.y, blockDim.x, 1, threadIdx.y + Xi[3], threadIdx.x + Xj[3]) - BUF_GET(Sf_ds, blockDim.y, blockDim.x, 2, threadIdx.y, threadIdx.x);
    h_next += BUF_GET(Sf_ds, blockDim.y, blockDim.x, 0, threadIdx.y + Xi[4], threadIdx.x + Xj[4]) - BUF_GET(Sf_ds, blockDim.y, blockDim.x, 3, threadIdx.y, threadIdx.x);

    SET(Sh, c, row_idx, col_idx, h_next);
  }
}

// ----------------------------------------------------------------------------
// Function main()
// ----------------------------------------------------------------------------
int main(int argc, char **argv)
{
  int rows, cols;
  double nodata;
  readHeaderInfo(argv[HEADER_PATH_ID], rows, cols, nodata);

  int r = rows;                     // r: grid rows
  int c = cols;                     // c: grid columns
  // int i_start = 1, i_end = r-1;   // [i_start,i_end[: kernel application range along rows
  // int j_start = 1, j_end = c-1;   // [i_start,i_end[: kernel application range along columns
  double *Sz;                       // Sz: substate (grid) containing cells' altitude a.s.l.
  double *Sh;                       // Sh: substate (grid) containing cells' flow thickness
  double *Sf;                       // Sf: 4 substates containing the flows towards the 4 neighbors
  int* Xi;                          // Xj: von Neuman neighborhood row coordinates (see below)
  int* Xj;                          // Xj: von Neuman neighborhood col coordinates (see below)
  double p_r = P_R;                 // p_r: minimization algorithm outflows dumping factor
  double p_epsilon = P_EPSILON;     // p_epsilon: frictional parameter threshold
  int steps = atoi(argv[STEPS_ID]); //steps: simulation steps

  // The adopted von Neuman neighborhood
  // Format: flow_index:cell_label:(row_index,col_index)
  //
  //   cell_label in [0,1,2,3,4]: label assigned to each cell in the neighborhood
  //   flow_index in   [0,1,2,3]: outgoing flow indices in Sf from cell 0 to the others
  //       (row_index,col_index): 2D relative indices of the cells
  //
  //               |0:1:(-1, 0)|
  //   |1:2:( 0,-1)| :0:( 0, 0)|2:3:( 0, 1)|
  //               |3:4:( 1, 0)|
  //
  //

  // printf("Allocating memory...\n");
  Sz = addLayer2D(r, c);                  // Allocates the Sz substate grid
  Sh = addLayer2D(r, c);                  // Allocates the Sh substate grid
  Sf = addLayer2D(ADJACENT_CELLS * r, c); // Allocates the Sf substates grid, having one layer for each adjacent cell
  checkError(cudaMallocManaged(&Xi, sizeof(int) * 5), __LINE__, "error allocating memory for Xi");
  Xi[0] = 0; Xi[1] = -1; Xi[2] = 0;  Xi[3] = 0; Xi[4] = 1;
  checkError(cudaMallocManaged(&Xj, sizeof(int) * 5), __LINE__, "error allocating memory for Xj");
  Xj[0] = 0; Xj[1] = 0;  Xj[2] = -1; Xj[3] = 1; Xj[4] = 0;

  // printf("Loading data from file...\n");
  loadGrid2D(Sz, r, c, argv[DEM_PATH_ID]);    // Load Sz from file
  loadGrid2D(Sh, r, c, argv[SOURCE_PATH_ID]); // Load Sh from file

  int n = rows * cols;
  int dim_x = 32;
  int dim_y = 32;
  dim3 block_size(dim_x, dim_y, 1);
  dim3 grid_size(ceil(sqrt(n / (dim_x * dim_y))), ceil(sqrt(n / (dim_x * dim_y))), 1);

  // printf("Problem size is %d elements\n", n);
  // printf("Block dimensions are %d, %d, %d\n", block_size.x, block_size.y, block_size.z);
  // printf("Grid dimensions are %d, %d, %d\n", grid_size.x, grid_size.y, grid_size.z);
  // printf("Total grid threads are: %d\n", block_size.x * block_size.y * grid_size.x * grid_size.y);

  dim3 tiled_block_size(TILED_BLOCK_WIDTH, TILED_BLOCK_WIDTH, 1);  // == TILED_BUFFER_SIZE
  dim3 tiled_grid_size(ceil(sqrt(n / (TILE_WIDTH * TILE_WIDTH))), ceil(sqrt(n / (TILE_WIDTH * TILE_WIDTH))), 1);

  printf("\n");
  printf("Mask width is %d\n", MASK_WIDTH);
  printf("Tile width is %d\n", TILE_WIDTH);
  printf("Problem size is %d elements\n", n);
  printf("Tiled block dimensions are %d, %d, %d\n", tiled_block_size.x, tiled_block_size.y, tiled_block_size.z);
  printf("Tiled grid dimensions are %d, %d, %d\n", tiled_grid_size.x, tiled_grid_size.y, tiled_grid_size.z);
  printf("Total blocks in tiled grid are: %d\n", tiled_grid_size.x * tiled_grid_size.y * tiled_grid_size.z);
  printf("Total tiled grid threads are: %d\n", tiled_block_size.x * tiled_block_size.y * tiled_block_size.z * tiled_grid_size.x * tiled_grid_size.y * tiled_grid_size.z);
  printf("Threads only involved in output: %d\n", TILE_WIDTH * TILE_WIDTH * tiled_grid_size.x * tiled_grid_size.y * tiled_grid_size.z);
  printf("One double precision buffer requires %lld bytes of shared memory\n", TILED_BUFFER_SIZE * sizeof(double));
  printf("\n");

  // TODO: define tile and by extension grid size individually depending on the kernel
  //       calculate them beforehand by querying device maxima

  //printf("Initializing...\n");
  sciddicaTSimulationInitKernel<<<grid_size, block_size>>>(r, c, Sz, Sh);
  checkError(__LINE__, "error executing sciddicaTSimulationInitKernel");
  checkError(cudaDeviceSynchronize(), __LINE__, "error syncing after sciddicaTSimulationInitKernel");

  printf("Running the simulation for %d steps...\n", steps);
  util::Timer cl_timer;
  for (int s = 0; s < steps; ++s) {
    // printf("step %d\n", s+1);

    sciddicaTResetFlowsKernel<<<grid_size, block_size>>>(r, c, nodata, Sf);
    checkError(__LINE__, "error executing sciddicaTSimulationInitKernel");
    checkError(cudaDeviceSynchronize(), __LINE__, "error syncing after sciddicaTResetFlowsKernel");

    // sciddicaTFlowsComputationKernel<<<grid_size, block_size>>>(r, c, nodata, Xi, Xj, Sz, Sh, Sf, p_r, p_epsilon);
    // checkError(__LINE__, "error executing sciddicaTFlowsComputationKernel");
    // checkError(cudaDeviceSynchronize(), __LINE__, "error syncing after sciddicaTFlowsComputationKernel");

    sciddicaTFlowsComputationHaloKernel<<<tiled_grid_size, tiled_block_size>>>(r, c, nodata, Xi, Xj, Sz, Sh, Sf, p_r, p_epsilon);
    checkError(__LINE__, "error executing sciddicaTFlowsComputationKernel");
    checkError(cudaDeviceSynchronize(), __LINE__, "error syncing after sciddicaTFlowsComputationHaloKernel");

    sciddicaTWidthUpdateKernel<<<grid_size, block_size>>>(r, c, nodata, Xi, Xj, Sz, Sh, Sf);
    checkError(__LINE__, "error executing sciddicaTWidthUpdateKernel");
    checkError(cudaDeviceSynchronize(), __LINE__, "error syncing after sciddicaTWidthUpdateKernel");

    // sciddicaTWidthUpdateHaloKernel<<<tiled_grid_size, tiled_block_size>>>(r, c, nodata, Xi, Xj, Sz, Sh, Sf);
    // checkError(__LINE__, "error executing sciddicaTWidthUpdateKernel");
    // checkError(cudaDeviceSynchronize(), __LINE__, "error syncing after sciddicaTWidthUpdateHaloKernel");
  }
  double cl_time = static_cast<double>(cl_timer.getTimeMilliseconds()) / 1000.0;
  printf("Elapsed time: %lf [s]\n", cl_time);

  saveGrid2Dr(Sh, r, c, argv[OUTPUT_PATH_ID]);

  //printf("Releasing memory...\n");
  checkError(cudaFree(Sz), __LINE__, "error deallocating memory for Sz");
  checkError(cudaFree(Sh), __LINE__, "error deallocating memory for Sh");
  checkError(cudaFree(Sf), __LINE__, "error deallocating memory for Sf");
  checkError(cudaFree(Xi), __LINE__, "error deallocating memory for Xi");
  checkError(cudaFree(Xj), __LINE__, "error deallocating memory for Xj");

  return 0;
}
