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
#define TILE_WIDTH 30

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

__global__ void sciddicaTFlowsComputationCachingKernel(int r, int c, double nodata, int* Xi, int* Xj, double *Sz, double *Sh, double *Sf, double p_r, double p_epsilon)
{
  int col_idx = threadIdx.x + blockDim.x * blockIdx.x;
  int row_idx = threadIdx.y + blockDim.y * blockIdx.y;

  bool eliminated_cells[5] = {false, false, false, false, false};
  bool again;
  int cells_count;
  double average;
  double m;
  double u[5];
  int n;
  double z, h;

  __shared__ double Sz_ds[TILE_WIDTH][TILE_WIDTH];
  __shared__ double Sh_ds[TILE_WIDTH][TILE_WIDTH];

  Sz_ds[threadIdx.y][threadIdx.x] = GET(Sz, c, row_idx, col_idx);
  Sh_ds[threadIdx.y][threadIdx.x] = GET(Sh, c, row_idx, col_idx);
  __syncthreads();

  int tile_start_x = blockIdx.x * blockDim.x;
  int next_tile_start_x = ((blockIdx.x + 1) * blockDim.x);
  int tile_start_y = blockIdx.y * blockDim.y;
  int next_tile_start_y = ((blockIdx.y + 1) * blockDim.y);

  if (row_idx > 0 && row_idx < r - 1 && col_idx > 0 && col_idx < c - 1) {
    m = Sh_ds[threadIdx.y][threadIdx.x] - p_epsilon;
    u[0] = Sz_ds[threadIdx.y][threadIdx.x] + p_epsilon;

    for(int cnt = 0; cnt <= MASK_WIDTH; ++cnt) {
      int n_index_x = row_idx + Xi[cnt + 1];
      int n_index_y = col_idx + Xj[cnt + 1];
      if((n_index_x >= 0) && (n_index_x < c) && (n_index_y >= 0) && (n_index_y < r)) {
        if((n_index_x >= tile_start_x) && (n_index_x < next_tile_start_x) && (n_index_y >= tile_start_y) && (n_index_y < next_tile_start_y)) {
          z = Sz_ds[threadIdx.y + Xi[cnt + 1]][threadIdx.x + Xj[cnt + 1]];
          h = Sh_ds[threadIdx.y + Xi[cnt + 1]][threadIdx.x + Xj[cnt + 1]];
        }
        else {  // try to get a L2 cache hit (best case, otherwise global memory in DRAM has to be accessed)
          z = GET(Sz, c, n_index_y, n_index_x);
          h = GET(Sh, c, n_index_y, n_index_x);
        }
      }
      u[cnt + 1] = z + h;
    }

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

    for(int cnt = 0; cnt <= MASK_WIDTH; ++cnt) {
      if (!eliminated_cells[cnt + 1]) BUF_SET(Sf, r, c, cnt, row_idx, col_idx, (average - u[cnt + 1]) * p_r);
    }
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

__global__ void sciddicaTWidthUpdateCachingKernel(int r, int c, double nodata, int* Xi, int* Xj, double *Sz, double *Sh, double *Sf)
{
  int col_idx = threadIdx.x + blockDim.x * blockIdx.x;
  int row_idx = threadIdx.y + blockDim.y * blockIdx.y;

  double h_next;
  
  __shared__ double Sf_ds[TILE_WIDTH * ADJACENT_CELLS][TILE_WIDTH];

  for(int cnt = 0; cnt < 4; ++cnt) {
    Sf_ds[threadIdx.y + cnt * TILE_WIDTH][threadIdx.x] = BUF_GET(Sf, r, c, cnt, row_idx, col_idx);
  }
  __syncthreads();

  int tile_start_x = blockIdx.x * blockDim.x;
  int next_tile_start_x = ((blockIdx.x + 1) * blockDim.x);
  int tile_start_y = blockIdx.y * blockDim.y;
  int next_tile_start_y = ((blockIdx.y + 1) * blockDim.y);

  if(col_idx > 0 && col_idx < c - 1 && row_idx > 0 && row_idx < r - 1) {
    h_next = GET(Sh, c, row_idx, col_idx);

    for(int cnt = 0; cnt <= MASK_WIDTH; ++cnt) {
      int n_index_x = col_idx + Xj[cnt + 1];
      int n_index_y = row_idx + Xi[cnt + 1];
      if((n_index_x >= 0) && (n_index_x < c) && (n_index_y >= 0) && (n_index_y < r)) {
        if((n_index_x >= tile_start_x) && (n_index_x < next_tile_start_x) && (n_index_y >= tile_start_y) && (n_index_y < next_tile_start_y)) {
          h_next += Sf_ds[threadIdx.y + Xi[cnt + 1] + (MASK_WIDTH - cnt) * TILE_WIDTH][threadIdx.x + Xj[cnt + 1]]
                    - Sf_ds[threadIdx.y + cnt * TILE_WIDTH][threadIdx.x];
        }
        else {  // try to get a L2 cache hit (best case, otherwise global memory in DRAM has to be accessed)
          h_next += BUF_GET(Sf, r, c, (MASK_WIDTH - cnt), n_index_y, n_index_x)
                    - BUF_GET(Sf, r, c, cnt, row_idx, col_idx);
        }
      }
    }

    SET(Sh, c, row_idx, col_idx, h_next);
  }
}

// This kernel benefits from a tiled implementation
__global__ void sciddicaTFlowsComputationKernelVerified(int r, int c, double nodata, int *Xi, int *Xj, double *Sz, double *Sh, double *Sf, double p_r, double p_epsilon)
{
  int col_index = threadIdx.x + blockDim.x * blockIdx.x;
  int row_index = threadIdx.y + blockDim.y * blockIdx.y;

  bool eliminated_cells[5] = {false, false, false, false, false};
  bool again;
  int cells_count;
  double average;
  double m;
  double u[5];
  int n;
  double z = 0, h = 0;

  __shared__ double Sz_ds[TILE_WIDTH][TILE_WIDTH];
  __shared__ double Sh_ds[TILE_WIDTH][TILE_WIDTH];

  Sz_ds[threadIdx.y][threadIdx.x] = GET(Sz, c, row_index, col_index);
  Sh_ds[threadIdx.y][threadIdx.x] = GET(Sh, c, row_index, col_index);
  __syncthreads();

  int tile_start_x = blockIdx.x * blockDim.x;
  int next_tile_start_x = ((blockIdx.x + 1) * blockDim.x);
  int tile_start_y = blockIdx.y * blockDim.y;
  int next_tile_start_y = ((blockIdx.y + 1) * blockDim.y);

  if (row_index > 0 && row_index < r - 1 && col_index > 0 && col_index < c - 1)
  {
    m = Sh_ds[threadIdx.y][threadIdx.x] - p_epsilon;
    u[0] = Sz_ds[threadIdx.y][threadIdx.x] + p_epsilon;

    int index_x;
    int index_y;

    for (int tmp = 0; tmp <= MASK_WIDTH; tmp++)
    {
      index_y = row_index + Xi[tmp + 1];
      index_x = col_index + Xj[tmp + 1];

      if ((index_x >= 0) && (index_x < c) && (index_y >= 0) && (index_y < r))
      {
        if ((index_x >= tile_start_x) && (index_x < next_tile_start_x) && (index_y >= tile_start_y) && (index_y < next_tile_start_y))
        {
          z = Sz_ds[threadIdx.y + Xi[tmp + 1]][threadIdx.x + Xj[tmp + 1]];
          h = Sh_ds[threadIdx.y + Xi[tmp + 1]][threadIdx.x + Xj[tmp + 1]];
        }
        else
        {
          z = GET(Sz, c, index_y, index_x);
          h = GET(Sh, c, index_y, index_x);
        }
        u[tmp + 1] = z + h;
      }
    }

    do
    {
      again = false;
      average = m;
      cells_count = 0;

      for (n = 0; n < 5; n++)
        if (!eliminated_cells[n])
        {
          average += u[n];
          cells_count++;
        }

      if (cells_count != 0)
        average /= cells_count;

      for (n = 0; n < 5; n++)
        if ((average <= u[n]) && (!eliminated_cells[n]))
        {
          eliminated_cells[n] = true;
          again = true;
        }
    } while (again);

    if (!eliminated_cells[1])
      BUF_SET(Sf, r, c, 0, row_index, col_index, (average - u[1]) * p_r);
    if (!eliminated_cells[2])
      BUF_SET(Sf, r, c, 1, row_index, col_index, (average - u[2]) * p_r);
    if (!eliminated_cells[3])
      BUF_SET(Sf, r, c, 2, row_index, col_index, (average - u[3]) * p_r);
    if (!eliminated_cells[4])
      BUF_SET(Sf, r, c, 3, row_index, col_index, (average - u[4]) * p_r);
  }
}

// This kernel benefits from a tiled implementation
__global__ void sciddicaTWidthUpdateKernelVerified(int r, int c, double nodata, int *Xi, int *Xj, double *Sz, double *Sh, double *Sf)
{
  int col_index = threadIdx.x + blockDim.x * blockIdx.x;
  int row_index = threadIdx.y + blockDim.y * blockIdx.y;

  double h_next;

  __shared__ double Sf_ds[TILE_WIDTH * ADJACENT_CELLS][TILE_WIDTH];

  Sf_ds[threadIdx.y][threadIdx.x] = BUF_GET(Sf, r, c, 0, row_index, col_index);
  Sf_ds[threadIdx.y + TILE_WIDTH][threadIdx.x] = BUF_GET(Sf, r, c, 1, row_index, col_index);
  Sf_ds[threadIdx.y + TILE_WIDTH * 2][threadIdx.x] = BUF_GET(Sf, r, c, 2, row_index, col_index);
  Sf_ds[threadIdx.y + TILE_WIDTH * 3][threadIdx.x] = BUF_GET(Sf, r, c, 3, row_index, col_index);
  __syncthreads();

  int tile_start_x = blockIdx.x * blockDim.x;
  int next_tile_start_x = ((blockIdx.x + 1) * blockDim.x);
  int tile_start_y = blockIdx.y * blockDim.y;
  int next_tile_start_y = ((blockIdx.y + 1) * blockDim.y);

  if (row_index > 0 && row_index < r - 1 && col_index > 0 && col_index < c - 1)
  {
    h_next = GET(Sh, c, row_index, col_index);

    for (int tmp = 0; tmp <= MASK_WIDTH; ++tmp)
    {
      int n_index_x = col_index + Xj[tmp + 1];
      int n_index_y = row_index + Xi[tmp + 1];
      if ((n_index_x >= 0) && (n_index_x < c) && (n_index_y >= 0) && (n_index_y < r))
      {
        if ((n_index_x >= tile_start_x) && (n_index_x < next_tile_start_x) && (n_index_y >= tile_start_y) && (n_index_y < next_tile_start_y))
        {
          h_next += Sf_ds[threadIdx.y + TILE_WIDTH * (MASK_WIDTH - tmp) + Xi[tmp + 1]][threadIdx.x + Xj[tmp + 1]] - Sf_ds[threadIdx.y + TILE_WIDTH * tmp][threadIdx.x];
        }
        else
        {
          h_next += BUF_GET(Sf, r, c, (MASK_WIDTH - tmp), n_index_y, n_index_x) - BUF_GET(Sf, r, c, tmp, row_index, col_index);
        }
      }
    }
    SET(Sh, c, row_index, col_index, h_next);
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

  dim3 tiled_block_size(TILE_WIDTH, TILE_WIDTH, 1);
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
  printf("One double precision buffer requires %lld bytes of shared memory\n", TILE_WIDTH * TILE_WIDTH * sizeof(double));
  printf("\n");

  //printf("Initializing...\n");
  sciddicaTSimulationInitKernel<<<grid_size, block_size>>>(r, c, Sz, Sh);
  checkError(__LINE__, "error executing sciddicaTSimulationInitKernel");
  checkError(cudaDeviceSynchronize(), __LINE__, "error syncing after sciddicaTSimulationInitKernel");

  // int loops = 100;  // TEST
  printf("Running the simulation for %d steps...\n", steps);
  // printf("... and %d times, determining the best time.\n", loops);
  // double best_time = 0.0;
  // for(int loop = 0; loop < loops; ++loop) {
    util::Timer cl_timer;
    for (int s = 0; s < steps; ++s) {
      //printf("step %d\n", s+1);

      sciddicaTResetFlowsKernel<<<grid_size, block_size>>>(r, c, nodata, Sf);
      checkError(__LINE__, "error executing sciddicaTSimulationInitKernel");
      checkError(cudaDeviceSynchronize(), __LINE__, "error syncing after sciddicaTResetFlowsKernel");

      // sciddicaTFlowsComputationKernel<<<grid_size, block_size>>>(r, c, nodata, Xi, Xj, Sz, Sh, Sf, p_r, p_epsilon);
      // checkError(__LINE__, "error executing sciddicaTFlowsComputationKernel");
      // checkError(cudaDeviceSynchronize(), __LINE__, "error syncing after sciddicaTFlowsComputationKernel");

      sciddicaTFlowsComputationKernelVerified<<<tiled_grid_size, tiled_block_size>>>(r, c, nodata, Xi, Xj, Sz, Sh, Sf, p_r, p_epsilon);
      checkError(__LINE__, "error executing sciddicaTFlowsComputationCachingKernel");
      checkError(cudaDeviceSynchronize(), __LINE__, "error syncing after sciddicaTFlowsComputationCachingKernel");

      // sciddicaTWidthUpdateKernel<<<grid_size, block_size>>>(r, c, nodata, Xi, Xj, Sz, Sh, Sf);
      // checkError(__LINE__, "error executing sciddicaTWidthUpdateKernel");
      // checkError(cudaDeviceSynchronize(), __LINE__, "error syncing after sciddicaTWidthUpdateKernel");

      sciddicaTWidthUpdateKernelVerified<<<tiled_grid_size, tiled_block_size>>>(r, c, nodata, Xi, Xj, Sz, Sh, Sf);
      checkError(__LINE__, "error executing sciddicaTWidthUpdateCachingKernel");
      checkError(cudaDeviceSynchronize(), __LINE__, "error syncing after sciddicaTWidthUpdateCachingKernel");
    }
    double cl_time = static_cast<double>(cl_timer.getTimeMilliseconds()) / 1000.0;
    // printf("[%d] ", loop);
    printf("Elapsed time: %lf [s]\n", cl_time);
    // if(cl_time < best_time || loop == 0) {
    //   best_time = cl_time;
    // }
  // }
  // printf("Best time: %lf [s]\n", best_time);

  saveGrid2Dr(Sh, r, c, argv[OUTPUT_PATH_ID]);

  //printf("Releasing memory...\n");
  checkError(cudaFree(Sz), __LINE__, "error deallocating memory for Sz");
  checkError(cudaFree(Sh), __LINE__, "error deallocating memory for Sh");
  checkError(cudaFree(Sf), __LINE__, "error deallocating memory for Sf");
  checkError(cudaFree(Xi), __LINE__, "error deallocating memory for Xi");
  checkError(cudaFree(Xj), __LINE__, "error deallocating memory for Xj");

  return 0;
}
