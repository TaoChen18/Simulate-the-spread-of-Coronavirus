// --------------------------------------------------------------
//   Author: Tao Chen
//   Citation:
//     Homework pdf;
//     Course slices;
//     Previous homework.
// --------------------------------------------------------------

#include<stdio.h>
#include<stdlib.h>
#include<unistd.h>
#include<stdbool.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Result from last compute of world.
unsigned char* g_resultData = NULL;

// Current state of world.
unsigned char* g_data = NULL;

// Current width of world.
size_t g_worldWidth = 0;

/// Current height of world.
size_t g_worldHeight = 0;

/// Current data length (product of width and height)
size_t g_dataLength = 0;  // g_worldWidth * g_worldHeight

// I added two ghost rows to the original world directly, so the worldHeight would become 2 more larger
void gol_initAllZeros(size_t worldWidth, size_t worldHeight)
{
    g_worldWidth = worldWidth;
    g_worldHeight = worldHeight+2;
    g_dataLength = g_worldWidth * g_worldHeight;
    // calloc init's to all zeros
    cudaMallocManaged(&g_data,(g_dataLength*sizeof(unsigned char)));
    cudaMallocManaged(&g_resultData,(g_dataLength*sizeof(unsigned char)));
}

void gol_initAllOnes(size_t worldWidth, size_t worldHeight)
{
    int i;

    g_worldWidth = worldWidth;
    g_worldHeight = worldHeight+2;
    g_dataLength = g_worldWidth * g_worldHeight;
    cudaMallocManaged(&g_data,(g_dataLength*sizeof(unsigned char)));
    // set all rows of world to true
    for (i = g_worldWidth; i < g_dataLength - g_worldWidth; i++)
    {
        g_data[i] = 1;
    }
    cudaMallocManaged(&g_resultData,(g_dataLength*sizeof(unsigned char)));
}

void gol_initOnesInMiddle(size_t worldWidth, size_t worldHeight)
{
    int i;

    g_worldWidth = worldWidth;
    g_worldHeight = worldHeight+2;
    g_dataLength = g_worldWidth * g_worldHeight;

    cudaMallocManaged(&g_data,(g_dataLength*sizeof(unsigned char)));
    // set the real last row's 127 to 137 to true
    for (i = (g_worldHeight-2) * g_worldWidth; i < (g_worldHeight - 1) * g_worldWidth; i++)
    {
        if ((i >= ((g_worldHeight - 2) * g_worldWidth + 127)) && (i < ((g_worldHeight - 2) * g_worldWidth + 137)))
        {
            g_data[i] = 1;
        }
    }

    cudaMallocManaged(&g_resultData,(g_dataLength*sizeof(unsigned char)));
}

void gol_initOnesAtCorners(size_t worldWidth, size_t worldHeight)
{
    g_worldWidth = worldWidth;
    g_worldHeight = worldHeight+2;
    g_dataLength = g_worldWidth * g_worldHeight;

    cudaMallocManaged(&g_data,(g_dataLength*sizeof(unsigned char)));

    g_data[worldWidth] = 1; // upper left
    g_data[2*worldWidth - 1] = 1; // upper right
    g_data[(g_worldHeight - 2) * worldWidth] = 1; // lower left
    g_data[(g_worldHeight - 2) * worldWidth + worldWidth - 1] = 1; // lower right

    cudaMallocManaged(&g_resultData,(g_dataLength*sizeof(unsigned char)));

}

void gol_initSpinnerAtCorner(size_t worldWidth, size_t worldHeight)
{
    g_worldWidth = worldWidth;
    g_worldHeight = worldHeight+2;
    g_dataLength = g_worldWidth * g_worldHeight;

    cudaMallocManaged(&g_data,(g_dataLength*sizeof(unsigned char)));

    g_data[worldWidth] = 1; // upper left
    g_data[worldWidth + 1] = 1; // upper left +1
    g_data[2 * worldWidth - 1] = 1; // upper right

    cudaMallocManaged(&g_resultData,(g_dataLength*sizeof(unsigned char)));
}

// called from main file to generate world
extern "C" void gol_initMaster(int rankid, int cudaDeviceCount, unsigned int pattern, size_t worldWidth, size_t worldHeight)
{
    // check if MPI can connect with CUDA correctly
    int cE;
    if( (cE = cudaGetDeviceCount( &cudaDeviceCount)) != cudaSuccess )
    {
      printf(" Unable to determine cuda device count, error is %d, count is %d\n",
      cE, cudaDeviceCount );
      exit(-1);
    }
    if( (cE = cudaSetDevice( rankid % cudaDeviceCount )) != cudaSuccess )
    {
      printf(" Unable to have rank %d set to cuda device %d, error is %d \n",
      rankid, (rankid % cudaDeviceCount), cE);
      exit(-1);
    }
    switch (pattern)
    {
    case 0:
        gol_initAllZeros(worldWidth, worldHeight);
        break;

    case 1:
        gol_initAllOnes(worldWidth, worldHeight);
        break;

    case 2:
        gol_initOnesInMiddle(worldWidth, worldHeight);
        break;

    case 3:
        gol_initOnesAtCorners(worldWidth, worldHeight);
        break;

    case 4:
        gol_initSpinnerAtCorner(worldWidth, worldHeight);
        break;

    default:
        printf("Pattern %u has not been implemented \n", pattern);
        exit(-1);
    }
}

void gol_swap(unsigned char** pA, unsigned char** pB)
{
  // You write this function - it should swap the pointers of pA and pB.
  unsigned char *temp = *pA; // set a temporary pointer to save pA
  *pA = *pB; // set pB to pA
  *pB = temp; // set temp to pB

}

__global__
void gol_kernel(const unsigned char* d_data, unsigned int worldWidth,
                unsigned int worldHeight,unsigned char* d_resultData){
  //  The CUDA parallel thread hierarchy can be seen as the 1-D array just like hw1
  size_t index = blockIdx.x *blockDim.x + threadIdx.x;
  // we just would like to update the inner part of the world
  if(index < worldWidth || index >= worldWidth*(worldHeight-1)) return;
  for(;index < worldWidth*worldHeight;index +=blockDim.x * gridDim.x){
    // Index is like the 1-D array index, it should be 0 when index < worldWidth and 1 when worldWidth <= index < 2*worldWidth
    size_t x = index % worldWidth;
    // y is exactly index % worldWidth. But to reduce computing time, I replace % in the following way
    size_t y = index / worldWidth;
    // the calculation of x0,x1,x2,y0,y1,y2 is the same with hw1, the only difference is that, we changeed "%" for reducing computing time
    size_t y0 = ((y+worldHeight-1)% worldHeight) * worldWidth;
    size_t y1 = y * worldWidth;
    size_t y2 = ((y + 1) % worldHeight) * worldWidth;

    size_t x1 = x;
    size_t x0 = (x1 + worldWidth - 1) % worldWidth;
    size_t x2 = (x1 + 1) % worldWidth;
    // I can add the surround cells together directly since 1 is live, 0 is dead. So, by adding them together, it's the number of living cells aournd x1,y1
    int count = d_data[x0+y0] + d_data[x0+y1] + d_data[x0+y2] + d_data[x1+y0] +
                d_data[x1+y2] + d_data[x2+y0] + d_data[x2+y1] + d_data[x2+y2];
    // if previous world's cell is alive
    if(d_data[x1+y1] == 1){
      // if the number of alive cells around it is less than 2 or greater than 3, it will die. set new world as 0
      if(count < 2) d_resultData[x1+y1] = 0;
      else if(count > 3) d_resultData[x1+y1] = 0;
      // else, it will still be alive. set new world as 1
      else d_resultData[x1+y1] = 1;
    }
    // if previous world's cell is dead
    else{
      // only if the number of alive cells around it is 3, it will be alive. set new world as 1
      if(count == 3) d_resultData[x1+y1] = 1;
      // else, it will still be dead.
      else d_resultData[x1+y1] = 0;
    }
  }
}

// called from main file to lauch CUDA kernel
extern "C" void gol_kernelLaunch(unsigned char** d_data, unsigned char** d_resultData,
    size_t worldWidth, size_t worldHeight,unsigned short threadsCount) {
    // calculate the number of blocks by divide the overall threads that are needed by the number of threads in one block
    size_t blocksCount = worldWidth * worldHeight / threadsCount;
    // call the CUDA Kernel
    gol_kernel <<<blocksCount, threadsCount>>> (*d_data, worldWidth, worldHeight, *d_resultData);
    // Wait for GPU to finish before accessing on host
    cudaDeviceSynchronize();
    // swap the old word with the new world
    gol_swap(d_data, d_resultData);
}
