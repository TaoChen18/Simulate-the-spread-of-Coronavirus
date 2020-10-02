#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <stdbool.h>
#include <math.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>


void gol_swap(int** pA,int** pB)
{
  // You write this function - it should swap the pointers of pA and pB.
  int *temp = *pA; // set a temporary pointer to save pA
  *pA = *pB; // set pB to pA
  *pB = temp; // set temp to pB

}

// kernel function has three steps:
// 1. transform 2-D array to be get their 1-D indexs
// 2. calculate number of each cell's neighbors
// 3. judge whether certain cell live or die based on its neighbor number

__global__ void init_stuff(curandState *state) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    curand_init(1337, idx, 0, &state[idx]);
}

__global__ void gol_kernel( int* g_data, int worldWidth,
                            int worldHeight, int* g_resultData, curandState *state)
{
    //According to blockIdx, blockDim, threadIdx, calculate index and stride
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if(index < worldWidth || index >= worldWidth*(worldHeight-1)) return;
    for(;index < worldWidth*worldHeight;index +=blockDim.x * gridDim.x)
    {
        // get 2-D coordinates obtained through mathematical operations
        int x = index % worldWidth;
        int y = index / worldWidth;

        // write code to set: y0, y1 and y2
        int y0 = ((y+worldHeight-1)% worldHeight) * worldWidth;
        int y1 = y * worldWidth;
        int y2 = ((y + 1) % worldHeight) * worldWidth;

        int x1 = x;
        int x0 = (x1 + worldWidth - 1) % worldWidth;
        int x2 = (x1 + 1) % worldWidth;

        int currentCell = g_data[(x1+y1)];
        if ( ( currentCell != 1) && (currentCell != 2) && (currentCell != 3) ){
            // check the patient is after 10 iterations
            if (currentCell >= 100){
                // create a random value from 0-99
                float random = curand_uniform(&state[idx]);
                // Any infected patient after 10 iterations will have 5% to die
                if (random < 0.05){
                    g_resultData[(x1+y1)] = 3;
                }
                // otherwise the infected patient will be recover from the COVID-19
                else {
                    g_resultData[(x1+y1)] = 2;
                }
            }
            else {
                // if g_data[(x1+y1)] is infected with COVID-19,
                // we should add 10 to show that patient go through one more iteration
                g_resultData[(x1+y1)] = g_data[(x1+y1)] + 10;
            }
        }
        // if the current cell is not infected by COVID-19,
        // we should detect how many its neighbours are infected so that
        // we can calculate the infection probability of the current cell
        else if (currentCell == 1){
          int infectProbability = 0;
          // consider about if the cell is at left or right border
          if(index % worldWidth == 0){
            int neighborIndex[5] = {(x1+y0),(x2+y0),(x2+y1),(x2+y2),(x1+y2)};
            for (int i = 0; i < 5; i++){
                int neighborValue = g_data[neighborIndex[i]];
                if (( neighborValue != 1) && (neighborValue != 2) && (neighborValue != 3)){
                    infectProbability += 10;
                }
            }
          }else if(index % worldWidth == worldWidth - 1){
            int neighborIndex[5] = {(x1+y0),(x0+y0),(x0+y1),(x0+y2),(x1+y2)};
            for (int i = 0; i < 5; i++){
                int neighborValue = g_data[neighborIndex[i]];
                if (( neighborValue != 1) && (neighborValue != 2) && (neighborValue != 3)){
                    infectProbability += 10;
                }
            }
          }else{
            int neighborIndex[8] = {(x0+y0),(x0+y1),(x0+y2),(x1+y0),(x1+y2),(x2+y0),(x2+y1),(x2+y2)};
            // Any live cell(value0) witha neighbor infected byC OVID-19 will have 10% to also infected by virus.
            // In other words, if all of its 8 neighbors are infected by virus, it will be 80% infected by virus.
            int infectProbability = 0;
            for (int index = 0; index < 8; index++){
                int neighborValue = g_data[neighborIndex[index]];
                if (( neighborValue != 1) && (neighborValue != 2) && (neighborValue != 3)){
                    infectProbability += 10;
                }
            }
          }
            // create a random value from 0-99
            float randomF = curand_uniform(&state[idx]) * 100;
            // if current cell is infected by COVID-19, change its value to 0
            // otherwise keep its value to be 1
            int random = (int) randomF;
            if (random < infectProbability){
                g_resultData[(x1+y1)] = 0;
            }
            else{
                g_resultData[(x1+y1)] = 1;
            }
        }
        // If a cell recovers or dies (with value 2 or 3), their value will never be changed in the future.
        else{
            g_resultData[(x1+y1)] = g_data[(x1+y1)];
        }

//        // return the number of alive cell for g_data[x1+y1]
//        size_t neighbor_count_live = g_data[(x0+y0)] + g_data[(x0+y1)] + g_data[(x0+y2)]
//                                + g_data[(x1+y0)] + g_data[(x1+y2)]
//                                + g_data[(x2+y0)] + g_data[(x2+y1)] + g_data[(x2+y2)];
//
//
//        // according to the logic from assignment1.pdf
//        // if a cell is live, it must has exactly 2 or 3 live neighbor to make itself live
//        if (g_data[(x1+y1)] == 1)
//        {
//            if (neighbor_count_live == 2 || neighbor_count_live == 3)
//            {
//                g_resultData[(x1+y1)] = 1;
//            }
//            else
//            {
//                g_resultData[(x1+y1)] = 0;
//            }
//        }
//        // if a cell is dead, it needs exactly 3 live neighbor to make itself live
//        else
//        {
//            if (neighbor_count_live == 3)
//            {
//                g_resultData[(x1+y1)] = 1;
//            }
//            else
//            {
//                g_resultData[(x1+y1)] = 0;
//            }
//        }
    }
}

extern "C" void kernel_function(int blockNum, int threadsCount, int** d_data, int** d_resultData, int state_size)
{
    curandState *d_state;
    cudaMalloc(&d_state, blockNum*threadsCount);
    init_stuff<<<blockNum, threadsCount>>>(d_state);
    gol_kernel <<<blockNum,threadsCount>>> (*d_data,state_size,state_size,*d_resultData,d_state);
    cudaDeviceSynchronize();
    gol_swap(d_data, d_resultData);
}

// extern "C" void cuda_free()
// {
//     cudaFree(g_data);
//     cudaFree(g_resultData);
// }

// Don't modify this function or your submitty autograding may incorrectly grade otherwise correct solutions.
// extern "C" void gol_printWorld(int myrank)
// {
//     char filename[50];
//     sprintf(filename, "output%d.txt", myrank);
//     FILE *fp = fopen(filename, "wb");
//     int i, j;
//
//     for( i = 1; i < g_worldHeight+1; i++)
//     {
//         fprintf(fp,"Row %2d: ", i);
//         for( j = 0; j < g_worldWidth; j++)
//         {
//             fprintf(fp,"%u ", (unsigned int)g_data[(i*g_worldWidth) + j]);
//         }
//         fprintf(fp,"\n");
//     }
//
//     fprintf(fp,"\n\n");
//     fclose(fp);
//
// }
