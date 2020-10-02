#include<stdio.h>
#include<stdlib.h>
#include<unistd.h>
#include<stdbool.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Result from last compute of world.
unsigned char *g_resultData=NULL;

// Current state of world. 
unsigned char *g_data=NULL;

// Current width of world.
size_t g_worldWidth=0;

/// Current height of world.
size_t g_worldHeight=0;

/// Current data length (product of width and height)
size_t g_dataLength=0;  // g_worldWidth * g_worldHeight

static inline void gol_initAllZeros( size_t worldWidth, size_t worldHeight)
{
    g_worldWidth = worldWidth;
    g_worldHeight = worldHeight;
    g_dataLength = g_worldWidth * (g_worldHeight+2);

    // cudaMallocManaged init's to all zeros
    cudaMallocManaged( &g_data, (g_dataLength * sizeof(unsigned char)));
    cudaMallocManaged( &g_resultData, (g_dataLength * sizeof(unsigned char))); 
}

static inline void gol_initAllOnes( size_t worldWidth, size_t worldHeight)
{
   
    g_worldWidth = worldWidth;
    g_worldHeight = worldHeight;
    g_dataLength = g_worldWidth * (g_worldHeight+2);

    cudaMallocManaged( &g_data, (g_dataLength * sizeof(unsigned char)));

    // set all rows of world to true
    int i;
    for( i = 0; i < g_dataLength; i++)
    {
		g_data[i] = 1;
    }
    
    cudaMallocManaged( &g_resultData, (g_dataLength * sizeof(unsigned char))); 
}

static inline void gol_initOnesInMiddle( size_t worldWidth, size_t worldHeight)
{
    g_worldWidth = worldWidth;
    g_worldHeight = worldHeight;
    // add two more rows: the first and last rows to be ghost rows
    g_dataLength = g_worldWidth * (g_worldHeight+2);

    cudaMallocManaged( &g_data, (g_dataLength * sizeof(unsigned char)));

    // set 10 ones at 128 columns and appears at the last row of each MPI
    int i;
    for( i = g_worldHeight*g_worldWidth; i < (g_worldHeight+1)*g_worldWidth; i++)
    {
    	if( (i >= ( g_worldHeight*g_worldWidth + 128 )) && (i < ( g_worldHeight*g_worldWidth + 138 )))
    	{
    	    g_data[i] = 1;
    	}
    }
    
    cudaMallocManaged( &g_resultData, (g_dataLength * sizeof(unsigned char))); 
}

static inline void gol_initOnesAtCorners( size_t worldWidth, size_t worldHeight, int myrank, int numranks )
{
    g_worldWidth = worldWidth;
    g_worldHeight = worldHeight;
    g_dataLength = g_worldWidth * (g_worldHeight+2);

    cudaMallocManaged( &g_data, (g_dataLength * sizeof(unsigned char)));
    // set ones at the corner of the world: the first row of rank 0 and last row of the last rank
    if (myrank == 0)
    {
        int i;
    	for( i = g_worldWidth; i < 2*g_worldWidth; i++)
	    {
	    	g_data[i] = 1;

	    } 
    }
    
    if (myrank == numranks-1)
    {
        int i;
        for (i = worldHeight*worldWidth; i < (worldHeight+1)*worldWidth; i++)
        {
            g_data[i] = 1;
        }
    }
    
    cudaMallocManaged( &g_resultData, (g_dataLength * sizeof(unsigned char))); 
}

static inline void gol_initSpinnerAtCorner( size_t worldWidth, size_t worldHeight, int myrank )
{
    g_worldWidth = worldWidth;
    g_worldHeight = worldHeight;
    g_dataLength = g_worldWidth * (g_worldHeight+2);

    cudaMallocManaged( &g_data, (g_dataLength * sizeof(unsigned char)));
    if ( myrank == 0 )
    {
	    g_data[g_worldWidth] = 1; // upper left
	    g_data[g_worldWidth+1] = 1; // upper left +1
	    g_data[g_worldWidth*2-1]=1; // upper right
    }
    
    cudaMallocManaged( &g_resultData, (g_dataLength * sizeof(unsigned char))); 
}

extern "C" void gol_initMaster( unsigned int pattern, size_t worldWidth, size_t worldHeight, int myrank, int numranks )
{
    int cE;
    int cudaDeviceCount;
    if( (cE = cudaGetDeviceCount( &cudaDeviceCount)) != cudaSuccess )
    {
        printf(" Unable to determine cuda device count, error is %d, count is %d\n", cE, cudaDeviceCount );
        exit(-1);
    }
    if( (cE = cudaSetDevice( myrank % cudaDeviceCount )) != cudaSuccess )
    {
        printf(" Unable to have rank %d set to cuda device %d, error is %d \n", myrank, (myrank % cudaDeviceCount), cE);
        exit(-1);
    }

    switch(pattern)
    {
        case 0:
    	gol_initAllZeros( worldWidth, worldHeight );
    	break;
    	
        case 1:
    	gol_initAllOnes( worldWidth, worldHeight );
    	break;
    	
        case 2:
    	gol_initOnesInMiddle( worldWidth, worldHeight );
    	break;
    	
        case 3:
    	gol_initOnesAtCorners( worldWidth, worldHeight, myrank, numranks );
    	break;

        case 4:
    	gol_initSpinnerAtCorner( worldWidth, worldHeight, myrank );
    	break;

        default:
    	printf("Pattern %u has not been implemented \n", pattern);
    	exit(-1);
    }
}

// kernel function has three steps:
// 1. transform 2-D array to be get their 1-D indexs
// 2. calculate number of each cell's neighbors
// 3. judge whether certain cell live or die based on its neighbor number
__global__ void gol_kernel( unsigned char* g_data, unsigned int worldWidth, 
                            unsigned int worldHeight, unsigned char* g_resultData)
{
    //According to blockIdx, blockDim, threadIdx, calculate index and stride
    size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x*gridDim.x;
    size_t indexPos;
    for ( indexPos = index+worldWidth; indexPos < (worldWidth * (worldHeight+1)); indexPos += stride )
    {
        // get 2-D coordinates obtained through mathematical operations
        int y = indexPos / worldWidth;
        int x = indexPos - y*worldWidth;

        // write code to set: y0, y1 and y2
        size_t y0 = ((y + (worldHeight+2) - 1) % (worldHeight+2)) * worldWidth; 
        size_t y1 = y * worldWidth;
        size_t y2 = ((y + 1) % (worldHeight+2)) * worldWidth;            
        // write the code here: set x0, x2, call countAliveCells and compute if resultsData[y1 + x] is 0 or 1
        size_t x1 = x;
        size_t x0 = (x1 + worldWidth - 1) % worldWidth; 
        size_t x2 = (x1 + 1) % worldWidth;

        
        // return the number of alive cell for g_data[x1+y1]
        size_t neighbor_count_live = g_data[(x0+y0)] + g_data[(x0+y1)] + g_data[(x0+y2)]
                                + g_data[(x1+y0)] + g_data[(x1+y2)]
                                + g_data[(x2+y0)] + g_data[(x2+y1)] + g_data[(x2+y2)];
        
        
        // according to the logic from assignment1.pdf
        // if a cell is live, it must has exactly 2 or 3 live neighbor to make itself live
        if (g_data[(x1+y1)] == 1)
        {
            if (neighbor_count_live == 2 || neighbor_count_live == 3)
            {
                g_resultData[(x1+y1)] = 1;
            }
            else
            {
                g_resultData[(x1+y1)] = 0;
            }
        }
        // if a cell is dead, it needs exactly 3 live neighbor to make itself live
        else
        {
            if (neighbor_count_live == 3)
            {
                g_resultData[(x1+y1)] = 1;
            }
            else
            {
                g_resultData[(x1+y1)] = 0;
            }
        }
    }
}

extern "C" void kernel_function(int blockNum, int threadsCount, unsigned char* g_data, unsigned char* g_resultData, int worldWidth, int worldHeight)
{
    gol_kernel <<<blockNum,threadsCount>>> (g_data,worldWidth,worldHeight,g_resultData);
    cudaDeviceSynchronize();
}

extern "C" void cuda_free()
{
    cudaFree(g_data);
    cudaFree(g_resultData);
}

// Don't modify this function or your submitty autograding may incorrectly grade otherwise correct solutions.
extern "C" void gol_printWorld(int myrank)
{
    char filename[50];
    sprintf(filename, "output%d.txt", myrank);
    FILE *fp = fopen(filename, "wb");
    int i, j;
    
    for( i = 1; i < g_worldHeight+1; i++)
    {
        fprintf(fp,"Row %2d: ", i);
        for( j = 0; j < g_worldWidth; j++)
        {
            fprintf(fp,"%u ", (unsigned int)g_data[(i*g_worldWidth) + j]);
        }
        fprintf(fp,"\n");
    }
    
    fprintf(fp,"\n\n");
    fclose(fp);
    
}


