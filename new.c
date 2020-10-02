#include<stdio.h>
#include<stdlib.h>
#include<unistd.h>
#include<stdbool.h>
#include <mpi.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <string.h>
#define MASTER 0		/* taskid of first task */
#define FROM_MASTER 1		/* setting a message type */
#define FROM_WORKER 2		/* setting a message type */
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

static inline void gol_initAllZeros( size_t worldWidth, size_t worldHeight )
{
    g_worldWidth = worldWidth;
    g_worldHeight = worldHeight+2;
    g_dataLength = g_worldWidth * g_worldHeight;
    int i;
    // calloc init's to all zeros
    g_data = calloc( g_dataLength, sizeof(unsigned char));
    for( i = g_worldWidth; i < g_dataLength-g_worldWidth; i++)
    {
	g_data[i] = 0;
    }
    g_resultData = calloc( g_dataLength, sizeof(unsigned char));
}

static inline void gol_initAllOnes( size_t worldWidth, size_t worldHeight )
{
    int i;

    g_worldWidth = worldWidth;
    g_worldHeight = worldHeight+2;
    g_dataLength = g_worldWidth * g_worldHeight;

    g_data = calloc( g_dataLength, sizeof(unsigned char));

    // set all rows of world to true
    for( i = g_worldWidth; i < g_dataLength-g_worldWidth; i++)
    {
	g_data[i] = 1;
    }

    g_resultData = calloc( g_dataLength, sizeof(unsigned char));
}

static inline void gol_initOnesInMiddle( size_t worldWidth, size_t worldHeight )
{
    int i;

    g_worldWidth = worldWidth;
    g_worldHeight = worldHeight+2;
    g_dataLength = g_worldWidth * g_worldHeight;

    g_data = calloc( g_dataLength, sizeof(unsigned char));
    for( i = g_worldWidth; i < g_dataLength-g_worldWidth; i++)
    {
	g_data[i] = 0;
    }
    for (i = (g_worldHeight-2) * g_worldWidth; i < (g_worldHeight - 1) * g_worldWidth; i++)
    {
        if ((i >= ((g_worldHeight - 2) * g_worldWidth + 17)) && (i < ((g_worldHeight - 2) * g_worldWidth + 27)))
        {
            g_data[i] = 1;
        }
    }
    // set first 1 rows of world to true

    g_resultData = calloc( g_dataLength, sizeof(unsigned char));
}

static inline void gol_initOnesAtCorners( size_t worldWidth, size_t worldHeight )
{
    g_worldWidth = worldWidth;
    g_worldHeight = worldHeight+2;
    g_dataLength = g_worldWidth * g_worldHeight;
    int i;
    g_data = calloc( g_dataLength, sizeof(unsigned char));
    for( i = g_worldWidth; i < g_dataLength-g_worldWidth; i++)
    {
	g_data[i] = 0;
    }
    g_data[worldWidth] = 1; // upper left
    g_data[2*worldWidth-1]=1; // upper right
    g_data[(g_worldHeight-2) * worldWidth]=1; // lower left
    g_data[(g_worldHeight-2) * worldWidth + worldWidth-1]=1; // lower right

    g_resultData = calloc( g_dataLength, sizeof(unsigned char));
}

static inline void gol_initSpinnerAtCorner( size_t worldWidth, size_t worldHeight )
{
    g_worldWidth = worldWidth;
    g_worldHeight = worldHeight;
    g_dataLength = g_worldWidth * g_worldHeight;
    int i;
    g_data = calloc( g_dataLength, sizeof(unsigned char));
    for( i = g_worldWidth; i < g_dataLength-g_worldWidth; i++)
    {
	g_data[i] = 0;
    }
    g_data[worldWidth] = 1; // upper left
    g_data[worldWidth+1] = 1; // upper left +1
    g_data[2*worldWidth-1]=1; // upper right

    g_resultData = calloc( g_dataLength, sizeof(unsigned char));
}

static inline void gol_initMaster( unsigned int pattern, size_t worldWidth, size_t worldHeight )
{
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
	gol_initOnesAtCorners( worldWidth, worldHeight );
	break;

    case 4:
	gol_initSpinnerAtCorner( worldWidth, worldHeight );
	break;

    default:
	printf("Pattern %u has not been implemented \n", pattern);
	exit(-1);
    }
}

static inline void gol_swap( unsigned char **pA, unsigned char **pB)
{
    // You write this function - it should swap the pointers of pA and pB.
    char *temp = *pA; // set a temporary pointer to save pA
    *pA = *pB; // set pB to pA
    *pB = temp; // set temp to pB
}

static inline unsigned int gol_countAliveCells(unsigned char* data,
					   size_t x0,
					   size_t x1,
					   size_t x2,
					   size_t y0,
					   size_t y1,
					   size_t y2)
{

    // You write this function - it should return the number of alive cell for data[x1+y1]
    // There are 8 neighbors - see the assignment description for more details.
    int count = 0;
    // count the number of alive around data[x1+y1]
    if(data[x0+y0] == 1) count++;
    if(data[x0+y1] == 1) count++;
    if(data[x0+y2] == 1) count++;
    if(data[x1+y0] == 1) count++;
    if(data[x1+y2] == 1) count++;
    if(data[x2+y0] == 1) count++;
    if(data[x2+y1] == 1) count++;
    if(data[x2+y2] == 1) count++;
    // return count
    return count;
}


// Don't modify this function or your submitty autograding may incorrectly grade otherwise correct solutions.
static inline void gol_printWorld()
{
    int i, j;

    for( i = 0; i < g_worldHeight; i++)
    {
	printf("Row %2d: ", i);
	for( j = 0; j < g_worldWidth; j++)
	{
	    printf("%u ", (unsigned int)g_data[(i*g_worldWidth) + j]);
	}
	printf("\n");
    }

    printf("\n\n");
}

static inline void gol_printresult()
{
    int i, j;
printf("this is result\n");
    for( i = 0; i < g_worldHeight; i++)
    {
	printf("Row %2d: ", i);
	for( j = 0; j < g_worldWidth; j++)
	{
	    printf("%u ", (unsigned int)g_resultData[(i*g_worldWidth) + j]);
	}
	printf("\n");
    }

    printf("\n\n");
}

/// Serial version of standard byte-per-cell life.
void gol_iterateSerial(size_t first_y, size_t last_y)
{
  size_t y,x;
    for (y = first_y; y < last_y; ++y)
  {
    // write code to set: y0, y1 and y2
    size_t y0 = ((y+g_worldHeight-1)% (g_worldHeight)) * g_worldWidth;
    size_t y1 = y * g_worldWidth;
    size_t y2 = ((y + 1) % (g_worldHeight)) * g_worldWidth;
    for (x = 0; x < g_worldWidth; ++x)
    {
  // write the code here: set x0, x2, call countAliveCells and compute if g_resultsData[y1 + x] is 0 or 1
      size_t x1 = x;
      size_t x0 = (x1 + g_worldWidth - 1) % g_worldWidth;
      size_t x2 = (x1 + 1) % g_worldWidth;
      // get the number of alive
      unsigned int count = gol_countAliveCells(g_data,x0,x1,x2,y0,y1,y2);
      // if previous world's cell is alive and the number of alive cells around
      // it is less than 2 or greater than 3, it will die. set new world as 0
      if(g_data[x1+y1] == 1 && (count < 2 || count > 3)) g_resultData[x1+y1] = 0;
      // if previous world's cell is dying and the number of alive cells around
      // it is 3, it will be alive. set new world as 1
      else if(g_data[x1+y1] == 0 && count == 3) g_resultData[x1+y1] = 1;
      // if previous world's cell is alive and the number of alive cells around
      // it is 2 or 3, it will still be alive. set new world as 1
      else if(g_data[x1+y1] == 1 && (count ==2 || count == 3)) g_resultData[x1+y1] = 1;
      // if previous world's cell is dying and the number of alive cells around
      // it is not 3, it will still be dying. set new world as 1
      else if(g_data[x1+y1] == 0 && count != 3) g_resultData[x1+y1] = 0;
    }
  }
}

char** alloc2d(int n, int m){
  char* data = calloc(n*m,sizeof(char));
  char** array = calloc(n,sizeof(char*));
  for(int i = 0; i < n; i++){
    array[i] = &(data[i*m]);
  }
  return array;
}


int main(int argc, char *argv[])
{
    unsigned int pattern = 0;
    unsigned int worldSize = 0;
    unsigned int itterations = 0;
    unsigned int output = 0;

    printf("This is the Game of Life running in serial on a CPU.\n");

    if( argc != 5 )
    {
	printf("GOL requires 5 arguments: pattern number, sq size of the world, the number of itterations and whether to print out, e.g. ./gol 0 32 2 0\n");
	exit(-1);
    }

    pattern = atoi(argv[1]);
    worldSize = atoi(argv[2]);
    itterations = atoi(argv[3]);
    output = atoi(argv[4]);

    int numtasks,			/* number of tasks in partition */
      taskid,			/* a task identifier */
      numworkers,			/* number of worker tasks */
      mtype;			/* message type */
    double timer;
    MPI_Status status;

    /******************************* init MPI_FILE AND MPI_DATATYPE *****************/
    MPI_File thefile;
    MPI_Datatype num_as_string;
    MPI_Datatype localarray;
    /******************************* end init MPI_FILE AND MPI_DATATYPE *****************/

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &taskid);
    MPI_Comm_size(MPI_COMM_WORLD, &numtasks);
    MPI_Request send_down,receive_down,send_up,receive_up;
    numworkers = numtasks;


    /* --------------- init MPI_Datatype ------------------ */


    /* --------------- end of init MPI_Datatype ------------------ */


    /**************************** master task ************************************/
    if (taskid == MASTER) {
      //printf("a:\n");
      timer = MPI_Wtime();
}
      /* send matrix data to the worker tasks */
      gol_initMaster(pattern, worldSize, worldSize);
      mtype = FROM_MASTER;
      size_t i;
      for(i = 0;i<itterations;i++){
        if(taskid != 0){
          MPI_Isend( &g_data[g_worldWidth], g_worldWidth, MPI_UNSIGNED_CHAR, taskid-1, mtype, MPI_COMM_WORLD, &send_up );
        }else{
          MPI_Isend( &g_data[g_worldWidth], g_worldWidth, MPI_UNSIGNED_CHAR, numworkers-1, mtype, MPI_COMM_WORLD, &send_up );
        }
        if(taskid != numworkers-1){
          MPI_Isend( &g_data[(g_worldHeight-2)*g_worldWidth], g_worldWidth, MPI_UNSIGNED_CHAR, taskid+1, mtype, MPI_COMM_WORLD, &send_down );
        }else{
          MPI_Isend( &g_data[(g_worldHeight-2)*g_worldWidth], g_worldWidth, MPI_UNSIGNED_CHAR, MASTER, mtype, MPI_COMM_WORLD, &send_down );
        }
        if(taskid != numworkers-1){
          MPI_Irecv( &g_data[(g_worldHeight-1)*g_worldWidth], g_worldWidth, MPI_UNSIGNED_CHAR, taskid+1, mtype, MPI_COMM_WORLD, &receive_up);
        }else{
          MPI_Irecv( &g_data[(g_worldHeight-1)*g_worldWidth], g_worldWidth, MPI_UNSIGNED_CHAR, MASTER, mtype, MPI_COMM_WORLD, &receive_up);
        }
        if(taskid != 0){
          MPI_Irecv( &g_data[0], g_worldWidth, MPI_UNSIGNED_CHAR, taskid-1, mtype, MPI_COMM_WORLD, &receive_down);
        }else{
          MPI_Irecv( &g_data[0], g_worldWidth, MPI_UNSIGNED_CHAR, numworkers-1, mtype, MPI_COMM_WORLD, &receive_down);
        }
        gol_iterateSerial(1,g_worldHeight-1);
        gol_swap(&g_resultData,&g_data);
        // insert function to swap resultData and data arrays
        // swap the previous world with the new world to continue generation
        MPI_Wait(&receive_up, &status);
        gol_iterateSerial(0,1);
        MPI_Wait(&receive_down, &status);
        gol_iterateSerial(g_worldHeight-1,g_worldHeight);
        MPI_Wait(&send_up, &status);
        MPI_Wait(&send_down, &status);
      }
      MPI_Barrier(MPI_COMM_WORLD);
      if(taskid == MASTER){
        timer = MPI_Wtime() - timer;
        printf("Over time: %f\n",timer);
      }
     if(output == 1){
       MPI_File_open(MPI_COMM_WORLD, "file.txt", MPI_MODE_CREATE | MPI_MODE_RDWR, MPI_INFO_NULL, &thefile);
       MPI_Offset offset = taskid * (g_worldWidth+1)*g_worldHeight;
       MPI_File_set_view(thefile, offset, MPI_CHAR, MPI_CHAR, "native", MPI_INFO_NULL);
       char* buf = calloc((g_worldWidth+1)*g_worldHeight,sizeof(char));
     	int count = 0;
     	for(int i =0; i< g_worldHeight; i++){
     		for(int j =0; j < g_worldWidth + 1;j++){
          if(j == g_worldWidth){
            buf[i*(g_worldWidth+1)+j] = '\n';
          }else{
            sprintf(&buf[i*(g_worldWidth+1)+j], "%u ",(unsigned int) g_data[(i*g_worldWidth) + j]);
          }
     		}
     	}
      MPI_File_write(thefile, buf, strlen(buf), MPI_CHAR, MPI_STATUS_IGNORE);
       MPI_File_close(&thefile);
     }
     free(g_data);
     free(g_resultData);
    MPI_Finalize();

    return 0;
}
