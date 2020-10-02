// --------------------------------------------------------------
//   Author: Tao Chen
//   Citation:
//     Homework pdf;
//     Course slices;
//     https://stackoverflow.com/questions/9084099/re-opening-stdout-and-stdin-file-descriptors-after-closing-them
//     https://github.com/yorgosk/game-of-life/blob/master/MPI/main_mpi.c
//     https://zhuanlan.zhihu.com/p/94753524
// --------------------------------------------------------------

#include<stdio.h>
#include<stdlib.h>
#include<unistd.h>
#include<stdbool.h>
#include <mpi.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <string.h>

// get the following global variables and functions from cuda file
extern unsigned char* g_data;
extern unsigned char* g_resultData;
extern size_t g_worldWidth;
extern size_t g_worldHeight;
extern size_t g_dataLength;

extern void gol_initMaster(int myrank, int cudaDeviceCount,unsigned int pattern, size_t worldWidth, size_t worldHeight);

extern void gol_kernelLaunch(unsigned char** d_data,
    unsigned char** d_resultData,
    size_t worldWidth,
    size_t worldHeight,
    unsigned short threadsCount);

// used to convert rankid into string in order to create different files
void itoa(int rankid,char* string,int radix){
  char temp[32] = {'\0'};
  int tempval = rankid;
  int i,j;
  for(i = 0;i<32;i++){
    temp[i] = (tempval%radix)+'0';
    tempval = tempval/radix;
    if(tempval==0) break;
  }
  for(j = 0;i>=0;i--) string[j++] = temp[i];
  string[j] = '\0';
}

// Don't modify this function or your submitty autograding may incorrectly grade otherwise correct solutions.
void gol_printWorld()
{
    int i, j;

    for( i = 1; i < g_worldHeight-1; i++)
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



int main(int argc, char *argv[])
{
  unsigned int pattern = 0;
  unsigned int worldSize = 0;
  unsigned int itterations = 0;
  unsigned int threadsCount = 0;
  unsigned int output = 0;  // control output

  printf("This is the Game of Life running in serial on GPU using MPI.\n");

  if( argc != 6 )
  {
printf("GOL requires 6 arguments: pattern number, sq size of the world, the number of itterations, blocksize and whether to print out, e.g. ./gol 1 64 4 3 0\n");
exit(-1);
  }

  pattern = atoi(argv[1]);
  worldSize = atoi(argv[2]);
  itterations = atoi(argv[3]);
  threadsCount = atoi(argv[4]);
  output = atoi(argv[5]);

  int rankid,			// a rank identifier
    ranknum,			// number of ranks
    mtype,			// message type
    down_dest,  // rankid down from the current rank
    up_dest;   // rankid up from the current rank
  double timer; // used to record time
  MPI_Status status; // store rank status
  MPI_File thefile;
  // initize MPI
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rankid);
  MPI_Comm_size(MPI_COMM_WORLD, &ranknum);
  // used to identify MPI requests
  MPI_Request send_down,receive_down,send_up,receive_up;

  // when it's rank 0, start to record time
  if (rankid == 0) {
    timer = MPI_Wtime();
  }
  int cudaDeviceCount;
  // generate the world
  gol_initMaster(rankid, cudaDeviceCount, pattern, worldSize, worldSize);

  mtype = 1;
  size_t i;
  // let each rank know the upper and down's rank id
  down_dest = (rankid+1)%ranknum;
  up_dest = (rankid-1+ranknum)%ranknum;
  // I added two rows based on the original world, so if the worldsize if 36*36. I would have a new world of 38*36 with 38 is number of row.
  for(i = 0;i<itterations;i++){
    // firstly, send send row and the penultimate row to other ranks' ghost rows, in the new iteration, this part can do the function of updating
    // borders.
    MPI_Isend( &g_data[g_worldWidth], g_worldWidth, MPI_UNSIGNED_CHAR, up_dest, mtype, MPI_COMM_WORLD, &send_up );
    MPI_Isend( &g_data[(g_worldHeight-2)*g_worldWidth], g_worldWidth, MPI_UNSIGNED_CHAR, down_dest, mtype, MPI_COMM_WORLD, &send_down );
    MPI_Irecv( &g_data[(g_worldHeight-1)*g_worldWidth], g_worldWidth, MPI_UNSIGNED_CHAR, down_dest, mtype, MPI_COMM_WORLD, &receive_up);
    MPI_Irecv( &g_data[0], g_worldWidth, MPI_UNSIGNED_CHAR, up_dest, mtype, MPI_COMM_WORLD, &receive_down);
    // wait until the rank sends and receives the data
    MPI_Wait(&send_up, &status);
    MPI_Wait(&send_down, &status);
    MPI_Wait(&receive_up, &status);
    MPI_Wait(&receive_down, &status);
    // update the inner part of the world
    gol_kernelLaunch(&g_data,&g_resultData, g_worldWidth,g_worldHeight,threadsCount);
  }
  // when the iteration is over, get the overall time
  if(rankid == 0){
    timer = MPI_Wtime() - timer;
    printf("Overall time: %f\n",timer);
  }
 // close stdout to put the content inside file
 if(output == 1){
   MPI_File_open(MPI_COMM_WORLD, "file.txt", MPI_MODE_CREATE | MPI_MODE_RDWR, MPI_INFO_NULL, &thefile);
   MPI_Offset offset = rankid * (g_worldWidth+1)*g_worldHeight;
   MPI_File_set_view(thefile, offset, MPI_CHAR, MPI_CHAR, "native", MPI_INFO_NULL);
   char* buf = calloc(ranknum*(g_worldWidth+1)*g_worldHeight,sizeof(char));
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
  MPI_Finalize();

  return 0;
}
