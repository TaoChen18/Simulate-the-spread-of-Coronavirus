// --------------------------------------------------------------
//   Authors: Tao Chen, Enzhe Lu, Maida Wu, Guanghan Cai
//   Citation:
//     Homework pdf;
//     Course slices;
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

// Current state of world.
extern unsigned char* g_data;
// result of the world
extern unsigned char* g_resultData;
// world width
extern size_t g_worldWidth;
// world height should be 2 larger than the actual height because of ghost rows
extern size_t g_worldHeight;
extern size_t g_dataLength;
// file data used to init the world
extern int* resource;

// init resource as cuda memory
extern void init_resource(int state_size);

// init world
extern void sov_initworld(int state_size);

// lanuch cuda functions
extern void kernel_function(size_t blockNum, size_t threadsCount, unsigned char** d_data, unsigned char** d_resultData, size_t worldWidth, size_t worldHeight);

// used to record time
typedef unsigned long long ticks;
static __inline__ ticks getticks(void)
{
unsigned int tbl, tbu0, tbu1;
do {
	__asm__ __volatile__ ("mftbu %0" : "=r"(tbu0));
	__asm__ __volatile__ ("mftb %0" : "=r"(tbl));
	__asm__ __volatile__ ("mftbu %0" : "=r"(tbu1));
	} while (tbu0 != tbu1);
	return ((((unsigned long long)tbu0) << 32) | tbl);
}


// transposite the matrix in order to make sure that each state adjacent with each other by left and right
char* exchange_row_col(char * g_data, int state_size){
    char* tmp_data=calloc((state_size+2)*state_size, sizeof(char));
    for(int i=0;i<state_size;++i){
        tmp_data[i]='1';
    }

    for(int i=state_size;i<(state_size+1)*state_size;++i){
        int row = (i-state_size)/state_size;
        int col = (i-state_size) - row*state_size;
        int tmp_row = col;
        int tmp_col = state_size-1- row;
        int tmp_index = tmp_row*state_size+tmp_col;
        tmp_data[tmp_index+state_size]= g_data[i-state_size];
    }

    for(int i= (state_size+1)*state_size;i<(state_size+2)*state_size;++i){
        tmp_data[i]='1';
    }
    return tmp_data;
}


 static inline void gol_printWorld()
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
  int num_state = atoi(argv[1]);
	int state_size = atoi(argv[2]);
	int threadsCount = atoi(argv[3]);
	int itterations = atoi(argv[4]);
	int output = atoi(argv[5]);
	// state real size
  int state_len = state_size * state_size;
	// bufsize used to write file since we have to consider about newline mark
	int bufsize = (state_size +1) * state_size;
	// number of char that needed to be read
	int nints = state_len/sizeof(char);

  char filename[20];
  strcpy(filename,"file_");

	int rankid,
      ranknum,			// number of ranks
      mtype,			  // message type
      down_dest,    // rankid down from the current rank
      up_dest;      // rankid up from the current rank
  MPI_File fh;     // read file descripter
	MPI_File oh;     // write file descripter
  MPI_Status status;
	// time recorders
	ticks compute_timer;
	ticks input_timer;
	ticks output_timer;
	ticks io_timer;

	// init MPI
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rankid);
  MPI_Comm_size(MPI_COMM_WORLD, &ranknum);
	// define direction of sending and receiving data
	MPI_Request send_down,receive_down,send_up,receive_up;
  char* buf=calloc(state_len,sizeof(char));
	// get file's names
	sprintf(filename+strlen(filename),"%d.txt",rankid);
	if(rankid == 0){
		input_timer = getticks();
	}
  MPI_File_open(MPI_COMM_WORLD, filename,MPI_MODE_RDONLY, MPI_INFO_NULL, &fh);
	MPI_File_seek(fh,0,MPI_SEEK_SET);
	MPI_File_read_all(fh, buf, nints, MPI_CHAR, &status);

	if(rankid == 0){
		input_timer = (getticks() - input_timer)/512000;
    printf("Overall input file time: %llu\n",input_timer);
	}
	// receive data from buf
	char* temp = calloc(state_len,sizeof(char));
  strcpy(temp,buf);
  free(buf);

  MPI_File_close(&fh);
	// transposite matrix and add ghost rows
	temp = exchange_row_col(temp,state_size);
  //init resource that needed to init world
  init_resource(state_size);

  // transfer data into resource
	for(int i = 0; i < strlen(temp);i++){
    resource[i] = temp[i] - '0';
  }

	// init world
  sov_initworld(state_size);



	if (rankid == 0) {
    compute_timer = getticks();
  }
	mtype = 1;
  size_t i;
	// let each rank know the upper and down's rank id
  down_dest = (rankid+1)%ranknum;
  up_dest = (rankid-1+ranknum)%ranknum;
	size_t blockNum = g_worldWidth*g_worldHeight / threadsCount;
	// I added two rows based on the original world, so if the worldsize if 36*36. I would have a new world of 38*36 with 38 is number of row.
  for(i = 0;i<itterations;i++){
		// states at bounderies don't need to transfer data to specific ghost row
		if(rankid == 0){
			MPI_Isend( &g_data[(g_worldHeight-2)*g_worldWidth], g_worldWidth, MPI_UNSIGNED_CHAR, down_dest, mtype, MPI_COMM_WORLD, &send_down );
			MPI_Irecv( &g_data[(g_worldHeight-1)*g_worldWidth], g_worldWidth, MPI_UNSIGNED_CHAR, down_dest, mtype, MPI_COMM_WORLD, &receive_up);
			// wait until the rank sends and receives the data
      MPI_Wait(&send_down, &status);
      MPI_Wait(&receive_up, &status);
		}else if(rankid == ranknum - 1){
			MPI_Isend( &g_data[g_worldWidth], g_worldWidth, MPI_UNSIGNED_CHAR, up_dest, mtype, MPI_COMM_WORLD, &send_up );
			MPI_Irecv( &g_data[0], g_worldWidth, MPI_UNSIGNED_CHAR, up_dest, mtype, MPI_COMM_WORLD, &receive_down);
			// wait until the rank sends and receives the data
      MPI_Wait(&send_up, &status);
      MPI_Wait(&receive_down, &status);
		}else{
      MPI_Isend( &g_data[g_worldWidth], g_worldWidth, MPI_UNSIGNED_CHAR, up_dest, mtype, MPI_COMM_WORLD, &send_up );
      MPI_Isend( &g_data[(g_worldHeight-2)*g_worldWidth], g_worldWidth, MPI_UNSIGNED_CHAR, down_dest, mtype, MPI_COMM_WORLD, &send_down );
      MPI_Irecv( &g_data[(g_worldHeight-1)*g_worldWidth], g_worldWidth, MPI_UNSIGNED_CHAR, down_dest, mtype, MPI_COMM_WORLD, &receive_up);
      MPI_Irecv( &g_data[0], g_worldWidth, MPI_UNSIGNED_CHAR, up_dest, mtype, MPI_COMM_WORLD, &receive_down);
			// wait until the rank sends and receives the data
      MPI_Wait(&send_up, &status);
      MPI_Wait(&send_down, &status);
      MPI_Wait(&receive_up, &status);
      MPI_Wait(&receive_down, &status);
		}
    // update the inner part of the world
    kernel_function(blockNum, threadsCount, &g_data, &g_resultData, g_worldWidth,g_worldHeight);
  }
// when the iteration is over, get the overall compute time
	if(rankid == 0){
		compute_timer = (getticks() - compute_timer)/512000;
		printf("Overall computation time: %llu\n",compute_timer);
	}
// we assume that if g_data is greater than 3, the cell should be infected
	for(int i = 0; i < g_dataLength; i++){
		if(g_data[i] > 3){
			g_data[i] = 0;
		}
	}
char* outfile = calloc(bufsize,sizeof(char));
  if(output == 1){
    /*************************** write file start here ****************************/
    if(rankid == 0){
      output_timer = getticks();
    }
		// open a shared file to write content
    MPI_File_open(MPI_COMM_WORLD, "result.txt", MPI_MODE_CREATE | MPI_MODE_RDWR, MPI_INFO_NULL, &oh);
		// use offset to let each rank know where to write
    MPI_Offset offset = rankid * bufsize;
    MPI_File_set_view(oh, offset, MPI_CHAR, MPI_CHAR, "native", MPI_INFO_NULL);
// convert g_data to string in order to write into the file
   int count = 0;
   for(int i =0; i< state_size; i++){
     for(int j =0; j < state_size + 1;j++){
       if(j == state_size){
         outfile[i*(state_size+1)+j] = '\n';
       }else{
         sprintf(&outfile[i*(state_size+1)+j], "%u ", (g_data[((i+1)*state_size) + j]));
       }
     }
   }
    MPI_File_write_all(oh, outfile, strlen(outfile), MPI_CHAR, MPI_STATUS_IGNORE);
    MPI_File_close(&oh);
    if(rankid == 0){
      output_timer = (getticks() - output_timer)/512000;
      printf("Overall output file time: %llu\n",output_timer);
    }
    /*************************** end of file write ********************************/
  }

  MPI_Finalize();
	// statistic the recover rate, death rate and infect rate.
	int num_infect = 0;
  int num_recover = 0;
  int num_death = 0;
  for(int i = 0; i < bufsize;i++){
    if(outfile[i] == '2') num_recover++;
    if(outfile[i] == '3') num_death++;
    if(outfile[i] != '1' && outfile[i] != '\n') num_infect++;
  }
  double recover_rate = (double)num_recover/(double)num_infect;
  double death_rate = (double)num_death/(double)num_infect;
  double infect_rate = (double)num_infect/(double)state_len;
  if(rankid == 0){
    printf("recover rate: %f\n",recover_rate);
    printf("death rate: %f\n",death_rate);
    printf("infect rate: %f\n",infect_rate);
  }
	return 0;
}
