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


unsigned char *g_resultData=NULL;

// Current state of world.
unsigned char *g_data=NULL;

// Current width of world.
size_t g_worldWidth=0;

/// Current height of world.
size_t g_worldHeight=0;

/// Current data length (product of width and height)
size_t g_dataLength=0;  // g_worldWidth * g_worldHeight


// record time
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

// transposite matrix and add ghost rows
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

// init world
void sov_initworld(int* resource,int state_size){
  g_worldWidth = state_size;
  g_worldHeight = state_size+2;
  g_dataLength = g_worldWidth * g_worldHeight;
  int i;
  g_data = calloc( g_dataLength, sizeof(unsigned char));
  for( i = 0; i < g_dataLength; i++)
  {
g_data[i] = resource[i];
  }
  g_resultData = calloc( g_dataLength, sizeof(unsigned char));
}


 void gol_swap(unsigned char** pA,unsigned char** pB)
 {
   // You write this function - it should swap the pointers of pA and pB.
   unsigned char *temp = *pA; // set a temporary pointer to save pA
   *pA = *pB; // set pB to pA
   *pB = temp; // set temp to pB

 }

 void gol_kernel( unsigned char* g_data, int worldWidth,
                             int worldHeight, unsigned char* g_resultData)
 {
     //According to blockIdx, blockDim, threadIdx, calculate index and stride
     int x, y;
     for(y = 1;y < worldHeight - 1;y++)
     {
         // get 2-D coordinates obtained through mathematical operations
         // write code to set: y0, y1 and y2

         int y0 = ((y+worldHeight-1)% worldHeight) * worldWidth;
         int y1 = y * worldWidth;
         int y2 = ((y + 1) % worldHeight) * worldWidth;
         for(x = 0; x < worldWidth; x ++){
           int x1 = x;
           int x0 = (x1 + worldWidth - 1) % worldWidth;
           int x2 = (x1 + 1) % worldWidth;
           int currentCell = g_data[(x1+y1)];
           if ( ( currentCell != 1) && (currentCell != 2) && (currentCell != 3) ){
               // check the patient is after 10 iterations
               if (currentCell >= 100){
                   // create a random value from 0-99
                   int random = rand() % 100;
                   // Any infected patient after 10 iterations will have 5% to die
                   if (random < 5){
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
                   //printf("condition 1: before currentcell %d\n",g_data[(x1+y1)]);
                   g_resultData[(x1+y1)] = g_data[(x1+y1)] + 10;
                  //printf("condition 1: after currentcell %d\n",g_resultData[(x1+y1)]);
               }
           }
           // if the current cell is not infected by COVID-19,
           // we should detect how many its neighbours are infected so that
           // we can calculate the infection probability of the current cell
           else if (currentCell == 1){
             //printf("condition 2: currentcell %d\n",currentCell);
             int infectProbability = 0;
             // consider about if the cell is at left or right border
             if(x % worldWidth == 0){
               int neighborIndex[5] = {(x1+y0),(x2+y0),(x2+y1),(x2+y2),(x1+y2)};
               for (int i = 0; i < 5; i++){
                   int neighborValue = g_data[neighborIndex[i]];
                   if (( neighborValue != 1) && (neighborValue != 2) && (neighborValue != 3)){
                       infectProbability += 30;
                   }
               }
             }else if(x % worldWidth == worldWidth - 1){
               int neighborIndex[5] = {(x1+y0),(x0+y0),(x0+y1),(x0+y2),(x1+y2)};
               for (int i = 0; i < 5; i++){
                   int neighborValue = g_data[neighborIndex[i]];
                   if (( neighborValue != 1) && (neighborValue != 2) && (neighborValue != 3)){
                       infectProbability += 30;
                   }
               }
             }else{
               int neighborIndex[8] = {(x0+y0),(x0+y1),(x0+y2),(x1+y0),(x1+y2),(x2+y0),(x2+y1),(x2+y2)};
               // Any live cell(value0) witha neighbor infected byC OVID-19 will have 10% to also infected by virus.
               // In other words, if all of its 8 neighbors are infected by virus, it will be 80% infected by virus.
               for (int index = 0; index < 8; index++){
                   int neighborValue = g_data[neighborIndex[index]];
                   if (( neighborValue != 1) && (neighborValue != 2) && (neighborValue != 3)){
                       infectProbability += 30;
                   }
               }
             }
               // create a random value from 0-99
               int random = rand() % 100;
               // if current cell is infected by COVID-19, change its value to 0
               // otherwise keep its value to be 1
               if (random < infectProbability){
                   g_resultData[(x1+y1)] = 0;
               }
               else{
                   g_resultData[(x1+y1)] = 1;
               }
           }
           // If a cell recovers or dies (with value 2 or 3), their value will never be changed in the future.
           else{
             //printf("condition 3: currentcell %d\n",currentCell);
               g_resultData[(x1+y1)] = g_data[(x1+y1)];
           }

         }
     }
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

void print_rate(int rankid,int state_len){
  int num_infect = 0;
  int num_recover = 0;
  int num_death = 0;
  for(int i = g_worldWidth; i < g_dataLength-g_worldWidth;i++){
    if(g_data[i] == 2) num_recover++;
    if(g_data[i] == 3) num_death++;
    if(g_data[i] != 1) num_infect++;
  }
  double recover_rate = (double)num_recover/(double)num_infect;
  double death_rate = (double)num_death/(double)num_infect;
  double infect_rate = (double)num_infect/(double)state_len;
  if(rankid == 0){
    printf("rank %d: recover rate: %f\n",rankid,recover_rate);
    printf("rank %d: death rate: %f\n",rankid,death_rate);
    printf("rank %d: infect rate: %f\n",rankid,infect_rate);
  }
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
	int bufsize = (state_size+1) * state_size;
	int ghostsize = (state_size + 2) * state_size;
  	// number of char that needed to be read
	int nints = state_len/sizeof(char);

  char filename[27];
  strcpy(filename,"data_5/file_");

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
	MPI_Request send_down,receive_down,send_up,receive_up;
  char* buf=calloc(state_len,sizeof(char));
	sprintf(filename+strlen(filename),"%d.txt",rankid);

  // if(rankid == 0){
	// 	input_timer = getticks();
	// }
  MPI_File_open(MPI_COMM_WORLD, filename,MPI_MODE_RDONLY, MPI_INFO_NULL, &fh);
  MPI_File_seek(fh,0,MPI_SEEK_SET);
  MPI_File_read_all(fh, buf, nints, MPI_CHAR, &status);

	// if(rankid == 0){
	// 	input_timer = (getticks() - input_timer)/512000;
  //   printf("Overall input file time: %llu\n",input_timer);
	// }
  char* temp = calloc(state_len,sizeof(char));
  strcpy(temp,buf);
  free(buf);

  MPI_File_close(&fh);

	// matrix transposition start here
  temp = exchange_row_col(temp,state_size);
  int* resource = calloc(ghostsize,sizeof(int));

	for(int i = 0; i < strlen(temp);i++){
    resource[i] = temp[i] - '0';
  }

  sov_initworld(resource,state_size);

  /********************** insert data handle part here **************************/
	// if (rankid == 0) {
  //   compute_timer = getticks();
  // }
	mtype = 1;
  // let each rank know the upper and down's rank id
  size_t i;
  down_dest = (rankid+1)%ranknum;
  up_dest = (rankid-1+ranknum)%ranknum;
  // I added two rows based on the original world, so if the worldsize if 36*36. I would have a new world of 38*36 with 38 is number of row.
  for(i = 0;i<itterations;i++){
    // states at bounderies don't need to transfer data to specific ghost row
		if(rankid == 0){
			MPI_Isend( &g_data[(g_worldHeight-2)*g_worldWidth], g_worldWidth, MPI_UNSIGNED_CHAR, down_dest, mtype, MPI_COMM_WORLD, &send_down );
			MPI_Irecv( &g_data[(g_worldHeight-1)*g_worldWidth], g_worldWidth, MPI_UNSIGNED_CHAR, down_dest, mtype, MPI_COMM_WORLD, &receive_up);
      MPI_Wait(&send_down, &status);
      MPI_Wait(&receive_up, &status);
		}else if(rankid == ranknum - 1){
			MPI_Isend( &g_data[g_worldWidth], g_worldWidth, MPI_UNSIGNED_CHAR, up_dest, mtype, MPI_COMM_WORLD, &send_up );
			MPI_Irecv( &g_data[0], g_worldWidth, MPI_UNSIGNED_CHAR, up_dest, mtype, MPI_COMM_WORLD, &receive_down);
      MPI_Wait(&send_up, &status);
      MPI_Wait(&receive_down, &status);
		}else{
      MPI_Isend( &g_data[g_worldWidth], g_worldWidth, MPI_UNSIGNED_CHAR, up_dest, mtype, MPI_COMM_WORLD, &send_up );
      MPI_Isend( &g_data[(g_worldHeight-2)*g_worldWidth], g_worldWidth, MPI_UNSIGNED_CHAR, down_dest, mtype, MPI_COMM_WORLD, &send_down );
      MPI_Irecv( &g_data[(g_worldHeight-1)*g_worldWidth], g_worldWidth, MPI_UNSIGNED_CHAR, down_dest, mtype, MPI_COMM_WORLD, &receive_up);
      MPI_Irecv( &g_data[0], g_worldWidth, MPI_UNSIGNED_CHAR, up_dest, mtype, MPI_COMM_WORLD, &receive_down);
      MPI_Wait(&send_up, &status);
      MPI_Wait(&send_down, &status);
      MPI_Wait(&receive_up, &status);
      MPI_Wait(&receive_down, &status);
		}
    // wait until the rank sends and receives the data
    gol_kernel(g_data,g_worldWidth,g_worldHeight,g_resultData);
    gol_swap(&g_data,&g_resultData);
  }
  // if(rankid == 0){
  //   compute_timer = (getticks() - compute_timer)/512000;
  //   printf("Overall computation time: %llu\n",compute_timer);
  // }
  for(int i = 0; i < g_dataLength; i++){
    if(g_data[i] > 3){
      g_data[i] = 0;
    }
  }
char* outfile = calloc(bufsize,sizeof(char));
  if(output == 1){
    /*************************** write file start here ****************************/
    // if(rankid == 0){
    //   output_timer = getticks();
    // }
    MPI_File_open(MPI_COMM_WORLD, "result.txt", MPI_MODE_CREATE | MPI_MODE_RDWR, MPI_INFO_NULL, &oh);
    MPI_Offset offset = rankid * bufsize;
    MPI_File_set_view(oh, offset, MPI_CHAR, MPI_CHAR, "native", MPI_INFO_NULL);

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
    // if(rankid == 0){
    //   output_timer = (getticks() - output_timer)/512000;
    //   printf("Overall output file time: %llu\n",output_timer);
    // }
    /*************************** end of file write ********************************/
  }

  MPI_Finalize();
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
