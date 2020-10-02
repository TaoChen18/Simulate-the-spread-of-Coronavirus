//
//  main.c
//  generate number
//
//  Created by 巫劢达 on 5/4/20.
//  Copyright © 2020 巫劢达. All rights reserved.
//

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

char* generate_number(int world_size);
int main(int argc, const char * argv[]) {
    // insert code here...
    int tmp_size = 256;
    int world_size = tmp_size*tmp_size;

    for(int i = 0;i<64;i++){
      char filename[30];
      sprintf(filename+strlen(filename),"data_5/file_%d.txt",i);
      char *g_data = NULL;
      g_data = generate_number(world_size);
      FILE *fp = fopen(filename,"w");

      for(int i=0;i<world_size;++i){
          fprintf(fp,"%c",g_data[i]);

      }
      fclose(fp);
    }
      printf("Hello, World!\n");
    return 0;
}
char* generate_number(int world_size){
    char* g_data=calloc(world_size, sizeof(char));
    srand((unsigned) (time(NULL)));
    int rate_0 = 5;
//    int rate_1 = 80;
//    int rate_2 = 90;
    for(int i=0; i<world_size;++i){
        int tmp_number = rand()%100;
        if(tmp_number<rate_0){
            g_data[i]='0';
        }
        else{
            g_data[i]='1';
        }
    }
    return g_data;
}
