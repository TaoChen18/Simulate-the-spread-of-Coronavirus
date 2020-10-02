//
//  main.c
//  exchange_row_col
//
//  Created by JOJO WU on 5/4/20.
//  Copyright © 2020 JOJO WU. All rights reserved.
//

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
void exchange_row_col(char * g_data, int state_size){
    char* tmp_data=calloc((state_size+2)*state_size, sizeof(char));
    for(int i=0;i<state_size;++i){
        tmp_data[i]='1';
    }
    for(int i=state_size;i<(state_size+1)*state_size;++i){
        int row = (i-state_size)/state_size;
        int col = (i-state_size) - row*state_size;
        int tmp_row = col;
        int tmp_col = state_size - row;
        int tmp_index = tmp_row*state_size+tmp_col;
        tmp_data[tmp_index+state_size]= g_data[i];
    }
    for(int i= (state_size+1)*state_size;i<(state_size+2)*state_size;++i){
        tmp_data[i]='1';
    }
    free(g_data);
    g_data = tmp_data;
}
int main(int argc, const char * argv[]) {
    // insert code here...
    int state_size = 5;
    char* g_data = calloc(state_size*state_size,sizeof(char));
    for(int i = 0; i < strlen(g_data);i++){
      if(g_data[i]%state_size == state_size-1) g_data[i] = '1';
      else g_data[i] = '0';
    }
    printf("%s\n",g_data);
    return 0;
}
