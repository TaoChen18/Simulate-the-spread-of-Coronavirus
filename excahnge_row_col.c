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
int* exchange_row_col(int * g_data, int state_size);
void my_print(int *g_data, int size_array);
int main(int argc, const char * argv[]) {
    // insert code here...
    printf("Hello, World!\n");
    int* g_data = calloc(9, sizeof(int));
    for(int i=0;i<9;++i){
        g_data[i]=i;
    }
    my_print(g_data, 9);
    free(g_data);
    g_data=exchange_row_col(g_data, 3);
    my_print(g_data, 15);
    return 0;
}
void my_print(int *g_data, int size_array){
    for(int i=0;i<size_array;++i){
        printf("%d",g_data[i]);
    }
    printf("\n");
}
int* exchange_row_col(int * g_data, int state_size){
    int* tmp_data=calloc((state_size+2)*state_size, sizeof(char));
    for(int i=0;i<state_size;++i){
        tmp_data[i]=1;
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
        tmp_data[i]=1;
    }
    return tmp_data;
}
