#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "../headers/preprocess.h"

void AssignPointsToBlocks(float *points, int number_of_points, int grid_d, int dimensions, int* block_of_point ){
    
    if(dimensions < 3){
        fprintf(stderr, "Not enough dimensions. TODO: implement special case\n");
        exit(0);
    }
    for(int i = 0; i < number_of_points; i++){
        int bIndex = 0;
        int gridMultiplier = 1;
        for(int j = 0; j < 3; j++){
            bIndex += gridMultiplier * (int)floor(points[i+j*number_of_points]*grid_d);
            //printf("%d\n",bIndex);
            gridMultiplier *= grid_d;
        }
        block_of_point[i] = bIndex;
    }
}



void CountPointsPerBlock(int* block_of_point, int number_of_points, 
                        int grid_d, int dimensions, 
                        int* points_per_block, int* intg_points_per_block)
{
    for(int i = 0; i < pow(grid_d,dimensions); i++){
        points_per_block[i] = 0;
        intg_points_per_block[i] = 0;
    }
    for(int i = 0; i < number_of_points; i++){
        points_per_block[block_of_point[i]]++;
    }

    for(int i = 1; i < pow(grid_d,dimensions); i++){
        intg_points_per_block[i] = intg_points_per_block[i-1] + points_per_block[i-1];
    }


}

/*
Int array, used for sorting the points based on an other array.
The comparatro function will access it and qsort will calculate the permutation
*/
int* g_sort_block_of_point;
int CompareFunc(const void* a, const void* b){
    int x = *((const int*)a);
    int y = *((const int*)b);
    return g_sort_block_of_point[x] - g_sort_block_of_point[y];
}

/*
    Sorts the points based on the block they belong, so points int the
    same block will be continuous in memory.
    Needs preallocated oreder_points array
    and perm_arr.
*/
void ReorderPointsByBlock(float* points, int* block_of_point, 
                int number_of_points, int dimensions, 
                float* ordered_points, int* perm_arr)
{   
    g_sort_block_of_point = block_of_point;
    for(int i = 0; i < number_of_points; i++){
        perm_arr[i] = i;
    }
    
    qsort(perm_arr, number_of_points, sizeof(float), CompareFunc);

    for(int i = 0; i < number_of_points; i++){
        for(int j = 0; j < dimensions; j++){
            ordered_points[i + j*number_of_points] = points[perm_arr[i] + j*number_of_points];
        }
    }
    g_sort_block_of_point = NULL;

    
}

