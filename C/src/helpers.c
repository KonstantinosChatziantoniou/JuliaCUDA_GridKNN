#include <stdio.h>
#include <stdlib.h>
#include "../headers/helpers.h"


void init_data(float* pts, size_t number_of_points, size_t dimensions){
    for(int i = 0; i < number_of_points; i++){
        for(int j = 0; j < dimensions; j++){
            pts[i + j*number_of_points] = (float)rand()/RAND_MAX;
        }
    }
}
void printTime(struct timeval start, struct timeval end, const char* str){
    unsigned long ss,es,su,eu,s,u;
    ss  =start.tv_sec;
    su = start.tv_usec;
    es = end.tv_sec;
    eu = end.tv_usec;
    s = es - ss;
    if(eu > su){
        u = eu - su;
    }else{
        s--;
        u = 1000000 + eu - su;
    }
   
    printf("%s,%lu,%lu\n",str,s,u);
}
void save_data(float* pts, size_t number_of_points, size_t dimensions, const char* name){
    FILE* f = fopen(name, "w+");
    fprintf(f, "%lu %lu\n", number_of_points, dimensions);
    for(int i = 0; i < number_of_points-1; i++){
        for(int j = 0; j < dimensions-1; j++){
            fprintf(f, "%.10f ", pts[i + j*number_of_points]);
        }
        fprintf(f, "%.10f\n",pts[i + (dimensions-1)*number_of_points]);
    }

    for(int j = 0; j < dimensions-1; j++){
            fprintf(f, "%.10f ", pts[(number_of_points-1) + j*number_of_points]);
        }
        fprintf(f, "%.10f\n",pts[(number_of_points-1) + (dimensions-1)*number_of_points]);
    fclose(f);
}

void save_int_data(int* pts, size_t number_of_points, size_t dimensions, const char* name){
    FILE* f = fopen(name, "w+");
    fprintf(f, "%lu %lu\n", number_of_points, dimensions);
     for(int i = 0; i < number_of_points-1; i++){
        for(int j = 0; j < dimensions-1; j++){
            fprintf(f, "%d ", pts[i + j*number_of_points]);
        }
        fprintf(f, "%d\n",pts[i + (dimensions-1)*number_of_points]);
    }

    for(int j = 0; j < dimensions-1; j++){
            fprintf(f, "%d ", pts[(number_of_points-1) + j*number_of_points]);
        }
        fprintf(f, "%d\n",pts[(number_of_points-1) + (dimensions-1)*number_of_points]);
    fclose(f);
}