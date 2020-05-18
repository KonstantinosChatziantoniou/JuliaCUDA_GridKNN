#ifndef HELP_H
#define HELP_H
#include <stdlib.h>
#include <sys/time.h>

#ifdef __cplusplus
extern "C" 
#endif 
void init_data(float* pts, size_t number_of_points, size_t dimensions);

#ifdef __cplusplus
extern "C" 
#endif 
void save_data(float* pts, size_t number_of_points, size_t dimensions, const char* name);

#ifdef __cplusplus
extern "C" 
#endif 
void save_int_data(int* pts, size_t number_of_points, size_t dimensions ,const char* name);

#ifdef __cplusplus
extern "C" 
#endif 
void printTime(struct timeval start, struct timeval end, const char* str);




#endif