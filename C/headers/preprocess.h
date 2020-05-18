#ifndef PREPRO_H
#define PREPRO_H
#ifdef __cplusplus
extern "C" 
#endif 
void AssignPointsToBlocks(float *points, int number_of_points, int grid_d, int dimensions, int* block_of_point);

#ifdef __cplusplus
extern "C" 
#endif 
void CountPointsPerBlock(int* block_of_point, int number_of_points, 
                        int grid_d, int dimensions, 
                        int* out_points_per_block, int* out_intg_points_per_block);
#ifdef __cplusplus
extern "C" 
#endif 
void ReorderPointsByBlock(float* points, int* block_of_point, 
                int number_of_points, int dimensions, 
                float* ordered_points, int* perm_arr);
#endif