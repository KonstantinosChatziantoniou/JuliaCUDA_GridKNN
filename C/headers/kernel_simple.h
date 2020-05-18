#ifndef H_KER_SIMP
#define H_KER_SIMP

#ifdef __cplusplus
extern "C" 
#endif 
void GridKNN(int number_of_points, int number_of_queries, int grid_d, int dimensions,
        float* ordered_ref_points, float* ordered_queries, 
        int* intg_points_per_block, int* intg_queries_per_block,
        int* points_per_block, int* queries_per_block,
        float** out_distances, int** out_neighbours);
#endif