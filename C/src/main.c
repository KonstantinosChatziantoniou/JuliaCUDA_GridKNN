#include <stdlib.h>
#include <stdio.h>
#include <math.h>

// #include <cuda.h>
// #include <cuda_runtime.h>
// #include <cuda_profiler_api.h>

#include "../headers/preprocess.h"
#include "../headers/helpers.h"
#include "../headers/kernel_simple.h"
#include "../headers/kernel_with_skip.h"



int main(int argc, char** argv){
    if(argc < 4){
        printf("Usage:\tNumber of Points\n\tNumber of grids\n\tRandom seed\n");
        return 0;
    }
    int input_num_points = atoi(argv[1]);
    int input_grid_d = atoi(argv[2]);
    int inpt_seed = atoi(argv[3]);
   
    srand(inpt_seed);
    size_t number_of_points = 1<<input_num_points;
    size_t number_of_queries = number_of_points;
    size_t grid_d = 1<<input_grid_d;
    size_t dimensions = 3;

    printf("KNN for points: %ld\n\tqueries: %ld\n\tdimensions: %ld\n\tgrids: %ld^3\n",number_of_points,number_of_queries, dimensions, grid_d);
    // ------ Init Points Array and relevant arays. ------
    float* ref_points = (float*)malloc(number_of_points*dimensions*sizeof(float));
    float* ordered_ref_points = (float*)malloc(number_of_points*dimensions*sizeof(float));
    int* block_of_point = (int*)malloc(number_of_points*sizeof(int));
    int* points_per_block = (int*)malloc(pow(grid_d,dimensions)*sizeof(int));
    int* intg_points_per_block = (int*)malloc(pow(grid_d,dimensions)*sizeof(int));
    int* perm_points = (int*)malloc(number_of_points*sizeof(int));

    // ------ Init Queries Array and relevant arays. ------
    float* queries = (float*)malloc(number_of_queries*dimensions*sizeof(float));
    float* ordered_queries = (float*)malloc(number_of_queries*dimensions*sizeof(float));
    int* block_of_query = (int*)malloc(number_of_queries*sizeof(int));
    int* queries_per_block = (int*)malloc(pow(grid_d,dimensions)*sizeof(int));
    int* intg_queries_per_block = (int*)malloc(pow(grid_d,dimensions)*sizeof(int));
    int* perm_queries = (int*)malloc(number_of_queries*sizeof(int));

    // --------------------- Preprocess data ---------------------
    init_data(ref_points, number_of_points, dimensions);
    init_data(queries, number_of_queries, dimensions);
    save_data(ref_points, number_of_points, dimensions, "pts.csv");
    save_data(queries, number_of_queries, dimensions, "qrs.csv");

    printf("Preprocessing...\n");
    AssignPointsToBlocks(ref_points, number_of_points, grid_d, dimensions, block_of_point);
    CountPointsPerBlock(block_of_point, number_of_points, grid_d, dimensions, points_per_block, intg_points_per_block);
    ReorderPointsByBlock(ref_points, block_of_point, number_of_points, dimensions, ordered_ref_points, perm_points);

    AssignPointsToBlocks(queries, number_of_queries, grid_d, dimensions, block_of_query);
    CountPointsPerBlock(block_of_query, number_of_queries, grid_d, dimensions, queries_per_block, intg_queries_per_block);
    ReorderPointsByBlock(queries, block_of_query, number_of_queries, dimensions, ordered_queries, perm_queries);
    
    save_data(ordered_ref_points, number_of_points, dimensions, "ord_pts.csv");
    save_data(ordered_queries, number_of_queries, dimensions, "ord_qrs.csv");
    printf("\t...Done\n");
    // --------------------- GPU ---------------------
	// cudaProfilerStart();
    float *distances, *dist_skip;
    int *neighbours, *nb_skip;

    printf("Running knn (simple)...\n");
    GridKNN(number_of_points, number_of_queries, grid_d, dimensions,
        ordered_ref_points, ordered_queries,
        intg_points_per_block, intg_queries_per_block, 
        points_per_block, queries_per_block,
        &distances, &neighbours);
    printf("\t...Done\n");

    printf("Running knn (simple)...\n");    
    GridKNNskip(number_of_points, number_of_queries, grid_d, dimensions,
        ordered_ref_points, ordered_queries,
        intg_points_per_block, intg_queries_per_block, 
        points_per_block, queries_per_block,
        &dist_skip, &nb_skip);
    printf("\t...Done\n");

    printf("Saving results to files...\n");
    save_data(distances, number_of_queries, 1, "dists.csv");
    save_int_data(neighbours, number_of_queries, 1, "nb.csv");   
    save_data(dist_skip,number_of_queries , 1, "dists2.csv");
    save_int_data(nb_skip,number_of_queries ,1 , "nb2.csv");         // Change for the index of unordered points.    
    printf("\t...Done\n");

    // -----------------------------------------------------------
    free(ref_points);
    free(ordered_ref_points);
    free(points_per_block);
    free(intg_points_per_block);
    free(block_of_point);
    free(perm_points);


    free(queries);
    free(ordered_queries);
    free(queries_per_block);
    free(intg_queries_per_block);
    free(block_of_query);
    free(perm_queries);

    free(dist_skip);
    free(nb_skip);

    free(distances);
    free(neighbours);
    
    return 0;

}

// inline void gpuAssert(cudaError_t code, const char *file, int line)
// {
//     if (code != cudaSuccess) 
//     {
//         fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
//         exit(code);
//     }
// }

