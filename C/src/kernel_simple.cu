/*
    Konstantinos Chatziantoniou

    This is the kernel for the simple implementation. The kernel searches for the nearest
    neighbour of a query in a particular block. The kernel is called multiple times to
    search all adjacent blocks. It searches ALL the blocks without any check for
    unnecessary searches.

*/

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_profiler_api.h>
#include <stdio.h>
#include <stdlib.h>
#include "../headers/kernel_simple.h"

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
#define TRUE 1
#define FALSE 0
inline void gpuAssert(cudaError_t code, const char *file, int line)
{
    if (code != cudaSuccess) 
    {
        fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        exit(code);
    }
}




__global__ 
void gpu_grid_knn(float* points, float* queries, 
    int* intgr_points_per_block, int* intgr_queries_per_block, 
    int* points_per_block, int* queries_per_block, 
    float* distsances, int* neighbours,
    int num_of_points, int num_of_queries, int dimensions, int grid_d,
    int offx, int offy, int offz)
{

    extern __shared__ float shared_array[];
    // Check if the block is inbounds
    if( (int)blockIdx.x + offx  < 0 || offx + (int)blockIdx.x >= grid_d || 
    (int)blockIdx.y + offy  < 0 || offy + (int)blockIdx.y >= grid_d ||
    (int)blockIdx.z + offz  < 0 || offz + (int)blockIdx.z >= grid_d) return;

    // Block of queries
    int q_bid = blockIdx.x + blockIdx.y*gridDim.x + blockIdx.z*gridDim.x*gridDim.y;
    // Block of points to search
    int p_bid = blockIdx.x+offx + (blockIdx.y+offy)*gridDim.x + (blockIdx.z+offz)*gridDim.x*gridDim.y;
    int tid = threadIdx.x + threadIdx.y*blockDim.x;
    int stride = blockDim.x*blockDim.y;

    float* sh_queries = shared_array;
    float* sh_points = &shared_array[stride*dimensions];
    int start_points = intgr_points_per_block[p_bid];
    int start_queries = intgr_queries_per_block[q_bid];
    int total_points = points_per_block[p_bid];
    int total_queries = queries_per_block[q_bid];
    float distance;
    int neighbour;


    for(int q = 0; q < total_queries; q += stride){
        // Read queries to shared memory
        int q_index = q + tid + start_queries;
        if(tid + q < total_queries){
            for(int d = 0; d < dimensions; d++){
                sh_queries[tid + d*stride] = queries[q_index + d*num_of_queries];
            }
            distance = distsances[q_index];
            neighbour = neighbours[q_index];
        }
        for(int p = 0; p < total_points; p+= stride){
            // Read points to shared memory
            __syncthreads();
            if(p + tid < total_points){
                for(int d = 0; d < dimensions; d++){
                    sh_points[tid + d*stride] = points[start_points + p + tid + d*num_of_points];
                }
            }
            __syncthreads();

            // For each point read, calculate distance and save the minimum.
            int bounds = stride < total_points-p ? stride : total_points-p;
            if(tid + q < total_queries){
                for(int i = 0; i < bounds; i++){
                    float tempdist = 0;
                    for(int d = 0; d < dimensions; d++){
                        float tempquery = sh_queries[tid + d*stride];
                        tempdist += powf(tempquery -  sh_points[(i+tid)%bounds + d*stride], 2); //(i+tid)%bounds
                    }
                    tempdist = sqrtf(tempdist);
                    if(tempdist < distance){
                        neighbour = start_points+ p +(i+tid)%bounds;
                        distance = tempdist;
                    }
                }
            }
        }
        // Save result to global memory
        if(tid + q < total_queries){
            distsances[q_index] = distance;
            neighbours[q_index] = neighbour;
        }
    }
}



void GridKNN(int number_of_points, int number_of_queries, int grid_d, int dimensions,
        float* ordered_ref_points, float* ordered_queries, 
        int* intg_points_per_block, int* intg_queries_per_block,
        int* points_per_block, int* queries_per_block,
        float** out_distances, int** out_neighbours)
{
    float* distances = (float*)malloc(number_of_queries*sizeof(float));
    int* neighbours = (int*)malloc(number_of_queries*sizeof(int));

    float* dev_points; 
    float* dev_queries; 
    int *dev_intg_points_per_block, *dev_points_per_block;
    int *dev_intg_queries_per_block, *dev_queries_per_block;
    float *dev_distances;
    int* dev_neighbours;

    gpuErrchk(cudaMalloc((void**)&dev_points, number_of_points*dimensions*sizeof(float)));
    gpuErrchk(cudaMalloc((void**)&dev_queries, number_of_queries*dimensions*sizeof(float)));
    gpuErrchk(cudaMalloc((void**)&dev_intg_points_per_block, pow(grid_d,dimensions)*sizeof(int)));
    gpuErrchk(cudaMalloc((void**)&dev_intg_queries_per_block, pow(grid_d,dimensions)*sizeof(int)));
    gpuErrchk(cudaMalloc((void**)&dev_points_per_block, pow(grid_d,dimensions)*sizeof(int)));
    gpuErrchk(cudaMalloc((void**)&dev_queries_per_block, pow(grid_d,dimensions)*sizeof(int)));
    gpuErrchk(cudaMalloc((void**)&dev_distances, number_of_queries*sizeof(float)));
    gpuErrchk(cudaMalloc((void**)&dev_neighbours, number_of_queries*sizeof(int)));

    gpuErrchk(cudaMemcpy(dev_points, ordered_ref_points, number_of_points*dimensions*sizeof(float), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(dev_queries, ordered_queries, number_of_queries*dimensions*sizeof(float), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(dev_intg_points_per_block, intg_points_per_block, pow(grid_d,dimensions)*sizeof(int), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(dev_intg_queries_per_block, intg_queries_per_block, pow(grid_d,dimensions)*sizeof(int), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(dev_points_per_block, points_per_block, pow(grid_d,dimensions)*sizeof(int), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(dev_queries_per_block, queries_per_block, pow(grid_d,dimensions)*sizeof(int), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemset(dev_distances, 100, number_of_queries*sizeof(float)))

    // Config
    int thread_groups = 2;
    dim3 blocks = {(unsigned int)grid_d, (unsigned int)grid_d, (unsigned int)grid_d};
    dim3 threads = {(unsigned int)32,(unsigned int)thread_groups,(unsigned int)1};
    uint64_t shmem = (2*thread_groups*32*dimensions)*sizeof(float);
    // Run knn for main block
    gpu_grid_knn<<<blocks, threads, shmem>>>(dev_points, dev_queries,
                dev_intg_points_per_block, dev_intg_queries_per_block,
                dev_points_per_block, dev_queries_per_block,
                dev_distances, dev_neighbours,
                number_of_points, number_of_queries, dimensions, grid_d, 0,0,0);
    // Run knn for neighbouring blocks
    for(int x = -1; x < 2; x++){
        for(int y = -1; y < 2; y++){
            for(int z = -1; z < 2; z++){
                if(x == 0 && y == 0 && z == 0) continue;
                  gpu_grid_knn<<<blocks, threads, shmem>>>(dev_points, dev_queries,
                    dev_intg_points_per_block, dev_intg_queries_per_block,
                    dev_points_per_block, dev_queries_per_block,
                    dev_distances, dev_neighbours,
                    number_of_points, number_of_queries, dimensions, grid_d, x,y,z);
            }
        }
    }
    gpuErrchk(cudaMemcpy(distances, dev_distances, number_of_points*sizeof(float), cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(neighbours, dev_neighbours, number_of_points*sizeof(int), cudaMemcpyDeviceToHost));
    cudaFree(dev_points);
    cudaFree(dev_queries);
    cudaFree(dev_points_per_block);
    cudaFree(dev_queries_per_block);
    cudaFree(dev_intg_points_per_block);
    cudaFree(dev_intg_queries_per_block);
    cudaFree(dev_distances);
    cudaFree(dev_neighbours);

    *out_distances = distances;
    *out_neighbours = neighbours;
}
