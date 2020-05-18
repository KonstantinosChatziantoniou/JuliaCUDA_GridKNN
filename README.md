# JuliaCUDA_GridKNN

An implementation of the grid knn algorithm for gpgpu, using `CUDAnative` and `CuArrays` for `Julia`.
The aim is to show that Julia for CUDA programming  can achieve not only faster development time, but also equivelent or even
better execution time than `C`(the goto language for CUDA kernel programming along with C++).

## Implementation
This implementation (both C and Julia) are for k = 1.
The Points and Queries are assigned to blocks (CPU). For each query, we will search the nearest neihgbor in the block of the query and 
all the adjacent blocks.

There are two implementations, named 'simple' and 'with_skip'. The first one search all the adjacent blocks, 
whereas the 'with_skip' checks if the distance of the current neighbour is bigger than the
distance from the boundaries of the box and skips some block searches.


## File structure

```Julia/```

+ `main.jl` Preprocesses the points and calls the kernels. Modify for different problem size and differrent kernel implementation. 
Returns Distances and Indices for each Nearest Neighbour

+ `validation.jl` Does everything `main.jl` does. It validates the results using the `NearestNeighbour` package.

+ `/kernel_**_view_function.jl/`

Contains a variation of the kernels,that use @view for every array that is possible and have the distance calculations
in an external function.


## C implementaion

The `C` implementation of the grid knn algorithm is taken and modified from this repo [https://github.com/KonstantinosChatziantoniou/GridKnn-Cuda]

```C/src/```

Contains the source code for the kernels.

## Execution

`Julia/main.jl` has two modes

+ 1 argument -> grids (2^ )^3. Reads the points and queries csv files located in the Julia/ folder
+ 3 arguments ->  number of points (2^), grids (2^ )^3, seed.  Generates random dataset with the given size.

You can uncomment the kernels you want to run.

`Julia/validate.jl`. Uncomment the ONE kernel you want to validate and run the script without arguments.

For the`C` implementation

                      run `make`
                      run `./mainProgram N B s` where N is the number of points (2^N) 
                          and B is the number of blocks per dimension (2^B,2^B,2^B) and s is the seed for the 
                          generation of random points.
The c implementation saves the points and the results to csv, so they can be given as input to Julia, to benchmark
on the same dataset and then check the results.

## Results
Google colab: Tesla P4
![res.png](https://raw.githubusercontent.com/KonstantinosChatziantoniou/JuliaCUDA_GridKNN/master/res.png)
