# JuliaCUDA_GridKNN

An implementation of the grid knn for gpgpu, using `CUDAnative` and `CuArrays` for `Julia`.
The aim is to show that Julia for CUDA programming  can achieve not only faster development time, but also equivelent or even
better results than `C`(the goto language for CUDA kernel programming along with C++).

## Implementation
The Points and Queries are assigned to blocks. For each query, we will search the nearest neihgbor in the block of the query and 
all the adjacent blocks.

The `Julia/Code/multi_kernel.jl` and the `C` implementations search all the adjacent blocks, 
whereas the `Julia/Code/multi_kernel_check.jl` checks if the distant of the current neighbour is bigger than the
distance from the boundaries of the box and skips some block searches.


## File structure

```Julia/```

+ `main.jl` Preprocesses the points and calls the kernels. Modify for different problem size and differrent kernel implementation. 
Returns Distances and Indices for each Nearest Neighbour

+ `validation.jl` Does everything `main.jl` does. It validates the results using the `NearestNeighbour` package.

```Julia/Code/```

Contains the preprocessing function definitions and the kernel definitions.


```C/```

Contains the `C` implementation of the grid knn algorithm from this repo [https://github.com/KonstantinosChatziantoniou/GridKnn-Cuda]


```Notebook```

Containg a Jyputer notebook to be executed in google colab for execution time benchmarking. The notebook includes scripts for 
installing cuda and julia in goole colab.


## Execution

`Julia/main.jl` and `Julia/validation.jl` are ready to be executed as standalone scripts.

For the`C` implementation

                      run `make`
                      run `./mainProgram N B` where N is the number of points (2^N) 
                          and B is the number of blocks per dimension (2^B,2^B,2^B)

For the `jl.ipynb`, upload as a new Notebook to google colab and follow the instructions inside

## Results
Google colab: Tesla P4
![res.png](https://raw.githubusercontent.com/KonstantinosChatziantoniou/JuliaCUDA_GridKNN/master/res.png)
