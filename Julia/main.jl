include("Code/preprocess.jl")
using BenchmarkTools, Statistics
using NearestNeighbors
using Random
## Modify for different problem size
Random.seed!(10)
numOfPoints = 1<<19
numOfQueries = 1<<19
dimensions = 3
numOfGrids = 1<<4 #PerDimension

## Dont modify
Points = rand(Float32, numOfPoints ,dimensions)
Queries = rand(Float32, numOfQueries, dimensions)

BlockOfPoint = AssignPointsToBlock(Points, numOfGrids, dimensions)
BlockOfQuery = AssignPointsToBlock(Queries, numOfGrids, dimensions)

PointsPerBlock, IntegralPointsPerBlock = CountPointsPerBlock(Points, numOfGrids, dimensions)
QueriesPerBlock, IntegralQueriesPerBlock = CountPointsPerBlock(Queries, numOfGrids, dimensions)

OrderedPoints = ReorderPointsByBlock(Points, BlockOfPoint)
OrderedQueries = ReorderPointsByBlock(Queries, BlockOfQuery)

## Change the kernel implementetion
include("Code/multi_kernel_check.jl")

## Modify code to add timing or benchmarking
gpu_idxs, gpu_dists = cuda_knn(OrderedPoints, OrderedQueries,PointsPerBlock,
    QueriesPerBlock, IntegralPointsPerBlock, IntegralQueriesPerBlock,numOfPoints,
    numOfQueries, numOfGrids, dimensions)

## Modify code to interact with the result
println(gpu_idxs[1:10])
println(gpu_dists[1:10])
