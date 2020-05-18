include("./preprocess.jl")
using BenchmarkTools, Statistics
using Random
## Modify for different problem size
inpGrid = 4
numOfPoints = 1<<19
numOfQueries = 1<<19
Points = 0
Queries = 0
dimensions = 3
if length(ARGS) == 1
    println("Running for .csv files")
    global inpGrid
    global numOfPoints
    global numOfQueries
    global Points
    global Queries
    global dimensions
    inpGrid = parse(Int64, ARGS[1])
    Points = ReadCSV("./pts.csv")
    Queries = ReadCSV("./qrs.csv")
    numOfPoints = size(Points)[1]
    numOfQueries = size(Queries)[1]
    dimensions = size(Points)[2]
elseif length(ARGS) == 3
    println("Running for random generated dataset")
    global rseed
    global numOfPoints
    global numOfQueries
    global inpGrid
    global Points
    global Queries
    global dimensions
    Random.seed!(parse(Int64, ARGS[3]))
    numOfPoints = 1<< parse(Int64, ARGS[1])
    numOfQueries = 1<<parse(Int64, ARGS[1])
    inpGrid = parse(Int64, ARGS[2])
    Points = rand(Float32, numOfPoints ,dimensions)
    Queries = rand(Float32, numOfQueries, dimensions)

end

numOfGrids = 1<<inpGrid


BlockOfPoint = AssignPointsToBlock(Points, numOfGrids, dimensions)
BlockOfQuery = AssignPointsToBlock(Queries, numOfGrids, dimensions)

PointsPerBlock, IntegralPointsPerBlock = CountPointsPerBlock(Points, numOfGrids, dimensions)
QueriesPerBlock, IntegralQueriesPerBlock = CountPointsPerBlock(Queries, numOfGrids, dimensions)

OrderedPoints = ReorderPointsByBlock(Points, BlockOfPoint)
OrderedQueries = ReorderPointsByBlock(Queries, BlockOfQuery)
println("Num pts: ", numOfPoints, " Gridsize: ", numOfGrids)

############################################################
#   Uncomment the implementations you want to benchmark.   #
############################################################
include("kernel_simple.jl")

println("Running for simple version: ")
gpu_idxs, gpu_dists = cuda_knn_simple(OrderedPoints, OrderedQueries,PointsPerBlock,
    QueriesPerBlock, IntegralPointsPerBlock, IntegralQueriesPerBlock,numOfPoints,
    numOfQueries, numOfGrids, dimensions)
#
println(gpu_idxs[1:10])
println(gpu_dists[1:10])
##############################################
include("kernel_simple_view_function.jl")

println("Running for simple version with @view and distance calcs in function: ")
gpu_idxs, gpu_dists = cuda_knn_simple_view_function(OrderedPoints, OrderedQueries,PointsPerBlock,
    QueriesPerBlock, IntegralPointsPerBlock, IntegralQueriesPerBlock,numOfPoints,
    numOfQueries, numOfGrids, dimensions)

println(gpu_idxs[1:10])
println(gpu_dists[1:10])
###############################################
include("kernel_with_skip.jl")

println("Running for version with skip: ")
gpu_idxs, gpu_dists = cuda_knn_with_skip(OrderedPoints, OrderedQueries,PointsPerBlock,
    QueriesPerBlock, IntegralPointsPerBlock, IntegralQueriesPerBlock,numOfPoints,
    numOfQueries, numOfGrids, dimensions)

println(gpu_idxs[1:10])
println(gpu_dists[1:10])
###############################################
include("kernel_with_skip_view_function.jl")
println("Running for version with skip and @view and distance calcs in function: ")
gpu_idxs, gpu_dists = cuda_knn_with_skip_view_function(OrderedPoints, OrderedQueries,PointsPerBlock,
    QueriesPerBlock, IntegralPointsPerBlock, IntegralQueriesPerBlock,numOfPoints,
    numOfQueries, numOfGrids, dimensions)

println(gpu_idxs[1:10])
println(gpu_dists[1:10])
