using NearestNeighbors
include("./preprocess.jl")

dimensions = 3
numOfPoints = 1<<19
numOfQueries = 1<<19
Points = rand(Float32, numOfPoints ,dimensions)
Queries = rand(Float32, numOfQueries, dimensions)

numOfGrids = 1<<4
BlockOfPoint = AssignPointsToBlock(Points, numOfGrids, dimensions)
BlockOfQuery = AssignPointsToBlock(Queries, numOfGrids, dimensions)

PointsPerBlock, IntegralPointsPerBlock = CountPointsPerBlock(Points, numOfGrids, dimensions)
QueriesPerBlock, IntegralQueriesPerBlock = CountPointsPerBlock(Queries, numOfGrids, dimensions)

OrderedPoints = ReorderPointsByBlock(Points, BlockOfPoint)
OrderedQueries = ReorderPointsByBlock(Queries, BlockOfQuery)

############################################################
# Uncomment only the implementation you want to validate.  #
############################################################
include("kernel_simple.jl")

println("Running for simple version: ")
gpu_idxs, gpu_dists = cuda_knn_simple(OrderedPoints, OrderedQueries,PointsPerBlock,
    QueriesPerBlock, IntegralPointsPerBlock, IntegralQueriesPerBlock,numOfPoints,
    numOfQueries, numOfGrids, dimensions)
##############################################
# include("kernel_simple_view_function.jl")
#
# println("Running for simple version with @view and distance calcs in function: ")
# gpu_idxs, gpu_dists = cuda_knn_simple_view_function(OrderedPoints, OrderedQueries,PointsPerBlock,
#     QueriesPerBlock, IntegralPointsPerBlock, IntegralQueriesPerBlock,numOfPoints,
#     numOfQueries, numOfGrids, dimensions)
###############################################
# include("kernel_with_skip.jl")
#
# println("Running for version with skip: ")
# gpu_idxs, gpu_dists = cuda_knn_with_skip(OrderedPoints, OrderedQueries,PointsPerBlock,
#     QueriesPerBlock, IntegralPointsPerBlock, IntegralQueriesPerBlock,numOfPoints,
#     numOfQueries, numOfGrids, dimensions)
###############################################
# include("kernel_with_skip_view_function.jl")
# println("Running for version with skip and @view and distance calcs in function: ")
# gpu_idxs, gpu_dists = cuda_knn_with_skip_view_function(OrderedPoints, OrderedQueries,PointsPerBlock,
#     QueriesPerBlock, IntegralPointsPerBlock, IntegralQueriesPerBlock,numOfPoints,
#     numOfQueries, numOfGrids, dimensions)






tr = NearestNeighbors.KDTree(OrderedPoints')
cpu_idxs, cpu_dists = NearestNeighbors.knn(tr, OrderedQueries', 1)

cpu_idxs = vcat(cpu_idxs...)

for i in 1:length(cpu_idxs)
    res = gpu_idxs[i]
    if cpu_idxs[i] != res
        println(i , " : " ,cpu_idxs[i], "     ", res)
        println("\tblock pt", BlockOfPoint[cpu_idxs[i]], " wrong pt ", BlockOfPoint[ res] ," block query: ", BlockOfQuery[i])
        end
end
