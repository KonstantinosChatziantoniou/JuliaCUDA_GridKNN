using NearestNeighbors
using Random
include("Code/preprocess.jl")
## Change seed
Random.seed!(10)

## Modify function to inspect errors
function CompareRes(l, b)
    numOfPoints = l
    numOfQueries = l
    dimensions = 3
    numOfGrids = b #PerDimension

    Points = rand(Float32, numOfPoints ,dimensions)
    Queries = rand(Float32, numOfQueries, dimensions)

    BlockOfPoint = AssignPointsToBlock(Points, numOfGrids, dimensions)
    BlockOfQuery = AssignPointsToBlock(Queries, numOfGrids, dimensions)

    PointsPerBlock, IntegralPointsPerBlock = CountPointsPerBlock(Points, numOfGrids, dimensions)
    QueriesPerBlock, IntegralQueriesPerBlock = CountPointsPerBlock(Queries, numOfGrids, dimensions)

    OrderedPoints = ReorderPointsByBlock(Points, BlockOfPoint)
    OrderedQueries = ReorderPointsByBlock(Queries, BlockOfQuery)
    gpu_idxs, gpu_dists = cuda_knn(OrderedPoints, OrderedQueries,PointsPerBlock,
        QueriesPerBlock, IntegralPointsPerBlock, IntegralQueriesPerBlock,numOfPoints,
        numOfQueries, numOfGrids, dimensions)

    tree = NearestNeighbors.KDTree(OrderedPoints')
    cpu_idxs, cpu_dists = NearestNeighbors.knn(tree,OrderedQueries', 1)
    cpu_dists = map(x -> x[1], cpu_dists)
    cpu_idxs = map(x -> x[1], cpu_idxs)
    wrong_idxs = findall(gpu_idxs .!= cpu_idxs)
    println("Number of different neighbours: ", length(wrong_idxs))
    if length(wrong_idxs) == 0
        return
    end
    ch_bl = AssignPointsToBlock(OrderedPoints[gpu_idxs[1:100], :],numOfGrids, 3)
    chq_bl = AssignPointsToBlock(OrderedQueries[1:100, :],numOfGrids, 3)
    # for i = 1:100
    #     println("Q: ", chq_bl[i], " P: ", ch_bl[i])
    # end
    w_points = OrderedPoints[gpu_idxs[wrong_idxs], :]
    c_points = OrderedPoints[cpu_idxs[wrong_idxs], :]
    w_queries = OrderedQueries[wrong_idxs, :]
    w_ppblock = AssignPointsToBlock(w_points, numOfGrids, 3)
    w_qpblock = AssignPointsToBlock(w_queries, numOfGrids, 3)
    c_ppblock = AssignPointsToBlock(c_points, numOfGrids, 3)
    w_dists = gpu_dists[wrong_idxs]
    c_dists = cpu_dists[wrong_idxs]
    w_id = gpu_idxs[wrong_idxs]
    c_id = cpu_idxs[wrong_idxs]
    # ----------------------------------
    #   DO ANY OTHER CALCULATION HERE
    # ----------------------------------
    actual_dists = mapslices(x-> sqrt(sum(x.^2)), w_points.-w_queries, dims=2)
    ccc_dists = mapslices(x-> sqrt(sum(x.^2)), c_points.-w_queries, dims=2)

end

## Change Problem size and implementetion
l = 1<<20
b = 1<<4
kernel_files = ["multi_kernel", "multi_kernel_check", "single_kernel", "single_kernel_check"]
include(string("Code/", kernel_files[2],".jl"))
CompareRes(l,b)
