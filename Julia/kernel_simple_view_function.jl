using CUDA

@inline function CheckBlockBounds(offset, block, grids)
        x::Int32 = offset.x + block.x
        if x < 1 || x > grids
                return true
        end
        x = offset.y + block.y
        if x < 1 || x > grids
                return true
        end
        x = offset.z + block.z
        if x < 1 || x > grids
                return true
        end
        return false
end

@inline function CalcDistance(query, point)
    tempdist::Float32 = 0
    for d = 1:length(query)
        @inbounds tempdist += CUDA.pow(query[d] - point[d], 2)
    end
    tempdist = CUDA.sqrt(tempdist)
    return tempdist
end



function cuGridKnnSimpleViewFunction(devPoints, devQueries,
        Pointsperblock, Queriesperblock,
        IntPointsperblock, IntQueriesperblock,
        devDistances, devNeighbours,
        dimensions, grids, offset)

    if CheckBlockBounds(offset, blockIdx(), grids)
        return nothing
    end
    q_bid::Int32 = ((blockIdx().z - 1)*gridDim().y*gridDim().x
            + (blockIdx().y - 1)*gridDim().x
            + blockIdx().x)
    p_bid::Int32 = ((blockIdx().z + offset.z- 1)*gridDim().y*gridDim().x
            + (blockIdx().y + offset.y - 1)*gridDim().x
            + blockIdx().x + offset.x)
    tid::Int32 = threadIdx().x + (threadIdx().y-1)*blockDim().x
    stride::Int32 = (blockDim().x)*(blockDim().y)

    SharedPoints = @cuDynamicSharedMem(Float32, (dimensions, stride))
    SharedQueries = @cuDynamicSharedMem(Float32, (dimensions, stride), dimensions*stride*sizeof(Float32))

    @inbounds startPoints::Int32 = IntPointsperblock[p_bid]
    @inbounds startQueries::Int32 = IntQueriesperblock[q_bid]
    @inbounds totalPoints::Int32 = Pointsperblock[p_bid]
    @inbounds totalQueries::Int32 = Queriesperblock[q_bid]
    @inbounds Points = @view devPoints[(startPoints+1):(startPoints+totalPoints), :]
    @inbounds Queries = @view devQueries[(startQueries+1):(startQueries+totalQueries), :]
    @inbounds Distances = @view devDistances[(startQueries+1):(startQueries+totalQueries)]
    @inbounds Neighbours = @view devNeighbours[(startQueries+1):(startQueries+totalQueries)]
    @inbounds query = @view SharedQueries[:, tid]
    dist::Float32 = 0
    nb::Int32 = 0
    for q::Int32 in 0:stride:(totalQueries-1)
        if tid + q <= totalQueries
            for d = 1:dimensions
                @inbounds  query[d] = Queries[tid+q, d]
            end
            @inbounds dist = Distances[tid+q]
            @inbounds nb = Neighbours[tid+q]
        end
        for p::Int32 in 0:stride:(totalPoints-1)
            sync_threads()
            if tid + p <= totalPoints
                for d::Int32 in 1:dimensions
                    @inbounds SharedPoints[d, tid] = Points[tid+p, d]
                end
            end
            sync_threads()
            bounds::Int32 = CUDA.min(stride, totalPoints-p)
            for i::Int32 in 1:bounds
                @inbounds point = @view SharedPoints[:, (i+tid-2)%bounds+1]
                tempdist::Float32 = CalcDistance(query, point)
                if tempdist < dist
                    dist = tempdist
                    nb = startPoints + p + (i+tid-2)%bounds+1
                end
            end
        end
        if tid + q <= totalQueries
            @inbounds Distances[tid+q] = dist
            @inbounds Neighbours[tid+q] = nb
        end
    end
    return nothing
end


function cuda_knn_simple_view_function(OrderedPoints, OrderedQueries,
    PointsPerBlock, QueriesPerBlock,
    IntegralPointsPerBlock, IntegralQueriesPerBlock,
    numOfPoints, numOfQueries, numOfGrids, dimensions)
    
    ## Data transfer
    devPoints = CuArray(OrderedPoints)
    devQueries = CuArray(OrderedQueries)
    devPointsPerBlock = CuArray(PointsPerBlock)
    devQueriesPerBlock = CuArray(QueriesPerBlock)
    devIntegralPointsPerBlock = CuArray(IntegralPointsPerBlock)
    devIntegralQueriesPerBlock = CuArray(IntegralQueriesPerBlock)
    devRes = CUDA.fill(Float32(100), numOfQueries)
    devNeighbours = CuArray(zeros(Int32, numOfQueries))
    ## Config
    dimensions = Int32(dimensions)
    grids = Int32(numOfGrids)
    offset = (x=Int32(0),y=Int32(0),z=Int32(0))
    thread_groups = 2
    shmem=2*thread_groups*32*dimensions*sizeof(Float32)
    ## Running for main block.
    @cuda(blocks=(grids,grids,grids), threads=(32,thread_groups), shmem=shmem,
      cuGridKnnSimpleViewFunction(devPoints, devQueries,
        devPointsPerBlock, devQueriesPerBlock,
        devIntegralPointsPerBlock, devIntegralQueriesPerBlock,
        devRes, devNeighbours, dimensions, grids, offset))
    #Running for neighbour blocks.
    for x::Int32 = -1:1
        for y::Int32 = -1:1
            for z::Int32 = -1:1
                if x == 0 && y == 0 && z == 0
                    continue
                end
                offset = (x=x,y=y,z=z)
                @cuda(blocks=(grids,grids,grids), threads=(32,thread_groups), shmem=shmem,
                  cuGridKnnSimpleViewFunction(devPoints, devQueries,
                    devPointsPerBlock, devQueriesPerBlock,
                    devIntegralPointsPerBlock, devIntegralQueriesPerBlock,
                    devRes, devNeighbours, dimensions, grids, offset))
            end
        end
    end

    return Array(devNeighbours), Array(devRes)

end
