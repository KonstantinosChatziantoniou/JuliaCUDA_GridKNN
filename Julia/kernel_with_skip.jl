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
@inline function BoundsDistance(offset, query, dist)

    dx::Float32 = (offset.x == 1)  ? (1/grids - mod(query[1], 1/grids)) : 0
    dx = (offset.x == -1) ? (mod(query[1], 1/grids)) : dx
    dx = CUDA.pow(dx,Int32(2))
    dy::Float32 = (offset.y == 1)  ? (1/grids - mod(query[2], 1/grids)) : 0
    dy = (offset.y == -1) ? (mod(query[2], 1/grids)) : dy
    dy = CUDA.pow(dy,Int32(2))
    dz::Float32 = (offset.z == 1)  ? (1/grids - mod(query[3], 1/grids)) : 0
    dz = (offset.z == -1) ? (mod(query[3], 1/grids)) : dz
    dz = CUDA.pow(dz,Int32(2))
    boundsDist::Float32 = CUDA.sqrt(dx+dy+dz)
    local_skip = boundsDist > dist

    return local_skip

end
function cuGridKnnWithSkip(Points, Queries,
        Pointsperblock, Queriesperblock,
        IntPointsperblock, IntQueriesperblock,
        Distances, Neighbours,
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
    SharedQueries = @cuDynamicSharedMem(Float32, (dimensions, stride),
                                dimensions*stride*sizeof(Float32))
    SharedSkip = @cuDynamicSharedMem(Bool, (blockDim().y),
                                2*dimensions*stride*sizeof(Float32))
    query = @view SharedQueries[:, tid]

    startPoints::Int32 = 0
    (threadIdx().x == 1)  && (@inbounds  startPoints = IntPointsperblock[p_bid])
    startPoints = shfl_sync(0xffffffff, startPoints, 1);

    startQueries::Int32 = 0
    (threadIdx().x == 1) && (@inbounds startQueries = IntQueriesperblock[q_bid])
    startQueries = shfl_sync(0xffffffff, startQueries, 1);

    totalPoints::Int32 = 0
    (threadIdx().x == 1) && (@inbounds totalPoints = Pointsperblock[p_bid])
    totalPoints = shfl_sync(0xffffffff, totalPoints, 1);

    totalQueries::Int32 = 0
    (threadIdx().x == 1)  && (@inbounds totalQueries = Queriesperblock[q_bid])
    totalQueries = shfl_sync(0xffffffff, totalQueries, 1);

    dist::Float32 = 0
    nb::Int32 = 0
    local_skip::Bool = true
    global_skip::Bool = true
    for q::Int32 = 0:stride:(totalQueries - 1)
        ## Read queries
        qIndex::Int32 = startQueries + q + tid
        if tid + q <= totalQueries
                for d::Int32 = 1:dimensions
                        @inbounds query[d] = Queries[qIndex, d]
                end
                @inbounds dist = Distances[qIndex]
                @inbounds nb = Neighbours[qIndex]

                dx::Float32 = (offset.x == 1)  ? (1/grids - mod(query[1], 1/grids)) : 0
                dx = (offset.x == -1) ? (mod(query[1], 1/grids)) : dx
                dx = CUDA.pow(dx,Int32(2))
                dy::Float32 = (offset.y == 1)  ? (1/grids - mod(query[2], 1/grids)) : 0
                dy = (offset.y == -1) ? (mod(query[2], 1/grids)) : dy
                dy = CUDA.pow(dy,Int32(2))
                dz::Float32 = (offset.z == 1)  ? (1/grids - mod(query[3], 1/grids)) : 0
                dz = (offset.z == -1) ? (mod(query[3], 1/grids)) : dz
                dz = CUDA.pow(dz,Int32(2))
                boundsDist::Float32 = CUDA.sqrt(dx+dy+dz)
                local_skip = boundsDist > dist
        end

        local_skip = vote_all(local_skip)
        (threadIdx().x==1) && (SharedSkip[threadIdx().y] = local_skip)
        sync_threads()
        if threadIdx().x==1
            for w = 1:blockDim().y
                global_skip = global_skip && SharedSkip[w]
            end
        end
        global_skip = shfl_sync(FULL_MASK, global_skip, 1)
        global_skip && continue
        for p::Int32 = 0:stride:(totalPoints - 1)
            ## Read points
            sync_threads()
            if tid + p <= totalPoints
                for d::Int32 = 1:dimensions
                    @inbounds SharedPoints[d, tid] = Points[startPoints + p + tid, d]
                end
            end
            sync_threads()
            ## Calculate and compare distance
            local_skip && continue
            bounds::Int32 = CUDA.min(stride, totalPoints-p)
            for i::Int32 = 1:bounds
                tempdist::Float32 = 0
                for d::Int32 = 1:dimensions
                    @inbounds tempdist += CUDA.pow(query[d]
                                        -SharedPoints[d, (i+tid-2)%bounds+1],2)
                end
                tempdist = CUDA.sqrt(tempdist)
                if tempdist < dist
                    dist = tempdist
                    nb = startPoints + p + (i+tid-2)%bounds+1
                end
            end
        end
        if tid + q <= totalQueries
            @inbounds Distances[qIndex] = dist
            @inbounds Neighbours[qIndex] = nb
        end
    end
    return nothing
end


function cuda_knn_with_skip(OrderedPoints, OrderedQueries,
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
    shmem=2*thread_groups*32*dimensions*sizeof(Float32) + thread_groups*sizeof(Int32)
    ## Running for main block.
    @cuda(blocks=(grids,grids,grids), threads=(32,thread_groups), shmem=shmem,
      cuGridKnnWithSkip(devPoints, devQueries,
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
                  cuGridKnnWithSkip(devPoints, devQueries,
                    devPointsPerBlock, devQueriesPerBlock,
                    devIntegralPointsPerBlock, devIntegralQueriesPerBlock,
                    devRes, devNeighbours, dimensions, grids, offset))
            end
        end
    end

    return Array(devNeighbours), Array(devRes)

end
