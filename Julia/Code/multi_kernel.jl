using CUDAdrv, CUDAnative, CuArrays
using StaticArrays

## Kernel

@inline function CheckBlockBounds(offset, block, grids)
        x = offset.x + block.x
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
function cuGridKnn(points, queries,
        pointsperblock, queriesperblock,
        intpointsperblock, intqueriesperblock,
        dists, nbs,
        d, grids, offset)
        if CheckBlockBounds(offset, blockIdx(), grids)
                return nothing
        end
        q_bid = ((blockIdx().z - 1)*gridDim().y*gridDim().x
                + (blockIdx().y - 1)*gridDim().x
                + blockIdx().x)
        p_bid = ((blockIdx().z + offset.z- 1)*gridDim().y*gridDim().x
                + (blockIdx().y + offset.y - 1)*gridDim().x
                + blockIdx().x + offset.x)
        tid = threadIdx().x + (threadIdx().y-1)*blockDim().x
        stride = (blockDim().x)*(blockDim().y)
        StartPoints = intpointsperblock[p_bid]
        StartQueries = intqueriesperblock[q_bid]
        TotalPoints = pointsperblock[p_bid]
        TotalQueries = queriesperblock[q_bid]
        SharedPoints = @cuStaticSharedMem(Float32, (64, 3))
        SharedQueries = @cuStaticSharedMem(Float32, (64, 3))
        query = @view SharedQueries[tid, :]
        SharedSkip = @cuStaticSharedMem(Bool, 2)
        dist = Float32(50.0)
        nb = 0
        for i = 0:stride:(TotalQueries - 1)
                qIndex = StartQueries + i + tid
                distCheck = true
                if tid + i <= TotalQueries
                        for j = 1:d
                                @inbounds query[j] = queries[qIndex, j]
                        end
                        dist = dists[qIndex]
                        nb = nbs[qIndex]
                end
                for j = 0:stride:(TotalPoints - 1)
                        pIndex = StartPoints + j + tid
                        sync_threads()
                        if tid + j <= TotalPoints
                                for k = 1:d
                                        @inbounds SharedPoints[tid, k] = points[pIndex, k]
                                        #dist = points[pIndex, k]
                                end
                        end
                        sync_threads()

                        bounds = CUDAnative.min(stride, TotalPoints-j)
                        for p = 1:bounds
                                tempdist = Float32(0)
                                for dm = 1:d
                                        tempdist += CUDAnative.pow(query[dm]-SharedPoints[p,dm],2)
                                end
                                tempdist = CUDAnative.sqrt(tempdist)
                                if tempdist < dist
                                        dist = tempdist
                                        nb = StartPoints + j + p
                                end
                        end


                end

                if tid + i <= TotalQueries
                        dists[qIndex] = dist
                        nbs[qIndex] = nb

                end
        end
        return nothing
end


function cuda_knn(OrderedPoints, OrderedQueries,
        PointsPerBlock, QueriesPerBlock,
        IntegralPointsPerBlock, IntegralQueriesPerBlock,
        numOfPoints, numOfQueries, numOfGrids, dimensions)

        devPoints = CuArray(OrderedPoints)
        devQueries = CuArray(OrderedQueries)
        devPointsPerBlock = CuArray(PointsPerBlock)
        devQueriesPerBlock = CuArray(QueriesPerBlock)
        devIntegralPointsPerBlock = CuArray(IntegralPointsPerBlock)
        devIntegralQueriesPerBlock = CuArray(IntegralQueriesPerBlock)
        devRes = CuArrays.fill(Float32(100), numOfQueries)
        devNeighbours = CuArray(zeros(Int64, numOfQueries))

        grids = [numOfGrids for i = 1:dimensions];
        offset = (x=0,y=0,z=0)
        println("running for main block",numOfQueries, " ", numOfGrids, " (mk)")
        # shmem=32*dimensions*sizeof(Float32)
        @cuda blocks=tuple(grids...) threads=(32,2)  cuGridKnn(devPoints, devQueries,
            devPointsPerBlock, devQueriesPerBlock,
            devIntegralPointsPerBlock, devIntegralQueriesPerBlock,
            devRes, devNeighbours, dimensions, numOfGrids, offset)
        println("\trunning for nb blocks")
        for x = -1:1
                for y = -1:1
                        for z = -1:1
                        #z = 0
                                if x == 0 && y == 0 && z == 0
                                        continue
                                end
                                offset = (x=x,y=y,z=z)
                                # println("\t running for offset ", offset)
                                # println("running for " , offset)
                                # shmem=32*dimensions*sizeof(Float32)
                                @cuda blocks=tuple(grids...) threads=(32,2)  cuGridKnn(devPoints, devQueries,
                                    devPointsPerBlock, devQueriesPerBlock,
                                    devIntegralPointsPerBlock, devIntegralQueriesPerBlock,
                                    devRes, devNeighbours, dimensions, numOfGrids, offset)



                        end
                end
        end

        return Array(devNeighbours), Array(devRes)

end
