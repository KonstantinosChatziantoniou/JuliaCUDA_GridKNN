using CUDAdrv, CUDAnative, CuArrays

## Kernel

function cuGridKnn(points, queries,
        pointsperblock, queriesperblock,
        intpointsperblock, intqueriesperblock,
        dists, nbs,
        d, grids)

        bid = ((blockIdx().z- 1)*gridDim().y*gridDim().x
                + (blockIdx().y - 1)*gridDim().x
                + blockIdx().x)
        SharedPoints = @cuStaticSharedMem(Float32, (64, 3))
        SharedQueries = @cuStaticSharedMem(Float32, (64, 3))
        p_bid = bid
        q_bid = bid
        tid = threadIdx().x
        query = @view SharedQueries[tid,:]
        stride = blockDim().x
        StartPoints = intpointsperblock[p_bid]
        StartQueries = intqueriesperblock[q_bid]
        TotalPoints = pointsperblock[p_bid]
        TotalQueries = queriesperblock[q_bid]
        dist = Float32(50.0)
        nb = 0
        for i = 0:stride:(TotalQueries - 1)
                qIndex = StartQueries + i + tid
                if tid + i <= TotalQueries
                        for j = 1:d
                                query[j] = queries[qIndex, j]
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
                        if   tid + i <= TotalQueries
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


                end

                if tid + i <= TotalQueries
                        dists[qIndex] = dist
                        nbs[qIndex] = nb

                end
        end

        return nothing
end


function ExtracuGridKnn(points, queries,
        pointsperblock, queriesperblock,
        intpointsperblock, intqueriesperblock,
        dists, nbs,
        d, grids)


        q_bid = ((blockIdx().z- 1)*gridDim().y*gridDim().x
                + (blockIdx().y - 1)*gridDim().x
                + blockIdx().x)
        SharedPoints = @cuStaticSharedMem(Float32, (64, 3))
        SharedQueries = @cuStaticSharedMem(Float32, (64, 3))
        SharedSkip = @cuStaticSharedMem(Bool, 2)
        tid = threadIdx().x + (threadIdx().y-1)*blockDim().x
        query = @view SharedQueries[tid, :]
        stride = (blockDim().x)*(blockDim().y)
        StartQueries = intqueriesperblock[q_bid]
        TotalQueries = queriesperblock[q_bid]
        dist = Float32(50.0)
        nb = 0
        ## For each query in main block
        for i = 0:stride:(TotalQueries - 1)
            qIndex = StartQueries + i + tid
            ## Read query to memory
            if tid + i <= TotalQueries
                for j = 1:d
                    @inbounds query[j] = queries[qIndex, j]
                end
                dist = dists[qIndex]
                nb = nbs[qIndex]
            end
            g = grids
            ## Scan all neighbouring blocks
            for x = -1:1
                ## Check if block is in the grid
                if blockIdx().x + x < 1 || blockIdx().x + x > g
                    continue
                end

                for y = -1:1
                    if blockIdx().y + y < 1 || blockIdx().y + y > g
                        continue
                    end

                    for z = -1:1
                        if blockIdx().z + z < 1 || blockIdx().z + z > g
                            continue
                        end
                        ## Check if it is the main block
                        if x == 0 && y == 0 && z == 0
                            continue
                        end


                        p_bid = ((blockIdx().z + z - 1)*gridDim().y*gridDim().x
                            + (blockIdx().y + y - 1)*gridDim().x
                            + blockIdx().x + x)
                        StartPoints = intpointsperblock[p_bid]
                        TotalPoints = pointsperblock[p_bid]
                        for j = 0:stride:(TotalPoints - 1)
                            pIndex = StartPoints + j + tid
                            sync_threads()
                            if tid + j <= TotalPoints
                                for k = 1:d
                                    @inbounds SharedPoints[tid, k] = points[pIndex, k]
                                end
                            end
                            sync_threads()

                                if   tid + i <= TotalQueries
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

                        end
                        if tid + i <= TotalQueries
                            dists[qIndex] = dist
                            nbs[qIndex] = nb
                        end

                    end
                end
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
        # shmem=32*dimensions*sizeof(Float32)
        println("running for main block",numOfQueries, " ", numOfGrids, " (sk)")
        @cuda blocks=tuple(grids...) threads=(64)  cuGridKnn(devPoints, devQueries,
            devPointsPerBlock, devQueriesPerBlock,
            devIntegralPointsPerBlock, devIntegralQueriesPerBlock,
            devRes, devNeighbours, dimensions, numOfGrids)
        println("\tStarting extra kernel")
        @cuda blocks=tuple(grids...) threads=(32,2)  ExtracuGridKnn(devPoints, devQueries,
            devPointsPerBlock, devQueriesPerBlock,
            devIntegralPointsPerBlock, devIntegralQueriesPerBlock,
            devRes, devNeighbours, dimensions, numOfGrids)

        return Array(devNeighbours), Array(devRes)

end
