
function AssignPointsToBlock(Points::Array{Float32,2},
    numOfGrids::Int64, dimensions::Int64)
    grids = [numOfGrids for i = 1:dimensions];
    gridMap1D = grids.^((0:(dimensions-1)));
    gridMap1Doffset = [0; -1*(2:dimensions).^0];
    BlockOfPoint = dropdims(
        mapslices(
        x -> Int64(sum( ((1 .+ div.(min.(x, 0.99999999), 1/numOfGrids)) + gridMap1Doffset ) .*gridMap1D )),
            Points, dims=2),
        dims=2);
    return BlockOfPoint
end


function CountPointsPerBlock(Points::Array{Float32,2},
    numOfGrids::Int64, dimensions::Int64)
    grids = [numOfGrids for i = 1:dimensions];
    gridMap1D = grids.^((0:(dimensions-1)));
    gridMap1Doffset = [0; -1*(2:dimensions).^0];
    PointsPerBlock = zeros(Int64, numOfGrids^dimensions)
    num = size(Points)[1]
    for i = 1:num
        x = Points[i,:]
        x = min.(x, 0.99999999)
        gid = Int64(sum(( (div.(x,1/numOfGrids) .+ 1) + gridMap1Doffset ) .*gridMap1D ))
        PointsPerBlock[gid...] += 1
    end
    IntegralPointsPerBlock = similar(PointsPerBlock)
    IntegralPointsPerBlock[1] = 0#PointsPerBlock[0]
    for i = 2:numOfGrids^dimensions
        IntegralPointsPerBlock[i] = PointsPerBlock[i-1] + IntegralPointsPerBlock[i-1]
    end
    return PointsPerBlock, IntegralPointsPerBlock
end


function ReorderPointsByBlock(Points::Array{Float32,2},
    BlockOfPoint::Array{Int64,1})
    BlockPermutationP = sortperm(BlockOfPoint);
    OrderedPoints = Points[BlockPermutationP,:];
    return OrderedPoints
end
