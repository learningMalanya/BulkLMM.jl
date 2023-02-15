module BulkLMM

    # dependent packages 
    using CSV, DelimitedFiles, DataFrames, Missings 
    using LinearAlgebra, Statistics, Optim
    using Random, Distributions, LoopVectorization
    
    include("./util.jl");
    include("./kinship.jl");
    export calcKinship

    include("./readData.jl");
    export readBXDpheno
    export readGenoProb_ExcludeComplements

    # code for (wls) weighted least squares
    include("./wls.jl");
    export wls
    export wls_multivar # multivariate version of WLS
    export LSEstimates
    export LSEstimates_multivar # multivariate version of WLS results

    # code for rorateData and flmm
    include("./lmm.jl");
    # data type we are exporting
    export LMMEstimates

    include("./scan.jl");
    export scan
    # export scan_perms
    export scan_perms_lite

    include("./bulkscan_helpers.jl");

    include("./bulkscan.jl");
    export bulkscan_null, bulkscan_null_grid, bulkscan_alt_grid

    include("./transform_helpers.jl");
    # export transform_rotation

end # module
