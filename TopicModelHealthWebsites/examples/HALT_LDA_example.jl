# import TopicModelHealthWebsites
include("/publichealth-websites/TopicModelHealthWebsites/src/TopicModelHealthWebsites.jl") # update path
TopicModelHealthWebsites.HALT_LDA(K = 10, datapath = "/data/dataanddict.jld", savepath = "/data/test.jld") # update path
