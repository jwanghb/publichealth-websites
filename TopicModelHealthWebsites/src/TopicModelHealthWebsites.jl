module TopicModelHealthWebsites
using Distributions, StatsBase, Dates, SpecialFunctions, JLD
export HALT_LDA
# Latent Dirichlet Allocation with hierarchical asymmetric prior + local topics (1 per website)
function HALT_LDA(; K::Integer, datapath::String, savepath::String, 
                n_iter = 100, n_burn = 0, n_lag = 5,
                c0alpha0_const = 10.0, beta = 0.05, gamma = 0.05, sigma0 = 5.0, sigmac = 0.05, a = 1.0, b = 1.0, c = 1.0, da = 1.0, db = 1.0)
    Ka = K + 1
    c0alpha0 = fill(c0alpha0_const, (Ka))
    data = load(datapath)
    words, V = data["words"], length(data["dictionary"])
    N = [[length(x) for x in y] for y in words]
    Mi = [length(x) for x in N]
    Mis = sum(Mi)
    M = length(Mi)
    topics, n, m0, m1, alpha = initialize(K, Ka, V, N, Mi, Mis, M, beta, gamma, c0alpha0, words)
    n_list, m0_list, m1_list, alpha_list, c0alpha0_list, c_list = sample_multiple(M, Mi, Mis, c, da, db, a, b, N, words, K, Ka, alpha, c0alpha0, gamma, beta, topics, V, n, m0, m1, sigma0, sigmac, n_iter, n_burn, n_lag)
    save(savepath, # MCMC samples
                    "n_list", n_list,
                    "m0_list", m0_list,
                    "m1_list", m1_list,
                    "alpha_list", alpha_list,
                    "c0alpha0_list", c0alpha0_list,
                    "c_list", c_list,
                    # Counts from data and number of topics
                    "K", K,
                    "Ka", Ka,
                    "V", V,
                    "M", M,
                    "Mi", Mi,
                    "N", N,
                    # Current state
                    "topics", topics,
                    "n", n,
                    "m0", m0,
                    "m1", m1,
                    "alpha", alpha,
                    "c0alpha0", c0alpha0,
                    # Fixed hyperparameters
                    "beta", beta,
                    "gamma", gamma,
                    "a", a,
                    "b", b,
                    "c", c,
                    "da", da,
                    "db", db,
                    # M-H
                    "sigma0", sigma0,
                    "sigmac", sigmac)
end
function initialize(K, Ka, V, N, Mi, Mis, M, beta, gamma, c0alpha0, words) # Initialize topic and counts
    # theta
    theta_init = fill(1/Ka, Ka)
    # Generate topics
    topics = [[rand(Categorical(theta_init), N[i][j]) for j in 1:Mi[i]] for i in 1:M]
    topics_nump = topics
    w_nump = words
    alpha = [rand(Dirichlet(c0alpha0)) for i in 1:M]
    # Page-topic counts
    m1 = [[counts(topics_nump[i][j], Ka) for j in 1:Mi[i]] for i in 1:M]
    # Website-topic-word counts
    topics_nump = [collect(Iterators.flatten(x)) for x in topics_nump]
    w_nump = [collect(Iterators.flatten(x)) for x in w_nump]
    m0 = [counts(w_nump[i][findall(x -> x == Ka, topics_nump[i])], V) for i in 1:M]
    # Global-topic-word counts
    topics_nump = collect(Iterators.flatten(topics_nump))
    w_nump = collect(Iterators.flatten(w_nump))
    n = [counts(w_nump[findall(x -> x == k, topics_nump)], V) for k in 1:K]
    # Add hyperparameters
    n = hcat(n...)' .+ beta
    m0 = hcat(m0...)' .+ gamma
    m1 = [hcat(m1[x]...)' .+ 0.0 for x in 1:M] # needs to update in sampling at each iteration
    return(topics, n, m0, m1, alpha)
end
function sample_antoniak(;N, calpha)
    if N > 0
        lambda = sum([rand(Bernoulli(calpha/(calpha+ind))) for ind in 0:N-1])
    else
        lambda = 0
    end
    return(lambda)
end
# likelihood ratio for c0alpha0
function lpiratio_c0alpha0(c0alpha0_new, c0alpha0, alpha, da, db, M, k)
    lpiratio = M*(loggamma(sum(c0alpha0_new)) - loggamma(c0alpha0_new[k]) - loggamma(sum(c0alpha0)) + loggamma(c0alpha0[k])) +
        sum(log.([x[k] for x in alpha])) * (c0alpha0_new[k] - c0alpha0[k]) +
        (da-1)*(log(c0alpha0_new[k]) - log(c0alpha0[k])) -
        db*(c0alpha0_new[k] - c0alpha0[k])
    return(lpiratio)
end
# likelihood ratio for c
function lpiratio_c(c_new, c, a, b, N, M, Mi, Mis, m1, alpha)
    lpiratio = (a-1)*(log(c_new)-log(c)) -
        b*(c_new-c) +
        Mis*(loggamma(c_new)-loggamma(c)) -
        sum([sum([loggamma(c_new + N[i][j]) - loggamma(c + N[i][j]) for j in 1:Mi[i]]) for i in 1:M]) +
        sum([sum([sum((loggamma.(c_new*alpha[i] .+ m1[i][j,:]) - loggamma.(c_new*alpha[i])) -
            (loggamma.(c*alpha[i] .+ m1[i][j,:]) - loggamma.(c*alpha[i]))) for j in 1:Mi[i]]) for i in 1:M])
    return(lpiratio)
end
function sample_multiple(M, Mi, Mis, c, da, db, a, b, N, words, K, Ka, alpha, c0alpha0, gamma, beta, topics, V, n, m0, m1, sigma0, sigmac,
                        n_iter = 100, n_burn = 0, n_lag = 5)
    alpha_list = []
    c0alpha0_list = []
    c_list = []
    n_list = []
    m0_list = []
    m1_list = []
    for iter in 1:n_iter
        keep = (iter % n_lag == 0) * (iter >= n_burn)
        # Sample topics for each word
        for i in 1:M
            for j in 1:Mi[i]
                for h in 1:N[i][j]
                    # Data + unknown categories
                    w = words[i][j][h]
                    z = topics[i][j][h]
                    # Remove word s,t,u from counts
                    m1[i][j,z] -= 1
                    if z < Ka
                        n[z,w] -= 1
                    else
                        m0[i,w] -= 1
                    end
                    # Sample levels and topics
                    u_1 = rand(Uniform(0,1))
                    # Calculate the level 1 sampling probability
                    part_1 = n[:,w] ./ sum(n, dims = 2) .* (m1[i][j,1:K] .+ c*alpha[i][1:K])
                    # Calculate the level 0 sampling probability
                    part_0 = m0[i,w] / sum(m0[i,:]) * (m1[i][j,Ka] + c*alpha[i][Ka])
                    sum_p = sum(part_1) + part_0
                    # Draw topic
                    p_cum = part_0
                    if u_1 > p_cum/sum_p
                        for k = 1:K
                            p_cum += part_1[k]
                            if u_1 <= p_cum/sum_p
                                topics[i][j][h] = k
                                n[k,w] += 1
                                m1[i][j,k] += 1
                                break
                            end
                        end
                    else
                        topics[i][j][h] = Ka
                        m0[i,w] += 1
                        m1[i][j,Ka] += 1
                    end
                end
            end
        end
        # Sample site means
        for i in 1:M
            lambdas = sum([[sample_antoniak(N = m1[i][j,k], calpha = c*alpha[i][k]) for k in 1:Ka] for j in 1:Mi[i]])
            alpha[i,] = rand(Dirichlet(c0alpha0 .+ lambdas))
        end
        # Sample global means (univariate proposals)
        for k in 1:Ka
            c0alpha0_k = c0alpha0[k]
            c0alpha0_k_new = rand(truncated(Normal(c0alpha0_k, sigma0), 0, c0alpha0_k*2))
            c0alpha0_new = copy(c0alpha0)
            c0alpha0_new[k] = c0alpha0_k_new
            diff = lpiratio_c0alpha0(c0alpha0_new, c0alpha0, alpha, da, db, M, k) + logpdf(truncated(Normal(c0alpha0_k_new, sigma0), 0, c0alpha0_k_new*2), c0alpha0_k) - logpdf(truncated(Normal(c0alpha0_k, sigma0), 0, c0alpha0_k*2), c0alpha0_k_new)
            ar = exp(diff)
            if ar >= rand(Uniform(0,1))
                c0alpha0[k] = c0alpha0_k_new
            end
        end
        # Sample c
        c_new = rand(truncated(Normal(c, sigmac), 0, c*2))
        ar = exp(lpiratio_c(c_new, c, a, b, N, M, Mi, Mis, m1, alpha) + logpdf(truncated(Normal(c_new, sigmac), 0, c_new*2), c) - logpdf(truncated(Normal(c, sigmac), 0, c*2), c_new))
        if ar >= rand(Uniform(0,1))
            c = c_new
        end
        # Save
        if keep == true
            push!(alpha_list, copy(alpha))
            push!(c_list, copy(c))
            push!(c0alpha0_list, copy(c0alpha0))
            push!(n_list, copy(n))
            push!(m0_list, copy(m0))
            push!(m1_list, copy([copy(x) for x in m1]))
            print(Dates.format(now(), "HH:MM:SS") * ": Iteration: " * string(iter) * "\n")
        end
    end
    return([n_list, m0_list, m1_list, alpha_list, c0alpha0_list, c_list])
end
end # module
