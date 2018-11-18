class Parameter:
    model_count = 0
    #DOC2VEC
    vector_size = [50, 75, 100, 125, 150]
    epoch = [10, 20, 30, 40]
    # alpha = [0.01, 0.05, 0.1, 0.15, 0.2]

    #Kmeans
    K = [10, 30, 50, 70, 100]

    #Birch
    threshold = [0.1, 0.3, 0.5, 0.7, 1]
    branching_factor = [50, 100, 150, 200]

    #CURE
    represent_point = [2, 5, 8, 10]

    #DBSCAN
    eps = [0.1, 0.3, 0.5, 0.7, 1]
    min_samples = [2, 5, 8, 10]
