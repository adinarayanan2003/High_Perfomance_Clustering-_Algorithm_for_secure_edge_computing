import numpy as np
import multiprocessing as mp
import time
from numba import cuda

from math import sqrt

@cuda.jit
def cuda_update_clusters(data, centroids, assignments):
    start = cuda.grid(1)
    stride = cuda.gridsize(1)

    for i in range(start, data.shape[0], stride):
        min_distance = np.inf
        for j in range(centroids.shape[0]):
            distance = 0.0
            for d in range(data.shape[1]):
                diff = data[i, d] - centroids[j, d]
                distance += diff * diff
            distance = sqrt(distance)
            if distance < min_distance:
                min_distance = distance
                assignments[i] = j

def update_clusters_gpu(data, centroids, assignments):
    threads_per_block = 256
    blocks_per_grid = (data.shape[0] + (threads_per_block - 1)) // threads_per_block
    blocks_per_grid = 256
    cuda_update_clusters[blocks_per_grid, threads_per_block](data, np.array(centroids), assignments)

def update_clusters_cpu(data, centroids):
    distances = np.linalg.norm(centroids - data[:, np.newaxis, :], axis=2)
    new_assignment = np.argmin(distances, axis=1)
    return new_assignment

def calculate_sd(data, clusters):
    sd_list = []
    for cluster in clusters:
        if len(clusters[cluster]) > 1:
            sd = np.std(data[clusters[cluster]], axis=0)
            sd_list.append(np.mean(sd))
        else:
            sd_list.append(0)
    return sd_list

def kmeans_plusplus_initializer(data, k):
    centroids = [data[np.random.choice(len(data))]]

    for _ in range(1, k):
        distances = np.array([min([np.linalg.norm(c - point) for c in centroids]) for point in data])
        probabilities = distances / distances.sum()
        cumulative_probabilities = probabilities.cumsum()
        r = np.random.rand()
        for j, p in enumerate(cumulative_probabilities):
            if r < p:
                centroids.append(data[j])
                break

    return {i: [] for i in range(k)}, centroids

def update_clusters(idx, point, centroids):
    distances = np.linalg.norm(centroids - point, axis=1)
    new_assignment = np.argmin(distances)
    return idx, new_assignment

def adaptive_kmeans(data, k_max=10, tolerance=1e-4):
    n_samples, n_features = data.shape
    rmssd_history = []
    best_clusters = None
    best_rmssd = float('inf')

    pool = mp.Pool(mp.cpu_count())

    for k in range(1, k_max + 1):
        clusters, centroids = kmeans_plusplus_initializer(data, k)
        rmssd_prev = float('inf')
        assignments = np.zeros(n_samples, dtype=np.int32)

        while True:
            # GPU: Parallelized cluster assignment

            update_clusters_gpu(data, np.array(centroids), assignments)

            new_clusters = {i: [] for i in range(k)}
            points_changed = False  # Track if any points changed cluster assignment

            # CPU: Parallelized cluster update
            results = pool.starmap(update_clusters_cpu, [(data, np.array(centroids))])
            new_assignment = np.concatenate(results)
            
            for idx, new_assign in enumerate(new_assignment):
                if new_assign != assignments[idx]:
                    points_changed = True
                new_clusters[new_assign].append(idx)

            # Check for convergence
            rmssd = np.sqrt(np.nanmean([np.sum((data[cluster] - np.mean(data[cluster], axis=0))**2) for cluster in new_clusters.values()]))
            if not points_changed or abs(rmssd - rmssd_prev) < tolerance:
                break

            clusters = new_clusters
            centroids = [np.mean(data[cluster], axis=0) for cluster in clusters.values()]
            rmssd_prev = rmssd

        rmssd_history.append(rmssd)

        # Update best clusters if the current RMSSD is better
        if rmssd < best_rmssd:
            best_rmssd = rmssd
            best_clusters = clusters.copy()

    pool.close()
    pool.join()

    return best_clusters, rmssd_history

if __name__ == "__main__":
    np.random.seed(42)
    data = np.random.rand(20000, 2)
    start_time = time.time()
    best_clusters, rmssd_history = adaptive_kmeans(data)
    end_time = time.time()

    #print("Best Clusters:", best_clusters)
    print("RMSSD History:", rmssd_history)
    print("Time with Hybrid GPU-CPU:", end_time - start_time)