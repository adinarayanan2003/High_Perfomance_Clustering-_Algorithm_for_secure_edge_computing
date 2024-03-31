import numpy as np
import time

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

def adaptive_kmeans(data, k_max=10, tolerance=1e-4):
    n_samples, n_features = data.shape
    rmssd_history = []
    best_clusters = None
    best_rmssd = float('inf')

    for k in range(1, k_max + 1):
        clusters, centroids = kmeans_plusplus_initializer(data, k)
        rmssd_prev = float('inf')

        while True:
            new_clusters = {i: [] for i in range(k)}
            points_changed = False  # Track if any points changed cluster assignment

            for idx, point in enumerate(data):
                distances = np.linalg.norm(centroids - point, axis=1)
                new_assignment = np.argmin(distances)
                
                if new_assignment != idx:
                    points_changed = True

                new_clusters[new_assignment].append(idx)

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

    return best_clusters, rmssd_history

# Example usage:
np.random.seed(42)
data = np.random.rand(20000, 2)
# print("Original Data:\n", data)

start_time = time.time()
best_clusters, rmssd_history = adaptive_kmeans(data)
end_time = time.time()

#print("Best Clusters:", best_clusters)
print("RMSSD History:", rmssd_history)
print("Time without parallelization:", end_time - start_time)
