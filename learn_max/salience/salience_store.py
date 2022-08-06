from typing import List


class SalienceStore:
    """
    So I think duplicates should be based on being within some percentile distance
    to a centroid generated with k-means on previous salient events. Then look at the duplicates
    and see if they make sense.
    Each event will be num-patches in size (so 123 currently).
    Keep 5 samples (random? different?) per cluster
    sampling to keep 10k or so 123 * 4B * 10k ~= 5MB of GPU
    """
    # cluster_points: List[list]  # list of clusters with list of points in cluster sorted by distance from centroid
    # cluster_centroids: list
    samples_per_cluster: int = 5  # Number of samples to keep per cluster to avoid running out of mem
    clustered_points: list
    unclustered_points: list
    max_num_clustered_points: int = 10_000  # Increase to get better salience deduplication at cost of GPU mem

    def __init__(self):
        # self.cluster_points = []
        # self.cluster_centroids = []
        self.clustered_points = []
        self.unclustered_points = []

    def add(self, x):
        self.unclustered_points.append(x)
        if len(self.unclustered_points) > 0.1 * self.max_num_clustered_points:
            self.cluster()

    def cluster(self):
        # TODO(for realtime salience detection): Remember to sort points by distance and sample samples_per_cluster
        #  randomly or with some heuristic. See get_distance_percentiles() for more.
        # Since we don't know the number of clusters up front, we'll need to do some search for it and measure
        # with unsupervised metrics like Silouette, Calinkski, and Davies, along with visualization to find a good k.
        # Considered DBSCAN, but we don't want noise/outlier points as new salient events will have few points in the
        # cluster. We could just count outliers as new events. DBSCAN needs a minPts param though and I'd want that
        # to be one as this is online clustering and data is not IID. But minPts=1 would prob give too many clusters.

        # We have 123 dimensions which I think is manageable with euclidean distance in traditional clustering algos,
        # but one thing to try may be dimensionality reduction if dimensions increase or results suck.
        # Also, we should view the points in t-sne.
        points = self.clustered_points + self.unclustered_points
        # These patch diffs don't have much meaning beyond the same gap should mean the same thing across instances.
        # A gap that is close to another gap though is not meaningful...unless we order the clusters linearly by
        # similarity somehow. If we represent the salience windows with z_q_emb though, the patch_diff will be more
        # semantic than the ints.

        self.unclustered_points = []
        pass

    def get_distance_percentiles(self):
        """
        Do this after clustering so lookups for new events are fast
        Immediate lookups based on percentiles will be less accurate than post-facto lookups based
        on cluster membership especially for new salient events as these won't have other examples yet.
        Discovery of a new salient event at runtime just lets you know that you're on the knowledge frontier.
        You may have known this already, say if you had low probabilities for next salients at that level.
        So I don't see a big advantage to this realtime new salience detection.
        """
        ret = []
        for i in range(self.samples_per_cluster):
            percentile = (i+1) / self.samples_per_cluster
            pct_distances = []
            for cluster in self.cluster_points:
                pct_distances.append(cluster[i])  # get i-th closest point
            avg_dist = sum(pct_distances) / len(pct_distances)
            ret.append((percentile, avg_dist))
        return ret
