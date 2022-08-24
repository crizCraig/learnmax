import math
import os
import pickle
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns


# sns.set_theme()  # Apply the default theme
from scipy.cluster.vq import kmeans2
from sklearn.cluster import DBSCAN

from learn_max.constants import RUN_ID, ROOT_DIR
from learn_max.utils import get_date_str, viz_experiences


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

    def __init__(self, replay_buffers, frames_in_sequence_window):
        # self.cluster_points = []
        # self.cluster_centroids = []
        self.clustered_points = []
        self.unclustered_points = []

        # For visualization
        self.replay_buffers = replay_buffers
        self.frames_in_sequence_window = frames_in_sequence_window

    def add(self, saliences, dvq_decoder, device):
        self.unclustered_points += saliences
        if len(self.unclustered_points) > 0.1 * self.max_num_clustered_points:
            self.cluster(dvq_decoder, device)

    def cluster(self, dvq_decoder, device):
        """
        Cluster points to detect duplicates. E.g. we can test if points are in the 90pct distance away from the
        closest cluster centroid and avoid detecting it as a salient event. Then after re-clustering, if the event
        is still 90pct or is in a new cluster that has not been added to the transformer softmax as a salient event,
        we can detect it as salient in the future.

        A problem with this if we get bad clusters at first (due to less data) is that we'd need to
        retrain the transformer with new clusters. It'd be nice if we had some sort of stable clustering
        where the same cluster index represented similar events across re-clusterings. We could do some distance
        measure from core samples in dbscan across clusterings to do that.
        """
        # TODO(for realtime salience detection): Remember to sort points by distance and sample samples_per_cluster
        #  randomly or with some heuristic. See get_distance_percentiles() for more.
        # Since we don't know the number of clusters up front, we'll need to do some search for it and measure
        # with unsupervised metrics like Silhouette, Calinkski, and Davies, along with visualization to find a good k.
        # Considered DBSCAN, but we don't want noise/outlier points as new salient events will have few points in the
        # cluster. We could just count outliers as new events. DBSCAN needs a minPts param though and I'd want that
        # to be one as this is online clustering and data is not IID. But minPts=1 would prob give too many clusters.

        # We have 123 dimensions which I think is manageable with euclidean distance in traditional clustering algos,
        # but one thing to try may be dimensionality reduction if dimensions increase or results suck.
        # Also, we should view the points in t-sne.
        points = self.clustered_points + self.unclustered_points
        patch_diff = np.array([p['patch_diff'].flatten().detach().cpu().numpy() for p in points]).T

        cluster_patch_diff(patch_diff, self.replay_buffers, points, self.frames_in_sequence_window, dvq_decoder, device)

        self.unclustered_points = []

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


def cluster_patch_diff(patch_diff, replay_buffers=None, points=None, frames_in_sequence_window=None, dvq_decoder=None,
                       device=None,):
    # TODO: Visualize the clusters dbscan produces
    dbscan = tune_dbscan(patch_diff)
    # TODO: We should also make sure that consecutive salient events are mostly clustered
    if replay_buffers is not None:
        viz_salience_clusters(dbscan, replay_buffers, points, patch_diff, frames_in_sequence_window, dvq_decoder,
                              device)
    show_non_consecutive_dupes(dbscan)


def tune_dbscan(patch_diff):
    cost = math.inf
    eps = 0.01
    eps_i = 0
    dbscan = None
    while True:
        new_eps = eps + 0.01
        new_dbscan = DBSCAN(eps=new_eps, min_samples=2).fit(patch_diff.T)
        scan_counts = Counter(new_dbscan.labels_)
        non_outlier_counts = scan_counts.copy()
        del non_outlier_counts[-1]
        largest_cluster = max(non_outlier_counts.values()) if non_outlier_counts else 0
        avg_cluster = sum(non_outlier_counts.values()) / len(non_outlier_counts) if non_outlier_counts else 0
        new_cost = (len(scan_counts) +  # minimize total number of clusters
                    largest_cluster +  # along with the smallest largest cluster size
                    avg_cluster +  # while also keeping the average cluster size down
                    2 * scan_counts[-1])  # and reducing the number of outliers
        print('len', len(scan_counts), 'max', largest_cluster, 'out', scan_counts[-1], 'avg', avg_cluster)
        print('eps', new_eps, 'cost', new_cost)
        if new_cost > cost and scan_counts[-1] < 0.9 * len(scan_counts):
            # TODO: Make sure we're not stopping too early (is outlier 0.9 check enough?)
            print('new cost greater', new_cost, '>', cost, 'use eps', eps)
            break
        cost = new_cost
        eps = new_eps
        dbscan = new_dbscan
        eps_i += 1
        if eps_i > 1000:
            raise RuntimeError('Could not find dbscan epsilon')
    return dbscan


def viz_salience_clusters(dbscan, replay_buffers, points, patch_diff, frames_in_sequence_window, dvq_decoder, device):
    """
    Create folders for each cluster (including an outlier folder) and a t-sne graph
    that colorizes by cluster and has a legend with the cluster index (-1 for outliers).
    """
    run_folder = f'{ROOT_DIR}/images/viz_salience_clusters/{get_date_str()}_r_{RUN_ID}'
    FiS = frames_in_sequence_window
    for i, lbl in enumerate(dbscan.labels_):
        replay_i = points[i]['replay_ind']
        folder = f'{run_folder}/cluster_{lbl}_rp_{replay_i}'
        Path(folder).mkdir(parents=True, exist_ok=True)

        # Visualize a movie around the index
        viz_experiences(replay_buffers.train_buf.get(start=replay_i - FiS + 1, length=FiS * 2), folder, dvq_decoder,
                        device)

        # TODO: Disable viz_salience as its redundant to this


    # for k in range(len(patch_diff)):
    #     kd = kmeans2(patch_diff, k, minit='points')
    # pca2 = PCA(n_components=2)
    # pca2.fit(patch_diff.T)
    # TODO: Check KNN or points in a cluster and see if they're sensible
    from sklearn.manifold import TSNE
    from sklearn.decomposition import PCA
    pca = PCA(n_components=50)
    pca.fit(patch_diff)
    tsne = TSNE()
    tsne.fit(pca.components_.T)
    # Create a visualization
    tsne_df = pd.DataFrame(tsne.embedding_, columns=['x', 'y'])
    sns.relplot(
        data=tsne_df,
        x='x',
        y='y',
    )
    # The t-sne showed some non-linear clusters, like strings, so maybe dbscan is the way to go.
    # https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html
    # We represent the patch_diffs with z_q_emb, so it will be more
    # semantic than say z_q_ind ints.
    pass  # TODO: Reduce dimensions 121x30 to 50 then, view points with t-sne per https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html
    # TODO: t-sne has issue with more than 2 duplicate points, we can try solver='dense' but may be slow.
    #   we can remove these points after PCA reduction
    pass


def show_non_consecutive_dupes(dbscan):
    scan_counts = Counter(dbscan.labels_)
    cons = 1
    for i, lb in enumerate(dbscan.labels_[:-1]):
        # print('lb', lb)
        ct = scan_counts[lb]
        # print('ct', ct)
        nxt = dbscan.labels_[i + 1]
        if lb == nxt:
            cons += 1
        else:
            if ct > cons and lb != -1:
                print(f'FOUND @ {lb}!')
            cons = 1
        # print('cons', cons)

def save_cluster_patch_diff_args():
    pickle.dump(
        dict(
            patch_diff=patch_diff,
            replay_buffers=self.replay_buffers,
            points=points,
            frames_in_sequence_window=self.frames_in_sequence_window,
            dvq_decoder=dvq_decoder,
            device=device,
        ),
        open('/home/a/src/learnmax/cluster_patch_diff', 'wb')
    )


def main():
    patch_diff = pickle.load(open('/home/a/src/learnmax/patch_diff.pickle', 'rb'))
    cluster_patch_diff(patch_diff)


if __name__ == '__main__':
    main()
