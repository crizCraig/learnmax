import math
import os
import pickle
from collections import Counter, defaultdict
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
# sns.set_theme()  # Apply the default theme
import scipy
import seaborn as sns
from loguru import logger as log
from scipy.spatial import KDTree
from sklearn.cluster import DBSCAN
from tdigest import TDigest

from learn_max.constants import RUN_ID, ROOT_DIR, PICKLE_DIR, DATE_STR
from learn_max.replay_buffer import ReplayBuffers, SalientExperience
from learn_max.utils import get_date_str, viz_experiences, viz_salience


class SalientEvent:
    def __init__(self, points_in_salient_cluster, cluster_index, dbscan):
        """
        @points_in_salient_cluster has (i, dict(patch_diff, replay_i)) of points
        TODO: Use these to map new experiences to previously detected salient events
            Do this by finding closest core sample to current patch_diff point.
            Then see if the distance to that point is less than some percentile of points.
        """
        # percentile distances of all points in cluster
        points = np.array([p[1]['patch_diff'].cpu().numpy()
                           for p in points_in_salient_cluster])
        points = points.reshape(points.shape[0], -1)
        distances = scipy.spatial.distance.pdist(points)  # TODO: compare points and core points?
        percentiles = [.01, .1, .25, .5, .8, .9, .95, .99, .999]
        self.distance_percentiles = {
            pct: val
            for pct, val in zip(percentiles, np.percentile(distances, percentiles))
        }
        self.point_index_in_cluster = [p[0] for p in points_in_salient_cluster]
        index_map = {p: i for i, p in enumerate(self.point_index_in_cluster)}
        self.core_sample_points = []
        for core_i in dbscan.core_sample_indices_:
            if core_i in index_map:
                self.core_sample_points.append(points[index_map[core_i]])
        self.core_sample_points = np.array(self.core_sample_points)
        if len(self.core_sample_points) < len(points) and cluster_index != -1:
            log.success(f'Cluster {cluster_index} has {len(points)} points, '
                        f'and {len(self.core_sample_points)} core sample points')
        self.core_sample_i = np.array([i for i in dbscan.core_sample_indices_ if i in index_map])
        self.cluster_index = cluster_index
        self.replay_i = np.array([p[1]['replay_ind'].cpu().numpy()
                                  for p in points_in_salient_cluster])
        self.RUN_ID = RUN_ID
        self.DATE_STR = DATE_STR
        # TODO: Store directory that replay buffer was saved to

class SalienceLevel:
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
    max_num_clustered_points: int = 100_000  # Increase to get better salience deduplication at cost of CPU mem
    clusters: List[SalientEvent]
    encountered_known_salients: list

    def __init__(self, level, replay_buffers, frames_in_sequence_window, salience_resume_path=None):
        # self.cluster_points = []
        # self.cluster_centroids = []
        self.level = level
        self.clustered_points = []
        self.unclustered_points = []
        self.kdtree: scipy.spatial.kdtree.KDTree = None  # TODO: Make this a list of kdtrees

        # Consider ddsketch to reduce error since we're interested in long tail salience
        # https://blog.acolyer.org/2019/09/06/ddsketch/
        self.tdigest = TDigest()

        # Map from DBSCAN core point to SalientEvent cluster
        self.core_point_map: dict = {}  # TODO: Make this a list of dicts

        # For visualization
        self.replay_buffers = replay_buffers
        self.frames_in_sequence_window = frames_in_sequence_window

        if salience_resume_path is not None:
            self.load(salience_resume_path)

    def add(self, saliences, dvq_decoder, device, FiS, lower_replay_buffers=None):
        """
        @param lower_replay_buffers: ReplayBuffers from which we extract salient events
        """
        # Move saliences to cpu
        for s in saliences:
            s['patch_diff'] = s['patch_diff'].cpu()
            s['replay_ind'] = s['replay_ind'].cpu()
        self.unclustered_points.extend(saliences)

        repeats = self.detect_repeats(
            FiS, device, dvq_decoder, lower_replay_buffers, saliences
        )
        if len(repeats) > 0:
            assert len(repeats) == 1  # TODO: Handle multiple repeats, but right now will always be one
            salient_event = repeats[0]
            # TODO: Go back through replay buffer and find all experiences
            #  that are salient events and save them for training
            self.replay_buffers.append(
                # TODO: Add cluster points (core points) to salient experience
                #   so that we can generalize to similar salient events (e.g. opening a new door
                #   is thought to be promising for learning because previous doors have been)
                SalientExperience(cluster_index=salient_event.cluster_index)
            )
        progress = len(self.unclustered_points) / (0.1 * self.max_num_clustered_points)
        if len(self.unclustered_points) % 10 == 0:
            log.info(f'Percentage of points needed to cluster: '
                     f'{100 * progress:.2f}%')
        ret = defaultdict(bool)
        if self.kdtree is not None:
            log.warning('Skipping clustering as we do not have stable clustering implemented')
        elif progress >= 1:
            self.cluster(dvq_decoder, device)
            ret['new_salience_level'] = True
        return ret

    def detect_repeats(self, FiS, device, dvq_decoder, replay_buffers, saliences):
        # Map saliences to existing clusters
        # TODO: Write unit test that checks for zero distance when detecting same points that were clustered
        ret = []
        if self.kdtree is not None:  # TODO: Visualize salient event and compare with orig closest points
            for salience in saliences:
                dist, ind = self.kdtree.query(salience['patch_diff'].cpu().numpy().reshape(1, -1))
                salient_event = self.core_point_map[ind[0]]
                if dist < salient_event.distance_percentiles[0.5]:
                    # TODO: ***Try z_q_emb again***  - 10/43 - log this

                    log.success(f"Found salient event {salient_event.cluster_index} with distance {dist}")
                    # TODO(salience): Inform parent transformer that new event occurred, store event, and train on
                    #   such events when we have enough data
                    ret.append(salient_event)
                    if replay_buffers is not None:
                        # Add salient event to replay buffer

                        viz_salience(FiS, [int(salience['replay_ind'])], replay_buffers,
                                     dvq_decoder, device,
                                     file_prefix=f'dupe_salient_event_{salient_event.cluster_index}_')
                else:
                    if replay_buffers is not None:
                        viz_salience(FiS, [int(salience['replay_ind'])], replay_buffers,
                                     dvq_decoder, device,
                                     file_prefix=f'new_salient_event_{salient_event.cluster_index}_')
        return ret

    def cluster(self, dvq_decoder, device):
        """
        Cluster points to detect duplicates. E.g. we can test if points are in the 90pct distance away from the
        closest cluster centroid and avoid detecting it as a NEW salient event (i.e. it's an existing one).
        Then after re-clustering, if the event is still within 90pct or is in a new cluster that has not
        been added to the transformer softmax as a salient event, we can detect it as salient in the future.

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
        # patch diff shape: (num_state/action/delim_embeddings, tokens_in_frame)
        self.kdtree, self.core_point_map = (
            cluster_patch_diff(patch_diff, self.replay_buffers, points,
                               self.frames_in_sequence_window, dvq_decoder, device)
        )
        self.save()
        self.unclustered_points = []

    def load(self, path):
        log.info(f'Loading salience store from {path}')
        self.kdtree = unpickle(f'{path}/kdtree.pickle')
        self.core_point_map = unpickle(f'{path}/core_point_map.pickle')
        self.clustered_points = unpickle(f'{path}/clustered_points.pickle')
        self.tdigest = unpickle(f'{path}/tdigest.pickle')

    def save(self):
        save_dir = f'{PICKLE_DIR}/salience_store/lvl_{self.level}'
        os.makedirs(save_dir, exist_ok=True)
        pickle_obj(self.kdtree, f'{save_dir}/kdtree.pickle')
        pickle_obj(self.core_point_map, f'{save_dir}/core_point_map.pickle')
        pickle_obj(self.clustered_points, f'{save_dir}/clustered_points.pickle')
        pickle_obj(self.tdigest, f'{save_dir}/tdigest.pickle')

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

def cluster_patch_diff(patch_diff, replay_buffers=None, points=None,
                       frames_in_sequence_window=None, dvq_decoder=None,
                       device=None, should_save_clusters=True):
    """
    patch diff (possibility, num_batches) where possibility is flattened logits for all patches,
    i.e. logits * patches... logits are predictions of the next embedding, so we're looking at the biggest differences
    in the expected future, i.e. the biggest difference in possibilities encountered in recently sampled batches
    """
    # TODO: Visualize the clusters dbscan produces
    # TODO: We should also make sure that consecutive salient events are mostly clustered

    dbscan = tune_dbscan(patch_diff)
    should_pickle = False
    if should_pickle:
        pickle.dump(patch_diff, open('/tmp/cluster_patch_diff.pickle', 'wb'))
        pickle.dump(dvq_decoder, open('/tmp/dvq_decoder.pickle', 'wb'))
        pickle.dump(replay_buffers, open('/tmp/replay_buffers.pickle', 'wb'))
        pickle.dump(points, open('/tmp/points.pickle', 'wb'))
        pickle.dump(device, open('/tmp/device.pickle', 'wb'))
        pickle.dump(dbscan, open('/tmp/dbscan.pickle', 'wb'))
        # frames_in_sequence_window = 8


    if replay_buffers is not None:
        viz_salience_clusters(
            dbscan, replay_buffers, points, patch_diff, frames_in_sequence_window,
            dvq_decoder, device, should_save_clusters)

    show_non_consecutive_dupes(dbscan)
    return create_clusters(dbscan, points)

def create_clusters(dbscan, points):
    salient_events = []
    cluster_map = defaultdict(list)
    for i, label in enumerate(dbscan.labels_):
        cluster_map[label].append((i, points[i]))

    for cluster, cluster_points in cluster_map.items():
        salient_events.append(SalientEvent(cluster_points, cluster_index=cluster, dbscan=dbscan))

    flat_points = []
    core_point_map = {}  # Maps flattened core points back to their original clusters
    for se in salient_events:
        for i, point in enumerate(se.core_sample_points):
            flat_points.append(point)
            core_point_map[len(flat_points) - 1] = se  # can just be append on list
    assert len(core_point_map) == len(flat_points)
    kdtree = KDTree(np.array(flat_points))  # Populate with flat list of core sample i
    return kdtree, core_point_map


def tune_dbscan(patch_diff):
    """@param patch_diff: logits * patches, num_salience_points
    """
    cost = math.inf
    distances = scipy.spatial.distance.pdist(patch_diff.T)
    MAX_ITERS = 200
    start_eps = np.percentile(distances, 0.01)
    eps = start_eps
    eps_incr = (np.percentile(distances, 10) - start_eps) / MAX_ITERS
    eps_i = 0
    dbscan = None
    print('tune_dbscan patch_diff', patch_diff.shape, '\n', patch_diff)
    while True:
        new_eps = eps + eps_incr
        new_dbscan = DBSCAN(eps=new_eps, min_samples=2).fit(patch_diff.T)
        clusters = Counter(new_dbscan.labels_)
        non_outlier_counts = clusters.copy()
        del non_outlier_counts[-1]
        largest_cluster = max(non_outlier_counts.values()) if non_outlier_counts else 0
        avg_cluster = sum(non_outlier_counts.values()) / len(non_outlier_counts) if non_outlier_counts else 0
        # new_cost = largest_cluster / avg_cluster  # copilot idea (simpler?)
        new_cost = (len(clusters) +  # minimize total number of clusters
                    largest_cluster +  # along with the smallest largest cluster size
                    avg_cluster +  # while also keeping the average cluster size down
                    2 * clusters[-1])  # and reducing the number of outliers
        print('len', len(clusters), 'max', largest_cluster, 'out', clusters[-1], 'avg', avg_cluster)
        print('eps', new_eps, 'cost', new_cost)
        if new_cost > cost and clusters[-1] < 0.99 * len(new_dbscan.labels_):
            # TODO: Make sure we're not stopping too early (is outlier 0.9 check enough?)
            print('new cost greater', new_cost, '>', cost, 'use eps', eps)
            break
        cost = new_cost
        eps = new_eps
        dbscan = new_dbscan
        eps_i += 1
        if eps_i >= MAX_ITERS:
            log.error('Could not find a good dbscan epsilon')
            break
    return dbscan

def viz_salience_clusters(dbscan, replay_buffers, points, patch_diff, frames_in_sequence_window, dvq_decoder, device,
                          should_save_clusters=True):
    """
    Create folders for each cluster (including an outlier folder) and a t-sne graph
    that colorizes by cluster and has a legend with the cluster index (-1 for outliers).

    TODO: Map these events cluster indices to new softmax outputs, then see if you can
        predict salient events. Then see what the longest chain of unique events is. We should be able
        to see the steps to get to the key. We can use RL for control instead of random actions too, to get
        longer chains.
    """
    if should_save_clusters:
        run_folder = f'{ROOT_DIR}/images/viz_salience_clusters/{get_date_str()}_r_{RUN_ID}'
        FiS = frames_in_sequence_window
        for i, lbl in enumerate(dbscan.labels_):
            replay_i = int(points[i]['replay_ind'])
            folder = f'{run_folder}/cluster_{lbl}'
            Path(folder).mkdir(parents=True, exist_ok=True)
            log.info(f'Saving salience clusters to {folder}')
            # Visualize a movie around the index
            viz_experiences(replay_buffers.train_buf.get(start=replay_i - FiS + 1, length=FiS * 2),
                            folder, dvq_decoder, device, file_prefix=f'cluster_{lbl}_rp_{replay_i}_')

        # TODO: Disable viz_salience as its redundant to this


    # for k in range(len(patch_diff)):
    #     kd = kmeans2(patch_diff, k, minit='points')
    # pca2 = PCA(n_components=2)
    # pca2.fit(patch_diff.T)
    # TODO: Check KNN or points in a cluster and see if they're sensible
    from sklearn.manifold import TSNE
    from sklearn.decomposition import PCA
    # clusters = Counter(dbscan.labels_)
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
        hue=dbscan.labels_,
        palette='tab10',
        legend='full',
    )
    # The t-sne showed some non-linear clusters, like strings, so maybe dbscan is the way to go.
    # https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html
    # We represent the patch_diffs with z_q_emb, so it will be more
    # semantic than say z_q_ind ints.
    pass  # TODO: Reduce dimensions 121x30 to 50 then, view points with t-sne per https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html
    # TODO: t-sne has issue with more than 2 duplicate points, we can try solver='dense' but may be slow.
    #   we can remove these points after PCA reduction


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
    FiS = 8

    # Instantiate SalienceStore
    salience_store = SalienceLevel(1, None, None, None)
    salience_store.kdtree = unpickle('/home/a/src/learnmax/pickles/2022.11.27_11:51:10.663231/kdtree.pickle')
    salience_store.core_point_map = unpickle(
        '/home/a/src/learnmax/pickles/2022.11.27_11:51:10.663231/core_point_map.pickle'
    )
    points = unpickle('/home/a/src/learnmax/pickles/points.pickle')
    dvq_decoder = unpickle('/home/a/src/learnmax/pickles/dvq_decoder.pickle')
    device = unpickle('/home/a/src/learnmax/pickles/device.pickle')

    salient_add = salience_store.add(
        saliences=points,
        dvq_decoder=dvq_decoder,
        device=device,
        FiS=FiS,
    )
    if salient_add['new_salience_level']:
        log.info(f'New salience level')



    patch_diff = unpickle('/home/a/src/learnmax/patch_diff.pickle')


    dbscan = unpickle('/home/a/src/learnmax/pickles/dbscan.pickle')
    replay_buffers = unpickle('/home/a/src/learnmax/pickles/replay_buffers.pickle')
    frames_in_sequence_window = 8

    kdtree, core_point_map = cluster_patch_diff(
        patch_diff=patch_diff, replay_buffers=replay_buffers, points=points,
        dvq_decoder=dvq_decoder, device=device,
        frames_in_sequence_window=frames_in_sequence_window,
        should_save_clusters=False)
    salience_store.kdtree = kdtree
    salience_store.core_point_map = core_point_map
    salience_store.save()
    # viz_salience_clusters(dbscan=dbscan, replay_buffers=replay_buffers, points=points, patch_diff=patch_diff,
    #                       frames_in_sequence_window=frames_in_sequence_window, dvq_decoder=dvq_decoder, device=device)
    # show_non_consecutive_dupes(dbscan)

def salience_level_factory(
        level, env_id, short_term_mem_max_length, frames_per_file,
        train_to_test_collection_ratio, should_overfit_gpt,
        frames_in_sequence_window, salience_resume_path=None,
):
    replay_buffers = ReplayBuffers(
        env_id=env_id,
        short_term_mem_max_length=short_term_mem_max_length,
        salience_level=level,
        frames_per_file=frames_per_file,
        train_to_test_collection_ratio=train_to_test_collection_ratio,
        overfit_to_short_term=should_overfit_gpt,
    )
    ret = SalienceLevel(level=level,
                        replay_buffers=replay_buffers,
                        frames_in_sequence_window=frames_in_sequence_window,
                        salience_resume_path=salience_resume_path)
    return ret

def unpickle(file):

    # TODO: Delete this module hack
    import sys
    sys.modules['learn_max.salience.salience_store'] = sys.modules[__name__]

    with open(file, 'rb') as fo:
        ret = pickle.load(fo)
    return ret

def pickle_obj(obj, file):
    with open(file, 'wb') as fo:
        pickle.dump(obj, fo)

if __name__ == '__main__':
    main()


