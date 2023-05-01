import math
import os
import pickle
import shutil
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Optional, Any, Dict, Tuple, List, Union, Iterable

from scipy import spatial
from typing_extensions import Self  # Will be in typing in 3.11

import numpy as np
import pandas as pd
# sns.set_theme()  # Apply the default theme
import scipy
import seaborn as sns
import torch
from ddsketch import DDSketch
from loguru import logger as log
from scipy.spatial import KDTree
from sklearn.cluster import DBSCAN
from tdigest import TDigest

from learn_max.constants import RUN_ID, ROOT_DIR, PICKLE_DIR, DATE_STR, NUM_DIFF_SALIENT, LEVEL_PREFIX_STR
from learn_max.percentile_estimator import PercentileEstimator
from learn_max.replay_buffer import ReplayBufferSplits, ReplayBuffer
from learn_max.salience.experience import SalientExperience, SensoryExperience, Experience
from learn_max.utils import get_date_str, viz_experiences, viz_salience

# Note that these contain logit based patch_diffs, which are not stable over time and
#   are much larger than z_q_emb based patch_diffs.
TEST_PICKLE_FILE = f'{ROOT_DIR}/test_data/exp_replay_i_and_patch_diff_for_salient_clusters.pickle'

class SalientCluster:
    def __init__(
        self,
        points_in_salient_cluster: Iterable,
        cluster_index: int,
        dbscan: DBSCAN,
        salience_level: int,
    ) -> None:
        """
        This object helps map the points in original salient cluster to the cluster index,
        through the dbscan core points. This as core points are used in a KDTree to map unseen points
        back to core points and through this object, the cluster index.
        (See core_point_map in SalienceLevel.)

        @points_in_salient_cluster has (i, dict(patch_diff, replay_i)) of points
        TODO: Use these to map new experiences to previously detected salient events
            Do this by finding closest core sample to current patch_diff point.
            Then see if the distance to that point is less than some percentile of points.
        """
        self.salience_level = salience_level
        # percentile distances of all points in cluster
        points = np.array(
            [p[1].patch_diff.cpu().numpy() for p in points_in_salient_cluster]
        )
        points = points.reshape(points.shape[0], -1)

        # TODO: Most points are core points https://en.wikipedia.org/wiki/DBSCAN#Preliminary
        #   so we don't need to treat them differently
        distances = scipy.spatial.distance.pdist(points)  # TODO: compare points and core points?
        percentiles = [1, 10, 25, 50, 80, 90, 95, 99, 99.9]
        self.distance_percentiles = {
            pct: val
            for pct, val in zip(
                percentiles, np.percentile(distances, percentiles)  # type: ignore
            )
        }

        # Rename this to self.salient_experience_index
        self.point_index_in_cluster = [p[0] for p in points_in_salient_cluster]
        index_map = {p: i for i, p in enumerate(self.point_index_in_cluster)}
        self.core_sample_points = []
        for core_i in dbscan.core_sample_indices_:
            if core_i in index_map:
                self.core_sample_points.append(points[index_map[core_i]])

        # Used for kdtree in salience level
        self.core_sample_points = np.array(self.core_sample_points)  # type: ignore

        if len(self.core_sample_points) < len(points) and cluster_index != -1:
            log.success(
                f'Cluster {cluster_index} has {len(points)} points, '
                f'and {len(self.core_sample_points)} core sample points'
            )

        # TODO: Is this just dbscan.core_sample_indices_?
        self.core_sample_i = np.array([i for i in dbscan.core_sample_indices_ if i in index_map])


        self.cluster_index = cluster_index
        self.below_replay_indexes = np.array(
            [p[1].below_replay_index for p in points_in_salient_cluster]
        )
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

    # list of clusters with list of points in cluster sorted by distance from centroid
    # cluster_points: List[list]
    # cluster_centroids: list

    # Number of samples to keep per cluster to avoid running out of mem
    samples_per_cluster: int = 5

    clustered_points: list
    unclustered_points: list

    # Increase to get better salience deduplication at cost of CPU mem
    num_points_to_cluster: Optional[int]

    encountered_known_salients: list
    salience_resume_path: Optional[str] = None  # Path to resume salience from
    pct_est: PercentileEstimator

    def __init__(
        self,
        input_exp_lvl: int,
        replay_buffers: ReplayBufferSplits,
        seq_len: int,
        salience_resume_path: Optional[str] = None,
        above: Optional[Self] = None,
        num_points_to_cluster: Optional[int] = None,
    ) -> None:
        # self.cluster_points = []
        # self.cluster_centroids = []
        self.input_exp_lvl = input_exp_lvl
        self.output_lvl = input_exp_lvl + 1
        self.clustered_points = []
        self.unclustered_points = []
        self.pct_est = PercentileEstimator()
        if num_points_to_cluster is not None:
            self.num_points_to_cluster = num_points_to_cluster
        elif self.input_exp_lvl == 0:
            self.num_points_to_cluster = 100_000
        else:
            self.num_points_to_cluster = None  # Needs to be set when we know the size of the replay buffer
        self.last_log_progress: float = 0
        self.clusters: List[SalientCluster] = []

        self.kdtree: Optional[scipy.spatial.kdtree.KDTree] = None

        self.above = above

        # Map from DBSCAN core point to SalientEvent cluster
        self.core_point_map: dict = {}  # TODO: Make this a list of dicts

        self.replay_splits: ReplayBufferSplits = replay_buffers
        self.seq_len = seq_len
        self.last_checked_replay_index = 0

        self.migrated_quantiles_to_percentiles: bool = True

        if salience_resume_path is not None:
            self.salience_resume_path = salience_resume_path
            self.load(salience_resume_path)

    def __len__(self) -> int:
        """Number of steps in replay buffer"""
        return len(self.replay_splits)

    def add(
        self,
        saliences: List[SalientExperience],
        device: Union[str, torch.device],
        FiS: int,
        lower_replay_buffers: Optional[ReplayBufferSplits] = None,
        dvq_decoder: Optional[torch.nn.Module] = None,
    ) -> Dict[str, Any]:
        """
        @param lower_replay_buffers: ReplayBuffers from which we extract salient events
        """
        # Move saliences to cpu
        for s in saliences:
            assert s.patch_diff is not None
            s.patch_diff = s.patch_diff.cpu()
        self.unclustered_points.extend(saliences)
        repeats = self.detect_repeats(
            FiS, device, saliences, dvq_decoder, lower_replay_buffers  # type: ignore
        )
        if len(repeats) > 0:
            assert len(repeats) == 1  # TODO: Handle multiple repeats, but right now will always be one
            salient_cluster, salience, dist = repeats[0]
            # TODO: Go back through replay buffer and find all experiences
            #  that are salient events and save them for training
            if self.above is None:
                log.warning('Salience level above is None')
            else:
                # Now that we have a salient event, add the cluster props and
                # append it to the level above
                salience.dist = dist
                salience.seq_len = self.seq_len
                salience.cluster_index = salient_cluster.cluster_index
                salience.below_cluster_replay_indexes = (
                    salient_cluster.below_replay_indexes  # type: ignore
                )
                salience.level = self.output_lvl
                splits = self.above.replay_splits
                raise NotImplementedError('add_to_below_map')
                if splits.is_train():
                    splits.train_buf.add_to_below_map(salience)
                else:
                    splits.test_buf.add_to_below_map(salience)
                self.above.replay_splits.append(salience)  # At least salience level 1
                # TODO: Add cluster points (core points) to salient experience
                #  (not just cluster index)
                #   so that we can generalize to similar salient events
                #   (e.g. opening a new door
                #   is thought to be promising for learning because
                #   previous doors have been)
                # SalientExperience(
                #
                #     # Avoid repeating mid_replay_ind by storing salient clusters separately
                #     below_replay_index=salience.below_replay_index,
                #
                #     patch_diff=salience.patch_diff,
                #     dist=dist,
                #     seq_len=self.seq_len,
                #     cluster_index=salient_cluster.cluster_index,
                #     below_cluster_replay_indexes=salient_cluster.below_replay_indexes,  # type: ignore
                #     level=self.output_lvl,
                #     done=salience.done,
                # )

        # TODO: Decrease for higher levels
        assert self.num_points_to_cluster is not None
        progress = len(self.unclustered_points) / self.num_points_to_cluster

        # Log every 10%
        if progress - self.last_log_progress >= 0.01:
            self.last_log_progress = progress
            log.info(f'Percentage of points needed to cluster: ' f'{100 * progress:.2f}%')
        ret = {'new_salience_level': False}
        if self.kdtree is not None:
            log.warning('Skipping clustering as we do not have stable clustering implemented')
        elif progress >= 1:
            if self.salience_resume_path is not None:
                log.warning(
                    'Skipping clustering as we resumed salience and only cluster once'
                            + 'currently')
            else:
                # TODO: Require fewer points for clusterings after the first
                # TODO: Deal with high level salience (i.e. no dvq_decoder)
                self.cluster(device, dvq_decoder)
                ret['new_salience_level'] = True
        ret['repeats'] = repeats
        return ret

    def detect_repeats(
        self,
        frames_in_seq: int,
        device: Union[str, torch.device],
        saliences: List[SalientExperience],
        dvq_decoder: Optional[torch.nn.Module] = None,
        replay_buffers: Optional[ReplayBufferSplits] = None,
    ) -> List[Tuple[SalientCluster, SalientExperience, float]]:
        # Map saliences to existing clusters
        # TODO: Write unit test that checks for zero distance when detecting same points that were clustered
        ret = []
        if (
            self.kdtree is not None
        ):  # TODO: Visualize salient event and compare with orig closest points
            for salience in saliences:
                dist, ind = self.kdtree.query(
                    salience.patch_diff.cpu().numpy().reshape(1, -1)
                )
                dist = dist[0]
                ind = ind[0]
                # TODO: We should capture the actual event, not just the cluster it maps to.
                #   This would mean calling salient_event, salient_cluster (more appropriate).
                #   Perhaps SalientExperience should hold the actual below_mid_replay_index in
                #   addition to the below_mid_replay_indexes in the salient cluster it was closest to.
                salient_cluster = self.core_point_map[ind]

                if dist < salient_cluster.distance_percentiles[50]:
                    # Here we compare the median distance (50th percentile)
                    # of all points within a cluster to the distance to the closest core point.
                    # This is okay to the extent core points are representative of the cluster.
                    # TODO: ***Try z_q_emb again***  - 10/43 - log this
                    log.success(
                        f'Found salient event {salient_cluster.cluster_index} with distance {dist}'
                    )
                    # TODO(salience): Inform parent transformer that new event occurred, store event, and train on
                    #   such events when we have enough data
                    ret.append((salient_cluster, salience, dist))
                    if replay_buffers is not None and dvq_decoder is not None:
                        # Add salient event to replay buffer

                        viz_salience(
                            frames_in_seq,
                            [int(salience.below_replay_index)],
                            replay_buffers,
                            dvq_decoder,
                            device,
                            file_prefix=f'repeat_salient_exp_for_cluster_{salient_cluster.cluster_index}_',
                        )
                else:
                    if replay_buffers is not None and dvq_decoder is not None:
                        viz_salience(
                            frames_in_seq,
                            [int(salience.below_replay_index)],
                            replay_buffers,
                            dvq_decoder,
                            device,
                            file_prefix=f'new_salient_exp__closest_was{salient_cluster.cluster_index}_',
                        )
        return ret

    def cluster(
            self, device: str, dvq_decoder: Optional[torch.nn.Module] = None
    ) -> None:
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
        points = self.clustered_points + self.unclustered_points  # Clustered points is always empty for now
        patch_diff = np.array([p.patch_diff.flatten().detach().cpu().numpy() for p in points]).T
        # patch diff shape: (num_state/action/delim_embeddings, tokens_in_frame)
        self.kdtree, self.core_point_map = cluster_patch_diff(
            patch_diff=patch_diff,
            points=points,
            replay_buffers=self.replay_splits,
            seq_len=self.seq_len,
            dvq_decoder=dvq_decoder,
            device=device,
            is_sensory=self.input_exp_lvl == 0,
        )
        self.save()
        self.clusters = self.get_clusters()

        # TODO: Some reservoir sampling or just keep the core points or merge old clusters with new ones
        # self.clustered_points = self.unclustered_points
        self.unclustered_points = []

    def get_clusters(self) -> List[SalientCluster]:
        return list(set(self.core_point_map.values()))

    def load(self, path: str) -> None:
        log.info(f'Loading salience store from {path}...')
        self.kdtree = unpickle(f'{path}/kdtree.pickle')
        self.core_point_map = unpickle(f'{path}/core_point_map.pickle')
        self.clustered_points = unpickle(f'{path}/clustered_points.pickle')
        if os.path.exists(f'{path}/tdigest.pickle'):
            tdigest: TDigest = unpickle(f'{path}/tdigest.pickle')
            self.pct_est = PercentileEstimator(tdigest)
        else:
            est = unpickle(f'{path}/sketch.pickle')
            if isinstance(est, (TDigest, DDSketch)):
                self.pct_est = PercentileEstimator(est)
            else:
                self.pct_est = est
        if not getattr(self, 'migrated_quantiles_to_percentiles', False):
            raise NotImplementedError('Make sure we do not need to migrate still')
            clusters = set(self.core_point_map.values())

            # TODO: Just get core points out, compute distances and recomputes percentiles
            # for cluster in clusters:

        self.clusters = self.get_clusters()
        # TODO: Load meta

        log.info(f'Loaded salience store from {path}')

    def save(self) -> str:
        save_dir = f'{PICKLE_DIR}/salience_store/{LEVEL_PREFIX_STR}{self.input_exp_lvl}/{get_date_str()}'
        os.makedirs(save_dir, exist_ok=True)
        replay_lineage = self.replay_splits.get_lineage()
        meta = dict(
            level=self.input_exp_lvl,
            seq_len=self.seq_len,
            replay_splits_data_dir=self.replay_splits.data_dir,
            run_id=RUN_ID,
            date_str=DATE_STR,
            replay_lineage=replay_lineage,
        )
        pickle_obj(meta, f'{save_dir}/meta.pickle')
        pickle_obj(self.kdtree, f'{save_dir}/kdtree.pickle')
        pickle_obj(self.core_point_map, f'{save_dir}/core_point_map.pickle')
        pickle_obj(self.clustered_points, f'{save_dir}/clustered_points.pickle')
        pickle_obj(self.pct_est, f'{save_dir}/sketch.pickle')
        log.success(f'Saved salience store to {save_dir}')
        return save_dir

    def get_distance_percentiles(self) -> List[Tuple[float, float]]:
        """
        NOT USED - Realtime salient event detection
        Do this after clustering so lookups for new events are fast
        Immediate lookups based on percentiles will be less accurate than post-facto lookups based
        on cluster membership especially for new salient events as these won't have other examples yet.
        Discovery of a new salient event at runtime just lets you know that you're on the knowledge frontier.
        You may have known this already, say if you had low probabilities for next salients at that level.
        So I don't see a big advantage to this realtime new salience detection.
        """
        ret: List[Tuple[float, float]] = []
        # for i in range(self.samples_per_cluster):
        #     percentile = (i+1) / self.samples_per_cluster * 100
        #     raise NotImplementedError('Need to check if this is valid percent')
        #     pct_distances = []
        #     for cluster in self.clustered_points:
        #         pct_distances.append(cluster[i])  # get i-th closest point
        #     avg_dist = sum(pct_distances) / len(pct_distances)
        #     ret.append((percentile, avg_dist))
        return ret


def cluster_patch_diff(
    patch_diff: np.ndarray,
    points: List[SalientExperience],
    replay_buffers: Optional[ReplayBufferSplits] = None,
    seq_len: Optional[int] = None,
    dvq_decoder: Optional[torch.nn.Module] = None,
    device: Optional[str] = None,
    should_save_clusters: bool = True,
    is_sensory: bool = True,
) -> Tuple[spatial.KDTree, Dict[int, SalientCluster]]:
    """
    patch diff (possibility, num_batches) where possibility is flattened logits for all patches,
    i.e. logits * patches... logits are predictions of the next embedding, so we're looking at the biggest differences
    in the expected future, i.e. the biggest difference in possibilities encountered in recently sampled batches
    """
    # TODO: Visualize the clusters dbscan produces
    # TODO: We should also make sure that consecutive salient events are mostly clustered

    dbscan = tune_dbscan(patch_diff, min_samples=2)
    should_pickle = False  # This is saved in the SalienceLevel, can delete
    if should_pickle:
        pickle.dump(patch_diff, open('/tmp/cluster_patch_diff.pickle', 'wb'))
        pickle.dump(dvq_decoder, open('/tmp/dvq_decoder.pickle', 'wb'))
        pickle.dump(replay_buffers, open('/tmp/replay_buffers.pickle', 'wb'))
        pickle.dump(points, open('/tmp/points.pickle', 'wb'))
        pickle.dump(device, open('/tmp/device.pickle', 'wb'))
        pickle.dump(dbscan, open('/tmp/dbscan.pickle', 'wb'))
        # frames_in_sequence_window = 8


    if replay_buffers is not None and points and points[0].level <= 1:
        viz_salience_clusters(
            dbscan,
            replay_buffers.train_buf,
            points,
            patch_diff,
            seq_len,
            dvq_decoder,
            device,
            should_save_clusters,
        )
    # TODO: Combine consecutive dupes into one cluster
    show_non_consecutive_dupes(dbscan)
    return create_clusters(dbscan, points)

def create_clusters(
        dbscan: DBSCAN, points: List[SalientExperience]
) -> Tuple[spatial.KDTree, Dict[int, SalientCluster]]:
    salient_clusters = []
    cluster_map = defaultdict(list)
    for i, label in enumerate(dbscan.labels_):
        cluster_map[label].append((i, points[i]))

    for cluster_index, cluster_points in cluster_map.items():
        salient_clusters.append(
            SalientCluster(cluster_points, cluster_index=cluster_index, dbscan=dbscan, salience_level=points[0].level)
        )

    flat_points = []
    core_point_map = {}  # Maps flattened core points back to their original clusters
    for sc in salient_clusters:
        for i, point in enumerate(sc.core_sample_points):
            flat_points.append(point)
            core_point_map[len(flat_points) - 1] = sc  # can just be append on list
    assert len(core_point_map) == len(flat_points)
    kdtree = KDTree(np.array(flat_points))  # Populate with flat list of core sample i
    # TODO: Create SalientExperiences
    return kdtree, core_point_map


def tune_dbscan(patch_diff: np.ndarray, min_samples: int = 2) -> DBSCAN:
    """
    @param patch_diff: logits * patches, num_salience_points
    @param min_samples: min points per DBSCAN cluster
    """
    distances = scipy.spatial.distance.pdist(patch_diff.T)
    MAX_ITERS = 200

    # This is .01%
    start_eps = np.percentile(distances, 0.01)  # type: ignore

    eps = start_eps
    eps_incr = (np.percentile(distances, 10) - start_eps) / MAX_ITERS  # type: ignore
    eps_i = 0
    best_eps_i = math.inf
    log.info(f'tune_dbscan patch_diff {patch_diff.shape}\n{patch_diff}')
    min_cost = math.inf
    ret = DBSCAN()  # placeholder
    while True:
        new_eps = eps + eps_incr
        new_dbscan = DBSCAN(eps=new_eps, min_samples=min_samples).fit(patch_diff.T)
        clusters = Counter(new_dbscan.labels_)
        non_outlier_counts = clusters.copy()
        del non_outlier_counts[-1]
        largest_cluster = max(non_outlier_counts.values()) if non_outlier_counts else 0
        avg_cluster = sum(non_outlier_counts.values()) / len(non_outlier_counts) if non_outlier_counts else 0

        # Spread points among non-outlier clusters
        # new_cost = -1 * len(clusters) / avg_cluster

        # Spread points among clusters and minimize outliers
        new_cost = (
            len(clusters) +  # minimize total number of clusters
            largest_cluster +  # along with the smallest largest cluster size
            avg_cluster +  # while also keeping the average cluster size down
            2 * clusters[-1]  # and reducing the number of outliers
        )
        print('len', len(clusters), 'max', largest_cluster, 'out', clusters[-1], 'avg', avg_cluster)
        print('eps', new_eps, 'cost', new_cost)
        print(Counter(new_dbscan.labels_))
        # if new_cost > cost and clusters[-1] < 0.99 * len(new_dbscan.labels_):
        #     # TODO: Make sure we're not stopping too early (is outlier 0.9 check enough?)
        #     print('new cost greater', new_cost, '>', cost, 'use eps', eps)
        #     return sorted(costs)[0][2]
        cost = new_cost
        eps = new_eps
        dbscan = new_dbscan
        eps_i += 1
        if cost < min_cost:
            log.success(f'Found new best dbscan: {Counter(dbscan.labels_)}')
            ret = dbscan
            min_cost = cost
            best_eps_i = eps_i
        if eps > max(distances) or (eps_i - best_eps_i) > 0.25 * MAX_ITERS:
            assert hasattr(ret, 'labels_'), 'No sub inf cost found, wth!'
            return ret
        # if eps_i >= MAX_ITERS:
        #     log.error('Could not find a good dbscan epsilon')
        #     break

def ensure_viz_dir(date_str: str, run_id: str, salience_level: int) -> str:
    ret = (
        f'{ROOT_DIR}/images/viz_salience_clusters'
        f'/{date_str}_r_{run_id}_{LEVEL_PREFIX_STR}{salience_level}'
    )
    Path(ret).mkdir(parents=True, exist_ok=True)
    return ret


def get_salient_viz_file_prefix(lbl: int, replay_i: int) -> str:
    return f'cluster_{lbl}_rp_{replay_i}_'


def viz_salience_clusters(
    dbscan: DBSCAN,
    replay_buffer: ReplayBuffer,
    points: List[SalientExperience],
    patch_diff: np.ndarray,
    frames_in_sequence_window: int,
    dvq_decoder: torch.nn.Module,
    device: torch.device,
    should_save_clusters: bool = True,
) -> None:
    """
    Create folders for each cluster (including an outlier folder) and a t-sne graph
    that colorizes by cluster and has a legend with the cluster index (-1 for outliers).

    TODO: Map these events cluster indices to new softmax outputs, then see if you can
        predict salient events. Then see what the longest chain of unique events is. We should be able
        to see the steps to get to the key. We can use RL for control instead of random actions too, to get
        longer chains.
    """
    if should_save_clusters:
        save_dir = ensure_viz_dir(date_str=get_date_str(), run_id=RUN_ID, level=buf.salience_level)
        FiS = frames_in_sequence_window
        for i, lbl in enumerate(dbscan.labels_):
            replay_i = points[i].below_replay_index
            folder = ensure_cluster_viz_dir(lbl, save_dir)
            log.info(f'Saving salience clusters to {folder}')

            # Visualize a movie around the index
            viz_experiences(
                get_sensory_sequence(replay_buffer, replay_i, FiS),
                folder,
                dvq_decoder,
                device,
                file_prefix=get_salient_viz_file_prefix(lbl, replay_i),
            )

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


def get_sensory_sequence(replay_buffer: ReplayBuffer, replay_i: int, FiS: int) -> List[Experience]:
    """Get a sequence of experiences around the salient event."""
    return replay_buffer.get(start=replay_i - FiS + 1, length=FiS * NUM_DIFF_SALIENT)


def ensure_cluster_viz_dir(lbl: int, save_dir: str) -> str:
    folder = f'{save_dir}/cluster_{lbl}'
    Path(folder).mkdir(parents=True, exist_ok=True)
    return folder


def show_non_consecutive_dupes(dbscan: DBSCAN) -> None:
    label_counts = Counter(dbscan.labels_)
    consecutive = 1
    non_consecutive_dupes = 0
    for i, lb in enumerate(dbscan.labels_[:-1]):
        # print('lb', lb)
        count = label_counts[lb]
        # print('ct', ct)
        nxt = dbscan.labels_[i + 1]
        if lb == nxt:
            consecutive += 1
        else:
            if count > consecutive and lb != -1:
                non_consecutive_dupes += 1
                log.info(f'FOUND @ {lb}!')
            consecutive = 1
        # print('cons', cons)
    # TODO: Combine consecutive dupes into one cluster
    log.info(
        f'Found {non_consecutive_dupes} non-consecutive '
        f'labels out of {len(dbscan.labels_)}'
    )

# def save_cluster_patch_diff_args():
#     pickle.dump(
#         dict(
#             patch_diff=patch_diff,
#             replay_buffers=self.replay_buffers,
#             points=points,
#             frames_in_sequence_window=self.steps_in_sequence_window,
#             dvq_decoder=dvq_decoder,
#             device=device,
#         ),
#         open('/home/a/src/learnmax/cluster_patch_diff', 'wb')
#     )


def main() -> None:
    FiS = 8

    # Instantiate SalienceStore
    salience_store = SalienceLevel(1, None, None, None)
    salience_store.kdtree = unpickle('/home/a/src/learnmax/pickles/2022.11.27_11:51:10.663231/kdtree.pickle')  # 246MB

    sys.modules[__name__].SalientEvent = SalientCluster  # Fix rename from when pickle was created
    salience_store.core_point_map = unpickle(
        '/home/a/src/learnmax/pickles/2022.11.27_11:51:10.663231/core_point_map.pickle'  # 122MB
    )
    points = unpickle('/home/a/src/learnmax/pickles/points.pickle')  # 130MB
    dvq_decoder = unpickle('/home/a/src/learnmax/pickles/dvq_decoder.pickle')  # 1MB
    device = unpickle('/home/a/src/learnmax/pickles/device.pickle')  # 44B

    # test add and detect_repeats
    test_add_and_detect_repeats = False
    if test_add_and_detect_repeats:
        salient_add = salience_store.add(
            saliences=points,
            dvq_decoder=dvq_decoder,
            device=device,
            FiS=FiS,
        )
        if salient_add['new_salience_level']:
            log.info(f'New salience level')

    patch_diff = unpickle('/home/a/src/learnmax/patch_diff.pickle')  # 125MB
    dbscan = unpickle('/home/a/src/learnmax/pickles/dbscan.pickle')
    replay_buffers = unpickle('/home/a/src/learnmax/pickles/replay_buffers.pickle')  # 2.6GB
    seq_len = 8

    kdtree, core_point_map = cluster_patch_diff(
        patch_diff=patch_diff,
        replay_buffers=replay_buffers,
        points=points,
        dvq_decoder=dvq_decoder,
        device=device,
        seq_len=seq_len,
        should_save_clusters=True,
    )
    salience_store.kdtree = kdtree
    salience_store.core_point_map = core_point_map
    salience_store.save()
    # viz_salience_clusters(dbscan=dbscan, replay_buffers=replay_buffers, points=points, patch_diff=patch_diff,
    #                       frames_in_sequence_window=frames_in_sequence_window, dvq_decoder=dvq_decoder, device=device)
    # show_non_consecutive_dupes(dbscan)

def salience_level_factory(
    input_exp_lvl: int,
    env_id: str,
    short_term_mem_max_length: int,

    # Steps per file should be set such that the
    # individual files are fast to load and small enough
    steps_per_file: int,

    train_to_test_collection_files: int,
    should_overfit_gpt: bool,
    steps_in_sequence_window: int,
    salience_resume_path: Optional[str] = None,
    above: Optional[SalienceLevel] = None,
    num_points_to_cluster: Optional[int] = None,
    replay_resume_paths: Optional[List[str]] = None,
) -> SalienceLevel:
    replay_buffers = ReplayBufferSplits(
        env_id=env_id,
        short_term_mem_max_length=short_term_mem_max_length,
        salience_level=input_exp_lvl,
        steps_per_file=steps_per_file,
        train_to_test_collection_files=train_to_test_collection_files,
        overfit_to_short_term=should_overfit_gpt,
        replay_resume_paths=replay_resume_paths,
    )
    ret = SalienceLevel(
        input_exp_lvl=input_exp_lvl,
        replay_buffers=replay_buffers,
        seq_len=steps_in_sequence_window,
        salience_resume_path=salience_resume_path,
        above=above,
        num_points_to_cluster=num_points_to_cluster,
    )
    return ret

def unpickle(filename: str) -> Any:

    # TODO: Delete this module hack
    import sys
    sys.modules['learn_max.salience.salience_store'] = sys.modules[__name__]

    with open(filename, 'rb') as fo:
        ret = pickle.load(fo)
    return ret

def pickle_obj(obj: Any, filename: str) -> None:
    """Needs to be in the same module for imports to resolve"""
    with open(filename, 'wb') as fo:
        pickle.dump(obj, fo)

def test_salient_cluster() -> None:
    """
    Sanity test salient event object

    6 clusters
    29 core sample points
    100 points
    """

    start = time.time()
    points = unpickle(TEST_PICKLE_FILE)
    patch_diff = [p.patch_diff.flatten().cpu().numpy() for p in points]
    compute_eps = False
    if compute_eps:
        distances = scipy.spatial.distance.pdist(patch_diff)

        # This is 0.01%
        eps = np.percentile(distances, 0.01)  # type: ignore
    else:
        eps = 48.5513148217899
    db = DBSCAN(eps=eps, min_samples=2).fit(patch_diff)
    cluster_count_mp = Counter(db.labels_)
    cluster_counts = Counter(db.labels_).most_common()
    assert cluster_counts[0][0] == -1, 'Biggest cluster should be noise (outliers)'
    biggest_cluster_i = cluster_counts[1][0]
    points_biggest_cluster = []
    for i, lb in enumerate(db.labels_):
        if lb == biggest_cluster_i:
            points_biggest_cluster.append((i, points[i]))
    sc = SalientCluster(points_biggest_cluster, 0, db, salience_level=1)
    assert len(sc.core_sample_i) == len(sc.core_sample_points)
    assert len(points) - cluster_count_mp[-1] >= len(db.core_sample_indices_)
    log.success(f'Salient event tests passed in {int((time.time() - start) * 1000)}ms')

def test_salience_level() -> None:
    salience_level = salience_level_factory(
        input_exp_lvl=1,
        env_id='testy-testerson',
        short_term_mem_max_length=32,
        steps_per_file=100,
        train_to_test_collection_files=1,
        should_overfit_gpt=False,
        steps_in_sequence_window=8,
        num_points_to_cluster=69,
    )
    save_dir = salience_level.save()
    meta = unpickle(f'{save_dir}/meta.pickle')
    assert meta['level'] == 1
    assert meta['seq_len'] == 8
    assert os.path.exists(meta['replay_splits_data_dir'])

    # remove the test directory
    shutil.rmtree(save_dir)

    # make sure replay buffers data dir is not bigger than 1MB,
    # to prevent deleting real data by accident
    assert os.path.getsize(meta['replay_splits_data_dir']) < 1_000_000

    # delete the replay buffers data dir
    shutil.rmtree(meta['replay_splits_data_dir'])

test_salience_level()
test_salient_cluster()



if __name__ == '__main__':
    main()
