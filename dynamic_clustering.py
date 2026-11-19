# SPDX-License-Identifier: MIT
# Copyright (c) 2026 ytbde

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.metrics import pairwise_distances
from sklearn.cluster import AgglomerativeClustering, MiniBatchKMeans
from sklearn.decomposition import PCA
from sentence_transformers import SentenceTransformer
import warnings
import hashlib
import json


class DynamicClustering:
    """
    Maintain AdaGATE structural-screening state and accepted feature clusters.
    """

    def __init__(
            self,
            min_cluster_size: int = 3,
            max_clusters: int = 20,
            stability_threshold: float = 0.7,
            drift_window_size: int = 5,
            update_frequency: int = 10,
            embedding_model: str = "sentence-transformers/clip-ViT-B-32",
            use_semantic: bool = True,
            use_statistical: bool = True,
            semantic_weight: float = 0.25,
            statistical_weight: float = 0.75,
    ):
        """
        Args:
            min_cluster_size: Minimum number of features per cluster
            max_clusters: Maximum number of clusters
            stability_threshold: Jaccard similarity threshold for cluster stability
            drift_window_size: Window size for monitoring drift scores
            update_frequency: Minimum partitions between cluster updates
            embedding_model: Model for semantic embeddings
            use_semantic: Whether to use semantic similarity
            use_statistical: Whether to use statistical similarity
            semantic_weight: Weight for semantic similarity
            statistical_weight: Weight for statistical similarity
        """
        self.min_cluster_size = min_cluster_size
        self.max_clusters = max_clusters
        self.stability_threshold = stability_threshold
        self.drift_window_size = drift_window_size
        self.update_frequency = update_frequency

        self.use_semantic = use_semantic
        self.use_statistical = use_statistical
        self.semantic_weight = semantic_weight
        self.statistical_weight = statistical_weight

        # Latest clustering structure accepted by the system.
        self.current_clusters = None
        self.cluster_history = []
        self.drift_scores_history = []
        self.partition_count = 0
        self.last_update_partition = 0
        self.update_count = 0

        self.alpha1 = 0.20  # continue boundary for low column-change ratio
        self.alpha2 = 0.30  # warn/review boundary
        self.alpha3 = 0.80  # review/reject boundary

        self.last_stability_score = None
        self.last_change_ratio = None
        self.last_structure_action = None
        self.last_structure_status = {
            'action': 'UNINITIALIZED',
            'allow_validation': False,
            'message': 'No partition has been processed yet.'
        }
        self.last_candidate_clusters = None

        if self.use_semantic:
            try:
                self.embedding_model = SentenceTransformer(embedding_model)
            except Exception as e:
                warnings.warn(f"Failed to load embedding model: {e}. Disabling semantic similarity.")
                self.use_semantic = False
                self.embedding_model = None
        else:
            self.embedding_model = None

    def compute_similarity_matrix(
            self,
            column_stats: pd.DataFrame,
            column_names: List[str],
            column_types: Dict[str, str],
            embedding_columns: List[str] = None,
    ) -> np.ndarray:
        """
        Args:
            column_stats: DataFrame with column statistics (columns x statistics)
            column_names: List of column names
            column_types: Dictionary mapping column names to types
            embedding_columns: List of embedding column names

        Returns:
            similarity_matrix: Combined similarity matrix
        """
        n_cols = len(column_names)
        similarity_matrix = np.zeros((n_cols, n_cols))

        if self.use_statistical and column_stats is not None:
            try:
                stats_normalized = (column_stats - column_stats.mean()) / (column_stats.std() + 1e-10)
                stats_sim = 1 - pairwise_distances(stats_normalized, metric='cosine')
                similarity_matrix += self.statistical_weight * stats_sim
            except Exception as e:
                warnings.warn(f"Failed to compute statistical similarity: {e}")

        if self.use_semantic and self.embedding_model is not None:
            try:
                descriptions = []
                for col in column_names:
                    col_type = column_types.get(col, "unknown")
                    if embedding_columns and any(col.startswith(ec) for ec in embedding_columns):
                        base_col = col.rsplit("_", 1)[0] if "_" in col else col
                        if base_col in embedding_columns:
                            descriptions.append(f"embedding dimension of {base_col}")
                        else:
                            descriptions.append(f"{col} column of type {col_type}")
                    else:
                        descriptions.append(f"{col} column of type {col_type}")

                embeddings = self.embedding_model.encode(descriptions)
                semantic_sim = 1 - pairwise_distances(embeddings, metric='cosine')
                similarity_matrix += self.semantic_weight * semantic_sim
            except Exception as e:
                warnings.warn(f"Failed to compute semantic similarity: {e}")

        if similarity_matrix.max() > similarity_matrix.min():
            similarity_matrix = (similarity_matrix - similarity_matrix.min()) / \
                                (similarity_matrix.max() - similarity_matrix.min())

        np.fill_diagonal(similarity_matrix, 1.0)
        return similarity_matrix

    def compute_clusters(
            self,
            similarity_matrix: np.ndarray,
            column_names: List[str],
    ) -> Dict[str, int]:
        """
        Compute clusters from similarity matrix using hierarchical clustering.

        Args:
            similarity_matrix: Similarity matrix
            column_names: List of column names

        Returns:
            Dictionary mapping column names to cluster labels
        """
        if len(column_names) == 0:
            return {}
        if len(column_names) == 1:
            return {column_names[0]: 1}

        distance_matrix = 1 - similarity_matrix
        np.fill_diagonal(distance_matrix, 0.0)

        linkage_matrix = linkage(squareform(distance_matrix), method='average')

        n_clusters = self._determine_optimal_clusters(linkage_matrix, len(column_names))

        clusters = fcluster(linkage_matrix, n_clusters, criterion='maxclust')

        cluster_mapping = {col: int(clusters[i]) for i, col in enumerate(column_names)}

        cluster_mapping = self._enforce_min_cluster_size(cluster_mapping, similarity_matrix)

        return cluster_mapping

    def _determine_optimal_clusters(self, linkage_matrix, n_items):
        """Determine optimal number of clusters using silhouette-like heuristic."""
        max_clusters = min(self.max_clusters, n_items // self.min_cluster_size)
        if max_clusters < 2:
            return 1

        best_score = -1
        best_k = 2

        for k in range(2, max_clusters + 1):
            clusters = fcluster(linkage_matrix, k, criterion='maxclust')

            unique, counts = np.unique(clusters, return_counts=True)

            size_score = np.sum(counts >= self.min_cluster_size) / k
            balance_score = 1 - np.std(counts) / np.mean(counts) if np.mean(counts) > 0 else 0

            total_score = 0.7 * size_score + 0.3 * balance_score

            if total_score > best_score:
                best_score = total_score
                best_k = k

        return best_k

    def _enforce_min_cluster_size(self, cluster_mapping, similarity_matrix):
        """Merge small clusters with their most similar neighbors."""
        columns = list(cluster_mapping.keys())
        cluster_to_cols = {}

        for col, cluster in cluster_mapping.items():
            if cluster not in cluster_to_cols:
                cluster_to_cols[cluster] = []
            cluster_to_cols[cluster].append(col)

        small_clusters = [c for c, cols in cluster_to_cols.items()
                          if len(cols) < self.min_cluster_size and len(cols) > 0]

        for small_cluster in small_clusters:
            small_cols = cluster_to_cols[small_cluster]

            best_similarity = -1
            best_target_cluster = None

            for target_cluster, target_cols in cluster_to_cols.items():
                if target_cluster == small_cluster or len(target_cols) == 0:
                    continue

                similarities = []
                for sc in small_cols:
                    sc_idx = columns.index(sc)
                    for tc in target_cols:
                        tc_idx = columns.index(tc)
                        similarities.append(similarity_matrix[sc_idx, tc_idx])

                if similarities:
                    avg_similarity = np.mean(similarities)
                    if avg_similarity > best_similarity:
                        best_similarity = avg_similarity
                        best_target_cluster = target_cluster

            if best_target_cluster is not None:
                for col in small_cols:
                    cluster_mapping[col] = best_target_cluster
                cluster_to_cols[best_target_cluster].extend(small_cols)
                cluster_to_cols[small_cluster] = []

        unique_clusters = sorted(set(cluster_mapping.values()))
        renumbered = {old: new for new, old in enumerate(unique_clusters, 1)}

        return {col: renumbered[cluster] for col, cluster in cluster_mapping.items()}

    def compute_cluster_stability(self, new_clusters: Dict[str, int]) -> float:
        """
        Compute stability of clusters compared to current clusters.

        Args:
            new_clusters: New cluster assignment

        Returns:
            Jaccard similarity score between old and new clusters
        """
        if self.current_clusters is None:
            return 1.0

        old_sets = {}
        new_sets = {}

        for col, cluster in self.current_clusters.items():
            if cluster not in old_sets:
                old_sets[cluster] = set()
            old_sets[cluster].add(col)

        for col, cluster in new_clusters.items():
            if cluster not in new_sets:
                new_sets[cluster] = set()
            new_sets[cluster].add(col)

        total_similarity = 0
        matched_clusters = set()

        for old_cluster, old_set in old_sets.items():
            best_similarity = 0
            best_new_cluster = None

            for new_cluster, new_set in new_sets.items():
                if new_cluster in matched_clusters:
                    continue

                jaccard = len(old_set & new_set) / len(old_set | new_set) if (old_set | new_set) else 0
                if jaccard > best_similarity:
                    best_similarity = jaccard
                    best_new_cluster = new_cluster

            if best_new_cluster is not None:
                matched_clusters.add(best_new_cluster)
                total_similarity += best_similarity

        avg_similarity = total_similarity / len(old_sets) if old_sets else 0

        return avg_similarity

    def should_update_clusters(
            self,
            drift_score: float,
            normalized_summaries: pd.DataFrame,
            column_names: List[str],
    ) -> bool:
        """Evaluate the AdaGATE structural-risk decision for one partition.

        Returns:
            True: downstream automatic validation may continue.
            False: the partition is routed to REVIEW or REJECT.
        """
        self.partition_count += 1
        self.drift_scores_history.append(drift_score)

        if len(self.drift_scores_history) > self.drift_window_size:
            self.drift_scores_history.pop(0)

        # rho_t compares the current schema with the latest accepted schema.
        previous_cols = set(self.current_clusters.keys()) if self.current_clusters else set()
        current_cols = set(column_names)
        union_cols = previous_cols | current_cols
        rho_t = len(previous_cols.symmetric_difference(current_cols)) / len(union_cols) if union_cols else 0.0
        self.last_change_ratio = rho_t
        self.last_candidate_clusters = None
        self.last_stability_score = None

        # Initialization builds the first accepted clustering candidate.
        if self.current_clusters is None:
            column_stats = self._extract_column_stats(normalized_summaries, column_names)
            column_types = self._infer_column_types(normalized_summaries, column_names)
            similarity_matrix = self.compute_similarity_matrix(column_stats, column_names, column_types)
            self.last_candidate_clusters = self.compute_clusters(similarity_matrix, column_names)
            self.last_stability_score = 1.0
            self.last_structure_action = 'INITIALIZE_AND_CONTINUE'
            self.last_structure_status = {
                'action': self.last_structure_action,
                'allow_validation': True,
                'stability': 1.0,
                'stability_threshold': self.stability_threshold,
                'change_ratio': rho_t,
                'drift_score': drift_score,
                'partition': self.partition_count,
                'message': 'Initial partition: initialize clustering and continue validation.',
            }
            return True

        # Unchanged schema: skip gamma_t and reuse the accepted clustering state.
        if rho_t == 0:
            self.last_structure_action = 'CONTINUE_NO_SCHEMA_CHANGE'
            self.last_structure_status = {
                'action': self.last_structure_action,
                'allow_validation': True,
                'stability': None,
                'stability_threshold': self.stability_threshold,
                'change_ratio': rho_t,
                'drift_score': drift_score,
                'partition': self.partition_count,
                'message': 'rho_t = 0: schema unchanged; skip gamma_t and continue validation.',
            }
            return True

        # Low-ratio evolution: compute C^t and gamma_t for structural stability.
        if 0 < rho_t < self.alpha1:
            column_stats = self._extract_column_stats(normalized_summaries, column_names)
            column_types = self._infer_column_types(normalized_summaries, column_names)
            similarity_matrix = self.compute_similarity_matrix(column_stats, column_names, column_types)
            new_clusters = self.compute_clusters(similarity_matrix, column_names)
            self.last_candidate_clusters = new_clusters
            gamma_t = self.compute_cluster_stability(new_clusters)
            self.last_stability_score = gamma_t

            if gamma_t >= self.stability_threshold:
                self.last_structure_action = 'CONTINUE_STABLE_LOW_CHANGE'
                allow_validation = True
                message = (
                    f'0 < rho_t={rho_t:.4f} < alpha1={self.alpha1:.4f}, and '
                    f'gamma_t={gamma_t:.4f} >= theta_s={self.stability_threshold:.4f}; '
                    f'use current clustering and continue validation.'
                )
            else:
                self.last_structure_action = 'REVIEW_LOW_STABILITY'
                allow_validation = False
                message = (
                    f'0 < rho_t={rho_t:.4f} < alpha1={self.alpha1:.4f}, but '
                    f'gamma_t={gamma_t:.4f} < theta_s={self.stability_threshold:.4f}; '
                    f'low-ratio column changes caused strong feature reorganization. '
                    f'Mark this partition as REVIEW.'
                )

            self.last_structure_status = {
                'action': self.last_structure_action,
                'allow_validation': allow_validation,
                'stability': gamma_t,
                'stability_threshold': self.stability_threshold,
                'change_ratio': rho_t,
                'alpha1': self.alpha1,
                'alpha2': self.alpha2,
                'alpha3': self.alpha3,
                'drift_score': drift_score,
                'partition': self.partition_count,
                'message': message,
            }
            return allow_validation

        # Moderate evolution: WARN and continue without computing gamma_t.
        if self.alpha1 <= rho_t < self.alpha2:
            self.last_structure_action = 'WARN_SCHEMA_CHANGE'
            self.last_structure_status = {
                'action': self.last_structure_action,
                'allow_validation': True,
                'stability': None,
                'stability_threshold': self.stability_threshold,
                'change_ratio': rho_t,
                'alpha1': self.alpha1,
                'alpha2': self.alpha2,
                'alpha3': self.alpha3,
                'drift_score': drift_score,
                'partition': self.partition_count,
                'warning': True,
                'message': (
                    f'alpha1={self.alpha1:.4f} <= rho_t={rho_t:.4f} < alpha2={self.alpha2:.4f}; '
                    f'continue validation with warning prompt. '
                ),
            }
            return True

        # High evolution requires manual review.
        if self.alpha2 <= rho_t < self.alpha3:
            self.last_structure_action = 'REVIEW_HIGH_SCHEMA_CHANGE'
            self.last_structure_status = {
                'action': self.last_structure_action,
                'allow_validation': False,
                'stability': None,
                'stability_threshold': self.stability_threshold,
                'change_ratio': rho_t,
                'alpha1': self.alpha1,
                'alpha2': self.alpha2,
                'alpha3': self.alpha3,
                'drift_score': drift_score,
                'partition': self.partition_count,
                'message': (
                    f'alpha2={self.alpha2:.4f} <= rho_t={rho_t:.4f} < alpha3={self.alpha3:.4f}; '
                    f'mark as REVIEW and pause automatic validation.'
                ),
            }
            return False

        # Extremely high evolution is rejected from automatic validation.
        self.last_structure_action = 'REJECT_EXTREME_SCHEMA_CHANGE'
        self.last_structure_status = {
            'action': self.last_structure_action,
            'allow_validation': False,
            'stability': None,
            'stability_threshold': self.stability_threshold,
            'change_ratio': rho_t,
            'alpha1': self.alpha1,
            'alpha2': self.alpha2,
            'alpha3': self.alpha3,
            'drift_score': drift_score,
            'partition': self.partition_count,
            'message': (
                f'rho_t={rho_t:.4f} >= alpha3={self.alpha3:.4f}; reject this partition.'
            ),
        }
        return False

    def _extract_column_stats(self, normalized_summaries, column_names):
        """Extract column statistics from normalized summaries."""
        try:
            pivoted = normalized_summaries.pivot_table(
                index='column',
                columns='statistic',
                values='value'
            ).fillna(0)

            missing_cols = set(column_names) - set(pivoted.index)
            if missing_cols:
                for col in missing_cols:
                    pivoted.loc[col] = 0

            return pivoted.loc[column_names] if column_names else None
        except Exception as e:
            warnings.warn(f"Failed to extract column stats: {e}")
            return None

    def _infer_column_types(self, normalized_summaries, column_names):
        """Infer column types from column names and statistics."""
        column_types = {}

        for col in column_names:
            if any(x in col.lower() for x in ['embed', 'vec', 'dim']):
                column_types[col] = 'embedding'
            elif col.endswith('_id') or col in ['id', 'index', 'key']:
                column_types[col] = 'id'
            elif any(x in col.lower() for x in ['date', 'time', 'year', 'month']):
                column_types[col] = 'timestamp'
            elif any(x in col.lower() for x in ['count', 'total', 'sum', 'num']):
                column_types[col] = 'integer'
            elif any(x in col.lower() for x in ['rate', 'ratio', 'prob', 'score']):
                column_types[col] = 'float'
            elif any(x in col.lower() for x in ['name', 'text', 'desc', 'label']):
                column_types[col] = 'string'
            else:
                column_types[col] = 'unknown'

        return column_types

    def update_clusters(
            self,
            normalized_summaries: pd.DataFrame,
            column_names: List[str],
            embedding_columns: List[str] = None,
            force_update: bool = False,
    ) -> Dict[str, int]:
        """Update the system's accepted clustering state after structural screening.
        """
        # First determine the structural action; then update accepted state if allowed.
        allow_validation = self.should_update_clusters(0.0, normalized_summaries, column_names)
        action = self.last_structure_action
        old_clusters = self.current_clusters.copy() if self.current_clusters is not None else None
        accepted_new_clusters = None

        if action in {'INITIALIZE_AND_CONTINUE', 'CONTINUE_STABLE_LOW_CHANGE'}:
            accepted_new_clusters = self.last_candidate_clusters

        # No schema change: keep exactly the latest accepted clustering.
        elif action == 'CONTINUE_NO_SCHEMA_CHANGE':
            accepted_new_clusters = self.current_clusters

        # WARN uses a newly computed clustering but deliberately skips gamma_t.
        elif action == 'WARN_SCHEMA_CHANGE':
            column_stats = self._extract_column_stats(normalized_summaries, column_names)
            column_types = self._infer_column_types(normalized_summaries, column_names)
            similarity_matrix = self.compute_similarity_matrix(column_stats, column_names, column_types, embedding_columns)
            accepted_new_clusters = self.compute_clusters(similarity_matrix, column_names)
            self.last_candidate_clusters = accepted_new_clusters

        else:
            accepted_new_clusters = self.current_clusters

        # Only processable partitions may replace the accepted clustering state.
        if allow_validation and accepted_new_clusters is not None:
            self.current_clusters = accepted_new_clusters
            self.last_update_partition = self.partition_count
            self.update_count += 1

        self.cluster_history.append({
            'partition': self.partition_count,
            'old_clusters': old_clusters,
            'clusters': self.current_clusters.copy() if self.current_clusters is not None else None,
            'candidate_clusters': self.last_candidate_clusters.copy() if self.last_candidate_clusters is not None else None,
            'stability': self.last_stability_score,
            'change_ratio': self.last_change_ratio,
            'action': self.last_structure_action,
            'allow_validation': allow_validation,
        })

        print(f"[DynamicClustering] Partition {self.partition_count}")
        print(f"[DynamicClustering] rho_t: {self.last_change_ratio:.4f}")
        print(f"[DynamicClustering] Action: {self.last_structure_action}")
        if self.last_stability_score is not None:
            print(f"[DynamicClustering] gamma_t: {self.last_stability_score:.4f}")
        if self.current_clusters is not None:
            print(f"[DynamicClustering] Number of accepted clusters: {len(set(self.current_clusters.values()))}")

        if not allow_validation:
            warnings.warn(
                f"{self.last_structure_action}: this partition should not proceed to "
                f"anomaly-matrix construction or drift detection unless manually approved."
            )

        return self.current_clusters if self.current_clusters else {}

    def get_cluster_mapping(self) -> Dict[str, int]:
        """Get current cluster mapping."""
        return self.current_clusters if self.current_clusters else {}

    def get_cluster_history(self) -> List[Dict]:
        """Get cluster update history."""
        return self.cluster_history

    def get_last_structure_status(self) -> Dict:
        """Get the latest structural-risk decision."""
        return self.last_structure_status

    def reset(self):
        """Reset the dynamic clustering state."""
        self.current_clusters = None
        self.cluster_history = []
        self.drift_scores_history = []
        self.partition_count = 0
        self.last_update_partition = 0
        self.update_count = 0
        self.last_stability_score = None
        self.last_change_ratio = None
        self.last_structure_action = None
        self.last_structure_status = {
            'action': 'UNINITIALIZED',
            'allow_validation': False,
            'message': 'No partition has been processed yet.'
        }
        self.last_candidate_clusters = None

    def save_state(self, filepath: str):
        """Save clustering state to file."""
        state = {
            'current_clusters': self.current_clusters,
            'cluster_history': self.cluster_history,
            'drift_scores_history': self.drift_scores_history,
            'partition_count': self.partition_count,
            'last_update_partition': self.last_update_partition,
            'update_count': self.update_count,
            'last_stability_score': self.last_stability_score,
            'last_change_ratio': self.last_change_ratio,
            'last_structure_action': self.last_structure_action,
            'last_structure_status': self.last_structure_status,
            'config': {
                'min_cluster_size': self.min_cluster_size,
                'max_clusters': self.max_clusters,
                'stability_threshold': self.stability_threshold,
                'drift_window_size': self.drift_window_size,
                'update_frequency': self.update_frequency,
                'alpha1': self.alpha1,
                'alpha2': self.alpha2,
                'alpha3': self.alpha3,
            }
        }

        with open(filepath, 'w') as f:
            json.dump(state, f, default=str)

    def load_state(self, filepath: str):
        """Load clustering state from file."""
        with open(filepath, 'r') as f:
            state = json.load(f)

        current = state.get('current_clusters')
        self.current_clusters = {k: int(v) for k, v in current.items()} if current else None
        self.cluster_history = state.get('cluster_history', [])
        self.drift_scores_history = state.get('drift_scores_history', [])
        self.partition_count = state.get('partition_count', 0)
        self.last_update_partition = state.get('last_update_partition', 0)
        self.update_count = state.get('update_count', 0)
        self.last_stability_score = state.get('last_stability_score')
        self.last_change_ratio = state.get('last_change_ratio')
        self.last_structure_action = state.get('last_structure_action')
        self.last_structure_status = state.get('last_structure_status', {
            'action': 'UNINITIALIZED',
            'allow_validation': False,
            'message': 'No partition has been processed yet.'
        })
        config = state.get('config', {})
        self.alpha1 = config.get('alpha1', getattr(self, 'alpha1', 0.20))
        self.alpha2 = config.get('alpha2', getattr(self, 'alpha2', 0.30))
        self.alpha3 = config.get('alpha3', getattr(self, 'alpha3', 0.80))
        self.last_candidate_clusters = None

    def decide_data_action(self, df, historical_df, change_ratio):
        quality_score = self._compute_quality_score(df)
        status = self.last_structure_status or {}
        action = status.get('action')
        allow_validation = status.get('allow_validation', True)

        if not allow_validation:
            if action == 'REJECT_EXTREME_SCHEMA_CHANGE':
                return {
                    'action': 'DROP_TABLE',
                    'message': f'ρ={change_ratio:.1%}, over α3={self.alpha3:.1%}, determined to be extremely high structural risk, partition discarded',
                    'quality_score': quality_score,
                    'should_retrain': False,
                    'stability': status.get('stability'),
                    'change_ratio': change_ratio,
                }
            return {
                'action': 'PAUSE_PIPELINE_AND_MANUAL_REVIEW',
                'message': status.get(
                    'message',
                    f'ρ={change_ratio:.1%},structural risk high, suspend subsequent construction of the abnormal matrix and drift detection'
                ),
                'quality_score': quality_score,
                'should_retrain': False,
                'stability': status.get('stability'),
                'change_ratio': change_ratio,
            }

        if action == 'WARN_SCHEMA_CHANGE' or (self.alpha1 <= change_ratio < self.alpha2):
            return {
                'action': 'CONTINUE_WITH_WARNING',
                'message': f'⚠️ ρ={change_ratio:.1%}, warn ,continue auto validation,but addicted warn info',
                'quality_score': quality_score,
                'should_retrain': True,
                'warning': True,
                'stability': status.get('stability'),
                'change_ratio': change_ratio,
            }

        return {
            'action': 'CONTINUE_VALIDATION',
            'message': status.get(
                'message',
                f'ρ={change_ratio:.1%},struct risk accepted, continue construct anomaly matrix and drift detection'
            ),
            'quality_score': quality_score,
            'should_retrain': True,
            'stability': status.get('stability'),
            'change_ratio': change_ratio,
        }

    def _compute_quality_score(self, df):
        scores = []

        missing_rate = df.isnull().sum().sum() / (df.shape[0] * df.shape[1])
        scores.append(1 - missing_rate)

        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            q1 = df[col].quantile(0.25)
            q3 = df[col].quantile(0.75)
            iqr = q3 - q1
            outlier_rate = ((df[col] < q1 - 1.5 * iqr) | (df[col] > q3 + 1.5 * iqr)).mean()
            scores.append(1 - outlier_rate)

        return np.mean(scores) if scores else 0.5
