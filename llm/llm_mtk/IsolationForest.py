"""PyTorch Isolation Forest used by MTK.

This is the repository implementation with a version-independent fitted-state
check.  The released implementation called scikit-learn's ``check_is_fitted``
on a plain ``torch.nn.Module``; recent scikit-learn versions reject that even
after the forest has been fitted.
"""

from __future__ import annotations

import numpy as np
import torch
from sklearn.utils import check_random_state
from torch import nn
from tqdm import tqdm


EULER_GAMMA = np.euler_gamma


def _average_path_length(n_samples_leaf):
    if isinstance(n_samples_leaf, (int, float)):
        n_samples_leaf = np.array([n_samples_leaf])
    elif isinstance(n_samples_leaf, torch.Tensor):
        n_samples_leaf = n_samples_leaf.detach().cpu().numpy()
    values = np.asarray(n_samples_leaf)
    original_shape = values.shape
    values = values.reshape(1, -1)
    result = np.zeros(values.shape)
    one_or_less = values <= 1
    exactly_two = values == 2
    remaining = ~(one_or_less | exactly_two)
    result[one_or_less] = 0.0
    result[exactly_two] = 1.0
    result[remaining] = (
        2.0 * (np.log(values[remaining] - 1.0) + EULER_GAMMA)
        - 2.0 * (values[remaining] - 1.0) / values[remaining]
    )
    return result.reshape(original_shape)


class PyTorchIsolationForest(nn.Module):
    """Compatibility-preserving copy of MTK's custom Isolation Forest."""

    def __init__(
        self,
        n_estimators=100,
        max_samples="auto",
        contamination="auto",
        max_features=1.0,
        bootstrap=False,
        random_state=None,
        verbose=0,
        warm_start=False,
    ):
        super().__init__()
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.contamination = contamination
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.random_state = random_state
        self.verbose = verbose
        self.warm_start = warm_start
        self.estimators_ = []
        self.estimators_features_ = []
        self.estimators_samples_ = []
        self.max_samples_ = None
        self.offset_ = None
        self.n_features_in_ = None
        self._max_features = None
        self._average_path_length_per_tree = []
        self._decision_path_lengths = []
        self.feature_importances_ = None
        self._compiled_trees = None
        self.random_state_ = check_random_state(random_state)

    def _check_fitted(self):
        # Do not use sklearn.check_is_fitted here.  This class intentionally is
        # a torch Module, not a scikit-learn BaseEstimator.
        if self.max_samples_ is None or self.offset_ is None or not self.estimators_:
            raise RuntimeError("PyTorchIsolationForest is not fitted")

    def fit(self, X, y=None, sample_weight=None):
        del y, sample_weight
        if not isinstance(X, torch.Tensor):
            X = torch.as_tensor(X, dtype=torch.float32)
        if X.dim() != 2:
            raise ValueError(f"X must be two-dimensional, got {X.dim()} dimensions")
        self.n_samples_, self.n_features_in_ = X.shape
        if self.max_samples == "auto":
            self.max_samples_ = min(256, self.n_samples_)
        elif isinstance(self.max_samples, int):
            self.max_samples_ = min(self.max_samples, self.n_samples_)
        else:
            self.max_samples_ = int(self.max_samples * self.n_samples_)
        self.max_samples_ = max(1, self.max_samples_)
        if isinstance(self.max_features, int):
            self._max_features = self.max_features
        else:
            self._max_features = max(1, int(self.max_features * self.n_features_in_))
        max_depth = int(np.ceil(np.log2(max(self.max_samples_, 2))))
        if not self.warm_start or not self.estimators_:
            self.estimators_ = []
            self.estimators_features_ = []
            self.estimators_samples_ = []
            self._average_path_length_per_tree = []
            self._decision_path_lengths = []
        for _ in tqdm(
            range(self.n_estimators - len(self.estimators_)),
            desc="Building isolation trees",
            disable=self.verbose == 0,
        ):
            sample_indices = self.random_state_.choice(
                self.n_samples_, self.max_samples_, replace=self.bootstrap
            )
            sample_indices = torch.as_tensor(sample_indices, device=X.device)
            X_sample = X[sample_indices]
            if self._max_features < self.n_features_in_:
                feature_indices = self.random_state_.choice(
                    self.n_features_in_, self._max_features, replace=False
                )
                feature_indices = torch.as_tensor(feature_indices, device=X.device)
            else:
                feature_indices = torch.arange(self.n_features_in_, device=X.device)
            tree = self._build_tree(X_sample, max_depth, feature_indices)
            self.estimators_.append(tree)
            self.estimators_features_.append(feature_indices)
            self.estimators_samples_.append(sample_indices)
            self._average_path_length_per_tree.append(
                _average_path_length(tree["node_samples"])
            )
            self._decision_path_lengths.append(tree["node_depths"])
        self._compute_feature_importances()
        self._compile_trees(X.device)
        if self.contamination == "auto":
            self.offset_ = -0.5
        else:
            # score_samples checks fitted state; all trees are available now,
            # so use a temporary offset while deriving the requested quantile.
            self.offset_ = 0.0
            scores = self.score_samples(X)
            self.offset_ = np.percentile(
                scores.detach().cpu().numpy(), 100.0 * self.contamination
            )
        return self

    def _build_tree(self, X, max_depth, feature_indices):
        n_samples, _ = X.shape
        root = {
            "left": None, "right": None, "feature": None, "threshold": None,
            "is_leaf": False, "node_id": 0, "depth": 0,
        }
        node_samples, node_depths = [], []
        queue, next_id = [root], 1
        while queue:
            node = queue.pop(0)
            depth, node_id = node["depth"], node["node_id"]
            count = n_samples if node_id == 0 else node["sample_count"]
            node_samples.append(count)
            node_depths.append(depth)
            if count <= 1 or depth >= max_depth:
                node["is_leaf"] = True
                continue
            if len(feature_indices) < self.n_features_in_:
                feature = feature_indices[
                    self.random_state_.choice(len(feature_indices))
                ].item()
            else:
                feature = self.random_state_.choice(self.n_features_in_)
            node["feature"] = feature
            values = X[:, feature]
            unique = torch.unique(values)
            if len(unique) == 1:
                node["is_leaf"] = True
                continue
            if len(unique) == 2:
                split = (unique[0] + unique[1]) / 2
            else:
                pos = self.random_state_.choice(len(unique) - 1)
                split = (unique[pos] + unique[pos + 1]) / 2
            node["threshold"] = split.item()
            if node_id == 0:
                left_mask = values < split
            else:
                left_mask = node["parent_mask"] & (values < split)
            right_mask = (torch.ones_like(left_mask) if node_id == 0 else node["parent_mask"]) & ~left_mask
            for side, mask in (("left", left_mask), ("right", right_mask)):
                child = {
                    "left": None, "right": None, "feature": None, "threshold": None,
                    "is_leaf": False, "node_id": next_id, "depth": depth + 1,
                    "sample_count": mask.sum().item(), "parent_mask": mask,
                }
                next_id += 1
                node[side] = child
                queue.append(child)
        return {
            "tree": root,
            "node_samples": np.asarray(node_samples),
            "node_depths": np.asarray(node_depths),
        }

    def _compute_feature_importances(self):
        counts = torch.zeros(self.n_features_in_)
        for tree in self.estimators_:
            queue = [tree["tree"]]
            while queue:
                node = queue.pop(0)
                if node["is_leaf"]:
                    continue
                counts[node["feature"]] += 1
                queue.extend([node["left"], node["right"]])
        total = counts.sum()
        self.feature_importances_ = (
            counts.numpy() / total.numpy() if total else counts.numpy()
        )

    def _compile_trees(self, device):
        """Compile the existing Python trees for equivalent parallel traversal."""
        tree_count = len(self.estimators_)
        max_nodes = max(len(tree["node_depths"]) for tree in self.estimators_)
        feature = torch.zeros((tree_count, max_nodes), dtype=torch.long, device=device)
        threshold = torch.zeros((tree_count, max_nodes), dtype=torch.float32, device=device)
        left = torch.zeros((tree_count, max_nodes), dtype=torch.long, device=device)
        right = torch.zeros((tree_count, max_nodes), dtype=torch.long, device=device)
        is_leaf = torch.ones((tree_count, max_nodes), dtype=torch.bool, device=device)
        path_length = torch.zeros((tree_count, max_nodes), dtype=torch.float32, device=device)

        for tree_index, (tree, selected_features) in enumerate(
            zip(self.estimators_, self.estimators_features_)
        ):
            nodes = [tree["tree"]]
            while nodes:
                node = nodes.pop(0)
                node_id = node["node_id"]
                is_leaf[tree_index, node_id] = node["is_leaf"]
                if not node["is_leaf"]:
                    original_feature = node["feature"]
                    if len(selected_features) < self.n_features_in_:
                        # Match _apply_tree's historical feature-position rule.
                        positions = torch.where(selected_features == original_feature)[0]
                        feature[tree_index, node_id] = positions[0] if len(positions) else 0
                    else:
                        feature[tree_index, node_id] = original_feature
                    threshold[tree_index, node_id] = node["threshold"]
                    left[tree_index, node_id] = node["left"]["node_id"]
                    right[tree_index, node_id] = node["right"]["node_id"]
                    nodes.extend([node["left"], node["right"]])
            node_count = len(tree["node_depths"])
            path_length[tree_index, :node_count] = torch.as_tensor(
                tree["node_depths"]
                + self._average_path_length_per_tree[tree_index]
                - 1.0,
                dtype=torch.float32,
                device=device,
            )
        self._compiled_trees = {
            "feature": feature,
            "threshold": threshold,
            "left": left,
            "right": right,
            "is_leaf": is_leaf,
            "path_length": path_length,
            "max_depth": int(max(tree["node_depths"].max() for tree in self.estimators_)),
        }

    def _apply_tree(self, x, tree_struct, features):
        node = tree_struct["tree"]
        while not node["is_leaf"]:
            feature = node["feature"]
            if features is not None and len(features) < self.n_features_in_:
                positions = torch.where(features == feature)[0]
                if len(positions) == 0:
                    return 0
                value = x[positions.item()]
            else:
                value = x[feature]
            node = node["left"] if value < node["threshold"] else node["right"]
        return node["node_id"]

    def decision_function(self, X):
        self._check_fitted()
        return self.score_samples(X) - self.offset_

    def score_samples(self, X):
        self._check_fitted()
        if not isinstance(X, torch.Tensor):
            X = torch.as_tensor(X, dtype=torch.float32)
        if X.dim() == 1:
            X = X.unsqueeze(0)
        if self._compiled_trees is None:
            self._compile_trees(X.device)
        elif self._compiled_trees["feature"].device != X.device:
            self._compiled_trees = {
                key: value.to(X.device) if isinstance(value, torch.Tensor) else value
                for key, value in self._compiled_trees.items()
            }
        compiled = self._compiled_trees
        tree_count = len(self.estimators_)
        batch_size = X.shape[0]
        tree_indices = torch.arange(tree_count, device=X.device).unsqueeze(0)
        tree_indices = tree_indices.expand(batch_size, -1)
        nodes = torch.zeros((batch_size, tree_count), dtype=torch.long, device=X.device)
        for _ in range(compiled["max_depth"] + 1):
            active = ~compiled["is_leaf"][tree_indices, nodes]
            if not bool(active.any()):
                break
            access_features = compiled["feature"][tree_indices, nodes]
            values = X.gather(1, access_features)
            go_left = values < compiled["threshold"][tree_indices, nodes]
            next_nodes = torch.where(
                go_left,
                compiled["left"][tree_indices, nodes],
                compiled["right"][tree_indices, nodes],
            )
            nodes = torch.where(active, next_nodes, nodes)
        per_tree_depths = compiled["path_length"][tree_indices, nodes]
        depths = torch.zeros(batch_size, dtype=torch.float32, device=X.device)
        for tree_index in range(tree_count):
            depths += per_tree_depths[:, tree_index]
        normalizer = _average_path_length(self.max_samples_)[0]
        denominator = tree_count * normalizer
        normalized = depths / denominator if denominator else torch.ones_like(depths)
        return -(2.0 ** (-normalized))

    def _score_samples_scalar(self, X):
        """Original traversal retained as an equivalence oracle for tests."""
        self._check_fitted()
        if not isinstance(X, torch.Tensor):
            X = torch.as_tensor(X, dtype=torch.float32)
        if X.dim() == 1:
            X = X.unsqueeze(0)
        depths = torch.zeros(X.shape[0], device=X.device)
        normalizer = _average_path_length(self.max_samples_)[0]
        for index, (tree, features) in enumerate(
            zip(self.estimators_, self.estimators_features_)
        ):
            for row in range(X.shape[0]):
                leaf = self._apply_tree(X[row], tree, features)
                depths[row] += (
                    tree["node_depths"][leaf]
                    + self._average_path_length_per_tree[index][leaf]
                    - 1.0
                )
        denominator = len(self.estimators_) * normalizer
        normalized = depths / denominator if denominator else torch.ones_like(depths)
        return -(2.0 ** (-normalized))

    def predict(self, X):
        return torch.where(self.decision_function(X) < 0, -1, 1)
