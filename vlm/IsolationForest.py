import torch
import numpy as np
from torch import nn
from tqdm import tqdm
from sklearn.utils import check_random_state
from sklearn.utils.validation import check_is_fitted

EULER_GAMMA = np.euler_gamma

def _average_path_length(n_samples_leaf):
    if isinstance(n_samples_leaf, (int, float)):
        n_samples_leaf = np.array([n_samples_leaf])
    elif isinstance(n_samples_leaf, torch.Tensor):
        n_samples_leaf = n_samples_leaf.cpu().numpy()

    n_samples_leaf = np.asarray(n_samples_leaf)
    n_samples_leaf_shape = n_samples_leaf.shape
    n_samples_leaf = n_samples_leaf.reshape((1, -1))
    average_path_length = np.zeros(n_samples_leaf.shape)

    mask_1 = n_samples_leaf <= 1
    mask_2 = n_samples_leaf == 2
    not_mask = ~np.logical_or(mask_1, mask_2)

    average_path_length[mask_1] = 0.0
    average_path_length[mask_2] = 1.0
    average_path_length[not_mask] = (
        2.0 * (np.log(n_samples_leaf[not_mask] - 1.0) + EULER_GAMMA)
        - 2.0 * (n_samples_leaf[not_mask] - 1.0) / n_samples_leaf[not_mask]
    )

    return average_path_length.reshape(n_samples_leaf_shape)

class PyTorchIsolationForest(nn.Module):
    def __init__(self,
                 n_estimators=100,
                 max_samples='auto',
                 contamination='auto',
                 max_features=1.0,
                 bootstrap=False,
                 random_state=None,
                 verbose=0,
                 warm_start=False):
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

        self.random_state_ = check_random_state(random_state)

    def fit(self, X, y=None, sample_weight=None):
        if not isinstance(X, torch.Tensor):
            X = torch.tensor(X, dtype=torch.float32)

        if X.dim() != 2:
            raise ValueError(f"Input must be 2D tensor, current dimensions: {X.dim()}")

        self.n_samples_, self.n_features_in_ = X.shape

        if self.max_samples == 'auto':
            self.max_samples_ = min(256, self.n_samples_)
        elif isinstance(self.max_samples, int):
            if self.max_samples > self.n_samples_:
                print(f"Warning: max_samples ({self.max_samples}) > total samples ({self.n_samples_}), using all samples")
                self.max_samples_ = self.n_samples_
            else:
                self.max_samples_ = self.max_samples
        else:
            self.max_samples_ = int(self.max_samples * self.n_samples_)

        self.max_samples_ = max(1, self.max_samples_)

        if isinstance(self.max_features, int):
            self._max_features = self.max_features
        else:
            self._max_features = max(1, int(self.max_features * self.n_features_in_))

        max_depth = int(np.ceil(np.log2(max(self.max_samples_, 2))))

        if not self.warm_start or len(self.estimators_) == 0:
            self.estimators_ = []
            self.estimators_features_ = []
            self.estimators_samples_ = []
            self._average_path_length_per_tree = []
            self._decision_path_lengths = []

        n_more_estimators = self.n_estimators - len(self.estimators_)
        if n_more_estimators <= 0:
            return self

        for _ in tqdm(range(n_more_estimators), desc="Building isolation trees", disable=self.verbose == 0):
            if self.bootstrap:
                sample_indices = self.random_state_.choice(
                    self.n_samples_, size=self.max_samples_, replace=True)
            else:
                sample_indices = self.random_state_.choice(
                    self.n_samples_, size=self.max_samples_, replace=False)

            sample_indices = torch.tensor(sample_indices, device=X.device)
            X_sample = X[sample_indices]

            if self._max_features < self.n_features_in_:
                feature_indices = self.random_state_.choice(
                    self.n_features_in_, size=self._max_features, replace=False)
                feature_indices = torch.tensor(feature_indices, device=X.device)
            else:
                feature_indices = torch.arange(self.n_features_in_, device=X.device)

            tree = self._build_tree(X_sample, max_depth, feature_indices)
            self.estimators_.append(tree)
            self.estimators_features_.append(feature_indices)
            self.estimators_samples_.append(sample_indices)

            node_samples = tree['node_samples']
            avg_path_length = _average_path_length(node_samples)
            decision_path_lengths = tree['node_depths']

            self._average_path_length_per_tree.append(avg_path_length)
            self._decision_path_lengths.append(decision_path_lengths)

        self._compute_feature_importances()

        if self.contamination == 'auto':
            self.offset_ = 0.5
        else:
            scores = self.score_samples(X)
            self.offset_ = np.percentile(scores.cpu().numpy(), 100.0 * (1.0 - self.contamination))

        return self

    def _build_tree(self, X, max_depth, feature_indices):
        n_samples, n_features = X.shape

        root = {
            'left': None,
            'right': None,
            'feature': None,
            'threshold': None,
            'is_leaf': False,
            'node_id': 0,
            'depth': 0
        }

        node_samples = []
        node_depths = []
        node_queue = [root]
        next_node_id = 1

        while node_queue:
            node = node_queue.pop(0)
            current_depth = node['depth']
            current_node_id = node['node_id']

            if current_node_id == 0:
                current_samples_mask = torch.ones(n_samples, dtype=torch.bool, device=X.device)
                current_samples_count = n_samples
            else:
                current_samples_count = node['sample_count']

            node_samples.append(current_samples_count)
            node_depths.append(current_depth)

            if current_samples_count <= 1 or current_depth >= max_depth:
                node['is_leaf'] = True
                continue

            if len(feature_indices) < self.n_features_in_:
                feature_idx = self.random_state_.choice(len(feature_indices))
                feature_idx = feature_indices[feature_idx].item()
            else:
                feature_idx = self.random_state_.choice(self.n_features_in_)

            node['feature'] = feature_idx

            feature_vals = X[:, feature_idx]

            unique_vals = torch.unique(feature_vals)

            if len(unique_vals) == 1:
                node['is_leaf'] = True
                continue

            if len(unique_vals) == 2:
                split_val = (unique_vals[0] + unique_vals[1]) / 2
            else:
                split_pos = self.random_state_.choice(len(unique_vals) - 1)
                split_val = (unique_vals[split_pos] + unique_vals[split_pos + 1]) / 2

            node['threshold'] = split_val.item()

            if current_node_id == 0:
                left_mask = feature_vals < split_val
                right_mask = ~left_mask
            else:
                parent_mask = node['parent_mask']
                left_mask = parent_mask & (feature_vals < split_val)
                right_mask = parent_mask & ~left_mask

            left_count = left_mask.sum().item()
            right_count = right_mask.sum().item()

            node['left'] = {
                'left': None,
                'right': None,
                'feature': None,
                'threshold': None,
                'is_leaf': False,
                'node_id': next_node_id,
                'depth': current_depth + 1,
                'sample_count': left_count,
                'parent_mask': left_mask if current_node_id == 0 else left_mask
            }
            next_node_id += 1

            node['right'] = {
                'left': None,
                'right': None,
                'feature': None,
                'threshold': None,
                'is_leaf': False,
                'node_id': next_node_id,
                'depth': current_depth + 1,
                'sample_count': right_count,
                'parent_mask': right_mask if current_node_id == 0 else right_mask
            }
            next_node_id += 1

            node_queue.append(node['left'])
            node_queue.append(node['right'])

        return {
            'tree': root,
            'node_samples': np.array(node_samples),
            'node_depths': np.array(node_depths)
        }

    def _compute_feature_importances(self):
        if self.n_features_in_ is None:
            return

        feature_counts = torch.zeros(self.n_features_in_)

        for tree in self.estimators_:
            queue = [tree['tree']]
            while queue:
                node = queue.pop(0)
                if node['is_leaf']:
                    continue

                feature_counts[node['feature']] += 1

                if node['left']:
                    queue.append(node['left'])
                if node['right']:
                    queue.append(node['right'])

        self.feature_importances_ = feature_counts.numpy() / feature_counts.sum().numpy()

    def _apply_tree(self, x, tree_struct, features):
        node = tree_struct['tree']

        while not node['is_leaf']:
            feature = node['feature']
            if features is not None and len(features) < self.n_features_in_:
                feature_pos = torch.where(features == feature)[0]
                if len(feature_pos) == 0:
                    return 0
                feature_pos = feature_pos.item()
                val = x[feature_pos]
            else:
                val = x[feature]

            if val < node['threshold']:
                node = node['left']
            else:
                node = node['right']

        return node['node_id']

    def decision_function(self, X):
        check_is_fitted(self)
        return self.score_samples(X) - self.offset_

    def score_samples(self, X):
        check_is_fitted(self)

        if not isinstance(X, torch.Tensor):
            X = torch.tensor(X, dtype=torch.float32, device=self.estimators_[0]['tree']['node_id'])

        if X.dim() == 1:
            X = X.unsqueeze(0)

        n_samples = X.shape[0]
        depths = torch.zeros(n_samples, device=X.device)

        average_path_length_max_samples = _average_path_length(self.max_samples_)[0]

        for tree_idx, (tree, features) in enumerate(zip(self.estimators_, self.estimators_features_)):
            for i in range(n_samples):
                x = X[i]
                leaf_idx = self._apply_tree(x, tree, features)

                depths[i] += (
                    tree['node_depths'][leaf_idx] +
                    self._average_path_length_per_tree[tree_idx][leaf_idx] -
                    1.0
                )

        denominator = len(self.estimators_) * average_path_length_max_samples

        if denominator != 0:
            depth_div = depths / denominator
        else:
            depth_div = torch.ones_like(depths)

        scores = 2.0 **(-depth_div)

        return scores

    def predict(self, X):
        decision_func = self.decision_function(X)
        return torch.where(decision_func >= 0, -1, 1)
