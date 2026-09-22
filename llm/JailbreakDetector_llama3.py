import torch

from IsolationForest import PyTorchIsolationForest


class JailbreakDetector:
    def __init__(self, rank_data, n_estimators=500, random_state=42, max_samples=512, k_nb=10):
        self.rows = rank_data["manifest"]["sample_rows"]
        labels = rank_data["labels"]
        train_ranks = rank_data["train_ranks"][k_nb]
        test_ranks = rank_data["test_ranks"][k_nb]
        benign_train = train_ranks[labels == 0]
        self.mean = benign_train.mean(dim=0, keepdim=True)
        self.std = benign_train.std(dim=0, keepdim=True) + 1e-8
        normalized_train = (benign_train - self.mean) / self.std
        self.normalized_test = (test_ranks - self.mean) / self.std
        self.if_model = PyTorchIsolationForest(
            n_estimators=n_estimators,
            max_samples=max_samples,
            random_state=random_state,
        ).fit(normalized_train)

    def score_all(self):
        with torch.no_grad():
            return self.if_model.decision_function(self.normalized_test).detach().cpu().numpy()
