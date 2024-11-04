import torch
import torch.nn as nn


class MultiShapelet_Learner(nn.Module):
    def __init__(self,
                 num_shapelet,
                 len_shapelet,
                 ts_channels,
                 input_size,
                 num_classes,
                 saturation,
                 alpha):
        super().__init__()
        self.L = num_shapelet
        self.K = len_shapelet
        self.C = ts_channels
        self.Q = input_size
        self.N = num_classes
        self.fc1 = nn.Linear(self.L, 2)
        self.loss_fn = torch.nn.CrossEntropyLoss()
        self.saturation = saturation
        self.shapelets = nn.ParameterList(
            nn.Parameter(torch.randn((self.K, self.C)))
            for _ in range(self.L))  # Init using Xaview Uniform init
        self.gating = nn.Parameter(torch.randn(self.L))
        # should be inited using Normal (.1 0.0001) basically.
        # Additionally we can use Kmeans centroids on each channel
        self.alpha = alpha  # Should be dynamic scheduler

    def _gating_values(self):
        sigmoid = nn.Sigmoid()
        values = sigmoid(self.saturation*self.gating)
        return values

    def _C_compute(self):
        return torch.sum(self._gating_values())/self.K

    def _E_compute(self, results):
        corr_matrix = torch.corrcoef(results.T)
        upp_right = torch.triu(corr_matrix, diagonal=1)
        return torch.max(torch.abs(upp_right))

    def compute_distance_matrix(self, ts):
        shapelet_distances = []
        for shapelet in self.shapelets:
            num_segments = self.Q - self.K + 1
            distances = []
            for j in range(num_segments):
                segment = ts[:, j: j+self.K, :]
                dist = torch.mean((segment - shapelet) ** 2, dim=(1, 2))
                if torch.isnan(dist).any():
                    print("Nan Detected In Distance Calcualtions")

                distances.append(dist)

            distances = torch.stack(distances, dim=1)
            hard_min, _ = torch.min(distances, dim=1)
            shapelet_distances.append(hard_min)

        result = torch.stack(shapelet_distances, dim=1)
        gating = self._gating_values()
        gating = torch.diag(gating)

        return torch.matmul(result, gating)

    def forward(self, ts):
        ts = ts.float()
        Min_DW = self.compute_distance_matrix(ts)
        output = self.fc1(Min_DW)
        return output

    def loss(self, pred, labels, Min_DW):
        E = self._E_compute(Min_DW)
        C = self._C_compute()
        H = self.loss_fn(pred, labels)
        return (1 - self.alpha) * H + self.alpha * (C + E)
