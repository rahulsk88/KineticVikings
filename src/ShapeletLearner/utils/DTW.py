import torch

def DTW_calc(ts1, ts2):
    """
    Computes the DTW similarity (with multidimensional time series) between two time series.
    Ensures that the entire process is differentiable.

    Will include the Path returner as well. Mostly for debugging.
    """
    if ts1.shape[1] != ts2.shape[1]:
        raise ValueError("The timeseries are multivariate in different ways")

    cost_mat = cost_matrix(ts1, ts2)
    return torch.sqrt(cost_mat[-1, -1])

def euclidean_diff(ts1, ts2):
    """
    Calculates the squared Euclidean distance between two points (or vectors) from two time series.
    This operation is differentiable.
    """
    return torch.sum((ts1 - ts2) ** 2)

def cost_matrix(ts1, ts2):
    """
    Computes the cost matrix for DTW with differentiable operations.
    """
    ts1_size = ts1.shape[0]
    ts2_size = ts2.shape[0]
    cum_sum = torch.zeros((ts1_size + 1, ts2_size + 1), device=ts1.device)
    cum_sum[1:, 0] = float('inf')
    cum_sum[0, 1:] = float('inf')
    for i in range(1, ts1_size + 1):
        for j in range(1, ts2_size + 1):
            cost = euclidean_diff(ts1[i - 1], ts2[j - 1])
            cum_sum[i, j] = cost + torch.min(torch.stack([
                cum_sum[i - 1, j],   ## Insertion
                cum_sum[i, j - 1],   ## Deletion
                cum_sum[i - 1, j - 1]  ## Match
            ]))
    return cum_sum[1:, 1:]