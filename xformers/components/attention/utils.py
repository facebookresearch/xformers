import torch


# Assumes that matrix passed in has had softmax applied to it.
def iterative_pinv(softmax_mat: torch.Tensor, n_iter=6, pinverse_original_init=False):
    """
    Computing the Moore-Penrose inverse.
    Use an iterative method from (Razavi et al. 2014) to approximate the Moore-Penrose inverse via efficient
    matrix-matrix multiplications.
    """

    i = torch.eye(
        softmax_mat.size(-1), device=softmax_mat.device, dtype=softmax_mat.dtype
    )
    k = softmax_mat

    # The entries of K are positive and ||K||_{\infty} = 1 due to softmax
    if pinverse_original_init:
        # This original implementation is more conservative to compute coefficient of Z_0.
        v = 1 / torch.max(torch.sum(k, dim=-2)) * k.transpose(-1, -2)
    else:
        # This is the exact coefficient computation, 1 / ||K||_1, of initialization of Z_0, leading to faster
        # convergence.
        v = (
            1
            / torch.max(torch.sum(k, dim=-2), dim=-1).values[:, None, None]
            * k.transpose(-1, -2)
        )

    for _ in range(n_iter):
        kv = torch.matmul(k, v)
        v = torch.matmul(
            0.25 * v,
            13 * i - torch.matmul(kv, 15 * i - torch.matmul(kv, 7 * i - kv)),
        )
    return v
