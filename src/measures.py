import torch; from torch import nn
import numpy as np
from scipy.stats import spearmanr

from ckatorch.core import cka_base, cka_batch
import cca_core

from logger import Logger

logger = Logger()


#----------
#utils



def knn(row: int, Z, k):
    # print(f"shape: {Z[row].unspeeze(0).shape}")

    # get knns for row in Z
    distances = torch.cdist(Z[row].unsqueeze(0), Z)
    # print(f"distances shape: {distances.shape}")
    knns_vals, knns_inds = torch.topk(distances, k=k+1, largest=False)  # include self
    # print(f"knn indices shape: {knns_inds.shape}")
    return knns_inds[0, 1:]

def pairwise_knn(
    X: torch.tensor,
    k:int
):
    distances = torch.cdist(X, X)

    # self is included
    knns_vals, knns_inds = torch.topk(distances, k=k+1, largest=False, dim=1)
    return knns_inds[:, 1:]  # remove self


#--------
#measures

def linear_r2_alignment(X, Y, test_ratio=0.2, ridge=1e-3):
    Xc = X - X.mean(dim=0, keepdim=True)
    Yc = Y - Y.mean(dim=0, keepdim=True)
    n = Xc.shape[0]
    idx = torch.randperm(n)

    # for the same data it was almost always 1.0 => nearly perfect alignment
    split = int(n * (1 - test_ratio))
    X_train, X_test = Xc[idx[:split]], Xc[idx[split:]]
    Y_train, Y_test = Yc[idx[:split]], Yc[idx[split:]]

    # ridge solution
    XtX = X_train.T @ X_train + ridge * torch.eye(X_train.shape[1], device=X.device)
    XtY = X_train.T @ Y_train
    W = torch.linalg.solve(XtX, XtY)
    Y_pred = X_test @ W

    ss_res = torch.sum((Y_test - Y_pred) ** 2)
    ss_tot = torch.sum((Y_test - Y_test.mean(dim=0, keepdim=True)) ** 2)
    return (1 - ss_res / ss_tot).item()


def rsa_similarity(X: torch.Tensor, Y: torch.Tensor, metric="cosine") -> float:
    """
    Representational Similarity Analysis (RSA) correlation.
    Args:
        X, Y: [n_samples, d]
        metric: 'cosine' or 'euclidean'
    Returns:
        Spearman correlation between RDMs.
    """
    X = X - X.mean(dim=0, keepdim=True)
    Y = Y - Y.mean(dim=0, keepdim=True)

    if metric == "cosine":
        sim_X = torch.nn.functional.cosine_similarity(
            X.unsqueeze(1), X.unsqueeze(0), dim=-1
        )
        sim_Y = torch.nn.functional.cosine_similarity(
            Y.unsqueeze(1), Y.unsqueeze(0), dim=-1
        )
        D_X = 1 - sim_X
        D_Y = 1 - sim_Y
    else:
        D_X = torch.cdist(X, X)
        D_Y = torch.cdist(Y, Y)

    # Vectorize upper triangle (excluding diagonal)
    iu = torch.triu_indices(D_X.shape[0], D_X.shape[1], offset=1)
    dx = D_X[iu[0], iu[1]].cpu().numpy()
    dy = D_Y[iu[0], iu[1]].cpu().numpy()

    # Compute correlation
    corr, _ = spearmanr(dx, dy)
    return float(corr)

def rank_similarity(
    X: torch.Tensor,        # [n, d]
    Y: torch.Tensor,        # [n, d]
    k: int = 10
) -> float:
    """
    Computes Rank Similarity, which measures the order of common neighbors.
    was optimized by genAI
    #TODO: is it really correct implementation; currently only for testing
    """
    n = X.shape[0]

    dists_X = torch.cdist(X, X)
    dists_Y = torch.cdist(Y, Y)

    # Get the indices of all other points sorted by distance (ascending)
    # ranks_X[i] contains indices of points sorted by distance to point i
    sorted_indices_X = torch.argsort(dists_X, dim=1)
    sorted_indices_Y = torch.argsort(dists_Y, dim=1)

    # Create a map from index to rank for efficient lookup
    # rank_map_X[i, j] will give the rank of point j in the neighborhood of point i
    rank_map_X = torch.zeros_like(sorted_indices_X)
    rank_map_Y = torch.zeros_like(sorted_indices_Y)

    # Populate the rank maps
    # This is an efficient way to get rank_of(j) for a given i
    ar = torch.arange(n, device=X.device)
    rank_map_X[ar, sorted_indices_X] = ar.unsqueeze(0).T
    rank_map_Y[ar, sorted_indices_Y] = ar.unsqueeze(0).T

    # Get the top k neighbors (excluding self, which is at rank 0)
    neighbors_X = sorted_indices_X[:, 1:k+1]
    neighbors_Y = sorted_indices_Y[:, 1:k+1]

    # --- Calculate instance-wise similarity ---
    instance_similarities = []
    for i in range(n):
        # Find the intersection of the neighbor sets
        set_X = set(neighbors_X[i].cpu().numpy())
        set_Y = set(neighbors_Y[i].cpu().numpy())
        common_neighbors = list(set_X.intersection(set_Y))

        if not common_neighbors:
            instance_similarities.append(0.0)
            continue

        # Calculate the score for the current instance i
        score_i = 0.0
        for j in common_neighbors:
            # Ranks are 1-based in the paper's formula
            rank_in_X = rank_map_X[i, j].item() + 1
            rank_in_Y = rank_map_Y[i, j].item() + 1

            # Formula term from the paper
            term = 2 / ( (1 + abs(rank_in_X - rank_in_Y)) * (rank_in_X + rank_in_Y) )
            score_i += term

        # --- Normalization factor ---
        K = len(common_neighbors)
        # v_max is the sum of 1/rank for the top K ranks
        v_max = sum(1.0 / r for r in range(1, K + 1))

        normalized_score = score_i / v_max if v_max > 0 else 0.0
        instance_similarities.append(normalized_score)

    # The final measure is the average over all instances
    return sum(instance_similarities) / n


def orthogonal_procrustes_distance(
    X: torch.Tensor,
    Y: torch.Tensor
) -> float:

    # formula: (|R|_F^2 + |S|_F^2 - 2*|R^T S|_*)^1/2

    assert X.shape == Y.shape

    X_centered = X - X.mean(dim=0, keepdim=True)
    Y_centered = Y - Y.mean(dim=0, keepdim=True)

    # |R|_F^2
    # squared frobenius norms
    norm_X_sq = torch.linalg.norm(X_centered, 'fro')**2
    norm_Y_sq = torch.linalg.norm(Y_centered, 'fro')**2

    # noclear norm
    # ||R^T S||_*
    M = Y_centered.T @ X_centered
    singular_values = torch.linalg.svdvals(M)
    nuclear_norm = torch.sum(singular_values)

    # formula 15 from paper similarity of neural networks: a survey
    distance_sq = norm_X_sq + norm_Y_sq - 2 * nuclear_norm

    distance = torch.sqrt(torch.clamp(distance_sq, min=0.0))

    return distance.item()

def jaccard_similarity(
    X: torch.Tensor,        # [n, d]
    Y: torch.Tensor,
    k: int = 10,
):
    n = X.shape[0]
    assert Y.shape[0] == n

    knns1 = pairwise_knn(X, k)   # [n, k]
    knns2 = pairwise_knn(Y, k)   # [n, k]

    sims = []

    for i in range(n):
        nearest_neighbors1 = set(knns1[i].cpu().numpy())
        nearest_neighbors2 = set(knns2[i].cpu().numpy())

        denom = len(nearest_neighbors1.intersection(nearest_neighbors2))
        nom = len(nearest_neighbors1.union(nearest_neighbors2))

        sim = 0.0
        if nom > 0:
            sim = denom / nom

        sims.append(sim)

    avg = sum(sims) / len(sims)
    return avg


def mutual_knn_alignment(
    Z1: torch.Tensor,
    Z2: torch.Tensor,
    k: int = 10
):
    # Z is matrix, each row is one sample

    def align(Z1, Z2, k):
        # a(i,j) = 1[j in knn(i, Z1) and i in knn(j, Z2); and i != j]
        counter = 0
        for i in range(Z1.shape[0]):
            knns_i_z1 = set(knn(i, Z1, k).cpu().numpy())  # explicit .cpu()
            for j in range(Z2.shape[0]):
                if i != j:
                    knns_j_z2 = set(knn(j, Z2, k).cpu().numpy())  # explicit .cpu()
                    if j in knns_i_z1 and i in knns_j_z2:
                        counter += 1
        return counter

    mknn = align(Z1, Z2, k) / ((align(Z1, Z1, k) * align(Z2, Z2, k)) ** 0.5)
    return mknn

# genai came up with the gpu optimized version of it
def mutual_knn_alignment_gpu(
    Z1: torch.Tensor,
    Z2: torch.Tensor,
    k: int = 10
):

    n = Z1.shape[0]

    def align_gpu(Z1, Z2, k):
        # Precompute all k-NN indices at once
        dists1 = torch.cdist(Z1, Z1)  # [n, n]
        dists2 = torch.cdist(Z2, Z2)  # [n, n]

        # Get k+1 nearest (including self), then remove self
        _, knn1 = torch.topk(dists1, k + 1, largest=False, dim=1)  # [n, k+1]
        _, knn2 = torch.topk(dists2, k + 1, largest=False, dim=1)  # [n, k+1]

        knn1 = knn1[:, 1:]  # [n, k] - remove self
        knn2 = knn2[:, 1:]  # [n, k] - remove self

        # Create masks for the conditions: j in knn(i, Z1) and i in knn(j, Z2)
        total = 0
        for i in range(n):
            for j in range(n):
                if i != j:
                    # Check if j in knn(i, Z1)
                    j_in_knn_i = (knn1[i] == j).any()
                    # Check if i in knn(j, Z2)
                    i_in_knn_j = (knn2[j] == i).any()

                    if j_in_knn_i and i_in_knn_j:
                        total += 1

        return total

    mknn = align_gpu(Z1, Z2, k) / ((align_gpu(Z1, Z1, k) * align_gpu(Z2, Z2, k)) ** 0.5)
    return mknn

# even morew sophisticated version works! from genai
def mutual_knn_alignment_gpu_advanced(
    Z1: torch.Tensor,
    Z2: torch.Tensor,
    k: int = 10
):
    def align_gpu_vectorized(Z1, Z2, k):
        n = Z1.shape[0]

        # Precompute all k-NN indices
        dists1 = torch.cdist(Z1, Z1)
        dists2 = torch.cdist(Z2, Z2)

        _, knn1 = torch.topk(dists1, k + 1, largest=False, dim=1)  # [n, k+1]
        _, knn2 = torch.topk(dists2, k + 1, largest=False, dim=1)  # [n, k+1]

        knn1 = knn1[:, 1:]  # [n, k] - remove self
        knn2 = knn2[:, 1:]  # [n, k] - remove self

        # Create boolean masks entirely on GPU
        # For each i,j: is j in knn(i, Z1)?
        mask1 = (knn1.unsqueeze(2) == torch.arange(n, device=Z1.device).unsqueeze(0).unsqueeze(0)).any(dim=1)  # [n, n]

        # For each i,j: is i in knn(j, Z2)?
        mask2 = (knn2.unsqueeze(2) == torch.arange(n, device=Z1.device).unsqueeze(0).unsqueeze(0)).any(dim=1)  # [n, n]

        # Exclude diagonal (i != j)
        eye_mask = ~torch.eye(n, dtype=torch.bool, device=Z1.device)

        # Count mutual k-NN: j in knn(i,Z1) AND i in knn(j,Z2) AND i!=j
        mutual_mask = mask1 & mask2.T & eye_mask

        return mutual_mask.sum().item()

    mknn = align_gpu_vectorized(Z1, Z2, k) / ((align_gpu_vectorized(Z1, Z1, k) * align_gpu_vectorized(Z2, Z2, k)) ** 0.5)
    return mknn


# simpler version
def mutual_knn_alignment_simple(
    Z1: torch.Tensor,
    Z2: torch.Tensor,
    k: int = 10
):
    """
    computes the mutual k-NN alignment metric as described in
    "Understanding the Emergence of Multimodal Representation Alignment".

    Args:
        Z1, Z2: [n_samples, embedding_dim] tensors
        k: number of nearest neighbors

    """
    assert Z1.shape[0] == Z2.shape[0]

    n_samples = Z1.shape[0]

    # Compute pairwise distances
    dists1 = torch.cdist(Z1, Z1)  # [n, n]
    dists2 = torch.cdist(Z2, Z2)  # [n, n]

    # Find k nearest neighbors (excluding self), genai gave me this function
    _, neighbors1 = torch.topk(dists1, k + 1, largest=False, dim=1)
    _, neighbors2 = torch.topk(dists2, k + 1, largest=False, dim=1)

    #rm self
    neighbors1 = neighbors1[:, 1:]  # [n, k]
    neighbors2 = neighbors2[:, 1:]  # [n, k]

    total_mutual = 0
    for i in range(n_samples):
        nn1_i = set(neighbors1[i].cpu().numpy())
        nn2_i = set(neighbors2[i].cpu().numpy())

        # nn1_i \cap nn2_i
        mutual_count = len(nn1_i.intersection(nn2_i))
        total_mutual += mutual_count


    align_mknn = total_mutual / (n_samples * k)
    return align_mknn

def cosine_similarity_indv(
    text_embedding:torch.Tensor,
    vision_embedding: torch.Tensor
) -> float:

    norm_text_embedding = nn.functional.normalize(text_embedding, dim=-1)
    norm_vision_embedding = nn.functional.normalize(vision_embedding, dim=-1)

    # print(f"norm Text embedding shape: {norm_text_embedding.shape},Vision embedding shape: {norm_vision_embedding.shape}")

    sim = torch.dot(norm_text_embedding, norm_vision_embedding)
    return sim.item()

def cosine_similarity_batch(
    text_embedding: torch.Tensor,
    vision_embedding: torch.Tensor
) -> torch.Tensor:


    # sim = torch.nn.functional.cosine_similarity(text_embedding, vision_embedding, dim=1)
    # return sim

    norm_text = nn.functional.normalize(text_embedding, dim=-1)
    norm_vision = nn.functional.normalize(vision_embedding, dim=-1)


    sim = torch.sum(norm_text * norm_vision, dim=-1)

    return sim  # [bs] tensor of similarities

def svcca_similarity(
    text_embedding: torch.Tensor,
    vision_embedding: torch.Tensor
):

    text_embedding_numpy = text_embedding.detach().cpu().numpy()
    vision_embedding_numpy = vision_embedding.detach().cpu().numpy()

    # print(f"text embedding shape: {text_embedding_numpy.shape}, vision embedding shape: {vision_embedding_numpy.shape}")


    #cca processing needs numpy input, numpy is different to tensors
    # needs shape [dim, tokens * bs]
    # each row: one feature dimension; dim[i]
    # each column: one datapoint/token/patch

    text_input = text_embedding_numpy.reshape(-1, text_embedding_numpy.shape[-1])
    text_input = text_input.transpose()     # [dim, tokens*bs]
    vision_input = vision_embedding_numpy.reshape(-1, vision_embedding_numpy.shape[-1])
    vision_input = vision_input.transpose()  # [dim, patches*bs]

    try:
        # print(f"reshaped text input shape: {text_input.shape}")
        result = cca_core.robust_cca_similarity(
            text_input,
            vision_input,
            compute_dirns=False
        )

        # single value result
        # from https://github.com/google/svcca/blob/master/tutorials/001_Introduction.ipynb
        result = np.mean(result["cca_coef1"])
        return result
    except np.linalg.LinAlgError as e:
        print(f"LinAlgError during SVCCA computation: {e}")
        logger.error(f"LinAlgError during SVCCA computation: {e}")
        return 0.0

def mutual_nearest_neighbor_alignment(text_embeds, vision_embeds, k=5):
    # this measurement is from the paper
    # The Platonic Representation Hypothesis
    # where they got values of about 0.16. theoretically the range of this measurement is 0 to 1.
    # but they also noted, that their alignment score was increasing, but not even near the top value of 1.
    batch_size = text_embeds.shape[0]

    text_dists = torch.cdist(text_embeds, text_embeds)
    vision_dists = torch.cdist(vision_embeds, vision_embeds)

    # largest=False => exclude self
    _, text_neighbors = text_dists.topk(k+1, largest=False)
    _, vision_neighbors = vision_dists.topk(k+1, largest=False)



    #remove self (index 0) from neighbors
    text_neighbors = text_neighbors[:, 1:]  # [batch_size, k]
    vision_neighbors = vision_neighbors[:, 1:]  # [batch_size, k]

    # was manually counting, genai gave me this more efficient method
    total_overlap = 0
    for i in range(batch_size):
        # Use torch operations to count intersections
        text_nn = text_neighbors[i]  # [k]
        vision_nn = vision_neighbors[i]  # [k]

        # Count how many elements in text_nn are also in vision_nn
        overlap = torch.isin(text_nn, vision_nn).sum().item()
        total_overlap += overlap

    return total_overlap / (batch_size * k)

def cka(
    text_embedding: torch.Tensor,
    vision_embedding: torch.Tensor
) -> float:
    """
    Compute Centered Kernel Alignment (CKA) between two representations.

    Args:
        text_embedding:  shape [batch_size, embedding_dim]
        vision_embedding: shape [batch_size, embedding_dim]

    """
    cka_score = cka_batch(text_embedding, vision_embedding)   # needs shape [bs, tokens, dim], but i have different tokens
    # => i could pad tokenizer to 197?
    # cka_score = cka_base(
    #     x=text_embedding,
    #     y=vision_embedding,
    #     kernel="linear",         # Use linear kernel (standard for CKA)
    #     unbiased=False,          # Use biased version (standard)
    #     method="fro_norm"        # Use Frobenius norm method
    # )
    return cka_score.item()

def max_similarity_token_patch(
    text_embedding, # [bs, num_tokens, dim]
    vision_embedding # [bs, num_patches+1, dim]
    ):

    text_embedding = text_embedding[:, 1:, :]  # [bs, num_tokens-1, dim]
    vision_embedding = vision_embedding[:, 1:, :]  # [bs, num_patches, dim]

    text_norm = nn.functional.normalize(text_embedding, dim=-1)
    vision_norm = nn.functional.normalize(vision_embedding, dim=-1)

    # pairwise similarities => matrix of similarities. here take the maximum for each token
    # basically doing:
    # sim_matrix[b, i, j] = text_norm[b, i, :] · vision_norm[b, j, :]
    # = cosine_similarity(text_token_i, image_patch_j)
    sim_matrix = torch.bmm(text_norm, vision_norm.transpose(1, 2))  # [bs, num_tokens-1, num_patches]

    max_sims, _ = sim_matrix.max(dim=2)
    return max_sims.mean(dim=1)

def max_similarity_patch_token(
    text_embedding,
    vision_embedding,
):
    max_sims = max_similarity_token_patch(
        text_embedding=vision_embedding,
        vision_embedding=text_embedding
    )

    return max_sims


if __name__ == "__main__":
    t1 = torch.rand((100,197, 768), )
    t2 = torch.rand((100, 197,768), )

    t_half = t1.clone()
    t_temp = t2.clone()
    # t_half = t_half + t_temp    # results in 0.5
    idx = 77
    t_half[:, :, :idx] = t1[:, :, :idx]
    t_half[:, :, idx:] = t2[:, :, idx:]

    cka_diff = cka(t1, t2)
    cka_identical = cka(t1, t1)
    cka_half = cka(t1, t_half)

    print(f"CKA identical: {cka_identical}, CKA different: {cka_diff}, CKA half: {cka_half}")