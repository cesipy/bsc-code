import torch ; import torch.nn.functional as F; import numpy as np
from config import *
import metrics
import vilbert

# path = "res/checkpoints/20251014-034432_pretrained_asymmetric_fusion/20251024-001100_finetuned_mm_imdb.pt"
path = "res/checkpoints/20251102-122009_pretrained_123.pt"
# res/checkpoints/20251102-122009_pretrained_123.pt src/test.py
def get_norm_rand_matrices():
    dim = torch.Size([768, EMBEDDING_DIM])
    X_poisson = torch.poisson(torch.ones(dim)) / EMBEDDING_DIM
    X_gaussian = torch.randn(dim)
    X_uniform = torch.rand(dim)
    # X_binomial = torch.binomial(torch.ones(dim)*10, 0.5435) / 1090
    X_exponential = torch.distributions.Exponential(112314.).sample(dim) / 10.0
    X_lognormal = torch.distributions.LogNormal(0.0, 1.0).sample(dim) / 10.0

    X = X_poisson @ torch.rand(EMBEDDING_DIM, 768)
    Y = X_gaussian @ torch.rand(EMBEDDING_DIM, 768)

    X = F.normalize(X, dim=-1)
    Y = F.normalize(Y, dim=-1)
    return X, Y

def main():
    dim = torch.Size([768, EMBEDDING_DIM])
    X, Y = get_norm_rand_matrices()
    basline_cka = metrics.AlignmentMetrics.cka(X, Y)
    print(f"baseline cka: {basline_cka}")

    model = vilbert.ViLBERT.load_model(load_path=path, device= "cuda")
    print(model.t_biattention_ids, model.v_biattention_ids)
    W_TQ = model.bert.encoder.layer[4].attention.self.query.weight
    W_VK = model.vit.blocks[1].attn.qkv.weight[EMBEDDING_DIM:2*EMBEDDING_DIM, :]  # K only
    W_VV = model.vit.blocks[1].attn.qkv.weight[2*EMBEDDING_DIM:, :]  # V only

    X_q = X @ W_TQ.T
    Y_k = Y @ W_VK.T
    Y_v = Y @ W_VV.T

    inner = (X_q @ Y_k.T) / torch.sqrt(torch.tensor(EMBEDDING_DIM, dtype=torch.float32))
    attention_weights = F.softmax(inner, dim=-1)  # [768, 768]
    print(f"cka attn weights vs X {metrics.AlignmentMetrics.cka(attention_weights, X)}")
    print(f"cka attn weights vs Y {metrics.AlignmentMetrics.cka(attention_weights, Y)}")
    X_out = attention_weights @ Y_v  # [768, 768]
    print(f"CKA X_q vs Y_k (after projection): {metrics.AlignmentMetrics.cka(X_q, Y_k)}")
    print(f"CKA attention_weights vs Y_v: {metrics.AlignmentMetrics.cka(attention_weights, Y_v)}")
    print(f"CKA X vs X_out (after cross-attention): {metrics.AlignmentMetrics.cka(X, X_out)}")
    print(f"CKA X_out vs Y (cross-modal alignment): {metrics.AlignmentMetrics.cka(X_out, Y)}")


if __name__ == "__main__":
    main()

