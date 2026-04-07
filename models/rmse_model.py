"""Modular DeepDTA-iBAM architecture optimized for RMSE-focused training."""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
import torch
import torch.nn as nn
import torch.nn.functional as F


def masked_softmax(logits: torch.Tensor, mask: torch.Tensor, dim: int) -> torch.Tensor:
    """Softmax that ignores masked positions."""

    logits = logits.masked_fill(~mask, float("-inf"))
    probs = F.softmax(logits, dim=dim)
    return torch.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0)


class AttentionPool(nn.Module):
    """Learned attention pooling over a masked token sequence."""

    def __init__(self, input_dim: int, hidden_dim: Optional[int] = None):
        super().__init__()
        hidden_dim = hidden_dim or input_dim
        self.score = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        logits = self.score(x).squeeze(-1)
        weights = masked_softmax(logits, mask, dim=1)
        pooled = torch.bmm(weights.unsqueeze(1), x).squeeze(1)
        return pooled, weights


class GraphAttentionLayer(nn.Module):
    """Batched graph attention with edge feature bias and masking."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        edge_features: int = 12,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        if out_features % num_heads != 0:
            raise ValueError("out_features must be divisible by num_heads")

        self.out_features = out_features
        self.num_heads = num_heads
        self.head_dim = out_features // num_heads

        self.node_proj = nn.Linear(in_features, out_features, bias=False)
        self.edge_proj = nn.Linear(edge_features, num_heads, bias=False)
        self.attn_src = nn.Parameter(torch.zeros(num_heads, self.head_dim))
        self.attn_dst = nn.Parameter(torch.zeros(num_heads, self.head_dim))
        nn.init.xavier_uniform_(self.attn_src.unsqueeze(0))
        nn.init.xavier_uniform_(self.attn_dst.unsqueeze(0))

        self.dropout = nn.Dropout(dropout)
        self.leaky_relu = nn.LeakyReLU(0.2)

    def forward(
        self,
        x: torch.Tensor,
        adjacency: torch.Tensor,
        node_mask: torch.Tensor,
        edge_features: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_size, num_nodes, _ = x.shape
        h = self.node_proj(x).view(batch_size, num_nodes, self.num_heads, self.head_dim)

        attn_src = (h * self.attn_src).sum(dim=-1)
        attn_dst = (h * self.attn_dst).sum(dim=-1)
        scores = attn_src.unsqueeze(2) + attn_dst.unsqueeze(1)

        if edge_features is not None:
            scores = scores + self.edge_proj(edge_features)

        scores = self.leaky_relu(scores)
        edge_mask = adjacency.bool()
        valid_mask = node_mask.unsqueeze(1) & node_mask.unsqueeze(2)
        full_mask = edge_mask & valid_mask
        scores = scores.masked_fill(~full_mask.unsqueeze(-1), float("-inf"))

        attn = F.softmax(scores, dim=2)
        attn = torch.nan_to_num(attn, nan=0.0, posinf=0.0, neginf=0.0)
        attn = attn * full_mask.unsqueeze(-1).float()
        attn = attn / attn.sum(dim=2, keepdim=True).clamp_min(1e-9)
        attn = self.dropout(attn)

        out = torch.einsum("bijh,bjhd->bihd", attn, h).reshape(batch_size, num_nodes, self.out_features)
        return out * node_mask.unsqueeze(-1).float()


class MolecularGAT(nn.Module):
    """Token-preserving molecular encoder with graph-level pooling."""

    def __init__(
        self,
        node_features: int = 78,
        edge_features: int = 12,
        hidden_dim: int = 512,
        num_layers: int = 6,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_proj = nn.Linear(node_features, hidden_dim)
        self.layers = nn.ModuleList(
            [
                GraphAttentionLayer(
                    in_features=hidden_dim,
                    out_features=hidden_dim,
                    edge_features=edge_features,
                    num_heads=num_heads,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )
        self.norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_layers)])
        self.dropout = nn.Dropout(dropout)
        self.pool = AttentionPool(hidden_dim, hidden_dim // 2)
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        x: torch.Tensor,
        adjacency: torch.Tensor,
        node_mask: torch.Tensor,
        edge_features: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h = self.dropout(self.input_proj(x))
        h = h * node_mask.unsqueeze(-1).float()

        for layer, norm in zip(self.layers, self.norms):
            residual = h
            h = layer(h, adjacency, node_mask, edge_features)
            h = norm(residual + self.dropout(F.gelu(h)))
            h = h * node_mask.unsqueeze(-1).float()

        pooled, pool_weights = self.pool(h, node_mask)
        graph_embedding = self.output_proj(pooled)
        return h, graph_embedding, pool_weights


class ProteinESM:
    """Pretrained protein embedder for offline cache generation."""

    def __init__(
        self,
        model_name: str = "esmc_600m",
        embedding_dim: Optional[int] = None,
        window_size: int = 1022,
        overlap: int = 128,
        cache_dtype: str = "bfloat16",
        device: str = "cpu",
    ):
        self.model_name = model_name
        self.window_size = window_size
        self.overlap = overlap
        self.cache_dtype = cache_dtype
        self.device = torch.device(device if torch.cuda.is_available() or device == "cpu" else "cpu")
        self.is_esmc = model_name.startswith("esmc_")
        self.model: Any = None
        self.tokenizer: Any = None
        self.batch_converter: Any = None
        self.repr_layer: Optional[int] = None
        self.embedding_dim = embedding_dim or self._infer_embedding_dim(model_name)
        self._load_model()

    @staticmethod
    def _infer_embedding_dim(model_name: str) -> int:
        dims = {
            "esmc_300m": 960,
            "esmc_600m": 1152,
            "esm2_t6_8M_UR50D": 320,
            "esm2_t12_35M_UR50D": 480,
            "esm2_t30_150M_UR50D": 640,
            "esm2_t33_650M_UR50D": 1280,
            "esm2_t36_3B_UR50D": 2560,
        }
        return dims.get(model_name, 640)

    def _load_model(self) -> None:
        if self.is_esmc:
            from esm.models.esmc import ESMC
            from esm.tokenization import EsmSequenceTokenizer

            self.model = ESMC.from_pretrained(self.model_name).to(self.device)
            self.tokenizer = EsmSequenceTokenizer()
            self.model.eval()
            return
        else:
            import esm

            self.model, self.alphabet = getattr(esm.pretrained, self.model_name)()
            self.model = self.model.to(self.device)
            self.model.eval()
            self.batch_converter = self.alphabet.get_batch_converter()
            self.repr_layer = self.model.num_layers
            self.embedding_dim = int(self.model.embed_dim)
            return

    def _align_embeddings(self, embeddings: torch.Tensor, target_length: int) -> torch.Tensor:
        if embeddings.size(0) == target_length:
            return embeddings
        if embeddings.size(0) >= target_length + 2:
            return embeddings[1 : 1 + target_length]
        if embeddings.size(0) > target_length:
            return embeddings[:target_length]
        if embeddings.size(0) < target_length:
            pad = torch.zeros(
                target_length - embeddings.size(0),
                embeddings.size(1),
                device=embeddings.device,
                dtype=embeddings.dtype,
            )
            return torch.cat([embeddings, pad], dim=0)
        return embeddings

    def embed_chunks(self, chunks: Sequence[str]) -> List[torch.Tensor]:
        if not chunks:
            return []
        if self.is_esmc:
            return self._embed_chunks_esmc(chunks)
        return self._embed_chunks_esm2(chunks)

    def _embed_chunks_esm2(self, chunks: Sequence[str]) -> List[torch.Tensor]:
        data = [(f"protein_{idx}", sequence) for idx, sequence in enumerate(chunks)]
        _, _, tokens = self.batch_converter(data)
        tokens = tokens.to(self.device)
        with torch.inference_mode():
            results = self.model(tokens, repr_layers=[self.repr_layer], return_contacts=False)
        representations = results["representations"][self.repr_layer]
        return [
            self._align_embeddings(representations[idx], len(sequence)).detach().cpu()
            for idx, sequence in enumerate(chunks)
        ]

    def _embed_chunks_esmc(self, chunks: Sequence[str]) -> List[torch.Tensor]:
        encoded = [self.tokenizer.encode(sequence) for sequence in chunks]
        max_len = max(len(tokens) for tokens in encoded)
        pad_id = 1
        padded = [tokens + [pad_id] * (max_len - len(tokens)) for tokens in encoded]
        batch_tokens = torch.tensor(padded, dtype=torch.long, device=self.device)
        with torch.inference_mode():
            outputs = self.model(batch_tokens)
        if hasattr(outputs, "embeddings") and outputs.embeddings is not None:
            representations = outputs.embeddings
        elif hasattr(outputs, "hidden_states") and outputs.hidden_states is not None:
            representations = outputs.hidden_states
        else:
            raise RuntimeError("Unsupported ESM-C output: no embeddings or hidden_states found.")
        return [
            self._align_embeddings(representations[idx], len(sequence)).detach().cpu()
            for idx, sequence in enumerate(chunks)
        ]


class AdapterBlock(nn.Module):
    """Small trainable refinement block for cached protein embeddings."""

    def __init__(self, dim: int, num_heads: int, dropout: float, ff_mult: int):
        super().__init__()
        self.attn_norm = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.ff_norm = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, dim * ff_mult),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * ff_mult, dim),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        attn_in = self.attn_norm(x)
        attn_out, _ = self.attn(
            attn_in,
            attn_in,
            attn_in,
            key_padding_mask=~mask,
            need_weights=False,
        )
        x = x + self.dropout(attn_out)
        x = x * mask.unsqueeze(-1).float()

        ff_out = self.ff(self.ff_norm(x))
        x = x + self.dropout(ff_out)
        return x * mask.unsqueeze(-1).float()


class ProteinAdapter(nn.Module):
    """Trainable adapter over cached residue embeddings with masked pooling."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 512,
        num_heads: int = 8,
        num_layers: int = 2,
        ff_mult: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.motif_mixer = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=9, padding=4, groups=hidden_dim)
        self.motif_norm = nn.LayerNorm(hidden_dim)
        self.layers = nn.ModuleList(
            [AdapterBlock(hidden_dim, num_heads, dropout, ff_mult) for _ in range(num_layers)]
        )
        self.pool = AttentionPool(hidden_dim, hidden_dim // 2)
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, embeddings: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self.dropout(self.input_proj(embeddings))
        x = x * mask.unsqueeze(-1).float()
        motif_update = self.motif_mixer(x.transpose(1, 2)).transpose(1, 2)
        x = self.motif_norm(x + self.dropout(motif_update))
        x = x * mask.unsqueeze(-1).float()
        for layer in self.layers:
            x = layer(x, mask)
        pooled, pool_weights = self.pool(x, mask)
        return x, self.output_norm(pooled), pool_weights


class CrossAttentionBlock(nn.Module):
    """Bidirectional masked atom↔residue cross-attention."""

    def __init__(self, dim: int, num_heads: int, dropout: float):
        super().__init__()
        self.atom_q_norm = nn.LayerNorm(dim)
        self.residue_q_norm = nn.LayerNorm(dim)
        self.atom_to_residue = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.residue_to_atom = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.atom_ff_norm = nn.LayerNorm(dim)
        self.residue_ff_norm = nn.LayerNorm(dim)
        self.atom_cross_gate = nn.Parameter(torch.tensor(0.1))
        self.residue_cross_gate = nn.Parameter(torch.tensor(0.1))
        self.atom_ff = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
        )
        self.residue_ff = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        atom_tokens: torch.Tensor,
        atom_mask: torch.Tensor,
        residue_tokens: torch.Tensor,
        residue_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        atom_in = self.atom_q_norm(atom_tokens)
        residue_in = self.residue_q_norm(residue_tokens)

        atom_update, atom_to_residue_weights = self.atom_to_residue(
            atom_in,
            residue_in,
            residue_in,
            key_padding_mask=~residue_mask,
            need_weights=True,
            average_attn_weights=False,
        )
        residue_update, residue_to_atom_weights = self.residue_to_atom(
            residue_in,
            atom_in,
            atom_in,
            key_padding_mask=~atom_mask,
            need_weights=True,
            average_attn_weights=False,
        )

        atom_tokens = atom_tokens + self.atom_cross_gate * self.dropout(atom_update)
        residue_tokens = residue_tokens + self.residue_cross_gate * self.dropout(residue_update)
        atom_tokens = atom_tokens * atom_mask.unsqueeze(-1).float()
        residue_tokens = residue_tokens * residue_mask.unsqueeze(-1).float()

        atom_tokens = atom_tokens + self.dropout(self.atom_ff(self.atom_ff_norm(atom_tokens)))
        residue_tokens = residue_tokens + self.dropout(self.residue_ff(self.residue_ff_norm(residue_tokens)))
        atom_tokens = atom_tokens * atom_mask.unsqueeze(-1).float()
        residue_tokens = residue_tokens * residue_mask.unsqueeze(-1).float()

        return atom_tokens, residue_tokens, {
            "atom_to_residue": atom_to_residue_weights,
            "residue_to_atom": residue_to_atom_weights,
        }


class BidirectionalCrossAttention(nn.Module):
    """Stacked bidirectional atom↔residue cross-attention with masked pooling."""

    def __init__(
        self,
        atom_dim: int,
        residue_dim: int,
        fusion_dim: int = 512,
        num_heads: int = 8,
        num_layers: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.atom_proj = nn.Linear(atom_dim, fusion_dim)
        self.residue_proj = nn.Linear(residue_dim, fusion_dim)
        self.blocks = nn.ModuleList([CrossAttentionBlock(fusion_dim, num_heads, dropout) for _ in range(num_layers)])
        self.atom_pool = AttentionPool(fusion_dim, fusion_dim // 2)
        self.residue_pool = AttentionPool(fusion_dim, fusion_dim // 2)

    def forward(
        self,
        atom_tokens: torch.Tensor,
        atom_mask: torch.Tensor,
        residue_tokens: torch.Tensor,
        residue_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        atom_tokens = self.atom_proj(atom_tokens) * atom_mask.unsqueeze(-1).float()
        residue_tokens = self.residue_proj(residue_tokens) * residue_mask.unsqueeze(-1).float()

        attention_maps: Dict[str, torch.Tensor] = {
            "atom_query_mask": atom_mask,
            "residue_query_mask": residue_mask,
        }
        for layer_idx, block in enumerate(self.blocks):
            atom_tokens, residue_tokens, layer_attention_maps = block(atom_tokens, atom_mask, residue_tokens, residue_mask)
            for name, value in layer_attention_maps.items():
                attention_maps[f"layer_{layer_idx}_{name}"] = value

        atom_pool, atom_pool_weights = self.atom_pool(atom_tokens, atom_mask)
        residue_pool, residue_pool_weights = self.residue_pool(residue_tokens, residue_mask)
        attention_maps["atom_pool"] = atom_pool_weights
        attention_maps["residue_pool"] = residue_pool_weights
        return atom_tokens, residue_tokens, atom_pool, residue_pool, attention_maps


class AffinityHead(nn.Module):
    """Residual MLP over fused drug/protein summaries."""

    def __init__(self, fusion_dim: int, graph_dim: int, protein_global_dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.graph_proj = nn.Linear(graph_dim, fusion_dim)
        self.protein_global_proj = nn.Linear(protein_global_dim, fusion_dim)
        combined_dim = fusion_dim * 6
        self.input_block = nn.Sequential(
            nn.Linear(combined_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.residual_block = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Dropout(dropout),
        )
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.output = nn.Linear(hidden_dim, 1)
        nn.init.xavier_uniform_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(
        self,
        atom_pool: torch.Tensor,
        protein_pool: torch.Tensor,
        graph_global: torch.Tensor,
        protein_global: torch.Tensor,
    ) -> torch.Tensor:
        graph_global = self.graph_proj(graph_global)
        protein_global = self.protein_global_proj(protein_global)
        features = torch.cat(
            [
                atom_pool,
                protein_pool,
                atom_pool * protein_pool,
                torch.abs(atom_pool - protein_pool),
                graph_global,
                protein_global,
            ],
            dim=-1,
        )
        hidden = self.input_block(features)
        hidden = self.output_norm(hidden + self.residual_block(hidden))
        return self.output(hidden)


class SinusoidalTimeEmbedding(nn.Module):
    """Sinusoidal timestep embedding for the diffusion head."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000.0) * torch.arange(half, dtype=torch.float32, device=t.device) / max(half, 1)
        )
        args = t.float().unsqueeze(1) * freqs.unsqueeze(0)
        return torch.cat([args.sin(), args.cos()], dim=-1)


class SimpleMessagePassing(nn.Module):
    """Mean-aggregation message passing used by the diffusion score network."""

    def __init__(self, dim: int):
        super().__init__()
        self.linear = nn.Linear(dim * 2, dim)
        self.norm = nn.LayerNorm(dim)

    def forward(self, h: torch.Tensor, adjacency: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        degree = adjacency.float().sum(-1, keepdim=True).clamp_min(1.0)
        agg = torch.bmm(adjacency.float(), h) / degree
        h_new = F.gelu(self.linear(torch.cat([h, agg], dim=-1)))
        h = self.norm(h + h_new)
        return h * mask.unsqueeze(-1).float()


class SimpleGraphDiffusion(nn.Module):
    """Protein-conditioned DDPM over atom features."""

    def __init__(
        self,
        node_features: int = 78,
        hidden_dim: int = 256,
        condition_dim: int = 512,
        T: int = 1000,
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
        inference_steps: int = 50,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.T = T
        self.node_features = node_features
        self.inference_steps = inference_steps

        betas = torch.linspace(beta_start, beta_end, T)
        alphas = 1.0 - betas
        alpha_bar = torch.cumprod(alphas, dim=0)
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_bar", alpha_bar)
        self.register_buffer("sqrt_alpha_bar", alpha_bar.sqrt())
        self.register_buffer("sqrt_one_minus_alpha_bar", (1.0 - alpha_bar).sqrt())

        self.node_proj = nn.Linear(node_features, hidden_dim)
        self.time_embed = nn.Sequential(
            SinusoidalTimeEmbedding(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
        )
        self.condition_proj = nn.Linear(condition_dim, hidden_dim * 2)
        self.mp1 = SimpleMessagePassing(hidden_dim)
        self.mp2 = SimpleMessagePassing(hidden_dim)
        self.output_proj = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, node_features),
        )

    def _film(self, h: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        scale, shift = self.condition_proj(condition).chunk(2, dim=-1)
        return h * (1.0 + scale.unsqueeze(1)) + shift.unsqueeze(1)

    def _score(
        self,
        x_t: torch.Tensor,
        adjacency: torch.Tensor,
        mask: torch.Tensor,
        t: torch.Tensor,
        condition: torch.Tensor,
    ) -> torch.Tensor:
        h = self.node_proj(x_t)
        h = h + self.time_embed(t).unsqueeze(1)
        h = self._film(h, condition)
        h = self.mp1(h, adjacency, mask)
        h = self.mp2(h, adjacency, mask)
        return self.output_proj(h) * mask.unsqueeze(-1).float()

    def get_loss(
        self,
        x0: torch.Tensor,
        adjacency: torch.Tensor,
        mask: torch.Tensor,
        condition: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = x0.size(0)
        timesteps = torch.randint(0, self.T, (batch_size,), device=x0.device)
        noise = torch.randn_like(x0)
        sqrt_ab = self.sqrt_alpha_bar[timesteps].view(batch_size, 1, 1)
        sqrt_omb = self.sqrt_one_minus_alpha_bar[timesteps].view(batch_size, 1, 1)
        x_t = sqrt_ab * x0 + sqrt_omb * noise
        noise_pred = self._score(x_t, adjacency, mask, timesteps, condition)
        mask_f = mask.unsqueeze(-1).float()
        return ((noise_pred - noise).pow(2) * mask_f).sum() / mask_f.sum().clamp_min(1.0)

    @torch.no_grad()
    def sample(self, adjacency: torch.Tensor, mask: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        batch_size, num_nodes = mask.shape
        device = mask.device
        x = torch.randn(batch_size, num_nodes, self.node_features, device=device)
        steps = min(self.inference_steps, self.T)
        schedule = torch.linspace(self.T - 1, 0, steps, dtype=torch.long, device=device)

        for idx, timestep in enumerate(schedule):
            t_batch = timestep.expand(batch_size)
            eps = self._score(x, adjacency, mask, t_batch, condition)
            ab = self.alpha_bar[timestep]
            x0_pred = (x - (1.0 - ab).sqrt() * eps) / ab.sqrt().clamp_min(1e-8)
            x0_pred = x0_pred.clamp(-3.0, 3.0)
            if idx < len(schedule) - 1:
                ab_next = self.alpha_bar[schedule[idx + 1]]
                x = ab_next.sqrt() * x0_pred + (1.0 - ab_next).sqrt() * eps
            else:
                x = x0_pred
        return x * mask.unsqueeze(-1).float()


class DeepDTAGenIBAM(nn.Module):
    """Predictor-first DeepDTA-iBAM model using cached protein embeddings."""

    def __init__(
        self,
        node_features: int = 78,
        edge_features: int = 12,
        gat_hidden_dim: int = 512,
        gat_layers: int = 6,
        gat_heads: int = 8,
        protein_embedding_dim: int = 1152,
        protein_adapter_dim: int = 512,
        protein_adapter_heads: int = 8,
        protein_adapter_layers: int = 2,
        protein_adapter_ff_mult: int = 4,
        fusion_mode: str = "bidirectional",
        fusion_dim: int = 512,
        fusion_layers: int = 3,
        fusion_heads: int = 8,
        fc_hidden_dim: int = 1024,
        dropout: float = 0.1,
        diff_hidden_dim: int = 256,
        diff_T: int = 1000,
        diff_inference_steps: int = 50,
        **_: Any,
    ):
        super().__init__()
        self.gat = MolecularGAT(
            node_features=node_features,
            edge_features=edge_features,
            hidden_dim=gat_hidden_dim,
            num_layers=gat_layers,
            num_heads=gat_heads,
            dropout=dropout,
        )
        self.protein_adapter = ProteinAdapter(
            input_dim=protein_embedding_dim,
            hidden_dim=protein_adapter_dim,
            num_heads=protein_adapter_heads,
            num_layers=protein_adapter_layers,
            ff_mult=protein_adapter_ff_mult,
            dropout=dropout,
        )
        self.fusion_mode = fusion_mode
        if fusion_mode == "bidirectional":
            self.fusion = BidirectionalCrossAttention(
                atom_dim=gat_hidden_dim,
                residue_dim=protein_adapter_dim,
                fusion_dim=fusion_dim,
                num_heads=fusion_heads,
                num_layers=fusion_layers,
                dropout=dropout,
            )
            self.no_fusion_drug_proj = None
            self.no_fusion_protein_proj = None
        elif fusion_mode == "none":
            self.fusion = None
            self.no_fusion_drug_proj = nn.Linear(gat_hidden_dim, fusion_dim)
            self.no_fusion_protein_proj = nn.Linear(protein_adapter_dim, fusion_dim)
        else:
            raise ValueError(f"Unsupported fusion_mode: {fusion_mode}")
        self.affinity_head = AffinityHead(
            fusion_dim=fusion_dim,
            graph_dim=gat_hidden_dim,
            protein_global_dim=protein_adapter_dim,
            hidden_dim=fc_hidden_dim,
            dropout=dropout,
        )
        self.diffusion = SimpleGraphDiffusion(
            node_features=node_features,
            hidden_dim=diff_hidden_dim,
            condition_dim=protein_adapter_dim,
            T=diff_T,
            inference_steps=diff_inference_steps,
            dropout=dropout,
        )

    def encode_drug(
        self,
        drug_x: torch.Tensor,
        drug_adj: torch.Tensor,
        drug_mask: torch.Tensor,
        drug_edge_features: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.gat(drug_x, drug_adj, drug_mask, drug_edge_features)

    def encode_protein(
        self,
        protein_embeddings: torch.Tensor,
        protein_mask: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if protein_embeddings.dim() != 3:
            raise ValueError(
                "DeepDTAGenIBAM expects cached protein embeddings with shape [batch, residues, dim]."
            )
        if protein_mask is None:
            protein_mask = torch.ones(
                protein_embeddings.shape[:2],
                dtype=torch.bool,
                device=protein_embeddings.device,
            )
        return self.protein_adapter(protein_embeddings, protein_mask)

    def encode_conditioned_pair(
        self,
        drug_x: torch.Tensor,
        drug_adj: torch.Tensor,
        drug_mask: torch.Tensor,
        protein_embeddings: torch.Tensor,
        protein_mask: Optional[torch.Tensor],
        drug_edge_features: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor | Dict[str, torch.Tensor]]:
        atom_tokens, graph_global, graph_pool = self.encode_drug(drug_x, drug_adj, drug_mask, drug_edge_features)
        residue_tokens, protein_global, protein_pool = self.encode_protein(protein_embeddings, protein_mask)

        if self.fusion_mode == "bidirectional":
            assert self.fusion is not None
            _, _, drug_pool, protein_pool_summary, attention_maps = self.fusion(
                atom_tokens, drug_mask, residue_tokens, protein_mask
            )
            attention_maps["graph_pool"] = graph_pool
            attention_maps["protein_adapter_pool"] = protein_pool
        else:
            assert self.no_fusion_drug_proj is not None
            assert self.no_fusion_protein_proj is not None
            drug_pool = self.no_fusion_drug_proj(graph_global)
            protein_pool_summary = self.no_fusion_protein_proj(protein_global)
            attention_maps = {
                "atom_query_mask": drug_mask,
                "residue_query_mask": protein_mask,
                "graph_pool": graph_pool,
                "protein_adapter_pool": protein_pool,
                "fusion_mode_none": torch.ones(
                    drug_mask.size(0),
                    1,
                    1,
                    1,
                    device=drug_mask.device,
                    dtype=drug_x.dtype,
                ),
            }
        return {
            "drug_pool": drug_pool,
            "protein_pool_summary": protein_pool_summary,
            "graph_global": graph_global,
            "protein_global": protein_global,
            "attention_maps": attention_maps,
        }

    def forward(
        self,
        drug_x: torch.Tensor,
        drug_adj: torch.Tensor,
        drug_mask: torch.Tensor,
        protein_embeddings: torch.Tensor,
        protein_mask: Optional[torch.Tensor],
        drug_edge_features: Optional[torch.Tensor] = None,
        compute_diff_loss: bool = False,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], Optional[torch.Tensor]]:
        pair_state = self.encode_conditioned_pair(
            drug_x,
            drug_adj,
            drug_mask,
            protein_embeddings,
            protein_mask,
            drug_edge_features=drug_edge_features,
        )
        drug_pool = pair_state["drug_pool"]
        protein_pool_summary = pair_state["protein_pool_summary"]
        graph_global = pair_state["graph_global"]
        protein_global = pair_state["protein_global"]
        attention_maps = pair_state["attention_maps"]
        affinity = self.affinity_head(drug_pool, protein_pool_summary, graph_global, protein_global)

        if compute_diff_loss:
            diff_loss = self.diffusion.get_loss(drug_x.float(), drug_adj, drug_mask, protein_global)
            return affinity, attention_maps, diff_loss
        return affinity, attention_maps, None

    @torch.no_grad()
    def generate_molecules(
        self,
        protein_embeddings: torch.Tensor,
        protein_mask: torch.Tensor,
        reference_adj: torch.Tensor,
        reference_mask: torch.Tensor,
    ) -> torch.Tensor:
        _, protein_global, _ = self.encode_protein(protein_embeddings, protein_mask)
        return self.diffusion.sample(reference_adj, reference_mask, protein_global)


def smiles_to_graph(
    smiles: str,
    max_atoms: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Convert a SMILES string into rich atom and bond tensors.

    If ``max_atoms`` is ``None`` the tensors are returned at native size.
    Otherwise the graph is padded or truncated to ``max_atoms``.
    """

    edge_features_dim = 12
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        max_atoms = max_atoms or 1
        return (
            torch.zeros((max_atoms, 78), dtype=torch.float32),
            torch.eye(max_atoms, dtype=torch.float32),
            torch.zeros(max_atoms, dtype=torch.bool),
            torch.zeros((max_atoms, max_atoms, edge_features_dim), dtype=torch.float32),
        )

    try:
        AllChem.ComputeGasteigerCharges(mol)
    except Exception:
        pass

    atom_types = ["C", "N", "O", "S", "F", "P", "Cl", "Br", "I"]
    hybridizations = [
        Chem.HybridizationType.SP,
        Chem.HybridizationType.SP2,
        Chem.HybridizationType.SP3,
        Chem.HybridizationType.SP3D,
        Chem.HybridizationType.SP3D2,
    ]
    electronegativity = {
        "H": 2.20,
        "C": 2.55,
        "N": 3.04,
        "O": 3.44,
        "S": 2.58,
        "F": 3.98,
        "P": 2.19,
        "Cl": 3.16,
        "Br": 2.96,
        "I": 2.66,
        "B": 2.04,
        "Si": 1.90,
        "Se": 2.55,
        "Na": 0.93,
        "K": 0.82,
    }
    covalent_radii = {
        "H": 0.31,
        "C": 0.76,
        "N": 0.71,
        "O": 0.66,
        "S": 1.05,
        "F": 0.57,
        "P": 1.07,
        "Cl": 1.02,
        "Br": 1.20,
        "I": 1.39,
    }

    node_features: List[List[float]] = []
    ring_info = mol.GetRingInfo()
    for atom in mol.GetAtoms():
        feat = [0.0] * 78
        symbol = atom.GetSymbol()

        if symbol in atom_types:
            feat[atom_types.index(symbol)] = 1.0
        feat[9 + min(atom.GetDegree(), 6)] = 1.0
        feat[16 + min(atom.GetImplicitValence(), 5)] = 1.0
        feat[22 + max(0, min(4, atom.GetFormalCharge() + 2))] = 1.0

        hyb = atom.GetHybridization()
        if hyb in hybridizations:
            feat[27 + hybridizations.index(hyb)] = 1.0

        feat[32] = 1.0 if atom.GetIsAromatic() else 0.0

        atom_idx = atom.GetIdx()
        for ring_size, feat_idx in [(3, 33), (4, 34), (5, 35), (6, 36)]:
            if ring_info.IsAtomInRingOfSize(atom_idx, ring_size):
                feat[feat_idx] = 1.0

        feat[37 + min(atom.GetTotalNumHs(), 4)] = 1.0

        total_val = min(max(atom.GetTotalValence(), 1), 6)
        feat[42 + total_val - 1] = 1.0
        feat[48 + min(atom.GetNumRadicalElectrons(), 4)] = 1.0
        feat[53] = 1.0 if atom.IsInRing() else 0.0
        feat[54] = 1.0 if atom.IsInRing() and symbol != "C" else 0.0

        chiral_tag = atom.GetChiralTag()
        if chiral_tag == Chem.ChiralType.CHI_TETRAHEDRAL_CW:
            feat[55] = 1.0
        elif chiral_tag == Chem.ChiralType.CHI_TETRAHEDRAL_CCW:
            feat[56] = 1.0
        elif chiral_tag == Chem.ChiralType.CHI_UNSPECIFIED:
            feat[57] = 1.0
        if atom.HasProp("_CIPCode"):
            cip = atom.GetProp("_CIPCode")
            feat[58] = 1.0 if cip == "R" else 0.0
            feat[59] = 1.0 if cip == "S" else 0.0
        feat[60] = 1.0 if atom.HasProp("_ChiralityPossible") else 0.0

        en = electronegativity.get(symbol, 2.5)
        if en < 2.0:
            feat[61] = 1.0
        elif en < 2.5:
            feat[62] = 1.0
        elif en < 3.0:
            feat[63] = 1.0
        elif en < 3.5:
            feat[64] = 1.0
        elif en < 4.0:
            feat[65] = 1.0
        else:
            feat[66] = 1.0

        feat[67] = atom.GetMass() / 200.0
        feat[68] = atom.GetAtomicNum() / 53.0
        feat[69] = covalent_radii.get(symbol, 0.8) / 1.5

        try:
            gasteiger = float(atom.GetProp("_GasteigerCharge"))
            if not np.isnan(gasteiger):
                if gasteiger < -0.25:
                    feat[70] = 1.0
                elif gasteiger < 0.0:
                    feat[71] = 1.0
                elif gasteiger < 0.25:
                    feat[72] = 1.0
                else:
                    feat[73] = 1.0
        except Exception:
            feat[72] = 1.0

        if symbol in ["N", "O"] and atom.GetTotalNumHs() > 0:
            feat[74] = 1.0
        if symbol in ["N", "O", "F"] and not (
            symbol in ["N", "O"] and atom.GetTotalNumHs() > 0 and atom.GetDegree() == 1
        ):
            feat[75] = 1.0

        if symbol == "C":
            hydrophobic = True
            for neighbor in atom.GetNeighbors():
                if neighbor.GetSymbol() in ["N", "O", "S", "P"]:
                    hydrophobic = False
                    break
            feat[76] = 1.0 if hydrophobic else 0.0
        elif symbol in ["F", "Cl", "Br", "I"]:
            feat[76] = 1.0

        if symbol == "N" and atom.GetFormalCharge() >= 0:
            if atom.GetTotalNumHs() > 0 or atom.GetFormalCharge() > 0:
                feat[77] = 1.0

        node_features.append(feat)

    num_atoms = len(node_features)
    node_tensor = torch.tensor(node_features, dtype=torch.float32)
    adjacency = torch.zeros((num_atoms, num_atoms), dtype=torch.float32)
    edge_tensor = torch.zeros((num_atoms, num_atoms, edge_features_dim), dtype=torch.float32)

    bond_type_map = {
        Chem.rdchem.BondType.SINGLE: 0,
        Chem.rdchem.BondType.DOUBLE: 1,
        Chem.rdchem.BondType.TRIPLE: 2,
        Chem.rdchem.BondType.AROMATIC: 3,
    }

    for bond in mol.GetBonds():
        begin = bond.GetBeginAtomIdx()
        end = bond.GetEndAtomIdx()
        adjacency[begin, end] = 1.0
        adjacency[end, begin] = 1.0

        edge_feat = [0.0] * edge_features_dim
        bond_type = bond.GetBondType()
        if bond_type in bond_type_map:
            edge_feat[bond_type_map[bond_type]] = 1.0
        edge_feat[4] = 1.0 if bond.GetIsConjugated() else 0.0
        edge_feat[5] = 1.0 if bond.IsInRing() else 0.0
        if bond.IsInRing():
            for ring_size, feat_idx in [(3, 6), (4, 6), (5, 6), (6, 7), (7, 8)]:
                if mol.GetRingInfo().IsBondInRingOfSize(bond.GetIdx(), ring_size):
                    edge_feat[feat_idx] = 1.0
                    break
        edge_feat[9] = 1.0 if bond.GetStereo() != Chem.rdchem.BondStereo.STEREONONE else 0.0
        edge_feat[10] = bond.GetBondTypeAsDouble() / 3.0
        edge_feat[11] = 1.0 if (
            bond_type == Chem.rdchem.BondType.SINGLE
            and not bond.IsInRing()
            and bond.GetBeginAtom().GetDegree() > 1
            and bond.GetEndAtom().GetDegree() > 1
        ) else 0.0

        edge_feat_tensor = torch.tensor(edge_feat, dtype=torch.float32)
        edge_tensor[begin, end] = edge_feat_tensor
        edge_tensor[end, begin] = edge_feat_tensor

    adjacency = adjacency + torch.eye(num_atoms, dtype=torch.float32)
    mask = torch.ones(num_atoms, dtype=torch.bool)

    if max_atoms is None:
        return node_tensor, adjacency, mask, edge_tensor

    if num_atoms > max_atoms:
        return (
            node_tensor[:max_atoms],
            adjacency[:max_atoms, :max_atoms],
            mask[:max_atoms],
            edge_tensor[:max_atoms, :max_atoms],
        )

    padded_nodes = torch.zeros((max_atoms, 78), dtype=torch.float32)
    padded_adj = torch.zeros((max_atoms, max_atoms), dtype=torch.float32)
    padded_mask = torch.zeros(max_atoms, dtype=torch.bool)
    padded_edges = torch.zeros((max_atoms, max_atoms, edge_features_dim), dtype=torch.float32)

    padded_nodes[:num_atoms] = node_tensor
    padded_adj[:num_atoms, :num_atoms] = adjacency
    padded_mask[:num_atoms] = mask
    padded_edges[:num_atoms, :num_atoms] = edge_tensor
    for idx in range(num_atoms, max_atoms):
        padded_adj[idx, idx] = 1.0

    return padded_nodes, padded_adj, padded_mask, padded_edges


__all__ = [
    "AttentionPool",
    "DeepDTAGenIBAM",
    "GraphAttentionLayer",
    "MolecularGAT",
    "ProteinAdapter",
    "ProteinESM",
    "SimpleGraphDiffusion",
    "smiles_to_graph",
]
