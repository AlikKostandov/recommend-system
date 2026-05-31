from __future__ import annotations

import os
import sys
import copy
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple, Any

import numpy as np
import pandas as pd

import torch
import torch.nn.functional as F
from torch import nn

from torch_geometric.data import HeteroData
from torch_geometric.nn import HGTConv, Linear

# Локальные модули проекта. Они должны лежать рядом с notebook или быть доступны в PYTHONPATH.
sys.path.append(str(Path.cwd()))

from common.data_prep import build_edges, filter_users_min_pos, interactions_stats
from common.split import temporal_train_test_split
from common.indexing import build_index_maps
from common.eval import build_user_item_dict
from common.cold_start import (
    make_synthetic_cold_start_split,
    get_eval_users,
    filter_interactions_by_users,
    summarize_split_result,
)


@dataclass
class HGTExperimentConfig:
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # preprocessing
    threshold_values: List[float] = field(default_factory=lambda: [3.0, 4.0, 5.0])
    min_pos_values: List[int] = field(default_factory=lambda: [5, 10, 20])
    # По умолчанию считаем Recall@20 и NDCG@20; список можно переопределить в notebook.
    k_values: List[int] = field(default_factory=lambda: [20])
    target_metrics: List[str] = field(default_factory=lambda: ["recall", "ndcg"])
    test_ratio: float = 0.2

    # context sizes
    top_actors: int = 200
    top_directors: int = 200
    top_tags: int = 200
    top_countries: Optional[int] = None

    # model
    hidden_channels: int = 128
    out_channels: int = 128
    num_heads: int = 4
    num_layers: int = 2
    dropout: float = 0.2

    # optimization
    epochs: int = 100
    batch_size: int = 4096
    lr: float = 5e-4
    weight_decay: float = 1e-5

    # cold start
    cold_user_fraction: float = 0.2
    cold_n_values: List[int] = field(default_factory=lambda: [1, 3, 5])
    min_interactions_for_cold: int = 20
    warm_last_n: int = 1


def log_step(message: str) -> None:
    """Единый формат логов для длинных запусков в Colab/Jupyter."""
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_hgt_tables(data_dir: str | Path = "..") -> Dict[str, pd.DataFrame]:
    """Загружает таблицы, которые нужны HGT-экспериментам.

    В ноутбуках меняй только DATA_DIR, остальной код экспериментов не трогай.
    """
    data_dir = Path(data_dir)
    tables = {
        "rates": pd.read_csv(data_dir / "user_movie_rates.csv"),
        "users": pd.read_csv(data_dir / "users.csv"),
        "movies": pd.read_csv(data_dir / "movies.csv"),
        "movie_genres": pd.read_csv(data_dir / "movie_genres.csv"),
        "movie_actors": pd.read_csv(data_dir / "movie_actors.csv"),
        "movie_directors": pd.read_csv(data_dir / "movie_directors.csv"),
        "movie_countries": pd.read_csv(data_dir / "movie_countries.csv"),
        "movie_tags": pd.read_csv(data_dir / "movie_tag.csv"),
    }
    for name, df in tables.items():
        log_step(f"loaded {name}: {df.shape}")
    return tables


def make_hgt_runner(tables: Dict[str, pd.DataFrame], config: HGTExperimentConfig) -> "HGTExperimentRunner":
    return HGTExperimentRunner(
        rates=tables["rates"],
        users=tables["users"],
        movie_genres=tables["movie_genres"],
        movie_actors=tables["movie_actors"],
        movie_directors=tables["movie_directors"],
        movie_countries=tables["movie_countries"],
        movie_tags=tables["movie_tags"],
        config=config,
    )


CONTEXT_ORDER = [
    "raw",
    "genres",
    "genres_directors",
    "genres_directors_actors",
    "genres_directors_actors_countries",
    "genres_directors_actors_countries_tags",
    "genres_directors_actors_countries_tags_age_group",
    "genres_directors_actors_countries_tags_age_group_occupation",
]


def _stage_feature_set(stage: str) -> Set[str]:
    stage = str(stage or "raw").lower()
    features = set()

    if "genres" in stage:
        features.add("genres")
    if "directors" in stage:
        features.add("directors")
    if "actors" in stage:
        features.add("actors")
    if "countries" in stage:
        features.add("countries")
    if "tags" in stage:
        features.add("tags")
    if "age_group" in stage:
        features.add("age_group")
    if "occupation" in stage:
        features.add("occupation")

    return features


def build_entity_index_map(df: pd.DataFrame, entity_col: str) -> Dict[int, int]:
    ids = sorted(df[entity_col].dropna().astype(int).unique())
    return {entity_id: idx for idx, entity_id in enumerate(ids)}


def build_string_index_map(df: pd.DataFrame, col: str) -> Dict[str, int]:
    values = sorted(df[col].dropna().astype(str).unique())
    return {value: idx for idx, value in enumerate(values)}


def keep_top_n_entities_by_frequency(
        relation_df: pd.DataFrame,
        entity_col: str,
        top_n: Optional[int] = None,
) -> pd.DataFrame:
    if top_n is None:
        return relation_df.copy()

    top_entities = (
        relation_df[entity_col]
        .value_counts()
        .head(top_n)
        .index
    )
    return relation_df[relation_df[entity_col].isin(top_entities)].copy()


def filter_context_by_train_items(
        relation_df: pd.DataFrame,
        train_item_ids: Iterable[int],
        item_col: str = "movie_id",
) -> pd.DataFrame:
    allowed_items = set(int(x) for x in train_item_ids)
    return relation_df[relation_df[item_col].isin(allowed_items)].copy()


def resolve_tag_column(movie_tags: pd.DataFrame) -> str:
    for candidate in ["tag_id", "tag", "tagId"]:
        if candidate in movie_tags.columns:
            return candidate
    raise ValueError(f"Cannot detect tag column in movie_tags. Columns: {movie_tags.columns.tolist()}")


def _add_bipartite_edges(
        data: HeteroData,
        src_type: str,
        rel: str,
        dst_type: str,
        src_idx: np.ndarray,
        dst_idx: np.ndarray,
) -> None:
    if len(src_idx) == 0:
        return

    edge_index = torch.tensor(np.vstack([src_idx, dst_idx]), dtype=torch.long)
    rev_edge_index = torch.tensor(np.vstack([dst_idx, src_idx]), dtype=torch.long)

    data[src_type, rel, dst_type].edge_index = edge_index
    data[dst_type, f"rev_{rel}", src_type].edge_index = rev_edge_index


def build_hgt_heterodata(
        stage: str,
        train_df: pd.DataFrame,
        user2idx: Dict[int, int],
        item2idx: Dict[int, int],
        movie_genres_train: Optional[pd.DataFrame] = None,
        genre2idx: Optional[Dict[int, int]] = None,
        movie_directors_train: Optional[pd.DataFrame] = None,
        director2idx: Optional[Dict[int, int]] = None,
        movie_actors_train: Optional[pd.DataFrame] = None,
        actor2idx: Optional[Dict[int, int]] = None,
        movie_countries_train: Optional[pd.DataFrame] = None,
        country2idx: Optional[Dict[int, int]] = None,
        movie_tags_train: Optional[pd.DataFrame] = None,
        tag2idx: Optional[Dict[int, int]] = None,
        tag_col: str = "tag_id",
        users_train: Optional[pd.DataFrame] = None,
        occupation2idx: Optional[Dict[str, int]] = None,
        age_group2idx: Optional[Dict[int, int]] = None,
) -> HeteroData:
    feature_set = _stage_feature_set(stage)
    data = HeteroData()

    data["user"].node_id = torch.arange(len(user2idx), dtype=torch.long)
    data["movie"].node_id = torch.arange(len(item2idx), dtype=torch.long)

    tmp = train_df.copy()
    tmp["user_idx"] = tmp["user_id"].map(user2idx)
    tmp["movie_idx"] = tmp["movie_id"].map(item2idx)
    tmp = tmp.dropna(subset=["user_idx", "movie_idx"])

    _add_bipartite_edges(
        data,
        "user",
        "interacts",
        "movie",
        tmp["user_idx"].astype(int).to_numpy(),
        tmp["movie_idx"].astype(int).to_numpy(),
    )

    if "genres" in feature_set and movie_genres_train is not None:
        data["genre"].node_id = torch.arange(len(genre2idx), dtype=torch.long)
        tmp = movie_genres_train.copy()
        tmp["movie_idx"] = tmp["movie_id"].map(item2idx)
        tmp["genre_idx"] = tmp["genre_id"].map(genre2idx)
        tmp = tmp.dropna(subset=["movie_idx", "genre_idx"])
        _add_bipartite_edges(data, "movie", "has_genre", "genre", tmp["movie_idx"].astype(int).to_numpy(),
                             tmp["genre_idx"].astype(int).to_numpy())

    if "directors" in feature_set and movie_directors_train is not None:
        data["director"].node_id = torch.arange(len(director2idx), dtype=torch.long)
        tmp = movie_directors_train.copy()
        tmp["movie_idx"] = tmp["movie_id"].map(item2idx)
        tmp["director_idx"] = tmp["director_id"].map(director2idx)
        tmp = tmp.dropna(subset=["movie_idx", "director_idx"])
        _add_bipartite_edges(data, "movie", "has_director", "director", tmp["movie_idx"].astype(int).to_numpy(),
                             tmp["director_idx"].astype(int).to_numpy())

    if "actors" in feature_set and movie_actors_train is not None:
        data["actor"].node_id = torch.arange(len(actor2idx), dtype=torch.long)
        tmp = movie_actors_train.copy()
        tmp["movie_idx"] = tmp["movie_id"].map(item2idx)
        tmp["actor_idx"] = tmp["actor_id"].map(actor2idx)
        tmp = tmp.dropna(subset=["movie_idx", "actor_idx"])
        _add_bipartite_edges(data, "movie", "has_actor", "actor", tmp["movie_idx"].astype(int).to_numpy(),
                             tmp["actor_idx"].astype(int).to_numpy())

    if "countries" in feature_set and movie_countries_train is not None:
        data["country"].node_id = torch.arange(len(country2idx), dtype=torch.long)
        tmp = movie_countries_train.copy()
        tmp["movie_idx"] = tmp["movie_id"].map(item2idx)
        tmp["country_idx"] = tmp["country_id"].map(country2idx)
        tmp = tmp.dropna(subset=["movie_idx", "country_idx"])
        _add_bipartite_edges(data, "movie", "has_country", "country", tmp["movie_idx"].astype(int).to_numpy(),
                             tmp["country_idx"].astype(int).to_numpy())

    if "tags" in feature_set and movie_tags_train is not None:
        data["tag"].node_id = torch.arange(len(tag2idx), dtype=torch.long)
        tmp = movie_tags_train.copy()
        tmp["movie_idx"] = tmp["movie_id"].map(item2idx)
        tmp["tag_idx"] = tmp[tag_col].astype(int).map(tag2idx)
        tmp = tmp.dropna(subset=["movie_idx", "tag_idx"])
        _add_bipartite_edges(data, "movie", "has_tag", "tag", tmp["movie_idx"].astype(int).to_numpy(),
                             tmp["tag_idx"].astype(int).to_numpy())

    if "age_group" in feature_set and users_train is not None:
        data["age_group"].node_id = torch.arange(len(age_group2idx), dtype=torch.long)

        tmp = users_train.copy()
        tmp["user_idx"] = tmp["user_id"].map(user2idx)
        tmp["age_group_idx"] = tmp["age_group_id"].map(age_group2idx)

        ua = tmp.dropna(subset=["user_idx", "age_group_idx"])
        _add_bipartite_edges(data, "user", "has_age_group", "age_group", ua["user_idx"].astype(int).to_numpy(),
                             ua["age_group_idx"].astype(int).to_numpy())

    if "occupation" in feature_set and users_train is not None:
        data["occupation"].node_id = torch.arange(len(occupation2idx), dtype=torch.long)

        tmp = users_train.copy()
        tmp["user_idx"] = tmp["user_id"].map(user2idx)
        tmp["occupation_idx"] = tmp["occupation"].astype(str).map(occupation2idx)

        uo = tmp.dropna(subset=["user_idx", "occupation_idx"])
        _add_bipartite_edges(data, "user", "has_occupation", "occupation", uo["user_idx"].astype(int).to_numpy(),
                             uo["occupation_idx"].astype(int).to_numpy())

    return data


def build_num_nodes_dict(
        stage: str,
        user2idx: Dict[int, int],
        item2idx: Dict[int, int],
        genre2idx: Optional[Dict[int, int]] = None,
        director2idx: Optional[Dict[int, int]] = None,
        actor2idx: Optional[Dict[int, int]] = None,
        country2idx: Optional[Dict[int, int]] = None,
        tag2idx: Optional[Dict[int, int]] = None,
        occupation2idx: Optional[Dict[str, int]] = None,
        age_group2idx: Optional[Dict[int, int]] = None,
) -> Dict[str, int]:
    feature_set = _stage_feature_set(stage)
    result = {"user": len(user2idx), "movie": len(item2idx)}

    if "genres" in feature_set:
        result["genre"] = len(genre2idx)
    if "directors" in feature_set:
        result["director"] = len(director2idx)
    if "actors" in feature_set:
        result["actor"] = len(actor2idx)
    if "countries" in feature_set:
        result["country"] = len(country2idx)
    if "tags" in feature_set:
        result["tag"] = len(tag2idx)
    if "age_group" in feature_set:
        result["age_group"] = len(age_group2idx)
    if "occupation" in feature_set:
        result["occupation"] = len(occupation2idx)

    return result


class CineContextHGT(nn.Module):
    def __init__(
            self,
            num_nodes_dict: Dict[str, int],
            metadata,
            hidden_channels: int = 128,
            out_channels: int = 128,
            num_heads: int = 4,
            num_layers: int = 2,
            dropout: float = 0.2,
    ):
        super().__init__()

        self.embeddings = nn.ModuleDict({
            node_type: nn.Embedding(num_nodes, hidden_channels)
            for node_type, num_nodes in num_nodes_dict.items()
        })

        self.convs = nn.ModuleList()
        self.residuals = nn.ModuleList()
        self.norms = nn.ModuleList()

        for _ in range(num_layers):
            self.convs.append(
                HGTConv(
                    in_channels=hidden_channels,
                    out_channels=hidden_channels,
                    metadata=metadata,
                    heads=num_heads,
                )
            )
            self.residuals.append(nn.Linear(hidden_channels, hidden_channels))
            self.norms.append(nn.LayerNorm(hidden_channels))

        self.dropout = nn.Dropout(dropout)
        self.user_projection = Linear(hidden_channels, out_channels)
        self.movie_projection = Linear(hidden_channels, out_channels)

    def encode(self, data: HeteroData) -> Dict[str, torch.Tensor]:
        x_dict = {
            node_type: self.embeddings[node_type](data[node_type].node_id)
            for node_type in data.node_types
        }

        for conv, residual, norm in zip(self.convs, self.residuals, self.norms):
            h_dict = conv(x_dict, data.edge_index_dict)
            updated = {}

            for node_type, x in x_dict.items():
                h = h_dict.get(node_type, x)
                h = h + residual(x)
                h = norm(h)
                h = F.relu(h)
                h = self.dropout(h)
                updated[node_type] = h

            x_dict = updated

        x_dict["user"] = self.user_projection(x_dict["user"])
        x_dict["movie"] = self.movie_projection(x_dict["movie"])
        return x_dict

    def forward(self, data: HeteroData) -> Tuple[torch.Tensor, torch.Tensor]:
        encoded = self.encode(data)
        return encoded["user"], encoded["movie"]

    def score_pairs(self, user_idx: torch.Tensor, item_idx: torch.Tensor, data: HeteroData) -> torch.Tensor:
        user_out, movie_out = self.forward(data)
        return (user_out[user_idx] * movie_out[item_idx]).sum(dim=-1)

    def full_score_matrix(self, data: HeteroData) -> torch.Tensor:
        user_out, movie_out = self.forward(data)
        return user_out @ movie_out.T


def bpr_loss(pos_scores: torch.Tensor, neg_scores: torch.Tensor) -> torch.Tensor:
    return -F.logsigmoid(pos_scores - neg_scores).mean()


def iterate_minibatches(pos_u: torch.Tensor, pos_i: torch.Tensor, batch_size: int, shuffle: bool = True):
    n = pos_u.size(0)
    indices = torch.arange(n, device=pos_u.device)
    if shuffle:
        indices = indices[torch.randperm(n, device=pos_u.device)]
    for start in range(0, n, batch_size):
        batch_idx = indices[start:start + batch_size]
        yield pos_u[batch_idx], pos_i[batch_idx]


def sample_negative_items_for_batch(
        batch_u: torch.Tensor,
        train_user_items: Dict[int, Set[int]],
        item2idx: Dict[int, int],
        idx2user: Dict[int, int],
        device: torch.device,
) -> torch.Tensor:
    all_item_ids = np.array(list(item2idx.keys()))
    neg_item_ids = []

    for u_idx in batch_u.detach().cpu().numpy():
        user_id = idx2user[int(u_idx)]
        seen_items = train_user_items.get(user_id, set())

        while True:
            item_id = int(np.random.choice(all_item_ids))
            if item_id not in seen_items:
                neg_item_ids.append(item2idx[item_id])
                break

    return torch.tensor(neg_item_ids, dtype=torch.long, device=device)


def train_one_epoch(
        model: CineContextHGT,
        data: HeteroData,
        train_df: pd.DataFrame,
        item2idx: Dict[int, int],
        idx2user: Dict[int, int],
        pos_u: torch.Tensor,
        pos_i: torch.Tensor,
        optimizer: torch.optim.Optimizer,
        batch_size: int = 4096,
) -> float:
    model.train()
    train_user_items = build_user_item_dict(train_df, user_col="user_id", item_col="movie_id")
    total_loss = 0.0
    total_examples = 0

    for batch_u, batch_pos_i in iterate_minibatches(pos_u, pos_i, batch_size=batch_size, shuffle=True):
        optimizer.zero_grad()

        batch_neg_i = sample_negative_items_for_batch(
            batch_u=batch_u,
            train_user_items=train_user_items,
            item2idx=item2idx,
            idx2user=idx2user,
            device=batch_u.device,
        )

        pos_scores = model.score_pairs(batch_u, batch_pos_i, data)
        neg_scores = model.score_pairs(batch_u, batch_neg_i, data)

        loss = bpr_loss(pos_scores, neg_scores)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * batch_u.size(0)
        total_examples += batch_u.size(0)

    return total_loss / max(total_examples, 1)


@torch.no_grad()
def generate_topk_recommendations(
        model: CineContextHGT,
        data: HeteroData,
        train_df: pd.DataFrame,
        test_df_eval: pd.DataFrame,
        user2idx: Dict[int, int],
        item2idx: Dict[int, int],
        idx2item: Dict[int, int],
        k: int = 20,
) -> Dict[int, List[int]]:
    model.eval()
    score_matrix = model.full_score_matrix(data).detach().cpu().numpy()
    train_user_items = build_user_item_dict(train_df, user_col="user_id", item_col="movie_id")

    recommendations = {}
    for user_id in sorted(test_df_eval["user_id"].unique()):
        if user_id not in user2idx:
            continue

        user_scores = score_matrix[user2idx[user_id]].copy()
        for item_id in train_user_items.get(user_id, set()):
            if item_id in item2idx:
                user_scores[item2idx[item_id]] = -1e9

        top_k = min(k, len(user_scores))
        top_idx = np.argpartition(-user_scores, top_k - 1)[:top_k]
        top_idx = top_idx[np.argsort(-user_scores[top_idx])]
        recommendations[user_id] = [idx2item[i] for i in top_idx]

    return recommendations


def _recall_at_k(recommended_items: List[int], relevant_items: Set[int], k: int) -> float:
    if not relevant_items:
        return 0.0
    recommended_top_k = recommended_items[:k]
    hits = len(set(recommended_top_k) & relevant_items)
    return hits / len(relevant_items)


def _ndcg_at_k(recommended_items: List[int], relevant_items: Set[int], k: int) -> float:
    if not relevant_items:
        return 0.0

    dcg = 0.0
    for rank, item_id in enumerate(recommended_items[:k], start=1):
        if item_id in relevant_items:
            dcg += 1.0 / np.log2(rank + 1)

    ideal_hits = min(len(relevant_items), k)
    idcg = sum(1.0 / np.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    return dcg / idcg if idcg > 0 else 0.0


def evaluate_model_at_k(
        model: CineContextHGT,
        data: HeteroData,
        train_df: pd.DataFrame,
        test_df_eval: pd.DataFrame,
        user2idx: Dict[int, int],
        item2idx: Dict[int, int],
        idx2item: Dict[int, int],
        k: int,
        target_metrics: Optional[List[str]] = None,
) -> Dict[str, float]:
    """Считает выбранные метрики качества для top-K рекомендаций.

    Поддерживаются `recall` и `ndcg`. Если нужны другие метрики, их лучше
    добавить сюда один раз, не размазывая расчёт по experiment-notebook'ам.
    """
    target_metrics = list(target_metrics or ["recall", "ndcg"])
    unsupported = set(target_metrics) - {"recall", "ndcg"}
    if unsupported:
        raise ValueError(f"Unsupported metrics for HGT evaluation: {sorted(unsupported)}")
    log_step(f"evaluation: generating top-{k} recommendations")
    recommendations = generate_topk_recommendations(
        model=model,
        data=data,
        train_df=train_df,
        test_df_eval=test_df_eval,
        user2idx=user2idx,
        item2idx=item2idx,
        idx2item=idx2item,
        k=k,
    )

    ground_truth = test_df_eval.groupby("user_id")["movie_id"].apply(set).to_dict()
    recall_values = []
    ndcg_values = []

    for user_id, relevant_items in ground_truth.items():
        recommended_items = recommendations.get(user_id, [])
        recall_values.append(_recall_at_k(recommended_items, relevant_items, k))
        ndcg_values.append(_ndcg_at_k(recommended_items, relevant_items, k))

    result = {"n_users_eval": len(ground_truth)}
    if "recall" in target_metrics:
        result["recall"] = float(np.mean(recall_values)) if recall_values else 0.0
    if "ndcg" in target_metrics:
        result["ndcg"] = float(np.mean(ndcg_values)) if ndcg_values else 0.0
    return result


class HGTExperimentRunner:
    def __init__(
            self,
            rates: pd.DataFrame,
            users: pd.DataFrame,
            movie_genres: pd.DataFrame,
            movie_actors: pd.DataFrame,
            movie_directors: pd.DataFrame,
            movie_countries: pd.DataFrame,
            movie_tags: pd.DataFrame,
            config: HGTExperimentConfig,
    ):
        self.rates = rates
        self.users = users
        self.movie_genres = movie_genres
        self.movie_actors = movie_actors
        self.movie_directors = movie_directors
        self.movie_countries = movie_countries
        self.movie_tags = movie_tags
        self.config = config
        self.device = torch.device(config.device)
        self.tag_col = resolve_tag_column(movie_tags)

    def prepare_context_tables(self, train_item_ids: Iterable[int]) -> Dict[str, Any]:
        log_step("context: filtering context tables by train movies")
        movie_genres_train = filter_context_by_train_items(self.movie_genres, train_item_ids)
        movie_directors_train = filter_context_by_train_items(self.movie_directors, train_item_ids)
        movie_actors_train = filter_context_by_train_items(self.movie_actors, train_item_ids)
        movie_countries_train = filter_context_by_train_items(self.movie_countries, train_item_ids)
        movie_tags_train = filter_context_by_train_items(self.movie_tags, train_item_ids)

        movie_actors_train = keep_top_n_entities_by_frequency(movie_actors_train, "actor_id", self.config.top_actors)
        movie_directors_train = keep_top_n_entities_by_frequency(movie_directors_train, "director_id",
                                                                 self.config.top_directors)
        movie_tags_train = keep_top_n_entities_by_frequency(movie_tags_train, self.tag_col, self.config.top_tags)
        movie_countries_train = keep_top_n_entities_by_frequency(movie_countries_train, "country_id",
                                                                 self.config.top_countries)

        log_step(
            "context sizes after filtering: "
            f"genres={len(movie_genres_train)}, "
            f"directors={len(movie_directors_train)}, "
            f"actors={len(movie_actors_train)}, "
            f"countries={len(movie_countries_train)}, "
            f"tags={len(movie_tags_train)}"
        )

        return {
            "movie_genres_train": movie_genres_train,
            "movie_directors_train": movie_directors_train,
            "movie_actors_train": movie_actors_train,
            "movie_countries_train": movie_countries_train,
            "movie_tags_train": movie_tags_train,
            "genre2idx": build_entity_index_map(movie_genres_train, "genre_id"),
            "director2idx": build_entity_index_map(movie_directors_train, "director_id"),
            "actor2idx": build_entity_index_map(movie_actors_train, "actor_id"),
            "country2idx": build_entity_index_map(movie_countries_train, "country_id"),
            "tag2idx": build_entity_index_map(movie_tags_train, self.tag_col),
            "occupation2idx": build_string_index_map(self.users, "occupation"),
            "age_group2idx": build_entity_index_map(self.users, "age_group_id"),
        }

    def build_artifacts(self, stage: str, train_df: pd.DataFrame) -> Dict[str, Any]:
        log_step(f"stage={stage}: building index maps")
        user2idx, idx2user, item2idx, idx2item = build_index_maps(train_df)
        train_item_ids = train_df["movie_id"].unique()
        context = self.prepare_context_tables(train_item_ids)

        users_train = self.users[self.users["user_id"].isin(user2idx.keys())].copy()

        log_step(f"stage={stage}: building HeteroData")
        data = build_hgt_heterodata(
            stage=stage,
            train_df=train_df,
            user2idx=user2idx,
            item2idx=item2idx,
            users_train=users_train,
            tag_col=self.tag_col,
            **context,
        )
        data = data.to(self.device)
        log_step(f"stage={stage}: graph moved to device={self.device}")
        log_step(f"stage={stage}: node types={list(data.node_types)}")
        log_step(f"stage={stage}: edge types={list(data.edge_types)}")

        num_nodes_dict = build_num_nodes_dict(
            stage=stage,
            user2idx=user2idx,
            item2idx=item2idx,
            genre2idx=context["genre2idx"],
            director2idx=context["director2idx"],
            actor2idx=context["actor2idx"],
            country2idx=context["country2idx"],
            tag2idx=context["tag2idx"],
            occupation2idx=context["occupation2idx"],
            age_group2idx=context["age_group2idx"],
        )

        return {
            "data": data,
            "user2idx": user2idx,
            "idx2user": idx2user,
            "item2idx": item2idx,
            "idx2item": idx2item,
            "num_nodes_dict": num_nodes_dict,
            "context": context,
            "users_train": users_train,
        }

    def fit_single(self, stage: str, train_df: pd.DataFrame, test_df_eval: pd.DataFrame, k_values: List[int]) -> Tuple[
        pd.DataFrame, Dict[str, Any]]:
        started_at = time.time()
        log_step(f"stage={stage}: fit started; k_values={k_values}; target_metrics={self.config.target_metrics}")
        artifacts = self.build_artifacts(stage, train_df)
        data = artifacts["data"]
        user2idx = artifacts["user2idx"]
        idx2user = artifacts["idx2user"]
        item2idx = artifacts["item2idx"]
        idx2item = artifacts["idx2item"]

        pos = train_df.copy()
        pos["user_idx"] = pos["user_id"].map(user2idx)
        pos["movie_idx"] = pos["movie_id"].map(item2idx)
        pos = pos.dropna(subset=["user_idx", "movie_idx"])

        pos_u = torch.tensor(pos["user_idx"].astype(int).to_numpy(), dtype=torch.long, device=self.device)
        pos_i = torch.tensor(pos["movie_idx"].astype(int).to_numpy(), dtype=torch.long, device=self.device)
        log_step(f"stage={stage}: positive train edges for BPR={len(pos)}")

        model = CineContextHGT(
            num_nodes_dict=artifacts["num_nodes_dict"],
            metadata=data.metadata(),
            hidden_channels=self.config.hidden_channels,
            out_channels=self.config.out_channels,
            num_heads=self.config.num_heads,
            num_layers=self.config.num_layers,
            dropout=self.config.dropout,
        ).to(self.device)

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=self.config.lr,
            weight_decay=self.config.weight_decay,
        )

        losses = []
        log_step(f"stage={stage}: training started for {self.config.epochs} epochs")
        for epoch in range(1, self.config.epochs + 1):
            loss = train_one_epoch(
                model=model,
                data=data,
                train_df=train_df,
                item2idx=item2idx,
                idx2user=idx2user,
                pos_u=pos_u,
                pos_i=pos_i,
                optimizer=optimizer,
                batch_size=self.config.batch_size,
            )
            losses.append({"epoch": epoch, "loss": loss})
            if epoch == 1 or epoch == self.config.epochs or epoch % 10 == 0:
                log_step(f"stage={stage}: epoch={epoch:03d}/{self.config.epochs} loss={loss:.5f}")

        log_step(f"stage={stage}: evaluation started")
        rows = []
        for k in k_values:
            metrics = evaluate_model_at_k(
                model=model,
                data=data,
                train_df=train_df,
                test_df_eval=test_df_eval,
                user2idx=user2idx,
                item2idx=item2idx,
                idx2item=idx2item,
                k=k,
                target_metrics=self.config.target_metrics,
            )
            rows.append({"stage": stage, "k": k, **metrics})

        artifacts["model"] = model
        artifacts["losses"] = pd.DataFrame(losses)
        log_step(f"stage={stage}: finished in {(time.time() - started_at) / 60:.2f} min")
        return pd.DataFrame(rows), artifacts

    def run_temporal_80_20(
            self,
            stage: str,
            threshold: float,
            min_pos: int,
            k_values: Optional[List[int]] = None,
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        log_step(f"stage={stage}: preparing temporal 80/20 split; threshold={threshold}, min_pos={min_pos}")
        df_pos = build_edges(self.rates, threshold=threshold)
        df_pos = filter_users_min_pos(df_pos, min_pos=min_pos)
        train_df, test_df = temporal_train_test_split(df_pos, test_ratio=self.config.test_ratio)
        log_step(f"stage={stage}: positive stats")
        print(interactions_stats(df_pos))
        log_step(f"stage={stage}: train/test shapes: train={train_df.shape}, test={test_df.shape}")
        return self.fit_single(stage=stage, train_df=train_df, test_df_eval=test_df, k_values=k_values or self.config.k_values)

    def run_context_research(self, threshold: float, min_pos: int, stages: List[str]) -> pd.DataFrame:
        all_rows = []
        for stage in stages:
            print()
            print("=== stage:", stage, "===")
            result_df, _ = self.run_temporal_80_20(stage=stage, threshold=threshold, min_pos=min_pos)
            result_df["threshold"] = threshold
            result_df["min_pos"] = min_pos
            all_rows.append(result_df)
        return pd.concat(all_rows, ignore_index=True)

    def run_cold_start(
            self,
            stage: str,
            threshold: float,
            min_pos: int,
            cold_n: int,
            k_values: Optional[List[int]] = None,
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        df_pos = build_edges(self.rates, threshold=threshold)
        df_pos = filter_users_min_pos(df_pos, min_pos=min_pos)

        split_result = make_synthetic_cold_start_split(
            interactions=df_pos,
            cold_user_fraction=self.config.cold_user_fraction,
            cold_n=cold_n,
            min_interactions_for_cold=self.config.min_interactions_for_cold,
            warm_last_n=self.config.warm_last_n,
            random_state=self.config.seed,
        )

        eval_users = get_eval_users(split_result, mode="cold_only")
        test_df_eval = filter_interactions_by_users(split_result.test_df, eval_users)
        print(summarize_split_result(split_result))
        result_df, artifacts = self.fit_single(
            stage=stage,
            train_df=split_result.train_df,
            test_df_eval=test_df_eval,
            k_values=k_values or self.config.k_values,
        )
        result_df["cold_n"] = cold_n
        artifacts["split_result"] = split_result
        return result_df, artifacts

    def run_cold_start_grid(
            self,
            stage: str,
            threshold: float,
            min_pos: int,
            cold_n_values: Optional[List[int]] = None,
            k_values: Optional[List[int]] = None,
    ) -> pd.DataFrame:
        rows = []
        for cold_n in (cold_n_values or self.config.cold_n_values):
            print()
            print("=== cold_n:", cold_n, "===")
            result_df, _ = self.run_cold_start(
                stage=stage,
                threshold=threshold,
                min_pos=min_pos,
                cold_n=cold_n,
                k_values=k_values or self.config.k_values,
            )
            result_df["threshold"] = threshold
            result_df["min_pos"] = min_pos
            rows.append(result_df)
        return pd.concat(rows, ignore_index=True)


EdgeType = Tuple[str, str, str]


@dataclass
class FeatureConfig:
    node_type: str
    edge_type: EdgeType
    reverse_edge_type: Optional[EdgeType] = None
    display_name: str = ""


@dataclass
class GraphSchema:
    node_features: Dict[str, Dict[str, FeatureConfig]]

    def get_features(self, node_type: str) -> Dict[str, FeatureConfig]:
        return self.node_features.get(node_type, {})


@dataclass
class EdgeGroupImportance:
    edge_type: EdgeType
    base_score: float
    masked_score: float
    importance: float


@dataclass
class LocalFeatureEvidence:
    feature_name: str
    feature_node_idx: int
    support_movies: List[int]
    support_count: int


@dataclass
class LocalExplanation:
    user_idx: int
    movie_idx: int
    base_score: float
    edge_group_importance: List[EdgeGroupImportance]
    local_feature_evidence: Dict[str, List[LocalFeatureEvidence]]


def build_default_graph_schema(data: HeteroData) -> GraphSchema:
    movie_features = {}
    candidates = {
        "genre": ("movie", "has_genre", "genre"),
        "actor": ("movie", "has_actor", "actor"),
        "director": ("movie", "has_director", "director"),
        "country": ("movie", "has_country", "country"),
        "tag": ("movie", "has_tag", "tag"),
    }
    for name, edge_type in candidates.items():
        if edge_type in data.edge_types:
            movie_features[name] = FeatureConfig(
                node_type=name,
                edge_type=edge_type,
                reverse_edge_type=(edge_type[2], f"rev_{edge_type[1]}", edge_type[0]),
                display_name=name,
            )
    return GraphSchema(node_features={"movie": movie_features})


class RecommendationSubgraphAnalyzer:


   def __init__(self, model, data: HeteroData, schema: GraphSchema, train_user_items_idx: Dict[int, Set[int]]):
       self.model = model
       self.data = data
       self.schema = schema
       self.train_user_items_idx = train_user_items_idx
       self.model.eval()


   @torch.no_grad()
   def score_pair(self, user_idx: int, movie_idx: int, data: Optional[HeteroData] = None) -> float:
       """
        Вычисляет score(u, i) итоговых embedding пользователя и фильма.
        """
       graph = data if data is not None else self.data
       embeddings = self.model.encode(graph)
       user_embedding = embeddings["user"][user_idx]
       movie_embedding = embeddings["movie"][movie_idx]
       return float((user_embedding * movie_embedding).sum().item())


   def mask_edge_type(self, edge_type: EdgeType, data: Optional[HeteroData] = None) -> HeteroData:
       """
        Удаляет все рёбра заданного типа.
        """
       graph = copy.deepcopy(data if data is not None else self.data)
       if edge_type in graph.edge_types:
           edge_index = graph[edge_type].edge_index
           graph[edge_type].edge_index = edge_index.new_empty((2, 0))
       return graph


   @torch.no_grad()
   def compute_edge_type_importance(self, user_idx: int, movie_idx: int, edge_type: EdgeType) -> EdgeGroupImportance:
       """Оценивает важность связи через изменение score после удаления рёбер данного типа."""
       base_score = self.score_pair(user_idx, movie_idx)
       masked_graph = self.mask_edge_type(edge_type)
       masked_score = self.score_pair(user_idx, movie_idx, masked_graph)
       return EdgeGroupImportance(
           edge_type=edge_type, base_score=base_score,
           masked_score=masked_score, importance=base_score - masked_score)


   def get_neighbors(self, node_type: str, node_idx: int, edge_type: EdgeType) -> Set[int]:
       """
        Возвращает соседей узла по заданному типу ребра.
        """
       if edge_type not in self.data.edge_types:
           return set()
       edge_index = self.data[edge_type].edge_index
       src_type, _, dst_type = edge_type


       if node_type == src_type:
           mask = edge_index[0] == node_idx
           return set(edge_index[1][mask].detach().cpu().tolist())


       if node_type == dst_type:
           mask = edge_index[1] == node_idx
           return set(edge_index[0][mask].detach().cpu().tolist())


       return set()


   def build_local_feature_evidence(self, user_idx: int, movie_idx: int) -> Dict[str, List[LocalFeatureEvidence]]:
       """
        Ищет локальные свидетельства: признаки рекомендованного фильма,
        которые также встречаются в истории пользователя.
        """
       watched_movies = self.train_user_items_idx.get(user_idx, set())
       movie_feature_configs = self.schema.get_features("movie")
       evidence_by_feature_type: Dict[str, List[LocalFeatureEvidence]] = {}
       for feature_name, feature_config in movie_feature_configs.items():
           recommended_movie_features = self.get_neighbors(
               node_type="movie", node_idx=movie_idx,
               edge_type=feature_config.edge_type)


           feature_evidences: List[LocalFeatureEvidence] = []
           for feature_node_idx in recommended_movie_features:
               support_movies = []


               for watched_movie_idx in watched_movies:
                   watched_movie_features = self.get_neighbors(
                       node_type="movie", node_idx=watched_movie_idx,
                       edge_type=feature_config.edge_type)


                   if feature_node_idx in watched_movie_features:
                       support_movies.append(watched_movie_idx)


               if support_movies:
                   feature_evidences.append(
                       LocalFeatureEvidence(
                           feature_name=feature_name,
                           feature_node_idx=feature_node_idx,
                           support_movies=sorted(support_movies),
                           support_count=len(support_movies)))


           evidence_by_feature_type[feature_name] = sorted(
               feature_evidences,key=lambda item: item.support_count,
               reverse=True)


       return evidence_by_feature_type


   def explain(self, user_idx: int, movie_idx: int, edge_types: Optional[List[EdgeType]] = None) -> LocalExplanation:
       """
        Формирует структурированное объяснение рекомендации.
        """
       if edge_types is None:
           edge_types = list(self.data.edge_types)


       base_score = self.score_pair(user_idx, movie_idx)
       edge_group_importance = [
           self.compute_edge_type_importance(
               user_idx=user_idx,movie_idx=movie_idx,
               edge_type=edge_type)
           for edge_type in edge_types]
       edge_group_importance = sorted(
           edge_group_importance,key=lambda item: item.importance,
           reverse=True)
       local_feature_evidence = self.build_local_feature_evidence(
           user_idx=user_idx,movie_idx=movie_idx)


       return LocalExplanation(
           user_idx=user_idx,movie_idx=movie_idx, base_score=base_score,
           edge_group_importance=edge_group_importance,
           local_feature_evidence=local_feature_evidence)
