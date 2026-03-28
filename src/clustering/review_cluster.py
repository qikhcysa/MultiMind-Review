"""Review clustering using DBSCAN + LLM-generated cluster summaries."""
from __future__ import annotations

import numpy as np

from src.models import ClusterResult, ClusterDetail
from src.rag import EmbeddingModel
from src.agents.llm_client import chat_complete


SUMMARY_SYSTEM = """你是一个评论总结专家。给定一组用户评论，提炼出最能代表这批评论核心主题的简短总结词或短语（不超过10个字）。
只返回总结词本身，不要解释。"""

SUMMARY_USER = """以下是关于"{dimension_name}"维度的一批评论：
{reviews}

请给出这批评论的核心总结词（≤10字）："""

# Minimum character counts for keyword extraction
_MIN_CHINESE_CHARS = 2   # minimum Chinese word length (characters)
_MIN_ENGLISH_CHARS = 4   # minimum English word length (letters)


class ReviewClusterer:
    """
    Clusters a set of reviews for a given dimension using DBSCAN on
    sentence-transformer embeddings, then generates cluster summary
    keywords using an LLM.
    """

    def __init__(
        self,
        embedding_model: EmbeddingModel | None = None,
        eps: float = 0.3,
        min_samples: int = 2,
        max_summary_samples: int = 10,
        use_llm: bool = True,
    ) -> None:
        self.embedding_model = embedding_model or EmbeddingModel()
        self.eps = eps
        self.min_samples = min_samples
        self.max_summary_samples = max_summary_samples
        self.use_llm = use_llm

    def cluster(
        self,
        reviews: list[str],
        dimension_id: str,
        dimension_name: str,
        sentiment_scores: list[float] | None = None,
    ) -> ClusterResult:
        """
        Cluster reviews for a specific dimension.

        Args:
            reviews:          List of review texts (already filtered for this dimension).
            dimension_id:     Dimension identifier.
            dimension_name:   Human-readable dimension name.
            sentiment_scores: Optional per-review sentiment scores (1-5).

        Returns:
            A :class:`ClusterResult` with cluster details and LLM summaries.
        """
        if not reviews:
            return ClusterResult(
                dimension_id=dimension_id,
                dimension_name=dimension_name,
                total_reviews=0,
                num_clusters=0,
                noise_count=0,
                clusters=[],
            )

        embeddings = self.embedding_model.encode(reviews)
        labels = self._dbscan(embeddings)

        unique_labels = set(labels)
        noise_count = int(np.sum(labels == -1))
        cluster_labels = sorted(l for l in unique_labels if l != -1)

        clusters: list[ClusterDetail] = []
        for cid in cluster_labels:
            mask = labels == cid
            cluster_reviews = [r for r, m in zip(reviews, mask) if m]
            cluster_scores = (
                [s for s, m in zip(sentiment_scores, mask) if m]
                if sentiment_scores
                else None
            )
            avg_score = (
                round(float(np.mean(cluster_scores)), 2) if cluster_scores else None
            )

            summary = self._generate_summary(cluster_reviews, dimension_name)

            # Pick up to 3 representative reviews (closest to cluster centroid)
            cluster_embs = embeddings[mask]
            centroid = cluster_embs.mean(axis=0)
            dists = np.linalg.norm(cluster_embs - centroid, axis=1)
            top_idx = np.argsort(dists)[:3]
            representative = [cluster_reviews[i] for i in top_idx]

            clusters.append(
                ClusterDetail(
                    cluster_id=int(cid),
                    size=int(mask.sum()),
                    summary=summary,
                    representative_reviews=representative,
                    avg_sentiment_score=avg_score,
                )
            )

        return ClusterResult(
            dimension_id=dimension_id,
            dimension_name=dimension_name,
            total_reviews=len(reviews),
            num_clusters=len(cluster_labels),
            noise_count=noise_count,
            clusters=clusters,
        )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _dbscan(self, embeddings: np.ndarray) -> np.ndarray:
        from sklearn.cluster import DBSCAN

        db = DBSCAN(eps=self.eps, min_samples=self.min_samples, metric="cosine")
        return db.fit_predict(embeddings)

    def _generate_summary(self, reviews: list[str], dimension_name: str) -> str:
        """Generate a cluster summary keyword using the LLM or a fallback."""
        if not reviews:
            return ""
        sample = reviews[: self.max_summary_samples]

        if self.use_llm:
            try:
                messages = [
                    {"role": "system", "content": SUMMARY_SYSTEM},
                    {
                        "role": "user",
                        "content": SUMMARY_USER.format(
                            dimension_name=dimension_name,
                            reviews="\n".join(f"- {r}" for r in sample),
                        ),
                    },
                ]
                result = chat_complete(messages, max_tokens=50)
                return result.strip()
            except Exception:
                pass  # fall through to keyword fallback

        # Simple keyword fallback: most common content words
        from collections import Counter
        import re

        pattern = (
            rf"[\u4e00-\u9fff]{{{_MIN_CHINESE_CHARS},}}"
            rf"|[a-zA-Z]{{{_MIN_ENGLISH_CHARS},}}"
        )
        words = re.findall(pattern, " ".join(sample))
        stopwords = {"这个", "一个", "非常", "感觉", "东西", "真的", "还是", "以及", "really", "very", "this", "that"}
        words = [w for w in words if w not in stopwords]
        if not words:
            return dimension_name
        most_common = Counter(words).most_common(3)
        return "、".join(w for w, _ in most_common)
