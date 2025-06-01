"""
Hybrid search implementation with vector similarity and BM25 ranking.

This module provides advanced search capabilities including:
- Hybrid search combining vector similarity and BM25
- Maximal Marginal Relevance (MMR) for result diversification
- Query expansion and reranking
- Configurable search strategies
"""

import asyncio
import logging
import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import math
from collections import defaultdict, Counter

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

from ..core.config import get_settings
from ..core.exceptions import SearchError, RetrievalError
from ..embeddings.generator import EmbeddingGenerator

logger = logging.getLogger(__name__)

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')


@dataclass
class SearchResult:
    """Represents a search result."""
    document_id: str
    content: str
    score: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    vector_score: Optional[float] = None
    bm25_score: Optional[float] = None
    mmr_score: Optional[float] = None
    rank: Optional[int] = None


@dataclass
class SearchQuery:
    """Represents a search query."""
    text: str
    embedding: Optional[np.ndarray] = None
    filters: Dict[str, Any] = field(default_factory=dict)
    expanded_terms: List[str] = field(default_factory=list)


class BM25Scorer:
    """BM25 scoring implementation."""
    
    def __init__(self, k1: float = 1.2, b: float = 0.75):
        """
        Initialize BM25 scorer.
        
        Args:
            k1: Term frequency saturation parameter
            b: Length normalization parameter
        """
        self.k1 = k1
        self.b = b
        self.stemmer = PorterStemmer()
        self.stop_words = set(stopwords.words('english'))
        
        # Document statistics
        self.documents: List[List[str]] = []
        self.doc_lengths: List[int] = []
        self.avg_doc_length: float = 0.0
        self.doc_freq: Dict[str, int] = defaultdict(int)
        self.idf_cache: Dict[str, float] = {}
        
        logger.debug(f"Initialized BM25 scorer with k1={k1}, b={b}")
    
    def _preprocess_text(self, text: str) -> List[str]:
        """Preprocess text for BM25 scoring."""
        # Tokenize and convert to lowercase
        tokens = word_tokenize(text.lower())
        
        # Remove stopwords and non-alphabetic tokens
        tokens = [
            self.stemmer.stem(token)
            for token in tokens
            if token.isalpha() and token not in self.stop_words
        ]
        
        return tokens
    
    def fit(self, documents: List[str]) -> None:
        """
        Fit BM25 scorer on document corpus.
        
        Args:
            documents: List of document texts
        """
        logger.info(f"Fitting BM25 scorer on {len(documents)} documents")
        
        self.documents = []
        self.doc_lengths = []
        self.doc_freq = defaultdict(int)
        
        # Process each document
        for doc in documents:
            tokens = self._preprocess_text(doc)
            self.documents.append(tokens)
            self.doc_lengths.append(len(tokens))
            
            # Count term frequencies
            unique_tokens = set(tokens)
            for token in unique_tokens:
                self.doc_freq[token] += 1
        
        # Calculate average document length
        self.avg_doc_length = sum(self.doc_lengths) / len(self.doc_lengths) if self.doc_lengths else 0
        
        # Pre-calculate IDF values
        total_docs = len(documents)
        for term, freq in self.doc_freq.items():
            self.idf_cache[term] = math.log((total_docs - freq + 0.5) / (freq + 0.5))
        
        logger.info(f"BM25 fitting complete. Vocabulary size: {len(self.doc_freq)}")
    
    def score(self, query: str, doc_idx: int) -> float:
        """
        Calculate BM25 score for query against specific document.
        
        Args:
            query: Query text
            doc_idx: Document index
            
        Returns:
            BM25 score
        """
        if doc_idx >= len(self.documents):
            return 0.0
        
        query_tokens = self._preprocess_text(query)
        doc_tokens = self.documents[doc_idx]
        doc_length = self.doc_lengths[doc_idx]
        
        score = 0.0
        doc_token_counts = Counter(doc_tokens)
        
        for token in query_tokens:
            if token in doc_token_counts:
                # Calculate term frequency component
                tf = doc_token_counts[token]
                tf_component = (tf * (self.k1 + 1)) / (
                    tf + self.k1 * (1 - self.b + self.b * (doc_length / self.avg_doc_length))
                )
                
                # Get IDF component
                idf = self.idf_cache.get(token, 0.0)
                
                score += idf * tf_component
        
        return score
    
    def score_all(self, query: str) -> List[float]:
        """
        Calculate BM25 scores for query against all documents.
        
        Args:
            query: Query text
            
        Returns:
            List of BM25 scores
        """
        return [self.score(query, i) for i in range(len(self.documents))]


class MMRSelector:
    """Maximal Marginal Relevance selector for result diversification."""
    
    def __init__(self, lambda_param: float = 0.5):
        """
        Initialize MMR selector.
        
        Args:
            lambda_param: Balance between relevance and diversity (0-1)
        """
        self.lambda_param = lambda_param
        logger.debug(f"Initialized MMR selector with lambda={lambda_param}")
    
    def select(
        self,
        query_embedding: np.ndarray,
        candidate_embeddings: List[np.ndarray],
        candidate_scores: List[float],
        num_results: int
    ) -> List[int]:
        """
        Select diverse results using MMR.
        
        Args:
            query_embedding: Query embedding vector
            candidate_embeddings: List of candidate embeddings
            candidate_scores: Initial relevance scores
            num_results: Number of results to select
            
        Returns:
            List of selected candidate indices
        """
        if not candidate_embeddings or num_results <= 0:
            return []
        
        selected_indices = []
        candidate_indices = list(range(len(candidate_embeddings)))
        
        # Normalize embeddings for cosine similarity
        query_norm = query_embedding / np.linalg.norm(query_embedding)
        candidate_norms = [
            emb / np.linalg.norm(emb) if np.linalg.norm(emb) > 0 else emb
            for emb in candidate_embeddings
        ]
        
        for _ in range(min(num_results, len(candidate_indices))):
            best_score = float('-inf')
            best_idx = None
            
            for idx in candidate_indices:
                if idx in selected_indices:
                    continue
                
                # Relevance score (cosine similarity with query)
                relevance = np.dot(query_norm, candidate_norms[idx])
                
                # Diversity score (max similarity with selected items)
                diversity = 0.0
                if selected_indices:
                    similarities = [
                        np.dot(candidate_norms[idx], candidate_norms[sel_idx])
                        for sel_idx in selected_indices
                    ]
                    diversity = max(similarities) if similarities else 0.0
                
                # MMR score
                mmr_score = (
                    self.lambda_param * relevance -
                    (1 - self.lambda_param) * diversity
                )
                
                if mmr_score > best_score:
                    best_score = mmr_score
                    best_idx = idx
            
            if best_idx is not None:
                selected_indices.append(best_idx)
        
        logger.debug(f"MMR selected {len(selected_indices)} results")
        return selected_indices


class QueryExpander:
    """Query expansion using pseudo-relevance feedback."""
    
    def __init__(self, num_expansion_terms: int = 3):
        """
        Initialize query expander.
        
        Args:
            num_expansion_terms: Number of terms to add for expansion
        """
        self.num_expansion_terms = num_expansion_terms
        self.stemmer = PorterStemmer()
        self.stop_words = set(stopwords.words('english'))
        
        logger.debug(f"Initialized query expander with {num_expansion_terms} expansion terms")
    
    def expand(
        self,
        query: str,
        top_documents: List[str],
        method: str = "tfidf"
    ) -> List[str]:
        """
        Expand query with relevant terms from top documents.
        
        Args:
            query: Original query
            top_documents: Top retrieved documents for expansion
            method: Expansion method ("tfidf", "frequency")
            
        Returns:
            List of expansion terms
        """
        if not top_documents:
            return []
        
        if method == "tfidf":
            return self._expand_tfidf(query, top_documents)
        elif method == "frequency":
            return self._expand_frequency(query, top_documents)
        else:
            raise ValueError(f"Unknown expansion method: {method}")
    
    def _expand_tfidf(self, query: str, documents: List[str]) -> List[str]:
        """Expand query using TF-IDF scores."""
        # Create TF-IDF vectorizer
        vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            lowercase=True,
            token_pattern=r'\b[a-zA-Z]{3,}\b'
        )
        
        try:
            # Fit on documents
            tfidf_matrix = vectorizer.fit_transform(documents)
            feature_names = vectorizer.get_feature_names_out()
            
            # Calculate average TF-IDF scores
            avg_scores = np.mean(tfidf_matrix.toarray(), axis=0)
            
            # Get top terms (excluding query terms)
            query_terms = set(word_tokenize(query.lower()))
            candidates = []
            
            for idx, score in enumerate(avg_scores):
                term = feature_names[idx]
                if term not in query_terms and score > 0:
                    candidates.append((term, score))
            
            # Sort by score and return top terms
            candidates.sort(key=lambda x: x[1], reverse=True)
            expansion_terms = [term for term, _ in candidates[:self.num_expansion_terms]]
            
            logger.debug(f"TF-IDF expansion terms: {expansion_terms}")
            return expansion_terms
            
        except Exception as e:
            logger.warning(f"TF-IDF expansion failed: {e}")
            return []
    
    def _expand_frequency(self, query: str, documents: List[str]) -> List[str]:
        """Expand query using term frequency."""
        # Tokenize and count terms
        term_counts = Counter()
        query_terms = set(word_tokenize(query.lower()))
        
        for doc in documents:
            tokens = word_tokenize(doc.lower())
            for token in tokens:
                if (token.isalpha() and 
                    len(token) > 2 and 
                    token not in self.stop_words and 
                    token not in query_terms):
                    term_counts[token] += 1
        
        # Get top terms
        expansion_terms = [
            term for term, _ in term_counts.most_common(self.num_expansion_terms)
        ]
        
        logger.debug(f"Frequency expansion terms: {expansion_terms}")
        return expansion_terms


class HybridSearcher:
    """Hybrid search combining vector similarity and BM25."""
    
    def __init__(
        self,
        embedding_generator: EmbeddingGenerator,
        vector_weight: float = 0.7,
        bm25_weight: float = 0.3,
        mmr_lambda: float = 0.5,
        enable_query_expansion: bool = True
    ):
        """
        Initialize hybrid searcher.
        
        Args:
            embedding_generator: Embedding generator instance
            vector_weight: Weight for vector similarity scores
            bm25_weight: Weight for BM25 scores
            mmr_lambda: MMR lambda parameter for diversity
            enable_query_expansion: Whether to enable query expansion
        """
        self.embedding_generator = embedding_generator
        self.vector_weight = vector_weight
        self.bm25_weight = bm25_weight
        
        self.bm25_scorer = BM25Scorer()
        self.mmr_selector = MMRSelector(lambda_param=mmr_lambda)
        self.query_expander = QueryExpander() if enable_query_expansion else None
        
        # Document storage
        self.documents: List[str] = []
        self.document_embeddings: List[np.ndarray] = []
        self.document_metadata: List[Dict[str, Any]] = []
        self.is_fitted = False
        
        logger.info(
            f"Initialized hybrid searcher with weights: "
            f"vector={vector_weight}, bm25={bm25_weight}"
        )
    
    async def fit(
        self,
        documents: List[str],
        metadata: Optional[List[Dict[str, Any]]] = None
    ) -> None:
        """
        Fit the searcher on document corpus.
        
        Args:
            documents: List of document texts
            metadata: Optional metadata for each document
        """
        logger.info(f"Fitting hybrid searcher on {len(documents)} documents")
        
        self.documents = documents
        self.document_metadata = metadata or [{} for _ in documents]
        
        # Generate embeddings
        self.document_embeddings = await self.embedding_generator.generate_batch(documents)
        
        # Fit BM25 scorer
        self.bm25_scorer.fit(documents)
        
        self.is_fitted = True
        logger.info("Hybrid searcher fitting complete")
    
    async def search(
        self,
        query: str,
        num_results: int = 10,
        enable_mmr: bool = True,
        expand_query: bool = True,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        """
        Perform hybrid search.
        
        Args:
            query: Search query
            num_results: Number of results to return
            enable_mmr: Whether to apply MMR for diversification
            expand_query: Whether to expand the query
            filters: Optional filters to apply
            
        Returns:
            List of search results
        """
        if not self.is_fitted:
            raise SearchError("Searcher must be fitted before searching")
        
        logger.debug(f"Searching for: '{query}' (top {num_results})")
        
        # Create search query object
        search_query = SearchQuery(text=query, filters=filters or {})
        
        # Generate query embedding
        search_query.embedding = await self.embedding_generator.generate_single(query)
        
        # Expand query if enabled
        if expand_query and self.query_expander:
            # Get initial top results for expansion
            initial_results = await self._search_without_expansion(search_query, num_results * 2)
            top_docs = [result.content for result in initial_results[:5]]
            
            search_query.expanded_terms = self.query_expander.expand(query, top_docs)
            
            if search_query.expanded_terms:
                expanded_query = f"{query} {' '.join(search_query.expanded_terms)}"
                search_query.embedding = await self.embedding_generator.generate_single(expanded_query)
                logger.debug(f"Expanded query with terms: {search_query.expanded_terms}")
        
        # Perform search
        return await self._search_without_expansion(search_query, num_results, enable_mmr)
    
    async def _search_without_expansion(
        self,
        search_query: SearchQuery,
        num_results: int,
        enable_mmr: bool = True
    ) -> List[SearchResult]:
        """Perform search without query expansion."""
        # Calculate vector similarities
        vector_scores = self._calculate_vector_scores(search_query.embedding)
        
        # Calculate BM25 scores
        bm25_scores = self.bm25_scorer.score_all(search_query.text)
        
        # Normalize scores
        vector_scores_norm = self._normalize_scores(vector_scores)
        bm25_scores_norm = self._normalize_scores(bm25_scores)
        
        # Combine scores
        combined_scores = [
            self.vector_weight * v_score + self.bm25_weight * b_score
            for v_score, b_score in zip(vector_scores_norm, bm25_scores_norm)
        ]
        
        # Create candidate results
        candidates = []
        for i, (doc, metadata) in enumerate(zip(self.documents, self.document_metadata)):
            # Apply filters
            if not self._apply_filters(metadata, search_query.filters):
                continue
            
            result = SearchResult(
                document_id=metadata.get('id', str(i)),
                content=doc,
                score=combined_scores[i],
                metadata=metadata,
                vector_score=vector_scores[i],
                bm25_score=bm25_scores[i]
            )
            candidates.append((i, result))
        
        # Sort by combined score
        candidates.sort(key=lambda x: x[1].score, reverse=True)
        
        # Apply MMR if enabled
        if enable_mmr and len(candidates) > num_results:
            candidate_indices = [idx for idx, _ in candidates[:num_results * 2]]
            candidate_embeddings = [self.document_embeddings[idx] for idx in candidate_indices]
            candidate_scores = [candidates[i][1].score for i in range(len(candidate_indices))]
            
            selected_indices = self.mmr_selector.select(
                search_query.embedding,
                candidate_embeddings,
                candidate_scores,
                num_results
            )
            
            # Update results with MMR scores
            final_results = []
            for rank, local_idx in enumerate(selected_indices):
                global_idx = candidate_indices[local_idx]
                result = candidates[local_idx][1]
                result.mmr_score = result.score  # Store original score as MMR score
                result.rank = rank + 1
                final_results.append(result)
            
            logger.debug(f"Applied MMR selection: {len(final_results)} results")
            return final_results
        
        else:
            # Return top results without MMR
            final_results = []
            for rank, (_, result) in enumerate(candidates[:num_results]):
                result.rank = rank + 1
                final_results.append(result)
            
            return final_results
    
    def _calculate_vector_scores(self, query_embedding: np.ndarray) -> List[float]:
        """Calculate cosine similarity scores."""
        if not self.document_embeddings:
            return []
        
        # Stack embeddings for batch computation
        doc_embeddings_matrix = np.vstack(self.document_embeddings)
        
        # Calculate cosine similarities
        similarities = cosine_similarity([query_embedding], doc_embeddings_matrix)[0]
        
        return similarities.tolist()
    
    def _normalize_scores(self, scores: List[float]) -> List[float]:
        """Normalize scores to [0, 1] range."""
        if not scores:
            return []
        
        min_score = min(scores)
        max_score = max(scores)
        
        if max_score == min_score:
            return [1.0] * len(scores)
        
        return [(score - min_score) / (max_score - min_score) for score in scores]
    
    def _apply_filters(
        self,
        metadata: Dict[str, Any],
        filters: Dict[str, Any]
    ) -> bool:
        """Apply filters to determine if document should be included."""
        for key, value in filters.items():
            if key not in metadata:
                return False
            
            if isinstance(value, list):
                if metadata[key] not in value:
                    return False
            else:
                if metadata[key] != value:
                    return False
        
        return True
    
    async def search_similar(
        self,
        document_id: str,
        num_results: int = 10,
        exclude_self: bool = True
    ) -> List[SearchResult]:
        """
        Find documents similar to a given document.
        
        Args:
            document_id: ID of the reference document
            num_results: Number of similar documents to return
            exclude_self: Whether to exclude the reference document
            
        Returns:
            List of similar documents
        """
        if not self.is_fitted:
            raise SearchError("Searcher must be fitted before searching")
        
        # Find the reference document
        ref_idx = None
        for i, metadata in enumerate(self.document_metadata):
            if metadata.get('id') == document_id:
                ref_idx = i
                break
        
        if ref_idx is None:
            raise SearchError(f"Document with ID '{document_id}' not found")
        
        # Use document embedding as query
        ref_embedding = self.document_embeddings[ref_idx]
        vector_scores = self._calculate_vector_scores(ref_embedding)
        
        # Create results
        results = []
        for i, (doc, metadata, score) in enumerate(
            zip(self.documents, self.document_metadata, vector_scores)
        ):
            if exclude_self and i == ref_idx:
                continue
            
            result = SearchResult(
                document_id=metadata.get('id', str(i)),
                content=doc,
                score=score,
                metadata=metadata,
                vector_score=score
            )
            results.append(result)
        
        # Sort by similarity
        results.sort(key=lambda x: x.score, reverse=True)
        
        # Add ranks
        for rank, result in enumerate(results[:num_results]):
            result.rank = rank + 1
        
        logger.debug(f"Found {len(results[:num_results])} similar documents")
        return results[:num_results]


# Factory function for creating configured searcher
async def create_hybrid_searcher(
    embedding_generator: Optional[EmbeddingGenerator] = None,
    **kwargs
) -> HybridSearcher:
    """
    Create a configured hybrid searcher.
    
    Args:
        embedding_generator: Optional embedding generator
        **kwargs: Additional configuration parameters
        
    Returns:
        Configured HybridSearcher instance
    """
    if embedding_generator is None:
        from ..embeddings.generator import EmbeddingGenerator
        embedding_generator = EmbeddingGenerator()
    
    searcher = HybridSearcher(embedding_generator, **kwargs)
    logger.info("Created hybrid searcher")
    return searcher