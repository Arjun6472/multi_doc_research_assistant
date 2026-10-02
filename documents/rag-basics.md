# Hybrid Retrieval Notes

Reciprocal Rank Fusion (RRF) combines ranked search results from multiple retrievers. For each result, it adds a reciprocal-rank contribution of `1 / (k + rank)`. This lets dense vector search and sparse BM25 search contribute rankings without directly comparing their differently scaled scores.

This project uses an RRF constant of 60. A result found near the top of both dense and sparse rankings receives contributions from both lists. The fused candidates can then be reranked before the best passages are sent to the answer generator.

Dense retrieval finds passages with similar meanings using embeddings. Sparse retrieval finds passages that share important terms with the query using BM25. Hybrid retrieval combines both approaches, which can help when a question contains either paraphrased concepts or exact names and phrases.