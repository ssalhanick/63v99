# Legal Citation Hallucination Detector
## Who, what and why?
- Mistrust in the output of LLMs is a major reason why the legal field has been slow to implement AI into their workflows.
    - Cases can be overturned
    - Lawyers can be disbarred
- LLMs can generate incorrect citations in 3 ways:
    - Corrupted metadata
    - Correctly cited, but incorrectly explained
    - Complete fabrication
- Verit aims to resolve this by establishing trust in the implementation of AI into legal brief writingimprove trust in AI-assisted legal drafting
- Target Audience: lawyers, law firms and legal researchers
## Data Sources
### Main Dataset (CourtListener.com/api)
- 2,000 4th amendment cases from 2010-2015 paired down to 1,353
    - Tossed out duplicates and cases that did not have full opinion text, citation metadata or court/year properties
### Benchmark Dataset 
- "Real Citations" (CLERC – Hugging Face)
    - Pre-chunked U.S. legal case retrieval dataset with labeled citation pairs. 
- Hallucinated Citations (Claude Haiku)
    - Prompted to generated fabricated citations
    - Corrupting real citations by altering year, volume or page number
    - Swapping real-but-unrelated cases from the corpus
## Preprocessing
1. HTML stripping, court header/footer removal and citation token normalization
2. Pruning cases with 200-50,000 characters of usable plain text
3. SpaCy lemmatization for BM25 token corpus
4. Paragraph chunking (1 paragraph overlap) before legal-BERT inference
5. L2 normalization of 768-dim embedding before Milvus insertion
## RAG Structure
### Chunking
- Paragraph level chunking with 512-token ceiling
- 1-paragraph overlap
- Paragraph overlap preserves context for citations that span boundaries
- Truncating 50,000 characters reduces noise of procedural tail content, strengthens legal reasoning in embedding
### Metadata (Milvus Vectors)
- case_id (+ chunk_index per chunk)
- 768-dim embeddings
- 20.2k chunk vectors
### Indexing
- Reciprocal Rank Fusion (RRF) combines dense and sparse
#### Dense (HNSW)
- 768-dim Legal-BERT
- Cosine similarity
##### Hyperparameters
- M = 16
- efConstruction = 200
#### Sparse (BM25Okapi)
- Term Frequency
    - How strongly query terms appear in that case
- Inverse document frequency
    - How rare the legal term is across the whole corpus
- Okapi normalizes length so longer docs do not get unfairly boosted
## Graph Structure
- MATCH (c:Case)-[r:CITES]->(cited:Case) Return c, r, cited
- 1,353 Full Nodes
    - court_id: enables corpus metadata mismatch detection
    - cite_count: sets node size in citation graph
    - landmark: true: enables visual difference in citation graph rendering
- 14, 773 Stub Nodes
    - Any cited opinion that does not coincide with an id in the corpus
    - stub: true: distinguishes full corpus node from placeholder nodes in citation density score
- 30, 806 [CITES] Edges
## Embedding Design
- Encoder: nlpaueb/legal-bert-base-uncased
    - Domain-adapted on US legal text, outperforms general BERT on legal benchmarks
- Training Cases
    - Paragraph chunks (≤512 tokens, 1-paragraph overlap)
    - Mean-pooling over paragraph chunks
    - L2 Normalization
    - HNSW/cosine on 20k chunk vectors stored in Milvus
- User Queries
    - Context window embedded with same encoder
    - L2 normalization
    - cosine/ANN
- Pre-cleaning
    - [CITATION] normalization reduces token budget waste from raw citation strings and prevents embedding noise from opinion metadata (volume, author, page number)
## Vector Store and Graph Queries
- Milvus ANN search (top-k=5) fused with BM25 via RRF = Semantic_result.rrf_score
- Existence Check
    - Match (c:Case {id: $id}) Return c
- Citation Density
    - Counts distinct corpus cases sharing outbound citation targets with queried node
    - If a case shares citation targets with many corpus cases, it is less likely to be hallucinated
- Metadata Validation
    - Extract year and court from citation string, compares to node properties
    - Court alias mapping handles natural-language circuit names
    - Ex: 4th Circ.  ca4
## Coding Demo
### demo_samples
#### Real (From Corpus)
```
Once we do that, it becomes irrelevant that Smith was recently a passenger in the Malibu. While the Supreme Court has held that passengers in cars are seized during traffic stops, see Brendlin v. California, 551 U.S. 249, 251 (2007), it has not extended that holding to former passengers who have since exited the vehicle. Brendlin’s holding rests in part on the recognition that once a police officer stops a car, “a sensible person would not expect a police officer to allow people to come and go freely..."
```
### neo4j statements
#### Graph Size:
```
MATCH (c:Case)
WITH
  count(c) AS total_cases,
  count(CASE WHEN c.stub = false THEN 1 END) AS full_cases,
  count(CASE WHEN c.stub = true THEN 1 END) AS stub_cases
MATCH ()-[r:CITES]->()
RETURN total_cases, full_cases, stub_cases, count(r) AS cites_edges
```
#### Landmark Nodes
```
MATCH (c:Case {landmark: true})
RETURN c.id AS case_id, c.name AS case_name, c.year AS year, c.court_id AS court_id
ORDER BY year
```
#### One case's 2-hop citation neighborhood
```
MATCH path = (c:Case {id: 2744142})-[:CITES*1..2]->(n:Case)
RETURN path
LIMIT 100
```
#### Most cited targets
```
MATCH (src:Case)-[:CITES]->(dst:Case{stub:FALSE})
RETURN dst.id AS case_id, dst.name AS case_name, count(src) AS cited_by_count
ORDER BY cited_by_count DESC
LIMIT 10
```
## Evaluation Metrics
### Failure Cases
- 6 false negatives (test set): all metadata-corrupted citations with high RRF scores (0.031–0.033) and high connectivity density (22–29)
- Semantic layer and graph signals actively passed them as REAL, and the Metadata layer either validated them correctly or did not check
## Limitations
- Domain Scope
    - The prediction veracity is only as robust as the dataset.
    - If a case is not in the corpus, then it will automatically trigger hallucinated, even if it is real
- The Semantic Layer doesn’t discriminate, it gatekeeps calls to the LLM
    - The RRF threshold is 0.02 which allows for pretty much everything that has exists=True to pass
    - The semantic search provides the retrieval context for the LLM in the RAG pipeline, but does not give an actual verdict
- Corpus data quality
    - Corrupted source data produces undetectable false negatives
## Innovation
- Hybrid Search in Semantic Layer
    - BM25 + HNSW dense fused together with RRF
    - BM25 recovers exact legal terms
    - HNSW captures semantic paraphrasing across variations
- Reranking via RRF
    - Produces a fused score from two ranked lists (dense and sparse)
    - RRF avoids cross-scale score normalization and weight tuning between dense/sparse scores.
- Feature Reduction
    - UMAP Visualization over PCA because Legal-BERT embeddings lie on non-linear plane
    - n_neighbors = 15 preserves neighborhood topography while allowing global structure to compress
    - StandardScaler applied before dimensionality reduction to prevent high-variance dimensions from dominating 2D layer






