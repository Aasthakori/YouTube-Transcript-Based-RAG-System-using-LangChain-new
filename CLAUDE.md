# YouTube Transcript RAG System

## What this project is
Upgraded version of a basic YouTube RAG chatbot. Built with LangChain (NOT LangGraph).
The old project is in reference_notebook.ipynb — read it to understand the starting point
but do NOT copy its patterns.

## Rules — follow these strictly
- Use Ollama (model: mistral) for LLM via ChatOllama — NEVER use OpenAI
- Use HuggingFaceEmbeddings (model: all-MiniLM-L6-v2) for embeddings
- NEVER hardcode any API keys or secrets in code
- Use .env file for all configuration
- All functions need type hints and docstrings
- Use LangChain LCEL patterns for all chains
- Persist FAISS index to disk after creation
- Every chunk must carry metadata: video_id, video_title, start_time, source_url
- Keep the project within LangChain — do NOT use LangGraph
- When ingesting a new video, merge into the existing FAISS index with add_documents()
  — never overwrite the whole index unless explicitly rebuilding from scratch

## Citation rules
- Every answer MUST cite sources using [Source N] format
- When formatting context for the prompt, label each chunk as [Source 1], [Source 2] etc.
- Each source label must include: video_title and start_time from chunk metadata
- After the answer, append a source list with clickable YouTube timestamp URLs
- Use format_with_sources() pattern to prepare context — never send raw chunks to the prompt
- If context is insufficient, respond "I don't have enough information from the videos"
  instead of guessing

## Tech stack
- LLM: Ollama (Mistral 7B) via langchain-ollama
- Embeddings: HuggingFace all-MiniLM-L6-v2
- Vector store: FAISS (persisted to disk)
- Retrieval: EnsembleRetriever (BM25 + FAISS) + ContextualCompressionRetriever with CrossEncoderReranker
- Framework: LangChain with LCEL
- Memory: RunnableWithMessageHistory + in-memory ChatMessageHistory (src/memory/session_store.py)
- Evaluation: RAGAS
- API: FastAPI
- UI: Streamlit
- Deployment: Docker

## Dependency notes (learned during build)
- Use `langchain_text_splitters` for RecursiveCharacterTextSplitter —
  `langchain.text_splitter` was removed in LangChain >= 0.3
- youtube-transcript-api v1.x changed from class-method `get_transcript()`
  to instance-based `YouTubeTranscriptApi().list(video_id)` + `.fetch()`;
  segments are now objects with `.text`, `.start`, `.duration` attributes,
  not plain dicts
- Exceptions live in `youtube_transcript_api._errors`, not the top-level package
  (VideoUnavailable, NoTranscriptFound, YouTubeRequestFailed etc.)
- yt-dlp is required for get_video_title() — add `yt-dlp>=2024.1.0` to requirements
- `langchain_text_splitters` must be listed explicitly in requirements.txt

## URL handling notes
- extract_video_id() must handle malformed URLs like `https://www.https://www.youtube.com/...`
  — these appear from copy-paste errors and should still parse correctly
- Supported formats: watch?v=, youtu.be/, embed/, and bare 11-char video IDs
- Playlist params (&list=, &index=) are silently ignored — only v= matters

## Multi-turn memory notes (learned during build)
- RunnableWithMessageHistory requires the inner chain to accept a dict input, not a plain str
  — use input_messages_key="question" and history_messages_key="chat_history"
- Invoke with config={"configurable": {"session_id": "..."}} — session_id is the key
- The condense step (rewriting follow-ups as standalone questions) must run BEFORE retrieval
  so the retriever gets a self-contained query, not a pronoun-heavy follow-up
- Mistral will sometimes emit its own **Sources:** block by copying the label format from
  the formatted context; strip any LLM-generated sources block in parse_citations() before
  appending the formatted one — otherwise the source list appears twice
- Build the chain once per session (outside the question loop), not once per query —
  BM25 index construction is expensive and session state must persist across turns
- hybrid.py has two functions: build_hybrid_retriever() (hardcoded k=6, weights=[0.4, 0.6])
  and build_ensemble_retriever() (uses config values). chains.py uses the former; keep them
  consistent or consolidate into one function that always reads from config

## Retrieval/reranker notes (learned during build)
- CrossEncoderReranker lives in langchain_classic, not langchain_community:
  `from langchain_classic.retrievers.document_compressors.cross_encoder_rerank import CrossEncoderReranker`
- EnsembleRetriever also lives in langchain_classic:
  `from langchain_classic.retrievers.ensemble import EnsembleRetriever`
- HuggingFaceCrossEncoder lives in langchain_community:
  `from langchain_community.cross_encoders import HuggingFaceCrossEncoder`
- ContextualCompressionRetriever wraps the EnsembleRetriever + CrossEncoderReranker;
  the reranker is the compressor, not an LLMChainExtractor
