import os
from functools import lru_cache

from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHROMA_DIR = os.path.join(BASE_DIR, "vector_db")   # absolute: same folder whatever directory you launch from
COLLECTION_NAME = "meetings"                       # new name: the old collection has no recording ids
EMBEDDING_MODEL = "all-MINILM-L6-v2"               # unchanged, so the cached model is reused
EMBED_CACHE = os.path.join(BASE_DIR, "models", "embeddings")
os.makedirs(EMBED_CACHE, exist_ok=True)


@lru_cache(maxsize=1)
def get_embeddings():
    # loaded once per process instead of on every build/load
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        cache_folder=EMBED_CACHE,
        model_kwargs={"device": "cpu"},
    )


def _store() -> Chroma:
    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=get_embeddings(),
        persist_directory=CHROMA_DIR,
    )


def build_vector_store(transcript: str, recording_id: str, title: str = "") -> Chroma:
    print("Building Vector Store...")
    vs = _store()
    vs._collection.delete(where={"recording_id": recording_id})   # re-run replaces, never duplicates
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    docs = [
        Document(
            page_content=chunk,
            metadata={"recording_id": recording_id, "title": title, "chunk_index": i},
        )
        for i, chunk in enumerate(splitter.split_text(transcript))
    ]
    vs.add_documents(docs)
    return vs


def load_vector_store() -> Chroma:
    return _store()


def delete_recording_vectors(recording_id: str) -> None:
    _store()._collection.delete(where={"recording_id": recording_id})


def get_retriever(vector_store: Chroma, k: int = 5, recording_id: str | None = None):
    """recording_id=None searches across every saved recording."""
    kwargs = {"k": k}
    if recording_id:
        kwargs["filter"] = {"recording_id": recording_id}
    return vector_store.as_retriever(search_type="similarity", search_kwargs=kwargs)