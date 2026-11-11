from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda

from core.vector_store import build_vector_store, load_vector_store, get_retriever
from core.llm import get_llm

SYSTEM_PROMPT = """You are an expert meeting assistant. Answer the user's question
based ONLY on the meeting transcript context provided below.
If the answer is not found in the context, say:
"I could not find this information in the meeting transcript."
Always be concise and precise. If quoting someone, mention it clearly.
Answer in Hinglish only if the user asks in Hinglish. Quotes stay in their original language.

Context from meeting transcript:
{context}"""


def format_docs(docs):
    parts = []
    for d in docs:
        title = d.metadata.get("title")
        parts.append(f"[{title}]\n{d.page_content}" if title else d.page_content)
    return "\n\n".join(parts)


def _make_chain(retriever):
    prompt = ChatPromptTemplate.from_messages([("system", SYSTEM_PROMPT), ("human", "{question}")])
    return (
        {"context": retriever | RunnableLambda(format_docs), "question": RunnablePassthrough()}
        | prompt | get_llm() | StrOutputParser()
    )


def build_rag_chain(transcript: str, recording_id: str, title: str = ""):
    vs = build_vector_store(transcript, recording_id, title)
    return _make_chain(get_retriever(vs, recording_id=recording_id))


def load_rag_chain(recording_id: str | None = None):
    """One recording if an id is given, otherwise search across all saved recordings."""
    return _make_chain(get_retriever(load_vector_store(), recording_id=recording_id))


def ask_question(rag_chain, question: str) -> str:
    return rag_chain.invoke(question)