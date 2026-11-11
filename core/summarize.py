from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
from core.llm import get_llm

def split_transcript(transcript: str) -> list:
    splitter = RecursiveCharacterTextSplitter(chunk_size=12000, chunk_overlap=300)  # 6000 if Devanagari
    return splitter.split_text(transcript)

def summarize(transcript: str) -> str:
    llm = get_llm()
    map_chain = ChatPromptTemplate.from_messages([
        ("system", "Summarize this part of a meeting transcript as bullet points. "
                   "Keep names, numbers, dates and decisions. Do not add anything that is not in the text."),
        ("human", "{text}"),
    ]) | llm | StrOutputParser()

    chunks = split_transcript(transcript)
    partials = []
    for i, chunk in enumerate(chunks):
        print(f"Summarizing part {i + 1}/{len(chunks)}...")
        partials.append(map_chain.invoke({"text": chunk}))

    if len(partials) == 1:
        return partials[0]

    combine_chain = ChatPromptTemplate.from_messages([
        ("system", "Merge these partial meeting summaries into one final summary in bullet points, "
                   "grouped by topic. Remove duplicates. Do not add anything that is not in the summaries."),
        ("human", "{text}"),
    ]) | llm | StrOutputParser()
    return combine_chain.invoke({"text": "\n\n".join(partials)})

def generate_title(text: str) -> str:
    llm = get_llm()
    chain = ChatPromptTemplate.from_messages([
        ("system", "Write a short professional meeting title (max 8 words). Return only the title."),
        ("human", "{text}"),
    ]) | llm | StrOutputParser()
    return chain.invoke({"text": text[:2000]}).strip().strip('"').strip()