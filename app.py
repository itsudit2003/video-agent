import datetime
import hashlib
import os
import tempfile
import time

import streamlit as st
from dotenv import load_dotenv

from utils.audio_processor import process_input
from core.transcriber import transcribe_all
from core.summarize import summarize, generate_title
from core.extractor import extract_action_items, extract_key_decisions, extract_questions
from core.rag_engine import build_rag_chain, load_rag_chain, ask_question
from core.vector_store import delete_recording_vectors
from core import db

load_dotenv(override=True)

st.set_page_config(
    page_title="Video Agent",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ------------------------------------------------------------------ styling
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;700&family=Space+Grotesk:wght@400;500;700&display=swap');

:root{
  --bg:#07090a; --lime:#c6ff3d; --lime-dim:rgba(198,255,61,.14);
  --ink:#e9eee4; --mute:#8b958a; --line:rgba(255,255,255,.09);
  --glass:rgba(255,255,255,.045);
}

/* canvas: obsidian + organic glows */
.stApp{
  background:
    radial-gradient(60% 50% at 88% -5%, rgba(198,255,61,.17), transparent 62%),
    radial-gradient(45% 40% at -5% 105%, rgba(70,255,170,.09), transparent 60%),
    var(--bg);
  color:var(--ink);
  font-family:'JetBrains Mono',monospace;
}
/* architectural grid */
.stApp::before{
  content:""; position:fixed; inset:0; z-index:0; pointer-events:none;
  background-image:
    linear-gradient(var(--line) 1px, transparent 1px),
    linear-gradient(90deg, var(--line) 1px, transparent 1px);
  background-size:56px 56px;
  -webkit-mask-image:radial-gradient(ellipse at 50% 0%, #000 15%, transparent 72%);
          mask-image:radial-gradient(ellipse at 50% 0%, #000 15%, transparent 72%);
}
/* grain overlay */
.stApp::after{
  content:""; position:fixed; inset:0; z-index:3; pointer-events:none; opacity:.07;
  mix-blend-mode:overlay;
  background-image:url("data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' width='160' height='160'><filter id='n'><feTurbulence type='fractalNoise' baseFrequency='.9' numOctaves='2' stitchTiles='stitch'/></filter><rect width='100%' height='100%' filter='url(%23n)'/></svg>");
}

header[data-testid="stHeader"], footer, #MainMenu, [data-testid="stToolbar"]{display:none !important;}

/* floating shell */
.block-container{
  position:relative; z-index:1;
  max-width:1080px; margin:2.2rem auto; padding:2.4rem 2.6rem 2.8rem !important;
  background:rgba(13,16,14,.55);
  border:1px solid var(--line); border-radius:28px;
  -webkit-backdrop-filter:blur(26px) saturate(140%); backdrop-filter:blur(26px) saturate(140%);
  box-shadow:0 40px 120px rgba(0,0,0,.65), inset 0 1px 0 rgba(255,255,255,.07);
}

h1,h2,h3{font-family:'Space Grotesk',sans-serif !important; letter-spacing:-.025em; color:var(--ink);}
p,li,label,input,textarea,button,.stMarkdown{font-family:'JetBrains Mono',monospace !important;}
.stMarkdown p, .stMarkdown li{font-size:.9rem; line-height:1.75; color:#d5dccf;}

/* top bar */
.va-bar{display:flex; justify-content:space-between; align-items:center; margin-bottom:2.4rem;}
.va-logo{font-family:'Space Grotesk',sans-serif; font-weight:700; font-size:1.05rem; display:flex; gap:.6rem; align-items:center;}
.va-logo i{width:11px; height:11px; border-radius:50%; background:var(--lime); box-shadow:0 0 14px 2px rgba(198,255,61,.7);}
.va-chip{font-size:.72rem; color:var(--lime); border:1px solid rgba(198,255,61,.35); background:var(--lime-dim);
  padding:.28rem .7rem; border-radius:999px;}

/* hero */
.va-hero h1{font-size:clamp(2.2rem,5vw,3.7rem); line-height:1.02; font-weight:700; margin:0 0 .9rem; max-width:15ch;}
.va-hero p{color:var(--mute); max-width:52ch; font-size:.92rem; line-height:1.7; margin-bottom:2rem;}

/* glass panels (st.container(border=True)) */
[data-testid="stVerticalBlockBorderWrapper"]{
  background:var(--glass) !important; border:1px solid var(--line) !important; border-radius:20px !important;
  -webkit-backdrop-filter:blur(14px); backdrop-filter:blur(14px);
}

/* inputs */
.stTextInput input, [data-testid="stFileUploaderDropzone"]{
  background:rgba(0,0,0,.35) !important; color:var(--ink) !important;
  border:1px solid var(--line) !important; border-radius:12px !important;
}
.stTextInput input:focus{border-color:var(--lime) !important; box-shadow:0 0 0 3px var(--lime-dim) !important;}
[data-testid="stFileUploaderDropzone"]{border-style:dashed !important;}
.stRadio label p{font-size:.85rem !important; color:var(--ink);}
.stRadio [role="radio"][aria-checked="true"] > div:first-child{background:var(--lime) !important; border-color:var(--lime) !important;}

/* buttons */
.stButton > button, .stDownloadButton > button{
  border-radius:12px; border:1px solid var(--line); background:var(--glass); color:var(--ink);
  font-weight:500; transition:all .18s ease;
}
.stButton > button:hover, .stDownloadButton > button:hover{border-color:var(--lime); color:var(--lime);}
.stButton > button[kind="primary"], [data-testid="stBaseButton-primary"]{
  background:var(--lime) !important; color:#0a0d05 !important; border:none !important; font-weight:700 !important;
  box-shadow:0 0 28px rgba(198,255,61,.32);
}
.stButton > button[kind="primary"]:hover{box-shadow:0 0 40px rgba(198,255,61,.5); color:#0a0d05 !important;}

/* tabs */
.stTabs [data-baseweb="tab-list"]{gap:.4rem; border-bottom:1px solid var(--line);}
.stTabs [data-baseweb="tab"]{background:transparent; color:var(--mute); font-size:.85rem; padding:.5rem .9rem;}
.stTabs [aria-selected="true"]{color:var(--lime) !important;}
.stTabs [data-baseweb="tab-highlight"]{background:var(--lime) !important;}

/* stats */
.va-stats{display:flex; gap:1rem; margin:1.4rem 0 1.6rem; flex-wrap:wrap;}
.va-stat{flex:1; min-width:140px; padding:1rem 1.2rem; border:1px solid var(--line); border-radius:16px; background:var(--glass);}
.va-stat b{display:block; font-family:'Space Grotesk',sans-serif; font-size:1.7rem; color:var(--lime);}
.va-stat span{font-size:.74rem; color:var(--mute);}

/* history rows */
.va-row-title{font-family:'Space Grotesk',sans-serif; font-weight:500; font-size:1rem; padding-top:.35rem;}
.va-row-meta{font-size:.74rem; color:var(--mute); padding-top:.55rem;}

/* transcript + chat */
.va-transcript{max-height:420px; overflow-y:auto; font-size:.82rem; line-height:1.8; color:#b9c2b3; white-space:pre-wrap;}
[data-testid="stChatMessage"]{background:var(--glass); border:1px solid var(--line); border-radius:16px;}
[data-testid="stChatInput"]{background:rgba(0,0,0,.4); border:1px solid var(--line); border-radius:14px;}
[data-testid="stStatusWidget"], [data-testid="stExpander"]{border-radius:16px;}

@media (max-width:700px){ .block-container{margin:.8rem; padding:1.4rem !important; border-radius:20px;} }
@media (prefers-reduced-motion:reduce){ *{transition:none !important;} }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

st.markdown(
    '<div class="va-bar"><div class="va-logo"><i></i>video agent</div>'
    '<span class="va-chip">pipeline ready</span></div>',
    unsafe_allow_html=True,
)


# ------------------------------------------------------------------ helpers
def analyze(source: str, language: str, rid: str) -> dict:
    """Runs the full pipeline with live progress and per-step timings."""
    t0 = time.time()
    with st.status("Analyzing recording", expanded=True) as status:
        t = time.time()
        st.write("Fetching audio and splitting it into chunks")
        chunks = process_input(source)
        st.write(f"Audio ready in {time.time() - t:.0f}s")

        t = time.time()
        st.write(f"Transcribing {len(chunks)} chunks ({language})")
        transcript = transcribe_all(chunks, language=language)
        st.write(f"Transcribed in {time.time() - t:.0f}s")

        t = time.time()
        st.write("Writing summary and title")
        summary = summarize(transcript)
        title = generate_title(summary)
        st.write(f"Summary and title in {time.time() - t:.0f}s")

        t = time.time()
        st.write("Extracting action items, decisions and open questions")
        actions = extract_action_items(transcript)
        decisions = extract_key_decisions(transcript)
        questions = extract_questions(transcript)
        st.write(f"Extraction in {time.time() - t:.0f}s")

        t = time.time()
        st.write("Indexing the transcript for chat")
        rag_chain = build_rag_chain(transcript, rid, title)
        st.write(f"Indexed in {time.time() - t:.0f}s")

        status.update(label="Analysis complete", state="complete", expanded=False)

    return {
        "title": title,
        "transcript": transcript,
        "summary": summary,
        "action_items": actions,
        "key_decisions": decisions,
        "open_questions": questions,
        "rag_chain": rag_chain,
        "chunks": len(chunks),
        "seconds": int(time.time() - t0),
    }


def report_md(r: dict) -> str:
    return (
        f"# {r['title']}\n\n## Summary\n{r['summary']}\n\n"
        f"## Action items\n{r['action_items']}\n\n"
        f"## Key decisions\n{r['key_decisions']}\n\n"
        f"## Open questions\n{r['open_questions']}\n\n"
        f"## Transcript\n{r['transcript']}\n"
    )


def open_recording(rid: str):
    r = db.get_recording(rid)
    r["rag_chain"] = load_rag_chain(rid)
    st.session_state.res = r
    st.session_state.rid = rid
    st.session_state.msgs = db.get_messages(rid)
    st.rerun()


# ------------------------------------------------------------------ state
if "res" not in st.session_state:
    st.session_state.res = None
    st.session_state.rid = None
    st.session_state.msgs = []

# ------------------------------------------------------------------ landing
if st.session_state.res is None:
    st.markdown(
        '<div class="va-hero"><h1>Drop in a recording. Leave with the decisions.</h1>'
        "<p>Paste a YouTube link or upload a meeting. You get a summary, action items, "
        "decisions, open questions, and a chat that answers from the transcript.</p></div>",
        unsafe_allow_html=True,
    )
    with st.container(border=True):
        mode = st.radio("Source", ["YouTube link", "Upload file"], horizontal=True, label_visibility="collapsed")
        url, upload = "", None
        if mode == "YouTube link":
            url = st.text_input("YouTube URL", placeholder="https://youtube.com/watch?v=...", label_visibility="collapsed")
        else:
            upload = st.file_uploader(
                "Audio or video file",
                type=["mp3", "wav", "m4a", "mp4", "mkv", "webm", "mov"],
                label_visibility="collapsed",
            )
        c1, c2 = st.columns([1, 2])
        language = c1.radio("Language", ["english", "hinglish"], horizontal=True)
        go = c2.button("Analyze recording", type="primary", use_container_width=True)

    if go:
        source = url.strip()
        raw = source.encode()
        label = source
        if upload is not None:
            raw = upload.getvalue()
            label = upload.name
            suffix = os.path.splitext(upload.name)[1]
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(raw)
                source = tmp.name

        if not source:
            st.warning("Add a YouTube link or upload a file first.")
        else:
            # same source + language = same id, so a repeat opens the saved result
            rid = hashlib.sha1(raw + language.encode()).hexdigest()[:12]
            if db.get_recording(rid):
                open_recording(rid)

            res = None
            try:
                res = analyze(source, language, rid)
            except Exception as e:
                st.error("The pipeline stopped. Full error below; also check .env keys, the Colab URL and ffmpeg.")
                st.exception(e)

            if res:
                db.save_recording(rid, res, label, language)
                st.session_state.res = res
                st.session_state.rid = rid
                st.session_state.msgs = []
                st.rerun()

    history = db.list_recordings()
    if history:
        st.markdown("### Past recordings")
        for row in history:
            a, b, c, d = st.columns([6, 3, 1.2, 1.2])
            a.markdown(f'<div class="va-row-title">{row["title"]}</div>', unsafe_allow_html=True)
            when = datetime.datetime.fromtimestamp(row["created_at"]).strftime("%d %b %Y, %H:%M")
            b.markdown(f'<div class="va-row-meta">{when} · {row["language"]}</div>', unsafe_allow_html=True)
            if c.button("Open", key=f"open_{row['id']}"):
                open_recording(row["id"])
            if d.button("Delete", key=f"del_{row['id']}"):
                db.delete_recording(row["id"])
                delete_recording_vectors(row["id"])
                st.rerun()

# ------------------------------------------------------------------ results
else:
    r = st.session_state.res
    rid = st.session_state.rid

    top_l, top_r = st.columns([4, 1])
    top_l.markdown(f"## {r['title']}")
    if top_r.button("All recordings", use_container_width=True):
        st.session_state.res = None
        st.session_state.rid = None
        st.session_state.msgs = []
        st.rerun()

    st.markdown(
        f'<div class="va-stats">'
        f'<div class="va-stat"><b>{len(r["transcript"].split()):,}</b><span>words transcribed</span></div>'
        f'<div class="va-stat"><b>{r["chunks"]}</b><span>audio chunks</span></div>'
        f'<div class="va-stat"><b>{r["seconds"]}s</b><span>processing time</span></div></div>',
        unsafe_allow_html=True,
    )

    t_sum, t_act, t_dec, t_q, t_tr = st.tabs(
        ["Summary", "Action items", "Key decisions", "Open questions", "Transcript"]
    )
    with t_sum:
        st.markdown(r["summary"])
    with t_act:
        st.markdown(r["action_items"])
    with t_dec:
        st.markdown(r["key_decisions"])
    with t_q:
        st.markdown(r["open_questions"])
    with t_tr:
        st.markdown(f'<div class="va-transcript">{r["transcript"]}</div>', unsafe_allow_html=True)

    st.download_button("Download report (.md)", report_md(r), file_name="video-agent-report.md", mime="text/markdown")

    st.markdown("### Chat with this recording")
    for m in st.session_state.msgs:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])

    if q := st.chat_input("Ask anything about this recording"):
        with st.chat_message("user"):
            st.markdown(q)
        with st.chat_message("assistant"):
            with st.spinner("Searching the transcript"):
                try:
                    ans = ask_question(r["rag_chain"], q)
                except Exception as e:
                    ans = f"I couldn't answer that: {e}"
            st.markdown(ans)
        st.session_state.msgs += [{"role": "user", "content": q}, {"role": "assistant", "content": ans}]
        db.add_message(rid, "user", q)
        db.add_message(rid, "assistant", ans)