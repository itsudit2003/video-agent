import os
import requests
from pydub import AudioSegment
from dotenv import load_dotenv
load_dotenv()

REMOTE_URL = os.getenv("WHISPER_REMOTE_URL")     
REMOTE_KEY = os.getenv("WHISPER_REMOTE_KEY")

SARVAM_PIECE_SECONDS = 25

# ---- local fallback (only imported if no remote URL is set) ----
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "small")
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "models", "whisper")
_model = None

SARVAM_API_KEY=os.getenv('SARVAM_API_KEY')

SARVAM_STT_TRANSLATE_URL = "https://api.sarvam.ai/speech-to-text-translate"
SARVAM_MODEL = os.getenv("SARVAM_STT_MODEL", "saaras:v2.5")

def load_model():
    global _model
    if _model is None:
        import whisper                      # lazy: torch not loaded when using remote
        os.makedirs(MODEL_DIR, exist_ok=True)
        _model = whisper.load_model(WHISPER_MODEL, download_root=MODEL_DIR)
    return _model

def _transcribe_whisper_local(chunk_path: str, translate: bool = False):
    task = "translate" if translate else "transcribe"
    return load_model().transcribe(chunk_path, task=task)["text"]

# ---- remote ----
def _transcribe_whisper_remote(chunk_path, translate=False, retries=3):
    for attempt in range(retries):
        try:
            with open(chunk_path, "rb") as f:
                r = requests.post(
                    f"{REMOTE_URL}/transcribe",
                    files={"file": f},
                    data={"translate": str(translate).lower()},
                    headers={"x-key": REMOTE_KEY, "ngrok-skip-browser-warning": "1"},
                    timeout=900,
                )
            r.raise_for_status()
            return r.json()["text"]
        except requests.RequestException as e:
            print(f"Remote attempt {attempt + 1} failed: {e}")
    raise RuntimeError(f"Remote transcription failed for {chunk_path}")


def _send_to_sarvam(piece_path: str) -> str:
    """Send one ≤30s WAV file to Sarvam and return the English transcript."""
    headers = {"api-subscription-key": SARVAM_API_KEY}

    with open(piece_path, "rb") as f:
        files = {"file": (os.path.basename(piece_path), f, "audio/wav")}
        data = {"model": SARVAM_MODEL, "with_diarization": "false"}
        response = requests.post(
            SARVAM_STT_TRANSLATE_URL,
            headers=headers,
            files=files,
            data=data,
            timeout=120,
        )

    if not response.ok:
        print(f"\n❌ Sarvam returned {response.status_code}")
        print(f"Response body: {response.text}\n")
        response.raise_for_status()

    return response.json().get("transcript", "")


def transcribe_chunk_sarvam(chunk_path: str) -> str:
    """
    Sarvam sync API only accepts ≤30s audio. We split this chunk into
    25-second pieces, send each separately, and join the transcripts.
    """
    if not SARVAM_API_KEY:
        raise RuntimeError("SARVAM_API_KEY is not set in environment / .env")

    audio = AudioSegment.from_wav(chunk_path)
    piece_ms = SARVAM_PIECE_SECONDS * 1000

    full_text = ""
    total_pieces = (len(audio) + piece_ms - 1) // piece_ms

    for i, start in enumerate(range(0, len(audio), piece_ms)):
        piece = audio[start: start + piece_ms]
        piece_path = f"{chunk_path}_sv_{i}.wav"
        piece.export(piece_path, format="wav")

        try:
            print(f"  → Sarvam piece {i + 1}/{total_pieces} ...")
            full_text += _send_to_sarvam(piece_path) + " "
        finally:
            if os.path.exists(piece_path):
                os.remove(piece_path)

    return full_text.strip()

def transcribe_chunk(chunk_path: str, language: str = 'english') -> str:
    if language.lower() != 'hinglish':
        if REMOTE_URL:
            print(f"[REMOTE] using Colab at {REMOTE_URL}")
            return _transcribe_whisper_remote(chunk_path)
        print("[LOCAL] using local Whisper")
        return _transcribe_whisper_local(chunk_path)
    return transcribe_chunk_sarvam(chunk_path)

def transcribe_all(chunks: list, language: str = "english") -> str:
    full_transcript = "" 
    engine = "Sarvam AI" if language.lower() == "hinglish" else "Whisper"
    print(f"Using {engine} for transcription.")
    for i, chunk in enumerate(chunks):  
        print(f"Transcribing chunk {i + 1}/{len(chunks)}...")
        text = transcribe_chunk(chunk, language=language)  
        full_transcript += text + " "  
    print("Transcription complete.")
    return full_transcript.strip()  