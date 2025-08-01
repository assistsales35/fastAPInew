from typing import Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import http.client
import json
from http.client import IncompleteRead
from fastapi.middleware.cors import CORSMiddleware
from sarvamai import SarvamAI
import logging
import base64
import io

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Request Models ---
class STTRequest(BaseModel):
    audio_data: str

class ChatRequest(BaseModel):
    client_id: Optional[int] = Field(None, description="(Optional) Client ID from login")
    message: str = Field(..., description="User's message to Watson")
    thread_id: Optional[str] = Field(None, description="Watson thread ID for continuation")
    response_language: str = Field("en", description="Desired response language code")

# --- Constants ---
SARVAM_API_KEY = "sk_da44glpq_OA9xCFHJXPaR04Tqby71r7yj"
IAM_API_KEY = "88txk_r9cjzdNcB0O-Ss6l5VSjVO7HGRbjsJz5E-mknX"
agent_id = "4d61b57a-8def-418a-868b-73f595665a2b"

SUPPORTED_LANGUAGES = {
    "hi-IN": "हिंदी (Hindi)", "bn-IN": "বাংলা (Bengali)", "ta-IN": "தமிழ் (Tamil)",
    "te-IN": "తెలుగు (Telugu)", "gu-IN": "ગુજરાતી (Gujarati)", "kn-IN": "ಕನ್ನಡ (Kannada)",
    "ml-IN": "മലയാളം (Malayalam)", "mr-IN": "मराठी (Marathi)", "pa-IN": "ਪੰਜਾਬੀ (Punjabi)",
    "od-IN": "ଓଡ଼ିଆ (Odia)", "en-IN": "English",
    "as-IN": "অসমীয়া (Assamese)", "brx-IN": "बड़ो (Bodo)", "doi-IN": "डोगरी (Dogri)",
    "ks-IN": "कॉशुर (Kashmiri)", "kok-IN": "कोंकणी (Konkani)", "mai-IN": "मैथिली (Maithili)",
    "mni-IN": "মৈতৈলোন্ (Manipuri)", "ne-IN": "नेपाली (Nepali)", "sa-IN": "संस्कृतम् (Sanskrit)",
    "sat-IN": "ᱥᱟᱱᱛᱟᱲᱤ (Santali)", "sd-IN": "سنڌي (Sindhi)", "ur-IN": "اردو (Urdu)"
}

# --- Watson Integration ---
def process_with_watson(payload: dict) -> tuple[Optional[str], Optional[str]]:
    conn = http.client.HTTPSConnection("api.us-south.watson-orchestrate.cloud.ibm.com")
    headers = {'IAM-API_KEY': IAM_API_KEY, 'Content-Type': 'application/json'}
    url = "/instances/9c78ae5d-f8b4-440a-87a5-c00e954bf3e4/v1/orchestrate/runs/stream"
    conn.request("POST", url, json.dumps(payload), headers)
    res = conn.getresponse()

    try:
        data = res.read()
    except IncompleteRead as e:
        logger.warning(f"IncompleteRead: {e}")
        data = e.partial

    try:
        raw = data.decode("utf-8")
    except UnicodeDecodeError:
        return None, None

    response_text = ""
    thread_id: Optional[str] = None

    for line in raw.splitlines():
        if not line.strip():
            continue
        try:
            event = json.loads(line)
            if thread_id is None and event.get("data", {}).get("thread_id"):
                thread_id = event["data"]["thread_id"]
            if event.get("event") == "message.delta":
                delta = event["data"]["delta"]
                if delta.get("content"):
                    response_text += delta["content"][0]["text"]
            elif event.get("event") == "message.created":
                msg = event["data"]["message"]["content"]
                if isinstance(msg, list):
                    response_text = msg[0]["text"]
        except Exception as parse_error:
            logger.warning(f"Parse error: {parse_error}")
            continue

    return response_text.strip(), thread_id

# --- SarvamAI Speech-to-Text ---
async def process_audio_with_sarvam(audio_bytes: bytes) -> dict:
    client = SarvamAI(api_subscription_key=SARVAM_API_KEY)
    audio_file = io.BytesIO(audio_bytes)
    audio_file.name = "audio.wav"
    response = client.speech_to_text.translate(file=audio_file, model="saaras:v2.5")
    return {"transcription": response.transcript or "", "language_code": response.language_code or "auto"}

# --- Endpoint: /stt ---
@app.post("/stt")
async def speech_to_text(req: STTRequest) -> dict:
    try:
        logger.info("Processing audio via SarvamAI")
        audio_bytes = base64.b64decode(req.audio_data)
        return await process_audio_with_sarvam(audio_bytes)
    except Exception as e:
        logger.error(f"STT error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# --- Endpoint: /chat_with_agent ---
@app.post("/chat_with_agent")
async def chat_with_agent(req: ChatRequest) -> dict:
    try:
        # Log input
        if req.client_id is not None:
            logger.info(f"[Client {req.client_id}] → WatsonX: {req.message}")
        else:
            logger.info(f"→ WatsonX: {req.message}")

        # Build system message
        if req.client_id is not None:
            system_msg = (
                f"You are Sales Mate Master agent for client {req.client_id}. "
                "Collaborate with: sales_assist_cart, inventory_lookup, authentication_agent, "
                "billing_agent, mailer, cross_sell_agent, buy_again_recommender, offer_agent."
            )
        else:
            system_msg = "You are Sales Mate Master agent."

        context = {"system_message": system_msg}

        # Prepare payload
        payload = {
            "message": {"role": "user", "content": req.message},
            "additional_properties": {},
            "context": context,
            "agent_id": agent_id
        }
        if req.thread_id:
            payload["thread_id"] = req.thread_id

        # Debug print
        print("Watson payload:", payload)

        # Call Watson
        response_text, thread_id = process_with_watson(payload)
        if response_text is None:
            raise HTTPException(status_code=500, detail="Watson failed")

        # Optional translation
        translated_response: Optional[str] = None
        lang = req.response_language
        if lang in SUPPORTED_LANGUAGES and not lang.startswith('en'):
            try:
                logger.info(f"Translating to {lang}")
                tr = SarvamAI(api_subscription_key=SARVAM_API_KEY).text.translate(
                    input=response_text,
                    source_language_code="en-IN",
                    target_language_code=lang,
                    speaker_gender="Male",
                    mode="classic-colloquial",
                    enable_preprocessing=False,
                )
                translated_response = tr.translated_text
            except Exception as e:
                logger.warning(f"Translation skipped: {e}")

        return {
            "response": response_text,
            "thread_id": thread_id or req.thread_id,
            "translated_response": translated_response,
        }

    except Exception as e:
        logger.error(f"/chat_with_agent error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# --- Endpoint: /supported_languages ---
@app.get("/supported_languages")
async def get_supported_languages() -> dict:
    return {
        "major": {code: name for code, name in SUPPORTED_LANGUAGES.items() if code in [
            "hi-IN","bn-IN","ta-IN","te-IN","gu-IN","kn-IN",
            "ml-IN","mr-IN","pa-IN","od-IN","en-IN"
        ]},
        "additional": {code: name for code, name in SUPPORTED_LANGUAGES.items() if code in [
            "as-IN","brx-IN","doi-IN","ks-IN","kok-IN",
            "mai-IN","mni-IN","ne-IN","sa-IN","sat-IN","sd-IN","ur-IN"
        ]}
    }
