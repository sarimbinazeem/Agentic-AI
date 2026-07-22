"""
It receives a whatsapp message
run the langgraph
send the reply back

"""

from __future__ import annotations
import asyncio
import hashlib
import hmac
#both are for webhook signature

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request #classes 

import httpx

from app.graph import build_graph
from app.openwa_client import OpenWAClient

# Load .env from the repo root when running natively (uvicorn app.main:app).
# Docker compose uses `env_file: .env`, so this is a no-op there.
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

log = logging.getLogger("app")  #for prettier printing


logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"),
                    format="%(asctime)s %(levelname)s %(message)s")

# Module-level client; lifespan manages its lifecycle.
_client: OpenWAClient | None = None

# HMAC secret OpenWA uses to sign outbound webhook payloads. If you don't
# want signature verification (e.g. running locally without a secret set),
# leave this unset and the verification step is skipped.
_WEBHOOK_SECRET = os.environ.get("OPENWA_WEBHOOK_SECRET", "").strip() #to check if someone SENT FAKE POST REQUEST


#the functioanlity of this function is to START PROGRAM, RUN CODE, START FASTAPI, END PROGRAM, CLEANUp
@asynccontextmanager
async def lifespan(app: FastAPI):
    #the    WA client is created only once and reused 

    global _client
    api_key = os.environ.get("OPENWA_API_KEY", "")
    if not api_key or api_key.startswith("replace-me"):
        # Fail loudly at boot rather than failing on first inbound POST.
        raise RuntimeError(
            "OPENWA_API_KEY is unset or still a placeholder. "
            "Create one in the OpenWA dashboard and put it in .env."
        )
    _client = OpenWAClient()
    log.info("OpenWA client ready: %s session=%s",_client.base_url, _client.session_id)
    if _WEBHOOK_SECRET:
        log.info("Webhook HMAC verification: ON")
    else:
        log.warning("Webhook HMAC verification: OFF (no OPENWA_WEBHOOK_SECRET)")

    #we build the graph in this lifespan so that AsyncSqliteSaver runs on the same event loop as ainvoke()
    app.state.graph = await build_graph()
    log.info("LangGraph compiled (async checkpointer ready)")

    yield
    await _client.aclose() #to close

#creates backend
app = FastAPI(title="whatsapp-bot-langgraph", lifespan=lifespan)


#to return dictrioy of health sttus
@app.get("/health")
async def health() -> dict[str, str]:
    """Cheap endpoint so we can curl from inside the compose network."""
    return {"status": "ok"}

#temporary function for debugging
@app.post("/debug-webhook")
async def debug_webhook(request: Request) -> dict[str, str]:
    """Temporary endpoint to see what headers OpenWA is sending."""
    headers = dict(request.headers)
    body = await request.body()
    log.info("DEBUG WEBHOOK - Headers: %s", headers)
    log.info("DEBUG WEBHOOK - Body: %s", body[:500])  # First 500 chars
    return {"status": "debug", "headers": str(headers)}


#it checks if WA Sends it *(is it trusted or not)
def _verify_signature(raw_body: bytes, signature_header: str | None) -> bool:
    """OpenWA sends `X-OpenWA-Signature: sha256=<hex>`. Compare in constant time."""
    if not _WEBHOOK_SECRET:
        return True  # verification disabled
    if not signature_header or not signature_header.startswith("sha256="):
        return False
    expected = hmac.new(
        _WEBHOOK_SECRET.encode("utf-8"), raw_body, hashlib.sha256
    ).hexdigest()
    provided = signature_header.split("=", 1)[1]
    return hmac.compare_digest(expected, provided)


#OpenWA sends the repyl to wahtsapp through this function
@app.post("/webhook")
async def webhook(request: Request) -> dict[str, str]:
    """Receive a single OpenWA event.

    OpenWA's payload (relevant fields):
      {
        "event": "message.received",
        "sessionId": "...",
        "payload": {
          "id": "...",
          "from": "1234567890@c.us",
          "body": "hello",
          "fromMe": false,
          ...
        }
      }
    """
    raw = await request.body() #get raw http BODY

    if not _verify_signature(raw, request.headers.get("X-OpenWA-Signature")):
        # 401 here will make OpenWA retry with backoff; safer than 200-ignore
        # when we *do* have a secret configured.
        raise HTTPException(status_code=401, detail="bad signature")

    try:
            
        #converts it into dictionary
        event: dict[str, Any] = await request.json()

    except Exception:
        # Probe / malformed body — return 200 so OpenWA doesn't retry,
        # but log loudly so we can see what came through.
        log.warning("non-JSON body on /webhook: %r", raw[:200])
        return {"status": "ignored-not-json"}
    

    #ignroe connection or irreleavnt message
    if event.get("event") != "message.received":
        # Acknowledge but do nothing — OpenWA expects 2xx on every event.
        return {"status": "ignored"}


    payload = event.get("payload") or event.get("data") or {}
    #ignore own emssages
    if payload.get("fromMe"):
        # Without this guard, our reply would be echoed back, generating
        # an infinite loop.
        return {"status": "ignored-our-own-message"}

    body = (payload.get("body") or "").strip()  #body is "How are you" and it removes User: from it

    #get chat id
    chat_id = payload.get("from")
    if not body or not chat_id:
        raise HTTPException(status_code=400, detail="missing body or from")
    
     # Newsletters and broadcasts can't be replied to (OpenWA returns 400).
    # @lid is newer WhatsApp privacy IDs — OpenWA *may* 500 on these, but
    # we attempt the send anyway and let _handle() catch the error gracefully.
    if chat_id.endswith(("@newsletter", "@broadcast")):
        log.warning("Skipping unreplyable chat_id=%s", chat_id)
        return {"status": "skipped-unreplyable-chat"}

    #Choosing Provider
    #we choose through / routing 
    #Anything else goes through FREELLMAPI
    provider = "free"
    persona: str | None = None
    lower = body.lower().lstrip()
    for prefix, name in (("/claude", "claude"), ("/gpt", "gpt")):
        if lower.startswith(prefix):
            provider = name
            body = body[len(prefix):].lstrip()
            log.info("routing to provider=%s", name)
            break

    # Then parse persona prefix (only one).
    for prefix, name in (("/resume", "resume"), ("/services", "services"), ("/personal", "personal")):
        if lower.startswith(prefix):
            persona = name
            lower = lower[len(prefix):].lstrip()
            log.info("routing to persona=%s", name)
            break
        
    # Whatever's left of text is what the LLM sees.
        body = lower.strip()
        if not body:
            # Slash command without any actual message.
            log.warning("empty body after slash parse (chat=%s)", chat_id)
            return {"status": "empty-after-slash"}

        # ACK fast — OpenWA's webhook timeout is ~10s, and an LLM call can
        # take 2–5s. If we block here, we'd risk hitting the timeout under
        # load and OpenWA would retry, doubling our work. Instead, hand off
        # to a background task and return immediately.
        initial_state = {"message": body, "reply": "", "provider": provider}
        if persona:
            initial_state["persona"] = persona

    #Open WA webhook takes alot of time that can cause timeout issues
    #we hand off it to a backgroudn task and let it run  and return immediately
    asyncio.create_task(_handle(request.app,chat_id, body, provider))
    return {"status": "queued"}

async def _handle(app, chat_id: str, initial_state: dict) -> None:
    """
    Background task: run the graph
    log the errors only dont raise it
    we do log only for debugging
    """
    try:
        """
        thread id is the one thing that keeps ocnversational history itnact
        checkpointer uses it to load previous information and save this session's hsitoy aswell
        so that next message sees the full history
        
        """
        config = {"configurable": {"thread_id": chat_id}}
        result = await app.state.graph.ainvoke(initial_state, config=config)
        reply = (result.get("reply") or "").strip()
        if not reply:
            log.warning("graph returned empty reply for chat=%s", chat_id)
            return

        log.info("chat=%s in=%r out=%r persona=%s provider=%s",
                 chat_id,
                 initial_state.get("message"),
                 reply,
                 result.get("persona", "?"),
                 initial_state.get("provider", "?"))
        try:
            await _client.send_text(chat_id=chat_id, text=reply)
        except httpx.HTTPStatusError as exc:
            body_text = ""
            try:
                body_text = exc.response.text[:200]
            except Exception:
                pass
            log.warning(
                "OpenWA send-text failed (chat_id=%s, status=%d): %s",
                chat_id, exc.response.status_code, body_text,
            )
        except httpx.HTTPError:
            log.exception("OpenWA send-text transport error for chat_id=%s", chat_id)

    except Exception:
        log.exception("Background handler crashed for chat_id=%s", chat_id)