"""
It is a service file.

We need to send a text reply to chat
"""

from __future__ import annotations

import logging
import os

import httpx #for async await

log = logging.getLogger("app.openwa")

#for connection of WA client we use this class
class OpenWAClient:
    #constructor 
    def __init__( self, base_url: str | None = None, api_key: str | None = None, session_id: str | None = None,  ) -> None:

        #either initliaze fro mconstructor or initlizae the default that is present inside .env file
        self.base_url = (base_url or os.environ["OPENWA_API_URL"]).rstrip("/") #to remove the last /
        self.api_key = api_key or os.environ["OPENWA_API_KEY"]
        requested = session_id or os.environ["OPENWA_SESSION_ID"]

        #to convert the session ID into UUID (Format that whatsapp allows)
        self.session_id = self._resolve_session_id(requested)

        # One client, reused — OpenWA sessions are long-lived.
        self._http = httpx.AsyncClient( base_url=self.base_url,headers={"X-API-Key": self.api_key},timeout=10.0, )

    def _resolve_session_id(self, requested: str) -> str:
        """UUIDs pass through unchanged. Names get resolved via /api/sessions."""
        # Looks like a UUID already (8-4-4-4-12 hex pattern).
        if len(requested) == 36 and requested.count("-") == 4:
            return requested

        # Synchronous httpx (we're in __init__, not async yet).
        try:
            #we get JSON from GET request
            with httpx.Client(timeout=10.0) as sync_http:
                r = sync_http.get(
                    f"{self.base_url}/api/sessions",
                    headers={"X-API-Key": self.api_key},
                )
                r.raise_for_status()

                #we loop through the json 
                for s in r.json():
                    #if name is requested then return ID
                    if s.get("name") == requested:
                        log.info("Resolved session %r -> %s", requested, s["id"])
                        return s["id"]
        except Exception as e:
            log.warning("Session-name resolution failed (%s); using %r as-is",
                        e, requested)

        log.warning("Session %r not found via API; passing it through anyway",
                    requested)
        return requested

    #for sending text to whatsapp
    async def send_text(self, chat_id: str, text: str) -> None:
        """POST /api/sessions/{sessionId}/messages/send-text

        `chat_id` comes straight from the inbound webhook payload
        (OpenWA delivers it as `payload.from`).
        """
        resp = await self._http.post(
            f"/api/sessions/{self.session_id}/messages/send-text",
            json={"chatId": chat_id, "text": text},
        )
        resp.raise_for_status() #it raises error if doesnt exists

    #it closes the connection
    async def aclose(self) -> None:
        await self._http.aclose()