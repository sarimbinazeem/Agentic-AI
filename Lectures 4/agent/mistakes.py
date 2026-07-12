"""
mistakes.py 

It automatically detects mistakes from histroy and extract rules from it

How it detects:
1)Verifier Fails To Diagonse
2)Tool Errors
3)User messages that depicts that the response is not satisfiable

"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

from agent import llm

@dataclass
class Mistake:
    """
    Mistake class from history
    """
    
    source:str 
    snippet:str #preview of the error
    why:str #reason of why the error happnened
    
    
#Error Prefixes lIst in regex string
_ERROR_PREFIX_RE = re.compile(
    r"\[error\]|\bTraceback \(most recent call last\)|\bUnicodeDecodeError"
    r"|\bFileNotFoundError\b|\bPermissionError\b|\bTimeoutExpired\b",
)


#User unsatisfiable remarks on the response of AI list as regex string
_USER_CORRECTION_RE = re.compile(
    r"^\s*(no|wrong|actually|instead|not what|i meant|incorrect|nah|nope"
    r"|that'?s not|not right)\b",
    re.IGNORECASE,
)


def detect_mistakes(history: list[dict], max_recent: int = 12) -> list[Mistake]:
    """
    check the mistakes of last 12 messages
    we check recent BECAUSE older messages are less important
    """
    
    mistakes: list[Mistake] = []
    
    #last 12 messages are stored in recent
    recent = history[-max_recent:]
    
    for msg in recent:
        role = msg.get("role")
        content = msg.get("content", "")
        if role not in ("user", "assistant"):
            continue #ignore system tool developer
        
        #Checking Verifier COach
        if role == "user" and "Verifier feedback" in content:
            m = re.search(r"Diagnosis:\s*(.+?)(?:\n|$)", content, re.DOTALL)
            if m:
                diag = m.group(1).strip()[:400]
                mistakes.append(Mistake(
                    source="verifier_fail",
                    snippet=diag,
                    why=f"Verifier rejected work: {diag[:120]}",
                ))
                
        #ToOOL Error
        if role == "user" and content.startswith("[Tool Result"):
            #Check if content have error prefix
            if _ERROR_PREFIX_RE.search(content):
                first_line = content.splitlines()[0][:120]
                mistakes.append(Mistake(
                    source="tool_error",
                    snippet=content[:400],
                    why=f"Tool returned an error: {first_line}",
                ))
                
        #User Pushbacks
        if role == "user":
            stripped = content.strip()
            if(
                len(stripped) <= 200 and
                len(stripped) >=3 and
                 _USER_CORRECTION_RE.match(stripped)
                and "Verifier feedback" not in stripped      # not our own feedback
                and "[Tool Result" not in stripped           # not a tool result echoed back
            ):
                mistakes.append(Mistake(
                    source="user_correction",
                    snippet=stripped[:200],
                    why=f"User pushed back: {stripped[:80]!r}",
                ))
                
    #We make a dupe of mistakes list
    seen: set[str] = set()
    deduped: list[Mistake] = []
    for m in mistakes:
        if m.snippet in seen:
            continue
        seen.add(m.snippet)
        deduped.append(m)
    return deduped


#EXTRACTION->  llm converts mistakes into rules
_EXTRACTION_PROMPT = """\
You are a rule-extractor for an AI agent harness. Given a list of
mistakes the agent made recently, produce 1-3 GENERALIZABLE rules
that would prevent the same mistake in future sessions.

Each rule must:
- Be specific and actionable (not vague like "be more careful")
- Use imperative voice ("Use X", "Avoid Y", "When Z, do W")
- Be tied to a CAUSE the agent can avoid next time
- Be a single sentence, ≤200 characters

INPUT (mistakes from recent history):
{mistakes_json}

OUTPUT — strict JSON, no prose, no markdown fences, no commentary:
{{
  "rules": [
    {{
      "text": "<imperative, single sentence, ≤200 chars>",
      "why": "<one-line trigger description>",
      "category": "<one of: tool_usage, verification, vague_tasks, prompt_injection, permission, general>"
    }}
  ]
}}

If the mistakes don't generalize (e.g. one-off typos), return
{{"rules": []}}.
Return ONLY the JSON object.
"""

def extract_rules(mistakes: list[Mistake]) -> list[dict]:
    """
    detect mistakes then create rules
    
    returned list's dictionary have text,why,category
    and returns empty list if there is no specifc rule 
    """
    
    if not mistakes:
        return []
    
    
    payload = [
        {"source": m.source, "why": m.why, "snippet": m.snippet}
        for m in mistakes
    ]
    
    #convert the paylod into json
    mistakes_json = json.dumps(payload, indent=2)
    
    #giving the extraction prompt to chat
    prompt = _EXTRACTION_PROMPT.format(mistakes_json=mistakes_json)
    raw = llm.chat(
        messages=[{"role": "user", "content": prompt}],
        plan_mode=True,    # JSON-only, no tools
    )
    
    try:
        parsed = json.loads(raw) #turns raw into python dict
    except json.JSONDecodeError:
        return []
    
    candidates = parsed.get("rules") or []
    #check if candidates and list are of same type or not
    if not isinstance(candidates, list):
        return [] 
    
    
    valid: list[dict] = [] #to hold good rules
    for c in candidates:
        if not isinstance(c, dict): #each rule should be a dictionar
            continue
        text = str(c.get("text", "")).strip()
        if not text or len(text) > 300: #ignore empty or large rules
            continue
        valid.append({
            "text": text,
            "why": str(c.get("why", "")).strip(),
            "category": str(c.get("category", "general")).strip() or "general",
        })
    return valid
     

    