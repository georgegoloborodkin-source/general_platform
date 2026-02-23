"""
Orbit Platform — Backend API

Company-agnostic intelligence system.
Dynamic system prompts generated from company descriptions.
"""

import os
import json
import asyncio
from typing import List, Dict, Any, Optional
from datetime import datetime

from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()

from prompt_builder import generate_system_prompt, FALLBACK_PROMPT

app = FastAPI(title="Orbit Platform API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
SUPABASE_URL = os.getenv("SUPABASE_URL", "")
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY", "")

# ──────────────────────────────────────────────
#  Models
# ──────────────────────────────────────────────

class CompanySetupRequest(BaseModel):
    """MD pastes company info during onboarding."""
    organization_id: str
    company_description: str
    company_name: Optional[str] = None

class CompanySetupResponse(BaseModel):
    system_prompt: str
    organization_id: str
    message: str

class ChatSource(BaseModel):
    title: Optional[str] = None
    file_name: Optional[str] = None
    snippet: Optional[str] = None

class ChatMessage(BaseModel):
    role: str
    content: str

class AskRequest(BaseModel):
    question: str
    sources: List[ChatSource] = []
    previous_messages: List[ChatMessage] = Field(default_factory=list)
    organization_id: Optional[str] = None

class AskResponse(BaseModel):
    answer: str


# ──────────────────────────────────────────────
#  Supabase helper
# ──────────────────────────────────────────────

async def get_supabase():
    """Lazy import to avoid issues if supabase not configured yet."""
    try:
        from supabase import create_client
        return create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)
    except Exception:
        return None


# ──────────────────────────────────────────────
#  Endpoints
# ──────────────────────────────────────────────

@app.get("/health")
async def health():
    return {"status": "ok", "service": "orbit-platform", "timestamp": datetime.utcnow().isoformat()}


@app.post("/company/setup", response_model=CompanySetupResponse)
async def setup_company(request: CompanySetupRequest):
    """
    MD pastes company description → we generate a system prompt → store it on the org.
    All team members in the same org will use this prompt.
    """
    if not request.company_description.strip():
        raise HTTPException(status_code=400, detail="Company description is required")

    # Generate the system prompt from the company description
    system_prompt = await generate_system_prompt(request.company_description)

    # Store on the organization
    sb = await get_supabase()
    if sb:
        try:
            sb.table("organizations").update({
                "company_description": request.company_description.strip(),
                "system_prompt": system_prompt,
                "updated_at": datetime.utcnow().isoformat(),
            }).eq("id", request.organization_id).execute()
        except Exception as e:
            print(f"[setup_company] DB update failed: {e}")

    return CompanySetupResponse(
        system_prompt=system_prompt,
        organization_id=request.organization_id,
        message="Company context saved. AI is now customized for your team.",
    )


@app.post("/company/regenerate-prompt", response_model=CompanySetupResponse)
async def regenerate_prompt(request: CompanySetupRequest):
    """Re-generate the system prompt (e.g., if MD updates the company description)."""
    return await setup_company(request)


@app.get("/company/{org_id}/context")
async def get_company_context(org_id: str):
    """
    Get the org's system prompt and company description.
    Team members call this to get the shared AI context.
    """
    sb = await get_supabase()
    if not sb:
        return {
            "organization_id": org_id,
            "company_description": "",
            "system_prompt": FALLBACK_PROMPT,
        }

    try:
        result = sb.table("organizations").select(
            "id, name, company_description, system_prompt, industry_hint"
        ).eq("id", org_id).single().execute()

        if result.data:
            return {
                "organization_id": result.data["id"],
                "company_description": result.data.get("company_description", ""),
                "system_prompt": result.data.get("system_prompt", "") or FALLBACK_PROMPT,
                "industry_hint": result.data.get("industry_hint", ""),
                "company_name": result.data.get("name", ""),
            }
    except Exception as e:
        print(f"[get_company_context] DB read failed: {e}")

    return {
        "organization_id": org_id,
        "company_description": "",
        "system_prompt": FALLBACK_PROMPT,
    }


@app.post("/ask", response_model=AskResponse)
async def ask(request: AskRequest):
    """Answer a question using the org's custom system prompt + provided sources."""
    system_prompt = FALLBACK_PROMPT

    # Load org-specific system prompt
    if request.organization_id:
        sb = await get_supabase()
        if sb:
            try:
                result = sb.table("organizations").select(
                    "system_prompt"
                ).eq("id", request.organization_id).single().execute()
                if result.data and result.data.get("system_prompt"):
                    system_prompt = result.data["system_prompt"]
            except Exception:
                pass

    # Build the full prompt
    prompt = _build_answer_prompt(
        question=request.question,
        sources=request.sources,
        previous_messages=request.previous_messages,
        system_prompt=system_prompt,
    )

    # Call Claude
    answer = await _call_claude(system_prompt, prompt)
    return AskResponse(answer=answer)


@app.post("/ask/stream")
async def ask_stream(request: AskRequest):
    """Streaming version of /ask."""
    system_prompt = FALLBACK_PROMPT

    if request.organization_id:
        sb = await get_supabase()
        if sb:
            try:
                result = sb.table("organizations").select(
                    "system_prompt"
                ).eq("id", request.organization_id).single().execute()
                if result.data and result.data.get("system_prompt"):
                    system_prompt = result.data["system_prompt"]
            except Exception:
                pass

    prompt = _build_answer_prompt(
        question=request.question,
        sources=request.sources,
        previous_messages=request.previous_messages,
        system_prompt=system_prompt,
    )

    async def generate():
        try:
            import httpx
            async with httpx.AsyncClient(timeout=120.0) as client:
                resp = await client.post(
                    "https://api.anthropic.com/v1/messages",
                    headers={
                        "x-api-key": ANTHROPIC_API_KEY,
                        "anthropic-version": "2023-06-01",
                        "content-type": "application/json",
                    },
                    json={
                        "model": "claude-sonnet-4-20250514",
                        "max_tokens": 4096,
                        "stream": True,
                        "system": system_prompt,
                        "messages": [{"role": "user", "content": prompt}],
                    },
                )
                async for line in resp.aiter_lines():
                    if line.startswith("data: "):
                        data = line[6:]
                        if data == "[DONE]":
                            break
                        try:
                            event = json.loads(data)
                            if event.get("type") == "content_block_delta":
                                text = event.get("delta", {}).get("text", "")
                                if text:
                                    yield f"data: {json.dumps({'text': text})}\n\n"
                        except json.JSONDecodeError:
                            pass
            yield "data: [DONE]\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")


# ──────────────────────────────────────────────
#  Internal helpers
# ──────────────────────────────────────────────

def _build_answer_prompt(
    question: str,
    sources: List[ChatSource],
    previous_messages: List[ChatMessage],
    system_prompt: str,
) -> str:
    """Build the user-facing prompt with sources and conversation history."""
    source_lines = []
    for idx, src in enumerate(sources[:20], start=1):
        title = src.title or src.file_name or f"Source {idx}"
        snippet = (src.snippet or "").strip()
        if len(snippet) > 2000:
            snippet = snippet[:2000] + "..."
        source_lines.append(f"[{idx}] {title}\n{snippet}")

    sources_block = "\n\n".join(source_lines) if source_lines else "(No sources provided)"

    history_block = ""
    if previous_messages:
        history_lines = []
        for msg in previous_messages[-10:]:
            role_label = "User" if msg.role == "user" else "Assistant"
            history_lines.append(f"{role_label}: {msg.content}")
        history_block = f"\n\n=== CONVERSATION HISTORY ===\n" + "\n".join(history_lines) + "\n=== END HISTORY ===\n"

    return f"""{history_block}

=== SOURCES ===
{sources_block}
=== END SOURCES ===

RULES:
1. Answer based on the sources provided. Cite using [1], [2], etc.
2. If conversation history is present, use it to resolve pronouns and context.
3. Be comprehensive and use the company's own terminology.
4. If sources don't contain relevant info, say so clearly.

Question: {question}"""


async def _call_claude(system_prompt: str, user_prompt: str) -> str:
    """Non-streaming Claude call."""
    import httpx
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(
                "https://api.anthropic.com/v1/messages",
                headers={
                    "x-api-key": ANTHROPIC_API_KEY,
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json",
                },
                json={
                    "model": "claude-sonnet-4-20250514",
                    "max_tokens": 4096,
                    "system": system_prompt,
                    "messages": [{"role": "user", "content": user_prompt}],
                },
            )
            resp.raise_for_status()
            data = resp.json()
            return data["content"][0]["text"]
    except Exception as e:
        return f"Sorry, I encountered an error: {str(e)}"


# ──────────────────────────────────────────────
#  Startup
# ──────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "10000"))
    print(f"Starting Orbit Platform API on port {port}")
    uvicorn.run(app, host="0.0.0.0", port=port)
