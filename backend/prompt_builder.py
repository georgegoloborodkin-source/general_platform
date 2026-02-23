"""
Dynamic System Prompt Builder

Takes a company description (pasted by the MD during onboarding)
and generates a full system prompt for the AI assistant.

The generated prompt is stored on the organization record so all
team members share the same AI personality and context.
"""

import os
import httpx

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")

PROMPT_GENERATOR_SYSTEM = """You are a system prompt engineer. Given a company description, generate a comprehensive system prompt for an AI assistant that will serve that company's team.

The system prompt you generate must:
1. Define the AI's identity (e.g., "You are Orbit AI, a [company type] intelligence system built for [team type]")
2. List 6-8 specific capabilities tailored to that company's work
3. Define what types of documents the team uploads (e.g., "strategy docs, risk reports" for trading; "campaign briefs, analytics" for marketing)
4. Define relationship/connection types relevant to the company (e.g., "Client, Vendor, Partner" etc.)
5. Define what KPIs/metrics to extract from documents
6. Define entity types to look for (e.g., "companies, people, projects, strategies")
7. Include a note about citing sources and being helpful

Output ONLY the system prompt text. No explanations, no markdown formatting, no code blocks. Just the raw prompt text that will be set as the AI's system message."""

FALLBACK_PROMPT = """You are Orbit AI, an enterprise intelligence system built for your team.

Your capabilities:
- Answer questions about uploaded documents (reports, proposals, contracts, notes)
- Extract structured information from unstructured documents
- Track decisions and outcomes across projects
- Provide insights from your team's knowledge base
- Search across all uploaded sources semantically
- Show relationships between companies, people, and projects
- Map connections between entities (Client, Vendor, Knowledge, Partnership, Internal)

When answering:
- Cite sources using [1], [2], etc. for every claim
- Be comprehensive and thorough
- If you don't have relevant information, say so clearly
- Focus on what the user is asking about"""


async def generate_system_prompt(company_description: str) -> str:
    """
    Call Claude to generate a tailored system prompt from the company description.
    Falls back to a generic prompt if the API call fails.
    """
    if not company_description or not company_description.strip():
        return FALLBACK_PROMPT

    if not ANTHROPIC_API_KEY:
        return _build_simple_prompt(company_description)

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(
                "https://api.anthropic.com/v1/messages",
                headers={
                    "x-api-key": ANTHROPIC_API_KEY,
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json",
                },
                json={
                    "model": "claude-sonnet-4-20250514",
                    "max_tokens": 1500,
                    "system": PROMPT_GENERATOR_SYSTEM,
                    "messages": [
                        {
                            "role": "user",
                            "content": f"Generate a system prompt for an AI assistant serving this company:\n\n{company_description}",
                        }
                    ],
                },
            )
            resp.raise_for_status()
            data = resp.json()
            generated = data["content"][0]["text"].strip()
            if len(generated) > 100:
                return generated
            return _build_simple_prompt(company_description)
    except Exception as e:
        print(f"[prompt_builder] Claude call failed: {e}, using fallback")
        return _build_simple_prompt(company_description)


def _build_simple_prompt(company_description: str) -> str:
    """Build a basic prompt from the description without an API call."""
    return f"""You are Orbit AI, an intelligent assistant built specifically for the following company:

{company_description}

Your capabilities:
- Answer questions about uploaded documents relevant to this company
- Extract structured information (entities, KPIs, relationships) from documents
- Track decisions and outcomes
- Provide insights from the team's knowledge base
- Search across all uploaded sources semantically
- Show relationships between entities mentioned in documents
- Help the team find information quickly

When answering:
- Cite sources using [1], [2], etc. for every claim
- Be comprehensive and thorough
- Use the company's own terminology and context
- If you don't have relevant information, say so clearly
- Focus on what the user is asking about"""
