# Orbit Platform — Company-Agnostic Intelligence System

A generalized version of VentureOS that works for **any company type**: trading firms, marketing agencies, consulting, real estate, VC, and more.

## How it works

1. **MD/Boss signs up** → selects role → **pastes company details** (what the company does, industry, key terms)
2. Backend **auto-generates a custom system prompt** from that company description
3. Team members join with an invitation code → they share the same org context and AI personality
4. All RAG, chat, document upload, embeddings, knowledge graph work identically — only the AI "persona" adapts

## Architecture

```
orbit-platform/
├── frontend/          # React + Vite + Tailwind + Supabase Auth
│   └── src/
│       ├── pages/     # Login, Onboarding, Dashboard
│       ├── hooks/     # useAuth, useOrgConfig
│       ├── utils/     # API client, prompt helpers
│       └── types/     # TypeScript interfaces
├── backend/           # FastAPI + Claude/OpenAI
│   ├── main.py        # API server with dynamic prompt builder
│   └── requirements.txt
└── supabase/
    └── migrations/    # DB schema
```

## Key difference from VentureOS

VentureOS = hardcoded VC prompts ("pitch decks", "portfolio", "Series A", etc.)
Orbit Platform = **company description → auto-generated prompt** → same backend

## Quick start

```bash
# Frontend
cd frontend && npm install && npm run dev

# Backend
cd backend && pip install -r requirements.txt && python main.py
```
