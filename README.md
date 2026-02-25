# Company Platform — Company-Agnostic Intelligence System

A platform that works for **any company type**: trading firms, marketing agencies, consulting, real estate, and more.

## How it works

1. **Admin signs up** → selects role → **pastes company details** (what the company does, industry, key terms)
2. Backend **auto-generates a custom system prompt** from that company description
3. Team members join with an invitation code → they share the same org context and AI personality
4. All RAG, chat, document upload, embeddings, knowledge graph work identically — only the AI "persona" adapts

## Architecture

```
company-platform/
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

## Quick start

```bash
# Frontend
cd frontend && npm install && npm run dev

# Backend
cd backend && pip install -r requirements.txt && python main.py
```

## Deploying (Vercel)

- **Connect Vercel to this repo.**
- **Root Directory:** set to `frontend` so Vercel builds the app (not the repo root). In Vercel: Project → Settings → General → Root Directory → `frontend`.
- Pushes to `main` deploy from here; the backend runs on Render.
