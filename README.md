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

---

## Next steps to test the new platform

### 1. Install and run (minimal test, no DB)

**Backend**
```bash
cd "c:\Users\User\Desktop\Orbit Ventures\orbit-platform\backend"
pip install -r requirements.txt
```
Create `backend/.env` with:
```
ANTHROPIC_API_KEY=sk-ant-api03-...   # Required for prompt generation + chat
PORT=10000
```
Then:
```bash
python main.py
```
You should see: `Starting Orbit Platform API on port 10000`

**Frontend** (new terminal)
```bash
cd "c:\Users\User\Desktop\Orbit Ventures\orbit-platform\frontend"
npm install
npm run dev
```
Create `frontend/.env` with:
```
VITE_API_URL=http://localhost:10000
```
Then open **http://localhost:5174**

**Test flow**
1. Click Continue on Login (simulated login).
2. Choose **Admin / Boss** → enter a company name → Create Organization.
3. On **Company Setup**: paste a description (or click an example) → **Generate AI Context**.
4. Wait for the prompt to generate (calls Claude). You’ll see the generated system prompt.
5. Click **Continue to Dashboard** → send a chat message. The backend will use the fallback prompt (org not in DB yet).

Without Supabase, the generated prompt is **not** persisted; only the UI flow and prompt-generation API are tested.

---

### 2. Full test with persistence (Supabase)

1. Create a Supabase project at https://supabase.com (or use an existing one).
2. Run the migration: in Supabase Dashboard → SQL Editor, run the contents of `supabase/migrations/20260223000001_initial_schema.sql`.
3. In **backend/.env** add:
   ```
   SUPABASE_URL=https://YOUR_PROJECT.supabase.co
   SUPABASE_SERVICE_KEY=eyJ...   # Project Settings → API → service_role key
   ```
4. Restart the backend. Now **Company Setup** will save `company_description` and `system_prompt` to the `organizations` table.
5. Wire the frontend to create/join orgs in Supabase (replace the fake `crypto.randomUUID()` in `RoleSelection.tsx` with real Supabase RPCs) so that:
   - Admin creates an org and gets a real `org_id` + invitation code.
   - Team member joins by code and gets the same `org_id`.
   - Dashboard chat sends that `org_id`; the backend will load and use the stored system prompt.

---

### 3. Optional: add Supabase Auth

Replace the simulated login in `Login.tsx` with `@supabase/supabase-js` (e.g. `signInWithOAuth` for Google or `signInWithOtp` for magic link). Use the same Supabase project as in step 2 so user profiles and org membership stay in one place.
