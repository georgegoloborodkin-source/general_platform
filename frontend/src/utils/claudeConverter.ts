/**
 * Claude API Converter for Document Extraction
 * 
 * Cost Analysis:
 * - Claude 3.5 Sonnet: $3.00 per 1M input tokens, $15.00 per 1M output tokens
 * - Average 15-page deck: ~3,000 tokens = $0.009 per deck (< 1 cent)
 * - 500 decks/month = ~$4.50
 * 
 * This module uses the Claude API for:
 * 1. Smart extraction (pitch decks → structured JSON)
 * 2. Deal memo generation
 * 3. Risk flag detection
 */

// Types for the Deal extraction
export interface DealMemo {
  company: {
    name: string;
    sector: string;
    stage: string;
    location: string;
    foundedYear?: number;
    website?: string;
  };
  team: {
    founders: Array<{
      name: string;
      role: string;
      background?: string;
      linkedIn?: string;
    }>;
    teamSize?: number;
  };
  financials: {
    revenue?: string;
    burn?: string;
    runway?: string;
    targetRaise?: string;
    valuation?: string;
    previousRounds?: Array<{
      round: string;
      amount: string;
      date?: string;
      investors?: string[];
    }>;
  };
  product: {
    description: string;
    traction?: string;
    metrics?: Record<string, string>;
  };
  market: {
    tam?: string;
    sam?: string;
    som?: string;
    competitors?: string[];
  };
  risks: Array<{
    category: string;
    description: string;
    severity: 'low' | 'medium' | 'high';
  }>;
  highlights: string[];
  redFlags: string[];
  rawConfidence: number;
}

export interface ExtractionResult {
  success: boolean;
  data?: DealMemo;
  error?: string;
  tokensUsed?: {
    input: number;
    output: number;
  };
  costEstimate?: string;
}

// Decision types for the Decision Logger
export interface Decision {
  id: string;
  timestamp: string;
  actor: string;
  actionType: 'intro' | 'pass' | 'follow_up' | 'invest' | 'meeting' | 'due_diligence';
  startupName: string;
  context: {
    sector?: string;
    stage?: string;
    geo?: string;
    source?: string;
  };
  confidenceScore: number;
  outcome?: 'positive' | 'negative' | 'pending';
  notes?: string;
  documentId?: string;
}

export interface DecisionStats {
  totalDecisions: number;
  byAction: Record<string, number>;
  byOutcome: Record<string, number>;
  averageConfidence: number;
  topActors: Array<{ actor: string; count: number; winRate: number }>;
}

// ============================================================================
// CLAUDE API CONVERTER
// ============================================================================

const EXTRACTION_PROMPT = `You are an expert VC analyst. Extract structured information from this pitch deck or company document.

Return ONLY valid JSON matching this exact schema (no markdown, no explanation):
{
  "company": {
    "name": "string",
    "sector": "string (e.g., FinTech, HealthTech, SaaS)",
    "stage": "string (Pre-seed, Seed, Series A, etc.)",
    "location": "string",
    "foundedYear": number or null,
    "website": "string or null"
  },
  "team": {
    "founders": [{ "name": "string", "role": "string", "background": "string or null" }],
    "teamSize": number or null
  },
  "financials": {
    "revenue": "string or null (e.g., '$1.2M ARR')",
    "burn": "string or null (e.g., '$150k/mo')",
    "runway": "string or null (e.g., '18 months')",
    "targetRaise": "string or null",
    "valuation": "string or null",
    "previousRounds": [{ "round": "string", "amount": "string", "investors": ["string"] }] or []
  },
  "product": {
    "description": "string (2-3 sentences)",
    "traction": "string or null",
    "metrics": { "key": "value" } or {}
  },
  "market": {
    "tam": "string or null",
    "sam": "string or null",
    "som": "string or null",
    "competitors": ["string"] or []
  },
  "risks": [{ "category": "string", "description": "string", "severity": "low|medium|high" }],
  "highlights": ["string (key positive points)"],
  "redFlags": ["string (concerns or warnings)"],
  "rawConfidence": number (0-100, your confidence in extraction accuracy)
}

Document content:
`;

/**
 * Extract structured deal information using Claude API
 * Requires a server-side API key (do not expose in production).
 */
export async function extractWithClaude(
  documentText: string,
  apiKey?: string
): Promise<ExtractionResult> {
  if (!apiKey) {
    return {
      success: false,
      error: "Claude API key is required. Configure a backend proxy for production use.",
    };
  }

  try {
    const response = await fetch('https://api.anthropic.com/v1/messages', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'x-api-key': apiKey,
        'anthropic-version': '2023-06-01',
        'anthropic-dangerous-direct-browser-access': 'true',
      },
      body: JSON.stringify({
        model: 'claude-3-5-sonnet-20241022',
        max_tokens: 4096,
        messages: [
          {
            role: 'user',
            content: EXTRACTION_PROMPT + documentText,
          },
        ],
      }),
    });

    if (!response.ok) {
      const error = await response.text();
      throw new Error(`Claude API error: ${error}`);
    }

    const result = await response.json();
    const content = result.content[0].text;
    
    // Parse the JSON response
    const data = JSON.parse(content) as DealMemo;
    
    // Calculate cost estimate
    const inputTokens = result.usage?.input_tokens || Math.ceil(documentText.length / 4);
    const outputTokens = result.usage?.output_tokens || Math.ceil(content.length / 4);
    const inputCost = (inputTokens / 1_000_000) * 3.00;
    const outputCost = (outputTokens / 1_000_000) * 15.00;
    const totalCost = inputCost + outputCost;

    return {
      success: true,
      data,
      tokensUsed: { input: inputTokens, output: outputTokens },
      costEstimate: `$${totalCost.toFixed(4)} (Input: $${inputCost.toFixed(4)}, Output: $${outputCost.toFixed(4)})`,
    };
  } catch (error) {
    console.error('Claude extraction error:', error);
    return {
      success: false,
      error: error instanceof Error ? error.message : 'Unknown error',
    };
  }
}

// ============================================================================
// DECISION LOGGER UTILITIES
// ============================================================================

const DECISIONS_STORAGE_KEY = 'cis_decisions';

/**
 * Save a new decision to local storage
 */
export function saveDecision(decision: Omit<Decision, 'id' | 'timestamp'>): Decision {
  const newDecision: Decision = {
    ...decision,
    id: `dec_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
    timestamp: new Date().toISOString(),
  };

  const existing = loadDecisions();
  existing.push(newDecision);
  localStorage.setItem(DECISIONS_STORAGE_KEY, JSON.stringify(existing));
  
  return newDecision;
}

/**
 * Load all decisions from storage
 */
export function loadDecisions(): Decision[] {
  try {
    const raw = localStorage.getItem(DECISIONS_STORAGE_KEY);
    return raw ? JSON.parse(raw) : [];
  } catch {
    return [];
  }
}

/**
 * Update an existing decision
 */
export function updateDecision(id: string, updates: Partial<Decision>): Decision | null {
  const decisions = loadDecisions();
  const index = decisions.findIndex(d => d.id === id);
  if (index === -1) return null;
  
  decisions[index] = { ...decisions[index], ...updates };
  localStorage.setItem(DECISIONS_STORAGE_KEY, JSON.stringify(decisions));
  return decisions[index];
}

/**
 * Delete a decision
 */
export function deleteDecision(id: string): boolean {
  const decisions = loadDecisions();
  const filtered = decisions.filter(d => d.id !== id);
  if (filtered.length === decisions.length) return false;
  
  localStorage.setItem(DECISIONS_STORAGE_KEY, JSON.stringify(filtered));
  return true;
}

/**
 * Calculate decision statistics
 */
export function calculateDecisionStats(decisions: Decision[]): DecisionStats {
  if (decisions.length === 0) {
    return {
      totalDecisions: 0,
      byAction: {},
      byOutcome: {},
      averageConfidence: 0,
      topActors: [],
    };
  }

  // Count by action
  const byAction: Record<string, number> = {};
  const byOutcome: Record<string, number> = {};
  const actorStats: Record<string, { total: number; wins: number }> = {};
  let totalConfidence = 0;

  for (const d of decisions) {
    byAction[d.actionType] = (byAction[d.actionType] || 0) + 1;
    if (d.outcome) {
      byOutcome[d.outcome] = (byOutcome[d.outcome] || 0) + 1;
    }
    totalConfidence += d.confidenceScore;

    if (!actorStats[d.actor]) {
      actorStats[d.actor] = { total: 0, wins: 0 };
    }
    actorStats[d.actor].total++;
    if (d.outcome === 'positive') {
      actorStats[d.actor].wins++;
    }
  }

  const topActors = Object.entries(actorStats)
    .map(([actor, stats]) => ({
      actor,
      count: stats.total,
      winRate: stats.total > 0 ? Math.round((stats.wins / stats.total) * 100) : 0,
    }))
    .sort((a, b) => b.count - a.count)
    .slice(0, 5);

  const avgConfidence = decisions.length > 0 && totalConfidence > 0 
    ? Math.round(totalConfidence / decisions.length) 
    : 0;

  return {
    totalDecisions: decisions.length,
    byAction,
    byOutcome,
    averageConfidence: avgConfidence,
    topActors,
  };
}

/**
 * Export decisions to CSV
 */
export function exportDecisionsToCSV(decisions: Decision[]): string {
  const headers = ['ID', 'Timestamp', 'Actor', 'Action', 'Startup', 'Sector', 'Stage', 'Geo', 'Confidence', 'Outcome', 'Notes'];
  
  // Escape CSV values properly (handle quotes and commas)
  const escapeCSV = (value: any): string => {
    if (value === null || value === undefined) return '';
    const str = String(value);
    // If contains comma, quote, or newline, wrap in quotes and escape internal quotes
    if (str.includes(',') || str.includes('"') || str.includes('\n')) {
      return `"${str.replace(/"/g, '""')}"`;
    }
    return str;
  };
  
  const rows = decisions.map(d => [
    escapeCSV(d.id),
    escapeCSV(d.timestamp),
    escapeCSV(d.actor),
    escapeCSV(d.actionType),
    escapeCSV(d.startupName),
    escapeCSV(d.context?.sector || ''),
    escapeCSV(d.context?.stage || ''),
    escapeCSV(d.context?.geo || ''),
    escapeCSV(d.confidenceScore),
    escapeCSV(d.outcome || ''),
    escapeCSV(d.notes || ''),
  ]);
  
  return [headers.join(','), ...rows.map(r => r.join(','))].join('\n');
}

