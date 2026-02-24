/**
 * AI converter client utility
 * Talks to the backend converter API (Claude or other provider).
 */

const ENV_CONVERTER_API_URL = import.meta.env.VITE_CONVERTER_API_URL as string | undefined;

function getDefaultBackendUrl(): string {
  if (typeof window !== "undefined" && window.location?.hostname === "localhost") return "http://localhost:10000";
  return "https://general-platform.onrender.com";
}

function buildCandidateBaseUrls(): string[] {
  if (ENV_CONVERTER_API_URL) return [ENV_CONVERTER_API_URL];
  return [getDefaultBackendUrl()];
}

let resolvedBaseUrl: string | null = null;

async function fetchWithTimeout(url: string, init?: RequestInit, timeoutMs = 800): Promise<Response> {
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), timeoutMs);
  try {
    return await fetch(url, { ...init, signal: controller.signal });
  } finally {
    clearTimeout(timeout);
  }
}

async function resolveConverterApiBaseUrl(): Promise<string> {
  if (resolvedBaseUrl) return resolvedBaseUrl;

  const candidates = buildCandidateBaseUrls();
  if (!candidates.length) {
    resolvedBaseUrl = getDefaultBackendUrl();
    return resolvedBaseUrl;
  }

  for (const base of candidates) {
    try {
      const res = await fetchWithTimeout(`${base}/health`, undefined, 800);
      if (res.ok) {
        resolvedBaseUrl = base;
        return base;
      }
    } catch {
      // try next
    }
  }

  // Nothing reachable; still return first candidate so error messages are consistent.
  resolvedBaseUrl = candidates[0];
  return resolvedBaseUrl;
}

export interface AIConversionRequest {
  data: string;
  dataType?: string;
  format?: string;
}

export interface AIConversionResponse {
  startups: Record<string, any>[];
  investors: Record<string, any>[];
  mentors: Record<string, any>[];
  corporates: Record<string, any>[];
  detectedType: string;
  confidence: number;
  warnings: string[];
  errors: string[];
  raw_content?: string | null;
}

export interface AskFundSource {
  title?: string | null;
  snippet?: string | null;
  file_name?: string | null;
}

export interface AskFundDecision {
  startup_name?: string | null;
  action_type?: string | null;
  outcome?: string | null;
  notes?: string | null;
}

export interface AskFundConnection {
  source_company_name: string;
  target_company_name: string;
  connection_type: string;
  connection_status: string;
  ai_reasoning?: string | null;
  notes?: string | null;
}

export interface ChatMessage {
  role: "user" | "assistant";
  content: string;
}

/**
 * Convert unstructured data using the converter API
 */
export async function convertWithAI(
  data: string,
  dataType?: "startup" | "investor"
): Promise<AIConversionResponse> {
  try {
    const baseUrl = await resolveConverterApiBaseUrl();
    const response = await fetch(`${baseUrl}/convert`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        data,
        dataType,
      } as AIConversionRequest),
    });

    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.detail || `HTTP error! status: ${response.status}`);
    }

    const result: AIConversionResponse = await response.json();

    // Convert to our internal types
    return {
      startups: (result.startups || []).map((s) => ({
        id: `startup-${Date.now()}-${Math.random()}`,
        companyName: s.companyName,
        geoMarkets: s.geoMarkets,
        industry: s.industry,
        fundingTarget: s.fundingTarget,
        fundingStage: s.fundingStage,
        availabilityStatus: s.availabilityStatus as "present" | "not-attending",
      })),
      investors: (result.investors || []).map((i) => ({
        id: `investor-${Date.now()}-${Math.random()}`,
        firmName: i.firmName,
        memberName: (i as any).memberName || "UNKNOWN",
        geoFocus: i.geoFocus,
        industryPreferences: i.industryPreferences,
        stagePreferences: i.stagePreferences,
        minTicketSize: i.minTicketSize,
        maxTicketSize: i.maxTicketSize,
        totalSlots: i.totalSlots,
        tableNumber: i.tableNumber,
        availabilityStatus: i.availabilityStatus as "present" | "not-attending",
      })),
      mentors: (result.mentors || []).map((m: any) => ({
        id: `mentor-${Date.now()}-${Math.random()}`,
        fullName: m.fullName,
        email: m.email,
        linkedinUrl: m.linkedinUrl,
        geoFocus: m.geoFocus || [],
        industryPreferences: m.industryPreferences || [],
        expertiseAreas: m.expertiseAreas || [],
        totalSlots: m.totalSlots || 3,
        availabilityStatus: (m.availabilityStatus as "present" | "not-attending") || "present",
      })),
      corporates: (result.corporates || []).map((c: any) => ({
        id: `corporate-${Date.now()}-${Math.random()}`,
        firmName: c.firmName,
        contactName: c.contactName,
        email: c.email,
        geoFocus: c.geoFocus || [],
        industryPreferences: c.industryPreferences || [],
        partnershipTypes: c.partnershipTypes || [],
        stages: c.stages || [],
        totalSlots: c.totalSlots || 3,
        availabilityStatus: (c.availabilityStatus as "present" | "not-attending") || "present",
      })),
      detectedType: result.detectedType,
      confidence: result.confidence,
      warnings: result.warnings,
      errors: result.errors,
      raw_content: result.raw_content ?? null,
    };
  } catch (error) {
    throw new Error(
      `AI conversion failed: ${error instanceof Error ? error.message : "Unknown error"}`
    );
  }
}

export async function askClaudeAnswer(input: {
  question: string;
  sources: AskFundSource[];
  decisions: AskFundDecision[];
}): Promise<{ answer: string }> {
  const baseUrl = await resolveConverterApiBaseUrl();
  const controller = new AbortController();
  // Increased timeout to 70 seconds to match backend (60s) + buffer
  const timeoutMs = 70000;
  const timeout = window.setTimeout(() => controller.abort(), timeoutMs);
  let response: Response | null = null;
  const sleep = (ms: number) => new Promise((resolve) => window.setTimeout(resolve, ms));
  try {
    for (let attempt = 0; attempt < 3; attempt += 1) {
      try {
        response = await fetch(`${baseUrl}/ask`, {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify(input),
          signal: controller.signal,
        });
        break;
      } catch (error) {
        if (error instanceof DOMException && error.name === "AbortError") {
          throw new Error("Claude request timed out after 70 seconds. The question may be too complex or the API is slow. Please try again with a simpler question.");
        }
        if (attempt < 2) {
          // Exponential backoff: 1s, 2s
          await sleep(1000 * (attempt + 1));
          continue;
        }
        throw error;
      }
    }
  } finally {
    window.clearTimeout(timeout);
  }

  if (!response || !response.ok) {
    const error = await response.json().catch(() => ({}));
    const errorMessage = error.detail || error.message || `HTTP error! status: ${response?.status || 'unknown'}`;
    throw new Error(errorMessage);
  }

  return await response.json();
}

export async function askClaudeAnswerStream(
  input: {
    question: string;
    sources: AskFundSource[];
    decisions: AskFundDecision[];
    connections?: AskFundConnection[];
    previousMessages?: ChatMessage[];
    webSearchEnabled?: boolean;
  },
  onChunk: (text: string) => void,
  onError?: (error: Error) => void,
  externalSignal?: AbortSignal
): Promise<void> {
  const baseUrl = await resolveConverterApiBaseUrl();
  const controller = new AbortController();
  // Link external signal so caller can cancel
  if (externalSignal) {
    externalSignal.addEventListener("abort", () => controller.abort());
  }
  // Give more time when web search is enabled (Claude may perform multiple searches)
  const timeoutMs = input.webSearchEnabled ? 120000 : 70000;
  let timeoutFired = false;
  const timeout = window.setTimeout(() => {
    timeoutFired = true;
    controller.abort();
  }, timeoutMs);
  
  // Store timeout value for error messages
  const timeoutSeconds = Math.round(timeoutMs / 1000);

  try {
    const payload = {
      question: input.question,
      sources: input.sources,
      decisions: input.decisions,
      connections: input.connections || [],
      // Backend expects snake_case
      previous_messages: input.previousMessages || [],
      web_search_enabled: input.webSearchEnabled || false,
    };
    const response = await fetch(`${baseUrl}/ask/stream`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(payload),
      signal: controller.signal,
    });

    if (!response.ok) {
      const error = await response.json().catch(() => ({}));
      throw new Error(error.detail || error.message || `HTTP error! status: ${response.status}`);
    }

    const reader = response.body?.getReader();
    const decoder = new TextDecoder();

    if (!reader) {
      throw new Error("No response body");
    }

    let buffer = "";
    let hasReceivedData = false;
    try {
      while (true) {
        const { done, value } = await reader.read();
        if (done) {
          // If stream ended without any data, it's an error
          if (!hasReceivedData && !timeoutFired) {
            onError?.(new Error("Stream ended without data. The server may have encountered an error."));
          }
          break;
        }

        // Check if timeout fired during read
        if (timeoutFired) {
          reader.cancel();
          onError?.(new Error(`Request timed out after ${timeoutSeconds} seconds. The response is taking too long. Please try again with a simpler question.`));
          return;
        }

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split("\n");
        buffer = lines.pop() || "";

        for (const line of lines) {
          if (line.startsWith("data: ")) {
            hasReceivedData = true;
            try {
              const dataStr = line.slice(6).trim();
              if (!dataStr) {
                continue;
              }
              if (dataStr === "[DONE]") {
                return;
              }
              const data = JSON.parse(dataStr);
              if (data.text) {
                onChunk(data.text);
              } else if (data.status) {
                // Status updates from backend (e.g. "🌐 Searching the web...", "tool_execution")
                // Handle tool_execution first so we never show the literal "tool_execution" to the user
                if (data.status === "tool_execution" || (typeof data.status === "object" && (data.status as { tools?: number }).tools) || data.tools !== undefined) {
                  const toolCount = typeof data.status === "object" ? (data.status as { tools?: number }).tools : data.tools ?? 1;
                  onChunk(`\n*Executing ${toolCount} tool${toolCount > 1 ? 's' : ''}...*\n`);
                } else if (typeof data.status === "string") {
                  // String status (e.g. "🌐 Searching the web...")
                  onChunk(`\n*${data.status}*\n`);
                }
              } else if (data.error) {
                onError?.(new Error(data.error));
                return;
              }
            } catch (e) {
              if (e instanceof Error && e.message !== "Unexpected end of JSON input") {
                onError?.(e);
                return;
              }
            }
          }
        }
      }
    } catch (readError) {
      // If timeout fired, we already handled it above
      if (!timeoutFired) {
        if (readError instanceof DOMException && readError.name === "AbortError") {
          onError?.(new Error(`Request timed out after ${timeoutSeconds} seconds. The response is taking too long. Please try again with a simpler question.`));
        } else {
          onError?.(readError instanceof Error ? readError : new Error("Stream read error"));
        }
      }
    }
  } catch (error) {
    if (timeoutFired) {
      onError?.(new Error(`Request timed out after ${timeoutSeconds} seconds. The response is taking too long. Please try again with a simpler question.`));
    } else if (error instanceof DOMException && error.name === "AbortError") {
      onError?.(new Error(`Request timed out after ${timeoutSeconds} seconds. The response is taking too long. Please try again with a simpler question.`));
    } else {
      onError?.(error instanceof Error ? error : new Error("Unknown error"));
    }
  } finally {
    window.clearTimeout(timeout);
  }
}

/**
 * Agentic RAG — calls the new /ask/agent/stream endpoint.
 * The backend handles all retrieval (SQL, vector search, graph) via Claude tool use.
 * Frontend just streams the final answer.
 */
/** Verifiable RAG: source citation from agent (doc_id + chunk for click-to-source) */
export type VerifiableSource = {
  type: "document" | "company_card" | "knowledge_graph";
  title: string;
  doc_id?: string;
  chunk?: number;
  entity_name?: string;
};

/** Simple source doc from agent (id + title for Sources strip) */
export type SourceDoc = { id: string; title: string };

export async function askAgentStream(
  input: {
    question: string;
    eventId: string;
    previousMessages?: ChatMessage[];
    webSearchEnabled?: boolean;
    folderIds?: string[];
  },
  onChunk: (text: string) => void,
  onStatus?: (status: string) => void,
  onError?: (error: Error) => void,
  externalSignal?: AbortSignal,
  onVerifiableSources?: (sources: VerifiableSource[]) => void,
  onCritic?: (text: string) => void,
  onSourceDocs?: (docs: SourceDoc[]) => void
): Promise<void> {
  const baseUrl = await resolveConverterApiBaseUrl();
  const controller = new AbortController();
  // Link external signal so caller can cancel
  if (externalSignal) {
    externalSignal.addEventListener("abort", () => controller.abort());
  }
  const timeoutMs = input.webSearchEnabled ? 120000 : 90000;
  let timeoutFired = false;
  const timeout = window.setTimeout(() => {
    timeoutFired = true;
    controller.abort();
  }, timeoutMs);
  const timeoutSeconds = Math.round(timeoutMs / 1000);

  try {
    const payload: Record<string, unknown> = {
      question: input.question,
      event_id: input.eventId,
      previous_messages: input.previousMessages || [],
      web_search_enabled: input.webSearchEnabled || false,
    };
    if (input.folderIds && input.folderIds.length > 0) {
      payload.folder_ids = input.folderIds;
    }
    const response = await fetch(`${baseUrl}/ask/agent/stream`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
      signal: controller.signal,
    });

    if (!response.ok) {
      const error = await response.json().catch(() => ({}));
      throw new Error(error.detail || error.message || `HTTP error! status: ${response.status}`);
    }

    const reader = response.body?.getReader();
    const decoder = new TextDecoder();
    if (!reader) throw new Error("No response body");

    let buffer = "";
    let hasReceivedData = false;
    try {
      while (true) {
        const { done, value } = await reader.read();
        if (done) {
          if (!hasReceivedData && !timeoutFired) {
            onError?.(new Error("Stream ended without data."));
          }
          break;
        }
        if (timeoutFired) {
          reader.cancel();
          onError?.(new Error(`Request timed out after ${timeoutSeconds}s. Try a simpler question.`));
          return;
        }

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split("\n");
        buffer = lines.pop() || "";

        for (const line of lines) {
          if (line.startsWith("data: ")) {
            hasReceivedData = true;
            const dataStr = line.slice(6).trim();
            if (!dataStr || dataStr === "[DONE]") {
              if (dataStr === "[DONE]") return;
              continue;
            }
            try {
              const data = JSON.parse(dataStr);
              if (data.text) {
                onChunk(data.text);
              } else if (data.status) {
                if (typeof data.status === "string") {
                  onStatus?.(data.status);
                  onChunk(`\n*${data.status}*\n`);
                }
              } else if (data.verifiable_sources && Array.isArray(data.verifiable_sources)) {
                onVerifiableSources?.(data.verifiable_sources as VerifiableSource[]);
              } else if (data.source_docs && Array.isArray(data.source_docs)) {
                onSourceDocs?.(data.source_docs as SourceDoc[]);
              } else if (data.critic && typeof data.critic === "string") {
                onCritic(data.critic);
              } else if (data.error) {
                onError?.(new Error(data.error));
                return;
              }
            } catch {
              // Partial JSON, ignore
            }
          }
        }
      }
    } catch (readError) {
      if (!timeoutFired) {
        if (readError instanceof DOMException && readError.name === "AbortError") {
          onError?.(new Error(`Timed out after ${timeoutSeconds}s.`));
        } else {
          onError?.(readError instanceof Error ? readError : new Error("Stream read error"));
        }
      }
    }
  } catch (error) {
    if (timeoutFired) {
      onError?.(new Error(`Timed out after ${timeoutSeconds}s.`));
    } else if (error instanceof DOMException && error.name === "AbortError") {
      onError?.(new Error(`Timed out after ${timeoutSeconds}s.`));
    } else {
      onError?.(error instanceof Error ? error : new Error("Unknown error"));
    }
  } finally {
    window.clearTimeout(timeout);
  }
}

/**
 * Delete redundant entity cards (keep one per core company name).
 * Call after sync to remove duplicate/empty cards.
 */
export async function deleteRedundantCards(eventId: string): Promise<{ deleted: number; message: string }> {
  const baseUrl = await resolveConverterApiBaseUrl();
  const response = await fetch(`${baseUrl}/kg/delete-redundant`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ event_id: eventId }),
  });
  if (response.status === 404) {
    return { deleted: 0, message: "Endpoint not available" };
  }
  if (!response.ok) {
    const error = await response.json().catch(() => ({}));
    throw new Error(error.detail || error.message || `HTTP error! status: ${response.status}`);
  }
  return response.json();
}

/**
 * Delete all entity cards for the event (company and fund).
 */
export async function deleteAllCards(eventId: string, entityTypes?: string[]): Promise<{ deleted: number; message: string }> {
  const baseUrl = await resolveConverterApiBaseUrl();
  const response = await fetch(`${baseUrl}/kg/delete-all`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ event_id: eventId, entity_types: entityTypes ?? ["company", "fund"] }),
  });
  if (!response.ok) {
    const error = await response.json().catch(() => ({}));
    throw new Error(error.detail || error.message || `HTTP error! status: ${response.status}`);
  }
  return response.json();
}

export async function rerankDocuments(input: {
  query: string;
  documents: Array<{ id: string; text: string }>;
  topN?: number;
}): Promise<Array<{ id: string; score: number }>> {
  const baseUrl = await resolveConverterApiBaseUrl();
  if (!input.documents.length) return [];
  const response = await fetch(`${baseUrl}/rerank`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      query: input.query,
      documents: input.documents,
      top_n: input.topN,
    }),
  });
  if (!response.ok) {
    const error = await response.json().catch(() => ({}));
    const errorMessage = error.detail || error.message || `HTTP error! status: ${response.status}`;
    throw new Error(errorMessage);
  }
  const data = await response.json();
  return data?.results || [];
}

// ---------------------------------------------------------------------------
// Web Search — calls /web-search on the backend (DuckDuckGo, no API key)
// ---------------------------------------------------------------------------

export interface WebSearchResult {
  title: string;
  snippet: string;
  url: string;
}

export async function webSearch(
  query: string,
  maxResults: number = 5
): Promise<WebSearchResult[]> {
  try {
    const baseUrl = await resolveConverterApiBaseUrl();
    const response = await fetchWithTimeout(
      `${baseUrl}/web-search`,
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query, max_results: maxResults }),
      },
      15000
    );

    if (!response.ok) {
      console.warn("[webSearch] HTTP error:", response.status);
      return [];
    }

    const data = await response.json();
    return data?.results || [];
  } catch (err) {
    console.warn("[webSearch] Error:", err);
    return [];
  }
}

export async function rewriteQueryWithLLM(
  question: string,
  previousMessages?: ChatMessage[]
): Promise<string> {
  const baseUrl = await resolveConverterApiBaseUrl();
  const response = await fetch(`${baseUrl}/rewrite-query`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      question,
      previous_messages: previousMessages || [],
    }),
  });

  if (!response.ok) {
    const error = await response.json().catch(() => ({}));
    throw new Error(error.detail || error.message || `HTTP error! status: ${response.status}`);
  }

  const data = await response.json();
  return data.rewritten_question || question;
}

// ---------------------------------------------------------------------------
//  Multi-Query Generation — produce diverse query variants for better recall
// ---------------------------------------------------------------------------

export interface MultiQueryResult {
  queries: string[];
  model_used: string;
}

export async function generateMultiQueries(
  question: string,
  maxVariants: number = 3
): Promise<MultiQueryResult> {
  try {
    const baseUrl = await resolveConverterApiBaseUrl();
    const response = await fetchWithTimeout(
      `${baseUrl}/multi-query`,
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question, max_variants: maxVariants }),
      },
      5000 // 5s — Haiku is fast, but network latency varies
    );
    if (!response.ok) {
      return { queries: [question], model_used: "" };
    }
    const data = await response.json();
    return {
      queries: data.queries?.length ? data.queries : [question],
      model_used: data.model_used || "",
    };
  } catch {
    // Fail silently — return original query only
    return { queries: [question], model_used: "" };
  }
}

export async function embedQuery(text: string, inputType: "query" | "document" = "query"): Promise<number[]> {
  const baseUrl = await resolveConverterApiBaseUrl();
  const response = await fetchWithTimeout(
    `${baseUrl}/embed/query`,
    {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ text, input_type: inputType }),
    },
    15000
  );

  if (!response.ok) {
    const error = await response.json().catch(() => ({}));
    throw new Error(error.detail || `HTTP error! status: ${response.status}`);
  }

  const result = await response.json();
  return result.embedding || [];
}

/**
 * Convert file using the converter API
 */
export async function convertFileWithAI(
  file: File,
  dataType?: "startup" | "investor"
): Promise<AIConversionResponse> {
  // Re-pack the file from bytes before uploading.
  const buf = await file.arrayBuffer();
  if (buf.byteLength === 0) {
    throw new Error(
      `Selected file is empty in the browser (0 bytes). filename="${file.name}", type="${file.type}". Re-select the file from disk.`
    );
  }

  const uploadFile = new File([buf], file.name, {
    type: file.type || "application/octet-stream",
  });

  const formData = new FormData();
  formData.append("file", uploadFile);
  if (dataType) {
    formData.append("dataType", dataType);
  }

  try {
    const baseUrl = await resolveConverterApiBaseUrl();
    const response = await fetchWithTimeout(
      `${baseUrl}/convert-file`,
      {
        method: "POST",
        body: formData,
      },
      20000 // 20 seconds — fail fast if backend is slow/down
    );

    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.detail || `HTTP error! status: ${response.status}`);
    }

    const result: AIConversionResponse = await response.json();

    // Convert to our internal types
    return {
      startups: (result.startups || []).map((s) => ({
        id: `startup-${Date.now()}-${Math.random()}`,
        companyName: s.companyName,
        geoMarkets: s.geoMarkets,
        industry: s.industry,
        fundingTarget: s.fundingTarget,
        fundingStage: s.fundingStage,
        availabilityStatus: s.availabilityStatus as "present" | "not-attending",
      })),
      investors: (result.investors || []).map((i) => ({
        id: `investor-${Date.now()}-${Math.random()}`,
        firmName: i.firmName,
        memberName: (i as any).memberName || "UNKNOWN",
        geoFocus: i.geoFocus,
        industryPreferences: i.industryPreferences,
        stagePreferences: i.stagePreferences,
        minTicketSize: i.minTicketSize,
        maxTicketSize: i.maxTicketSize,
        totalSlots: i.totalSlots,
        tableNumber: i.tableNumber,
        availabilityStatus: i.availabilityStatus as "present" | "not-attending",
      })),
      mentors: (result.mentors || []).map((m: any) => ({
        id: `mentor-${Date.now()}-${Math.random()}`,
        fullName: m.fullName,
        email: m.email,
        linkedinUrl: m.linkedinUrl,
        geoFocus: m.geoFocus || [],
        industryPreferences: m.industryPreferences || [],
        expertiseAreas: m.expertiseAreas || [],
        totalSlots: m.totalSlots || 3,
        availabilityStatus: (m.availabilityStatus as "present" | "not-attending") || "present",
      })),
      corporates: (result.corporates || []).map((c: any) => ({
        id: `corporate-${Date.now()}-${Math.random()}`,
        firmName: c.firmName,
        contactName: c.contactName,
        email: c.email,
        geoFocus: c.geoFocus || [],
        industryPreferences: c.industryPreferences || [],
        partnershipTypes: c.partnershipTypes || [],
        stages: c.stages || [],
        totalSlots: c.totalSlots || 3,
        availabilityStatus: (c.availabilityStatus as "present" | "not-attending") || "present",
      })),
      detectedType: result.detectedType,
      confidence: result.confidence,
      warnings: result.warnings,
      errors: result.errors,
      raw_content: result.raw_content ?? null,
    };
  } catch (error) {
    const baseUrl = resolvedBaseUrl ?? "(unresolved)";
    throw new Error(
      `AI file conversion failed (API: ${baseUrl}): ${error instanceof Error ? error.message : "Unknown error"}`
    );
  }
}

// ---------------------------------------------------------------------------
//  Step 1: Contextual chunking — call /contextualize-chunk before embedding
// ---------------------------------------------------------------------------

export interface ContextualizeChunkInput {
  document_title: string;
  document_summary: string;
  chunk_text: string;
  chunk_index: number;
  total_chunks: number;
}

export interface ContextualizeChunkResult {
  enriched_chunk: string;
  contextual_header: string;
}

export async function contextualizeChunk(
  input: ContextualizeChunkInput
): Promise<ContextualizeChunkResult> {
  try {
    const baseUrl = await resolveConverterApiBaseUrl();
    // Fast timeout: 3s max - if backend is slow/down, skip enrichment immediately
    // This prevents blocking document uploads
    const response = await fetchWithTimeout(
      `${baseUrl}/contextualize-chunk`,
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(input),
      },
      3000 // 3s — fail fast if backend is slow/down
    );
    if (!response.ok) {
      // Fall back to raw chunk
      return { enriched_chunk: input.chunk_text, contextual_header: "" };
    }
    return await response.json();
  } catch {
    // Fail silently — return raw chunk (non-blocking)
    return { enriched_chunk: input.chunk_text, contextual_header: "" };
  }
}

// ---------------------------------------------------------------------------
//  Step 1b: Agentic chunking — call /agentic-chunk for LLM-driven splitting
// ---------------------------------------------------------------------------

export interface AgenticSection {
  label: string;
  text: string;
}

export interface AgenticChunkInput {
  document_title: string;
  document_text: string;
  max_sections?: number;
}

export interface AgenticChunkResult {
  sections: AgenticSection[];
  model_used: string;
  fallback: boolean;
}

export async function agenticChunk(
  input: AgenticChunkInput
): Promise<AgenticChunkResult> {
  try {
    const baseUrl = await resolveConverterApiBaseUrl();
    const response = await fetchWithTimeout(
      `${baseUrl}/agentic-chunk`,
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          document_title: input.document_title,
          document_text: input.document_text,
          max_sections: input.max_sections ?? 8,
        }),
      },
      15000 // 15s — LLM needs time to read and split the full document
    );
    if (!response.ok) {
      return { sections: [], model_used: "", fallback: true };
    }
    return await response.json();
  } catch {
    return { sections: [], model_used: "", fallback: true };
  }
}

// ---------------------------------------------------------------------------
//  Step 2: GraphRAG retrieval — call /graphrag/retrieve for relevance filtering
// ---------------------------------------------------------------------------

export interface GraphRAGChunk {
  id: string;
  text: string;
  score?: number;
  metadata?: Record<string, unknown>;
}

export interface GraphRAGResult {
  relevant_chunks: GraphRAGChunk[];
  expanded: boolean;
  total_assessed: number;
}

export async function graphragRetrieve(input: {
  query: string;
  initial_chunks: GraphRAGChunk[];
  neighboring_chunks?: GraphRAGChunk[];
  min_relevant_chunks?: number;
}): Promise<GraphRAGResult> {
  try {
    const baseUrl = await resolveConverterApiBaseUrl();
    const response = await fetchWithTimeout(
      `${baseUrl}/graphrag/retrieve`,
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          query: input.query,
          initial_chunks: input.initial_chunks,
          neighboring_chunks: input.neighboring_chunks || [],
          min_relevant_chunks: input.min_relevant_chunks ?? 2,
        }),
      },
      30000 // 30s — Claude needs to assess each chunk
    );
    if (!response.ok) {
      return { relevant_chunks: input.initial_chunks, expanded: false, total_assessed: 0 };
    }
    return await response.json();
  } catch {
    return { relevant_chunks: input.initial_chunks, expanded: false, total_assessed: 0 };
  }
}

// ---------------------------------------------------------------------------
//  Step 3: Query router — entity extraction, intent, complexity, routing
// ---------------------------------------------------------------------------

export type QueryIntent =
  | "factual"       // Simple lookup: "What is Giga Energy?"
  | "compare"       // Compare entities: "Compare Ridelink vs Weego"
  | "summarize"     // Summarize a doc/company
  | "diligence"     // Due diligence: "risks of investing in X"
  | "forecast"      // Forward-looking: "What's the growth potential"
  | "relationship"  // About connections: "Who is connected to X"
  | "meta"          // About the system: "What can you do?"
  | "conversational"; // Greeting/chat

export interface QueryAnalysis {
  entities: Array<{ name: string; type: "company" | "person" | "fund" | "metric" | "sector" | "unknown" }>;
  intent: QueryIntent;
  complexity: number; // 0.0–1.0
  retrieval_strategy: "vector" | "vector+graph" | "vector+graph+structured" | "none";
  rewritten_query: string;
}

export async function analyzeQuery(
  question: string,
  previousMessages?: ChatMessage[]
): Promise<QueryAnalysis> {
  try {
    const baseUrl = await resolveConverterApiBaseUrl();
    const response = await fetchWithTimeout(
      `${baseUrl}/analyze-query`,
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question, previous_messages: previousMessages || [] }),
      },
      10000
    );
    if (!response.ok) {
      // Fallback: simple heuristic
      return fallbackQueryAnalysis(question);
    }
    return await response.json();
  } catch {
    return fallbackQueryAnalysis(question);
  }
}

function fallbackQueryAnalysis(question: string): QueryAnalysis {
  const q = question.toLowerCase();
  const connectionWords = ["connect", "partner", "introduce", "relationship", "linked"];
  const compareWords = ["compare", "vs", "versus", "difference", "better"];
  const diligenceWords = ["risk", "diligence", "red flag", "concern", "weakness"];
  const forecastWords = ["growth", "potential", "forecast", "predict", "future"];
  const metaWords = ["what can you", "your purpose", "help me", "what do you"];

  let intent: QueryIntent = "factual";
  let strategy: QueryAnalysis["retrieval_strategy"] = "vector";

  if (metaWords.some((w) => q.includes(w))) { intent = "meta"; strategy = "none"; }
  else if (connectionWords.some((w) => q.includes(w))) { intent = "relationship"; strategy = "vector+graph"; }
  else if (compareWords.some((w) => q.includes(w))) { intent = "compare"; strategy = "vector+graph"; }
  else if (diligenceWords.some((w) => q.includes(w))) { intent = "diligence"; strategy = "vector+graph+structured"; }
  else if (forecastWords.some((w) => q.includes(w))) { intent = "forecast"; strategy = "vector+graph+structured"; }
  else { intent = "factual"; strategy = "vector"; }

  return {
    entities: [],
    intent,
    complexity: q.split(" ").length > 15 ? 0.8 : 0.3,
    retrieval_strategy: strategy,
    rewritten_query: question,
  };
}

// ---------------------------------------------------------------------------
//  Step 7 (partial): RAG eval logging
// ---------------------------------------------------------------------------

export interface RAGEvalEntry {
  question: string;
  retrieval_strategy: string;
  chunks_retrieved: number;
  chunks_cited: number;
  model_used: string;
  latency_ms: number;
  user_feedback?: "helpful" | "not_helpful" | null;
}

export async function logRAGEval(entry: RAGEvalEntry): Promise<void> {
  try {
    const baseUrl = await resolveConverterApiBaseUrl();
    await fetchWithTimeout(
      `${baseUrl}/rag-eval/log`,
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(entry),
      },
      5000
    );
  } catch {
    // Non-critical — don't block the user
    console.warn("[RAG eval] Failed to log entry");
  }
}

// ---------------------------------------------------------------------------
//  Entity Extraction — auto-populate knowledge graph + KPIs from documents
// ---------------------------------------------------------------------------

export interface ExtractedEntity {
  name: string;
  type: "company" | "person" | "fund" | "round" | "sector" | "metric" | "location";
  properties: Record<string, unknown>;
  confidence: number;
}

export interface ExtractedRelationship {
  source_name: string;
  target_name: string;
  relation_type: string;
  properties: Record<string, unknown>;
  confidence: number;
}

export interface ExtractedKPI {
  company_name: string;
  metric_name: string;
  value: number;
  unit: string;
  period: string;
  category: string;
  confidence: number;
}

export interface EntityExtractionResult {
  entities: ExtractedEntity[];
  relationships: ExtractedRelationship[];
  kpis: ExtractedKPI[];
}

/**
 * Parse an SSE stream from a /stream endpoint.
 * Returns the final result from the "done" event.
 * Handles keepalive pings and errors gracefully.
 */
async function parseSSEExtractionStream<T>(response: Response, label: string): Promise<T | null> {
  if (!response.body) {
    console.warn(`[${label}] No response body`);
    return null;
  }
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });

      // Process complete SSE events (separated by double newlines)
      const events = buffer.split("\n\n");
      buffer = events.pop() || ""; // Keep incomplete event in buffer

      for (const event of events) {
        const dataLine = event.trim();
        if (!dataLine.startsWith("data: ")) continue;
        const jsonStr = dataLine.slice(6);
        try {
          const parsed = JSON.parse(jsonStr);
          if (parsed.status === "processing") {
            // Keepalive ping — connection is alive
            continue;
          }
          if (parsed.status === "done" && parsed.result) {
            console.log(`[${label}] ✅ Extraction complete via stream`);
            return parsed.result as T;
          }
          if (parsed.status === "error") {
            console.error(`[${label}] Backend error:`, parsed.error);
            return null;
          }
        } catch {
          // Ignore non-JSON lines
        }
      }
    }
  } catch (err) {
    console.error(`[${label}] Stream read error:`, err);
  }
  return null;
}

export async function extractEntities(input: {
  document_title: string;
  document_text: string;
  document_type?: string;
  pdf_base64?: string;
}): Promise<EntityExtractionResult> {
  const empty: EntityExtractionResult = { entities: [], relationships: [], kpis: [] };
  const hasPdf = !!input.pdf_base64;

  try {
    const baseUrl = await resolveConverterApiBaseUrl();

    // ── Try streaming endpoint first (avoids Render.com 30s timeout) ──
    if (hasPdf) {
      console.log(`[extractEntities] Using streaming PDF path (${Math.round((input.pdf_base64?.length || 0) / 1024)}KB base64)`);
      try {
        const streamResp = await fetchWithTimeout(
          `${baseUrl}/extract-entities/stream`,
          {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(input),
          },
          120000 // 2 min — stream keepalives prevent Render timeout
        );
        if (streamResp.ok) {
          const result = await parseSSEExtractionStream<EntityExtractionResult>(streamResp, "extractEntities-stream");
          if (result && (result.entities?.length || result.relationships?.length || result.kpis?.length)) {
            return result;
          }
          console.warn("[extractEntities] Stream returned empty, falling back to text-only");
        } else {
          console.warn(`[extractEntities] Stream endpoint returned HTTP ${streamResp.status}, falling back`);
        }
      } catch (streamErr) {
        console.warn("[extractEntities] Stream endpoint failed, falling back:", streamErr);
      }
    }

    // ── Fallback: regular (non-streaming) endpoint, text-only ──
    const textOnlyInput = { ...input, pdf_base64: undefined };
    console.log(`[extractEntities] Using regular text-only path (${(input.document_text?.length || 0)} chars)`);
    const response = await fetchWithTimeout(
      `${baseUrl}/extract-entities`,
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(textOnlyInput),
      },
      45000
    );
    if (!response.ok) {
      console.error(`[extractEntities] HTTP ${response.status}: ${await response.text().catch(() => "")}`);
      return empty;
    }
    return await response.json();
  } catch (err) {
    console.error("[extractEntities] Failed:", err);
    return empty;
  }
}

/**
 * Check if converter API is available
 */
export async function checkConverterHealth(): Promise<{
  available: boolean;
  provider?: string;
  models?: string[];
  error?: string;
}> {
  try {
    // Re-resolve each time; if user starts API later, we can find it.
    resolvedBaseUrl = null;
    const baseUrl = await resolveConverterApiBaseUrl();
    const response = await fetch(`${baseUrl}/health`);
    const data = await response.json();
    const available = data.available === true || data.status === "healthy";
    return {
      available,
      provider: data.provider,
      models: data.models,
      error: data.error,
    };
  } catch (error) {
    return {
      available: false,
      error: error instanceof Error ? error.message : "Connection failed",
    };
  }
}

/**
 * Ask Claude to suggest company connections based on documents + existing graph
 */
export interface SuggestedConnection {
  source_company: string;
  target_company: string;
  connection_type: string;
  reasoning: string;
  confidence: number;
}

export async function suggestConnections(input: {
  companyName?: string;
  question?: string;
  sources: AskFundSource[];
  existingConnections: AskFundConnection[];
  maxSuggestions?: number;
}): Promise<{ suggestions: SuggestedConnection[]; contextSummary: string }> {
  try {
    const baseUrl = await resolveConverterApiBaseUrl();
    const response = await fetch(`${baseUrl}/suggest-connections`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        company_name: input.companyName ?? "",
        question: input.question ?? "",
        sources: input.sources,
        existing_connections: input.existingConnections,
        max_suggestions: input.maxSuggestions ?? 5,
      }),
    });
    if (!response.ok) {
      const err = await response.json().catch(() => ({}));
      throw new Error(err.detail || `HTTP ${response.status}`);
    }
    const data = await response.json();
    return {
      suggestions: data.suggestions || [],
      contextSummary: data.context_summary || "",
    };
  } catch (error) {
    console.error("[suggestConnections] Error:", error);
    // Return empty suggestions if backend is unavailable
    const errorMessage = error instanceof Error ? error.message : String(error);
    if (errorMessage.includes("Anthropic SDK") || errorMessage.includes("API key")) {
      return { suggestions: [], contextSummary: "AI suggestions require Anthropic API configuration. Please set ANTHROPIC_API_KEY in your backend environment." };
    }
    return { suggestions: [], contextSummary: "Connection suggestions unavailable. Please check backend configuration." };
  }
}

/**
 * Extract structured company card properties from document text.
 * Calls the backend /extract-company-properties endpoint.
 */
export interface CompanyPropertyExtractionResult {
  properties: Record<string, any>;
  confidence: Record<string, number>;
  document_type_detected: string;
}

export async function extractCompanyProperties(input: {
  rawContent: string;
  documentTitle?: string;
  documentType?: string;
  existingProperties?: Record<string, any>;
  pdfBase64?: string;
}): Promise<CompanyPropertyExtractionResult> {
  const empty: CompanyPropertyExtractionResult = { properties: {}, confidence: {}, document_type_detected: "" };
  const hasPdf = !!input.pdfBase64;

  const payload = {
    raw_content: input.rawContent,
    document_title: input.documentTitle || "",
    document_type: input.documentType || "",
    existing_properties: input.existingProperties || {},
    pdf_base64: input.pdfBase64 || null,
  };

  try {
    const baseUrl = await resolveConverterApiBaseUrl();

    // ── Try streaming endpoint first when PDF (avoids Render 30s timeout) ──
    if (hasPdf) {
      console.log(`[extractCompanyProperties] Using streaming PDF path (${Math.round((input.pdfBase64?.length || 0) / 1024)}KB base64)`);
      try {
        const streamResp = await fetchWithTimeout(
          `${baseUrl}/extract-company-properties/stream`,
          {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload),
          },
          120000 // 2 min — stream keepalives prevent Render timeout
        );
        if (streamResp.ok) {
          const result = await parseSSEExtractionStream<CompanyPropertyExtractionResult>(streamResp, "extractProps-stream");
          if (result && Object.keys(result.properties || {}).length > 0) {
            return {
              properties: result.properties || {},
              confidence: result.confidence || {},
              document_type_detected: result.document_type_detected || "",
            };
          }
          console.warn("[extractCompanyProperties] Stream returned empty, falling back to text-only");
        } else {
          console.warn(`[extractCompanyProperties] Stream HTTP ${streamResp.status}, falling back`);
        }
      } catch (streamErr) {
        console.warn("[extractCompanyProperties] Stream failed, falling back:", streamErr);
      }
    }

    // ── Fallback: regular endpoint, text-only (no PDF) ──
    const textPayload = { ...payload, pdf_base64: null };
    console.log(`[extractCompanyProperties] Using regular text-only path (${(input.rawContent?.length || 0)} chars)`);
    const response = await fetchWithTimeout(
      `${baseUrl}/extract-company-properties`,
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(textPayload),
      },
      45000
    );
    if (!response.ok) {
      const errText = await response.text().catch(() => "");
      console.error(`[extractCompanyProperties] HTTP ${response.status}: ${errText}`);
      return empty;
    }
    const data = await response.json();
    return {
      properties: data.properties || {},
      confidence: data.confidence || {},
      document_type_detected: data.document_type_detected || "",
    };
  } catch (error) {
    console.error("[extractCompanyProperties] Error:", error);
    return empty;
  }
}

// ---------------------------------------------------------------------------
//  Multi-Agent RAG — Orchestrator, Critic
// ---------------------------------------------------------------------------

export interface OrchestrateResult {
  use_vector: boolean;
  use_graph: boolean;
  use_kpis: boolean;
  use_web: boolean;
  reasoning: string;
  sub_queries: Record<string, string>;
}

/**
 * Multi-Agent RAG — ORCHESTRATOR (Router Agent).
 * Analyzes the user question and returns a routing plan.
 */
export async function orchestrateQuery(input: {
  question: string;
  previousMessages?: ChatMessage[];
}): Promise<OrchestrateResult> {
  const fallback: OrchestrateResult = {
    use_vector: true,
    use_graph: false,
    use_kpis: false,
    use_web: false,
    reasoning: "Fallback: vector only",
    sub_queries: {},
  };
  try {
    const baseUrl = await resolveConverterApiBaseUrl();
    const response = await fetchWithTimeout(
      `${baseUrl}/orchestrate-query`,
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          question: input.question,
          previousMessages: input.previousMessages || [],
        }),
      },
      8000 // 8s timeout — router should be fast
    );
    if (!response.ok) return fallback;
    const data = await response.json();
    return {
      use_vector: data.use_vector ?? true,
      use_graph: data.use_graph ?? false,
      use_kpis: data.use_kpis ?? false,
      use_web: data.use_web ?? false,
      reasoning: data.reasoning || "",
      sub_queries: data.sub_queries || {},
    };
  } catch (error) {
    console.error("[orchestrateQuery] Error:", error);
    return fallback;
  }
}

export interface CriticResult {
  issues: string[];
  is_grounded: boolean;
  confidence: number;
}

/**
 * Multi-Agent RAG — CRITIC (Verifier Agent).
 * Checks whether the answer is grounded in the provided context.
 */
export async function criticCheck(input: {
  question: string;
  answer: string;
  contextVector?: string;
  contextGraph?: string;
  contextKpis?: string;
  contextWeb?: string;
}): Promise<CriticResult> {
  try {
    const baseUrl = await resolveConverterApiBaseUrl();
    const response = await fetchWithTimeout(
      `${baseUrl}/critic-check`,
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          question: input.question,
          answer: input.answer,
          context_vector: input.contextVector || "",
          context_graph: input.contextGraph || "",
          context_kpis: input.contextKpis || "",
          context_web: input.contextWeb || "",
        }),
      },
      15000
    );
    if (!response.ok) return { issues: [], is_grounded: true, confidence: 0.5 };
    return await response.json();
  } catch (error) {
    console.error("[criticCheck] Error:", error);
    return { issues: [], is_grounded: true, confidence: 0.5 };
  }
}

// ---------------------------------------------------------------------------
//  System 2 RAG — Test-Time Compute (Reflect → Search → Refine)
// ---------------------------------------------------------------------------

export interface System2ReflectResult {
  needs_more_data: boolean;
  missing_data_types: string[];
  follow_up_queries: string[];
  reasoning: string;
  refined_answer: string;
  confidence: number;
}

/**
 * System 2 RAG — REFLECTOR.
 * Analyzes a draft answer and identifies gaps that need more data.
 */
export async function system2Reflect(input: {
  question: string;
  draftAnswer: string;
  vectorContext?: string;
  graphContext?: string;
  kpiContext?: string;
  iteration?: number;
  maxIterations?: number;
}): Promise<System2ReflectResult> {
  const fallback: System2ReflectResult = {
    needs_more_data: false,
    missing_data_types: [],
    follow_up_queries: [],
    reasoning: "Reflection unavailable",
    refined_answer: "",
    confidence: 0.7,
  };
  try {
    const baseUrl = await resolveConverterApiBaseUrl();
    const response = await fetchWithTimeout(
      `${baseUrl}/system2-reflect`,
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          question: input.question,
          draft_answer: input.draftAnswer,
          vector_context: input.vectorContext || "",
          graph_context: input.graphContext || "",
          kpi_context: input.kpiContext || "",
          iteration: input.iteration ?? 0,
          max_iterations: input.maxIterations ?? 3,
        }),
      },
      12000,
    );
    if (!response.ok) return fallback;
    return await response.json();
  } catch (error) {
    console.error("[system2Reflect] Error:", error);
    return fallback;
  }
}

/**
 * System 2 RAG — REFINER (streaming).
 * Produces a refined answer incorporating additional context.
 * Returns an SSE ReadableStream.
 */
export async function system2RefineStream(input: {
  question: string;
  draftAnswer: string;
  originalContext?: string;
  additionalContext?: string;
  reflectionReasoning?: string;
  previousMessages?: ChatMessage[];
}): Promise<Response> {
  const baseUrl = await resolveConverterApiBaseUrl();
  return fetch(`${baseUrl}/system2-refine/stream`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      question: input.question,
      draft_answer: input.draftAnswer,
      original_context: input.originalContext || "",
      additional_context: input.additionalContext || "",
      reflection_reasoning: input.reflectionReasoning || "",
      previousMessages: input.previousMessages || [],
    }),
  });
}

// ---------------------------------------------------------------------------
//  ColBERT Late Interaction Reranking
// ---------------------------------------------------------------------------

export interface ColBERTRerankResult {
  results: Array<{
    index: number;
    doc_id: string;
    relevance_score: number;
    voyage_score: number;
    maxsim_score: number;
    document: string;
  }>;
  method: string;
}

/**
 * ColBERT-style late-interaction reranking.
 * Combines Voyage cross-encoder with token-level MaxSim scoring.
 */
export async function colbertRerank(input: {
  query: string;
  documents: string[];
  docIds?: string[];
  topK?: number;
}): Promise<ColBERTRerankResult> {
  const fallback: ColBERTRerankResult = { results: [], method: "error" };
  try {
    const baseUrl = await resolveConverterApiBaseUrl();
    const response = await fetchWithTimeout(
      `${baseUrl}/colbert-rerank`,
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          query: input.query,
          documents: input.documents,
          doc_ids: input.docIds || [],
          top_k: input.topK ?? 10,
        }),
      },
      30000,
    );
    if (!response.ok) return fallback;
    return await response.json();
  } catch (error) {
    console.error("[colbertRerank] Error:", error);
    return fallback;
  }
}