import { supabase } from "@/integrations/supabase/client";
import type { DecisionLog, DocumentRecord, Event, SourceRecord, Task, TaskStatus, UserProfile } from "@/types";

type SupabaseResult<T> = { data: T | null; error: any };

const DEFAULT_EVENT_NAME = "Main Event";

function slugifyOrgName(value: string) {
  return value
    .toLowerCase()
    .replace(/[^a-z0-9\s-]/g, "")
    .trim()
    .replace(/\s+/g, "-")
    .slice(0, 50);
}

/**
 * Normalize company name for matching/deduplication so "Trashcoin" and "Trashcoin Limited"
 * map to the same canonical key and resolve to one entity.
 */
export function normalizeCompanyNameForMatch(name: string): string {
  if (!name || typeof name !== "string") return "";
  let s = name.toLowerCase().trim();
  // Strip common company suffixes (with optional trailing dot and whitespace)
  s = s.replace(/\s+(limited|ltd\.?|inc\.?|llc\.?|corp\.?|corporation|plc\.?|gmbh|co\.?|company|group)\s*$/i, "").trim();
  return s.replace(/\s+/g, " ").trim();
}

/**
 * Extract a fuzzy "core" company name by handling document-title patterns
 * and stripping ALL corporate suffixes.
 *
 * Examples:
 *   "Chhaya Technologies PTE. LTD."          → "chhaya"
 *   "[Chari] GPT Demo Day"                   → "chari"
 *   "Template for Chari Side Letter to Chari SPA" → "chari"
 *   "V1 . Chari"                             → "chari"
 */
export function extractCoreCompanyName(name: string): string {
  if (!name || typeof name !== "string") return "";
  let s = name.trim();

  // 1) If name starts with [Something], extract bracket content as the company name
  const bracketMatch = s.match(/^\[([^\]]+)\]/);
  if (bracketMatch) {
    s = bracketMatch[1].trim();
  }

  // 2) Strip "V1 .", "V2." etc. prefix
  s = s.replace(/^v\d+\.?\s*\.?\s*/i, "").trim();

  // 3) Strip "Copy of " prefix
  s = s.replace(/^copy of\s+/i, "").trim();

  // 4) Lowercase
  s = s.toLowerCase();

  // 5) Remove corporate suffixes
  s = s.replace(
    /\b(technologies|technology|tech|pte\.?|ltd\.?|limited|inc\.?|incorporated|corp\.?|corporation|plc\.?|gmbh|co\.?|company|group|holdings|solutions|services|ventures|capital|partners|llc\.?|llp\.?|lp\.?|sa|s\.a\.?|ag|bv|nv|pvt\.?|private)\b/gi,
    " "
  );

  // 6) Remove document-title junk words
  s = s.replace(
    /\b(template|draft|final|copy|summary|report|update|email|memo|letter|deck|pitch|presentation|agenda|minutes|notes|side letter|spa|sha|ssa|term sheet|nda|mou|loi|weekly|monthly|quarterly|annual|demo day|gpt)\b/gi,
    " "
  );

  // 7) Clean up punctuation and filler words
  s = s.replace(/[.,():\[\]{}"'`/\\|_-]+/g, " ");
  s = s.replace(/\b(for|to|of|the|a|an|and|or|in|on|at|by|from|with)\b/g, " ");
  return s.replace(/\s+/g, " ").trim();
}

export async function ensureOrganizationForUser(profile: UserProfile): Promise<SupabaseResult<{ organization: any; updatedProfile: UserProfile }>> {
  if (profile.organization_id) {
    const { data, error } = await supabase.from("organizations").select("*").eq("id", profile.organization_id).maybeSingle();
    if (error) return { data: null, error };
    if (!data) return { data: null, error: { message: "Organization not found or access denied. Try signing out and back in.", code: "PGRST116" } as any };
    return { data: { organization: data, updatedProfile: profile }, error: null };
  }

  const fallbackName = profile.full_name || profile.email || "Default Organization";
  const orgSlug = slugifyOrgName(fallbackName) || `org-${profile.id.slice(0, 8)}`;

  // Use RPC to avoid RLS issues during signup
  const { data: org, error: orgError } = await supabase.rpc("ensure_user_organization", {
    org_name: fallbackName,
    org_slug: orgSlug,
  });

  if (orgError || !org) {
    return { data: null, error: orgError };
  }

  const { data: updatedProfile, error: profileError } = await supabase
    .from("user_profiles")
    .select("*")
    .eq("id", profile.id)
    .single();

  return {
    data: {
      organization: org,
      updatedProfile: (updatedProfile as UserProfile) || { ...profile, organization_id: org.id },
    },
    error: profileError,
  };
}

export async function ensureActiveEventForOrg(orgId: string): Promise<SupabaseResult<Event>> {
  // Prefer RPC so event creation succeeds even when client token/RLS is flaky (e.g. after 429 refresh)
  const { data: rpcEvent, error: rpcError } = await supabase.rpc("ensure_active_event");
  if (!rpcError && rpcEvent && typeof rpcEvent === "object" && rpcEvent.id) {
    return { data: rpcEvent as Event, error: null };
  }

  const { data: events, error } = await supabase
    .from("events")
    .select("*")
    .eq("organization_id", orgId)
    .eq("status", "active")
    .order("created_at", { ascending: false });

  if (error) return { data: null, error };
  if (events && events.length > 0) return { data: events[0] as Event, error: null };

  const { data: created, error: createError } = await supabase
    .from("events")
    .insert({ organization_id: orgId, name: DEFAULT_EVENT_NAME, status: "active" })
    .select("*")
    .single();
  return { data: created as Event, error: createError };
}

export async function getActiveEvents(organizationId: string) {
  return supabase
    .from("events")
    .select("*")
    .eq("organization_id", organizationId)
    .eq("status", "active")
    .order("created_at", { ascending: false });
}

export async function getEvent(eventId: string) {
  return supabase.from("events").select("*").eq("id", eventId).single();
}

export async function getDecisionsByEvent(eventId: string) {
  return supabase
    .from("decisions")
    .select("*")
    .eq("event_id", eventId)
    .order("created_at", { ascending: false });
}

export async function insertDecision(
  eventId: string,
  payload: {
    actor_id: string | null;
    actor_name: string;
    action_type: string;
    startup_name: string;
    context: Record<string, any> | null;
    confidence_score: number;
    outcome: string | null;
    notes: string | null;
    document_id?: string | null;
  }
) {
  return supabase.from("decisions").insert({ event_id: eventId, ...payload }).select("*").single();
}

export async function updateDecision(decisionId: string, updates: Partial<DecisionLog>) {
  return supabase.from("decisions").update(updates).eq("id", decisionId);
}

export async function deleteDecision(decisionId: string) {
  return supabase.from("decisions").delete().eq("id", decisionId);
}

export async function insertDocument(
  eventId: string,
  payload: {
    title: string | null;
    source_type: string;
    file_name: string | null;
    storage_path: string | null;
    detected_type: string | null;
    extracted_json: Record<string, any>;
    raw_content?: string | null;
    created_by: string | null;
    folder_id?: string | null;
  }
) {
  return supabase
    .from("documents")
    .insert({ event_id: eventId, ...payload })
    .select("*")
    .single();
}

export async function getDocumentById(documentId: string) {
  return supabase
    .from("documents")
    .select("*")
    .eq("id", documentId)
    .single();
}

export async function getDocumentsByEvent(eventId: string) {
  // Join with user_profiles to get uploader info
  return supabase
    .from("documents")
    .select("*")
    .eq("event_id", eventId)
    .order("created_at", { ascending: false });
}

export async function getSourcesByEvent(eventId: string) {
  return supabase
    .from("sources")
    .select("*")
    .eq("event_id", eventId)
    .order("created_at", { ascending: false });
}

export async function getSourceFoldersByEvent(eventId: string) {
  return supabase
    .from("source_folders")
    .select("*")
    .eq("event_id", eventId)
    .order("created_at", { ascending: false });
}

export async function ensureDefaultFoldersForEvent(eventId: string): Promise<void> {
  const { error } = await supabase.rpc("ensure_default_folders_for_event", {
    p_event_id: eventId,
  });
  if (error) {
    const defaultFolders: { name: string; category: string }[] = [
      { name: 'Companies', category: 'Companies' },
      { name: 'Partners', category: 'Partners' },
      { name: 'Deals', category: 'Sourcing' },
      { name: 'Market Research', category: 'Sourcing' },
      { name: 'Due Diligence', category: 'Companies' },
      { name: 'BD', category: 'BD' },
    ];

    const { data: existingFolders } = await supabase
      .from("source_folders")
      .select("name")
      .eq("event_id", eventId);

    const existingNames = new Set((existingFolders || []).map((f: any) => f.name.toLowerCase()));
    const foldersToCreate = defaultFolders.filter(d => !existingNames.has(d.name.toLowerCase()));

    if (foldersToCreate.length > 0) {
      const { data: { user } } = await supabase.auth.getUser();
      await supabase
        .from("source_folders")
        .insert(
          foldersToCreate.map(d => ({
            event_id: eventId,
            name: d.name,
            created_by: user?.id || null,
            category: d.category,
          }))
        );
    }
  }
}

export async function updateFolderCategory(folderId: string, category: string) {
  return supabase
    .from("source_folders")
    .update({ category })
    .eq("id", folderId)
    .select("*")
    .single();
}

export async function insertSourceFolder(
  eventId: string,
  payload: { name: string; created_by: string | null; category?: string | null }
) {
  const row: { event_id: string; name: string; created_by: string | null; category?: string } = {
    event_id: eventId,
    name: payload.name,
    created_by: payload.created_by,
  };
  if (payload.category != null && payload.category !== "") {
    row.category = payload.category;
  }
  return supabase
    .from("source_folders")
    .insert(row)
    .select("*")
    .single();
}

/**
 * Delete a folder and all documents that are in it (via document_folder_links).
 * Permanently removes those documents (and their embeddings via CASCADE).
 * Use with confirmation in the UI.
 */
export async function deleteFolderAndContents(folderId: string): Promise<{ docCount: number; error?: string }> {
  const { data: links, error: linksError } = await supabase
    .from("document_folder_links")
    .select("document_id")
    .eq("folder_id", folderId);

  if (linksError) {
    return { docCount: 0, error: linksError.message };
  }

  const docIds = (links || []).map((r: { document_id: string }) => r.document_id);
  const docCount = docIds.length;

  if (docIds.length > 0) {
    const { error: deleteDocsError } = await supabase
      .from("documents")
      .delete()
      .in("id", docIds);

    if (deleteDocsError) {
      return { docCount, error: deleteDocsError.message };
    }
  }

  const { error: deleteFolderError } = await supabase
    .from("source_folders")
    .delete()
    .eq("id", folderId);

  if (deleteFolderError) {
    return { docCount, error: deleteFolderError.message };
  }

  return { docCount };
}

// Company Connections helpers
export type ConnectionType = "BD" | "INV" | "Knowledge" | "Partnership" | "Portfolio";
export type ConnectionStatus = "To Connect" | "Connected" | "Rejected" | "In Progress" | "Completed";

export interface CompanyConnection {
  id: string;
  event_id: string;
  created_by: string | null;
  source_company_name: string;
  target_company_name: string;
  source_document_id: string | null;
  target_document_id: string | null;
  connection_type: ConnectionType;
  connection_status: ConnectionStatus;
  ai_reasoning: string | null;
  notes: string | null;
  created_at: string;
  updated_at: string;
}

export async function getCompanyConnectionsByEvent(eventId: string) {
  return supabase
    .from("company_connections")
    .select("*")
    .eq("event_id", eventId)
    .order("created_at", { ascending: false });
}

// Get pending relationship reviews from knowledge graph
export async function getCompanyCards(eventId: string) {
  // Get all company entities for this event
  const { data: entities, error: entitiesError } = await supabase
    .from("kg_entities")
    .select("id, name, properties, event_id")
    .eq("event_id", eventId)
    .eq("entity_type", "company")
    .order("name", { ascending: true });
  
  if (entitiesError || !entities) {
    return { data: [], error: entitiesError };
  }
  
  // For each company, get its card data
  const cards = await Promise.all(
    entities.map(async (entity: any) => {
      const { data: cardData, error: cardError } = await supabase
        .rpc("get_company_card", {
          p_company_entity_id: entity.id,
          p_filter_event_id: eventId,
        } as any)
        .single();
      
      if (cardError || !cardData) {
        // Fallback: return basic card if RPC fails
        return {
          company_id: entity.id,
          company_name: entity.name,
          company_properties: entity.properties || {},
          document_count: 0,
          document_ids: [],
          connection_count: 0,
          connection_ids: [],
          kpi_count: 0,
          kpi_summary: {},
          relationship_count: 0,
          related_companies: [],
        };
      }
      
      return cardData;
    })
  );
  
  return { data: cards, error: null };
}

export async function getCompanyCardById(companyEntityId: string, eventId: string) {
  const { data, error } = await supabase
    .rpc("get_company_card", { p_company_entity_id: companyEntityId, p_filter_event_id: eventId } as any)
    .single();
  return { data, error };
}

export async function updateCompanyCardProperties(entityId: string, newProperties: Record<string, any>) {
  const { error } = await supabase
    .rpc("update_company_card_properties", {
      p_entity_id: entityId,
      p_new_properties: newProperties,
    } as any);
  return { error };
}

// ─────────────────────────────────────────────────────────────────────────
// SMART PROPERTY MERGE — auto-populate company cards from AI extraction
// ─────────────────────────────────────────────────────────────────────────

export interface PropertyConflict {
  field: string;
  oldValue: any;
  newValue: any;
  newDocumentId: string;
  newConfidence: number;
}

export interface MergeResult {
  updated: string[];
  skipped: string[];
  conflicts: PropertyConflict[];
  entityId: string;
  companyName: string;
}

/**
 * Fetch the company_entity_id linked to a document (set by the DB trigger).
 * Returns null if the document has no linked entity.
 */
export async function getDocumentCompanyEntityId(documentId: string): Promise<string | null> {
  const { data, error } = await supabase
    .from("documents")
    .select("company_entity_id")
    .eq("id", documentId)
    .single();
  if (error || !data) return null;
  return (data as any).company_entity_id || null;
}

/**
 * Fetch current properties for a kg_entity.
 */
export async function getEntityProperties(entityId: string): Promise<{ name: string; properties: Record<string, any> } | null> {
  const { data, error } = await supabase
    .from("kg_entities")
    .select("name, properties")
    .eq("id", entityId)
    .single();
  if (error || !data) return null;
  return { name: (data as any).name, properties: (data as any).properties || {} };
}

// ─────────────────────────────────────────────────────────────────────────
// MULTI-AGENT RAG — Graph & KPI Retrieval
// ─────────────────────────────────────────────────────────────────────────

export interface GraphRetrievalResult {
  entities: Array<{ id: string; name: string; type: string; properties: Record<string, any> }>;
  edges: Array<{ source: string; target: string; relation: string; properties: Record<string, any> }>;
  connections: Array<{ source: string; target: string; type: string; status: string; reasoning: string }>;
  summary: string;
}

/**
 * Retrieve entities, relationships, AND company connections from the knowledge graph.
 * - Searches kg_entities by entity name (prioritizes exact names over generic words)
 * - Fetches kg_edges for those entities
 * - Also fetches company_connections (the user-facing connections table)
 */
export async function retrieveGraphContext(
  eventId: string,
  searchTerms: string[],
  entityNames?: string[],
): Promise<GraphRetrievalResult> {
  const entities: GraphRetrievalResult["entities"] = [];
  const edges: GraphRetrievalResult["edges"] = [];
  const connections: GraphRetrievalResult["connections"] = [];

  // Filter out noise words — only keep likely entity names
  const noiseWords = new Set([
    "about", "tell", "what", "which", "where", "when", "how", "could", "would", "should",
    "they", "them", "their", "this", "that", "with", "from", "into", "have", "been",
    "more", "most", "some", "also", "just", "make", "like", "want", "need", "give",
    "strategy", "expand", "connect", "know", "information", "details", "explain",
  ]);

  // Prioritize entity names from query analysis, then use meaningful search terms
  const cleanedEntityNames = (entityNames || []).filter(n => n && n.length > 1);
  const cleanedSearchTerms = (searchTerms || [])
    .filter(t => t && t.length > 2 && !noiseWords.has(t.toLowerCase()))
    .slice(0, 8);

  // Entity names get priority; if we have them, they're the primary search
  const primaryTerms = cleanedEntityNames.length > 0 ? cleanedEntityNames : cleanedSearchTerms;
  const allTerms = [...new Set([...primaryTerms, ...cleanedSearchTerms])].filter(Boolean);

  if (allTerms.length === 0) {
    return { entities: [], edges: [], connections: [], summary: "No search terms provided." };
  }


  try {
    // ── Step 1: Find entities by name ──
    const nameFilters = allTerms.map((t) => `name.ilike.%${t}%`).join(",");

    const { data: entityData, error: entityErr } = await supabase
      .from("kg_entities")
      .select("id, name, entity_type, properties")
      .eq("event_id", eventId)
      .or(nameFilters)
      .limit(30);

    if (entityErr) {
      console.error("[GraphRetrieval] Entity query error:", entityErr.message);
    }

    if (entityData && entityData.length > 0) {
      for (const e of entityData as any[]) {
        entities.push({
          id: e.id,
          name: e.name,
          type: e.entity_type,
          properties: e.properties || {},
        });
      }

      // ── Step 2: Get kg_edges connecting these entities (use .in() separately) ──
      const entityIds = entities.map((e) => e.id);
      try {
        // Query edges where source OR target is in our entity set
        const [sourceEdges, targetEdges] = await Promise.all([
          supabase
            .from("kg_edges")
            .select("source_entity_id, target_entity_id, relation_type, properties")
            .eq("event_id", eventId)
            .in("source_entity_id", entityIds)
            .limit(50),
          supabase
            .from("kg_edges")
            .select("source_entity_id, target_entity_id, relation_type, properties")
            .eq("event_id", eventId)
            .in("target_entity_id", entityIds)
            .limit(50),
        ]);

        const edgeMap = new Map<string, any>();
        for (const result of [sourceEdges, targetEdges]) {
          if (result.data) {
            for (const edge of result.data as any[]) {
              const key = `${edge.source_entity_id}-${edge.target_entity_id}-${edge.relation_type}`;
              if (!edgeMap.has(key)) edgeMap.set(key, edge);
            }
          }
        }

        // Also fetch names for entities we found via edges but aren't in our initial set
        const allEdgeEntityIds = new Set<string>();
        for (const edge of edgeMap.values()) {
          allEdgeEntityIds.add(edge.source_entity_id);
          allEdgeEntityIds.add(edge.target_entity_id);
        }
        const missingIds = [...allEdgeEntityIds].filter(id => !entities.find(e => e.id === id));
        const entityNameMap = new Map(entities.map((e) => [e.id, e.name]));

        if (missingIds.length > 0) {
          const { data: extraEntities } = await supabase
            .from("kg_entities")
            .select("id, name")
            .in("id", missingIds);
          if (extraEntities) {
            for (const e of extraEntities as any[]) {
              entityNameMap.set(e.id, e.name);
            }
          }
        }

        for (const edge of edgeMap.values()) {
          edges.push({
            source: entityNameMap.get(edge.source_entity_id) || edge.source_entity_id,
            target: entityNameMap.get(edge.target_entity_id) || edge.target_entity_id,
            relation: edge.relation_type,
            properties: edge.properties || {},
          });
        }
      } catch (edgeErr) {
        console.error("[GraphRetrieval] Edge query error:", edgeErr);
      }
    }

    // ── Step 3: Also search company_connections (user-facing connections table) ──
    try {
      const connFilters = allTerms
        .map((t) => `source_company_name.ilike.%${t}%,target_company_name.ilike.%${t}%`)
        .join(",");

      const { data: connData, error: connErr } = await supabase
        .from("company_connections")
        .select("source_company_name, target_company_name, connection_type, connection_status, ai_reasoning, notes")
        .eq("event_id", eventId)
        .or(connFilters)
        .limit(30);

      if (connErr) {
        console.error("[GraphRetrieval] Connection query error:", connErr.message);
      }

      if (connData) {
        for (const c of connData as any[]) {
          connections.push({
            source: c.source_company_name,
            target: c.target_company_name,
            type: c.connection_type,
            status: c.connection_status,
            reasoning: c.ai_reasoning || c.notes || "",
          });
        }
      }
    } catch (connErr) {
      console.error("[GraphRetrieval] Connection query error:", connErr);
    }

  } catch (err) {
    console.error("[GraphRetrieval] Error:", err);
  }

  // Build summary text
  const entitySummaries = entities.map((e) => {
    const props = e.properties || {};
    const details = [
      props.bio,
      props.funding_stage ? `Stage: ${props.funding_stage}` : null,
      props.industry ? `Industry: ${props.industry}` : null,
      props.website ? `Website: ${props.website}` : null,
      props.hq || props.location ? `Location: ${props.hq || props.location}` : null,
    ].filter(Boolean).join(", ");
    return `${e.name} (${e.type})${details ? ": " + details : ""}`;
  });

  const edgeSummaries = edges.map(
    (e) => `${e.source} --[${e.relation}]--> ${e.target}`
  );

  const connectionSummaries = connections.map(
    (c) => `${c.source} --[${c.type}, ${c.status}]--> ${c.target}${c.reasoning ? ` (${c.reasoning.slice(0, 150)})` : ""}`
  );

  const summary = [
    entities.length > 0 ? `Entities found:\n${entitySummaries.join("\n")}` : "No entities found in knowledge graph.",
    edges.length > 0 ? `Knowledge graph relationships:\n${edgeSummaries.join("\n")}` : "No knowledge graph relationships found.",
    connections.length > 0 ? `Company connections:\n${connectionSummaries.join("\n")}` : "No company connections found.",
  ].join("\n\n");

  return { entities, edges, connections, summary };
}


export interface KpiRetrievalResult {
  kpis: Array<{
    company: string;
    metric: string;
    value: number;
    unit: string;
    period: string;
    category: string;
    confidence: number;
  }>;
  summary: string;
}

/**
 * Retrieve structured KPIs/metrics from the company_kpis table.
 * Searches by company name and/or metric name.
 */
export async function retrieveKpiContext(
  eventId: string,
  companyNames?: string[],
  metricNames?: string[],
): Promise<KpiRetrievalResult> {
  const kpis: KpiRetrievalResult["kpis"] = [];

  try {
    let query = supabase
      .from("company_kpis")
      .select("company_name, metric_name, value, unit, period, metric_category, confidence")
      .eq("event_id", eventId);

    // Filter by company names if provided
    if (companyNames && companyNames.length > 0) {
      const filters = companyNames.map((n) => `company_name.ilike.%${n}%`).join(",");
      query = query.or(filters);
    }

    // Filter by metric names if provided
    if (metricNames && metricNames.length > 0) {
      const filters = metricNames.map((m) => `metric_name.ilike.%${m}%`).join(",");
      query = query.or(filters);
    }

    const { data } = await query.order("period", { ascending: false }).limit(50);

    if (data) {
      for (const row of data as any[]) {
        kpis.push({
          company: row.company_name,
          metric: row.metric_name,
          value: row.value,
          unit: row.unit || "",
          period: row.period || "",
          category: row.metric_category || "",
          confidence: row.confidence || 0,
        });
      }
    }
  } catch (err) {
    console.error("[KpiRetrieval] Error:", err);
  }

  // Build summary
  const summary = kpis.length > 0
    ? kpis.map((k) => `${k.company}: ${k.metric} = ${k.value} ${k.unit} (${k.period})`).join("\n")
    : "No KPI data found.";

  return { kpis, summary };
}

/**
 * Smart merge: fill empty fields from AI extraction, never overwrite user edits,
 * detect conflicts, and track property sources.
 *
 * Merge strategy:
 * - Empty field on card + extracted value exists → fill it
 * - Non-empty field on card + extracted value differs → record as conflict, don't overwrite
 * - Fields in _edited_fields → never overwrite (user wins)
 * - Store _property_sources: { field: { document_id, confidence, extracted_at } }
 */
export async function mergeCompanyCardFromExtraction(
  entityId: string,
  extractedProps: Record<string, any>,
  confidence: Record<string, number>,
  sourceDocumentId: string,
  options?: { overwriteExisting?: boolean; isMeetingNotes?: boolean }
): Promise<MergeResult> {
  const entity = await getEntityProperties(entityId);
  if (!entity) {
    return { updated: [], skipped: [], conflicts: [], entityId, companyName: "" };
  }

  const current = entity.properties;
  const editedFields: string[] = current._edited_fields || [];
  const propertySources: Record<string, any> = current._property_sources || {};
  const propertyConflicts: any[] = current._property_conflicts || [];

  const updated: string[] = [];
  const skipped: string[] = [];
  const conflicts: PropertyConflict[] = [];
  const mergedProps: Record<string, any> = {};
  let removedConflicts = false;

  // Fields that are internal metadata — skip them
  const metaFields = new Set([
    "_edited_fields", "_property_sources", "_property_conflicts",
    "auto_created", "source", "first_seen_document", "last_seen_document", "document_count",
  ]);

  // Meeting-note–specific fields: always overwrite with latest when source is meeting notes
  const MEETING_OVERWRITE_FIELDS = new Set([
    "meeting_summary", "last_meeting", "next_steps", "concerns",
    "decision", "meeting_notes", "action_items", "key_decisions",
    "last_meeting_date", "meeting_date",
  ]);

  for (const [field, newValue] of Object.entries(extractedProps)) {
    if (metaFields.has(field)) continue;

    // Skip if no meaningful value extracted
    if (newValue === "" || newValue === null || newValue === undefined) continue;
    if (Array.isArray(newValue) && newValue.length === 0) continue;

    // For meeting notes: always overwrite meeting-specific fields, even if user edited
    const isMeetingOverwrite = options?.isMeetingNotes && MEETING_OVERWRITE_FIELDS.has(field);

    // Skip if user manually edited this field (unless it's a meeting overwrite field)
    if (editedFields.includes(field) && !isMeetingOverwrite) {
      skipped.push(field);
      continue;
    }

    const currentValue = current[field];
    const isEmpty = (v: any) => {
      if (v === "" || v === null || v === undefined) return true;
      if (Array.isArray(v) && v.length === 0) return true;
      if (typeof v === "string" && v === "[]") return true;
      return false;
    };

    if (isEmpty(currentValue)) {
      // Empty → fill it
      mergedProps[field] = newValue;
      updated.push(field);
      propertySources[field] = {
        document_id: sourceDocumentId,
        confidence: confidence[field] ?? 0.5,
        extracted_at: new Date().toISOString(),
      };
    } else if (options?.overwriteExisting || isMeetingOverwrite) {
      // Force overwrite mode OR meeting-note overwrite for specific fields
      mergedProps[field] = newValue;
      updated.push(field);
      propertySources[field] = {
        document_id: sourceDocumentId,
        confidence: confidence[field] ?? 0.5,
        extracted_at: new Date().toISOString(),
      };
    } else {
      // Non-empty → check if values differ; auto-overwrite with higher confidence (no conflict UI)
      const valuesMatch = JSON.stringify(currentValue) === JSON.stringify(newValue);
      if (!valuesMatch) {
        const existingConf = propertySources[field]?.confidence ?? 0;
        const newConf = confidence[field] ?? 0.5;
        if (newConf >= existingConf) {
          mergedProps[field] = newValue;
          updated.push(field);
          propertySources[field] = {
            document_id: sourceDocumentId,
            confidence: newConf,
            extracted_at: new Date().toISOString(),
          };
        } else {
          skipped.push(field);
        }
        // Remove any existing conflict for this field (we resolved by confidence)
        const idx = propertyConflicts.findIndex((c) => c.field === field);
        if (idx !== -1) {
          propertyConflicts.splice(idx, 1);
          removedConflicts = true;
        }
      } else {
        skipped.push(field);
      }
    }
  }

  // Build the update payload
  if (updated.length > 0 || removedConflicts) {
    const updatePayload: Record<string, any> = {
      ...mergedProps,
      _property_sources: propertySources,
      _property_conflicts: propertyConflicts,
    };

    const { error } = await supabase.rpc("update_company_card_properties", {
      p_entity_id: entityId,
      p_new_properties: updatePayload,
    } as any);

    if (error) {
      console.error("[mergeCompanyCard] Update failed:", error);
    }
  }

  return { updated, skipped, conflicts, entityId, companyName: entity.name };
}

// ─────────────────────────────────────────────────────────────────────────
// STRUCTURED CSV INGESTION → kg_entities + kg_edges
// Takes parsed rows from CSV conversion and creates proper structured records
// ─────────────────────────────────────────────────────────────────────────

export interface StructuredCSVIngestionResult {
  entitiesCreated: number;
  entitiesUpdated: number;
  edgesCreated: number;
  skipped: number;
  errors: string[];
}

/**
 * Ingest structured investor rows from CSV conversion into kg_entities.
 * Creates a 'fund' entity for each unique firm, a 'person' entity for each team member,
 * and edges linking them.
 */
export async function ingestInvestorCSVRows(
  eventId: string,
  investors: Array<{
    firmName: string;
    memberName?: string;
    geoFocus?: string[];
    industryPreferences?: string[];
    stagePreferences?: string[];
    minTicketSize?: number;
    maxTicketSize?: number;
  }>,
  documentId: string | null,
  createdBy: string | null
): Promise<StructuredCSVIngestionResult> {
  const result: StructuredCSVIngestionResult = { entitiesCreated: 0, entitiesUpdated: 0, edgesCreated: 0, skipped: 0, errors: [] };

  // Group investors by firm name (one firm can have multiple team members)
  const firmMap = new Map<string, typeof investors>();
  for (const inv of investors) {
    if (!inv.firmName || inv.firmName.trim().length < 2) {
      result.skipped++;
      continue;
    }
    const key = inv.firmName.trim().toLowerCase();
    if (!firmMap.has(key)) firmMap.set(key, []);
    firmMap.get(key)!.push(inv);
  }

  for (const [normalizedName, firmInvestors] of firmMap) {
    const firstInv = firmInvestors[0];
    const firmName = firstInv.firmName.trim();

    // Collect all team members across rows for this firm
    const teamMembers = firmInvestors
      .map((i) => i.memberName?.trim())
      .filter((n): n is string => !!n && n !== "UNKNOWN" && n.length > 1);
    const uniqueTeamMembers = [...new Set(teamMembers)];

    // Merge all geo/industry/stage from all rows
    const allGeo = [...new Set(firmInvestors.flatMap((i) => i.geoFocus || []))];
    const allIndustry = [...new Set(firmInvestors.flatMap((i) => i.industryPreferences || []))];
    const allStage = [...new Set(firmInvestors.flatMap((i) => i.stagePreferences || []))];

    // Determine cheque size range
    const minTickets = firmInvestors.map((i) => i.minTicketSize || 0).filter((v) => v > 0);
    const maxTickets = firmInvestors.map((i) => i.maxTicketSize || 0).filter((v) => v > 0);
    const minTicket = minTickets.length ? Math.min(...minTickets) : 0;
    const maxTicket = maxTickets.length ? Math.max(...maxTickets) : 0;

    // Format cheque size for display
    const formatAmount = (n: number) => {
      if (n >= 1_000_000) return `$${(n / 1_000_000).toFixed(n % 1_000_000 === 0 ? 0 : 1)}M`;
      if (n >= 1_000) return `$${(n / 1_000).toFixed(0)}K`;
      return `$${n}`;
    };
    const chequeDisplay =
      minTicket && maxTicket && minTicket !== maxTicket
        ? `${formatAmount(minTicket)} - ${formatAmount(maxTicket)}`
        : maxTicket
        ? formatAmount(maxTicket)
        : minTicket
        ? formatAmount(minTicket)
        : "";

    // Check if fund entity already exists (use array query to avoid 406 from PostgREST)
    const { data: existingArr } = await supabase
      .from("kg_entities")
      .select("id")
      .eq("event_id", eventId)
      .eq("normalized_name", normalizedName)
      .eq("entity_type", "fund")
      .limit(1);
    const existing = existingArr?.[0] ?? null;

    let fundEntityId: string;

    if (existing?.id) {
      fundEntityId = existing.id;
      // Fetch current properties so we can merge without overwriting user edits
      const { data: currentArr } = await supabase
        .from("kg_entities")
        .select("properties")
        .eq("id", fundEntityId)
        .limit(1);
      const currentProps = (currentArr?.[0] as any)?.properties || {};
      
      // Merge: CSV data fills in gaps but doesn't overwrite non-empty user edits
      await supabase
        .from("kg_entities")
        .update({
          properties: {
            ...currentProps,
            auto_created: true,
            source: "csv_import",
            geo_focus: allGeo.length ? allGeo : (currentProps.geo_focus || []),
            industry_preferences: allIndustry.length ? allIndustry : (currentProps.industry_preferences || []),
            stage_preferences: allStage.length ? allStage : (currentProps.stage_preferences || []),
            cheque_size: chequeDisplay || currentProps.cheque_size || "",
            min_ticket_size: minTicket || currentProps.min_ticket_size || 0,
            max_ticket_size: maxTicket || currentProps.max_ticket_size || 0,
            team_members: uniqueTeamMembers.length ? uniqueTeamMembers : (currentProps.team_members || []),
            // Preserve user-edited fields — only set if currently empty
            bio: currentProps.bio || "",
            website: currentProps.website || "",
            logo_url: currentProps.logo_url || "",
          },
          updated_at: new Date().toISOString(),
        })
        .eq("id", fundEntityId);
      result.entitiesUpdated++;
    } else {
      const { data: newEntity, error: insertErr } = await supabase
        .from("kg_entities")
        .insert({
          event_id: eventId,
          entity_type: "fund",
          name: firmName,
          normalized_name: normalizedName,
          properties: {
            auto_created: true,
            source: "csv_import",
            geo_focus: allGeo,
            industry_preferences: allIndustry,
            stage_preferences: allStage,
            cheque_size: chequeDisplay,
            min_ticket_size: minTicket,
            max_ticket_size: maxTicket,
            team_members: uniqueTeamMembers,
            bio: "",
            website: "",
            logo_url: "",
          },
          source_document_id: documentId,
          confidence: 0.9,
          created_by: createdBy,
        })
        .select("id")
        .single();

      if (insertErr || !newEntity) {
        result.errors.push(`Failed to create fund entity for "${firmName}": ${insertErr?.message || "unknown"}`);
        continue;
      }
      fundEntityId = newEntity.id;
      result.entitiesCreated++;
    }

    // Create person entities and edges for team members
    for (const memberName of uniqueTeamMembers) {
      const memberNormalized = memberName.toLowerCase().trim();

      // Use array query to avoid 406 from PostgREST when no rows found
      const { data: existingPersonArr } = await supabase
        .from("kg_entities")
        .select("id")
        .eq("event_id", eventId)
        .eq("normalized_name", memberNormalized)
        .eq("entity_type", "person")
        .limit(1);
      const existingPerson = existingPersonArr?.[0] ?? null;

      let personEntityId: string;
      if (existingPerson?.id) {
        personEntityId = existingPerson.id;
      } else {
        const { data: newPerson, error: personErr } = await supabase
          .from("kg_entities")
          .insert({
            event_id: eventId,
            entity_type: "person",
            name: memberName,
            normalized_name: memberNormalized,
            properties: { auto_created: true, source: "csv_import", role: "team_member" },
            source_document_id: documentId,
            confidence: 0.9,
            created_by: createdBy,
          })
          .select("id")
          .single();
        if (personErr || !newPerson) continue;
        personEntityId = newPerson.id;
        result.entitiesCreated++;
      }

      // Create "works_at" edge: person → fund
      const { error: edgeErr } = await supabase.from("kg_edges").insert({
        event_id: eventId,
        source_entity_id: personEntityId,
        target_entity_id: fundEntityId,
        relation_type: "works_at",
        properties: { role: "team_member" },
        source_document_id: documentId,
        confidence: 0.9,
        created_by: createdBy,
      });
      if (!edgeErr) result.edgesCreated++;
    }
  }

  return result;
}

/**
 * Ingest structured startup rows from CSV conversion into kg_entities.
 */
export async function ingestStartupCSVRows(
  eventId: string,
  startups: Array<{
    companyName: string;
    geoMarkets?: string[];
    industry?: string;
    fundingTarget?: number;
    fundingStage?: string;
  }>,
  documentId: string | null,
  createdBy: string | null
): Promise<StructuredCSVIngestionResult> {
  const result: StructuredCSVIngestionResult = { entitiesCreated: 0, entitiesUpdated: 0, edgesCreated: 0, skipped: 0, errors: [] };

  for (const startup of startups) {
    if (!startup.companyName || startup.companyName.trim().length < 2) {
      result.skipped++;
      continue;
    }
    const name = startup.companyName.trim();
    const normalizedName = name.toLowerCase();

    // Use array query to avoid 406 from PostgREST when no rows found
    const { data: existingArr } = await supabase
      .from("kg_entities")
      .select("id")
      .eq("event_id", eventId)
      .eq("normalized_name", normalizedName)
      .eq("entity_type", "company")
      .limit(1);
    const existing = existingArr?.[0] ?? null;

    if (existing?.id) {
      result.entitiesUpdated++;
      continue;
    }

    const { error: insertErr } = await supabase.from("kg_entities").insert({
      event_id: eventId,
      entity_type: "company",
      name,
      normalized_name: normalizedName,
      properties: {
        auto_created: true,
        source: "csv_import",
        bio: "",
        funding_stage: startup.fundingStage || "",
        amount_seeking: startup.fundingTarget ? `$${startup.fundingTarget}` : "",
        valuation: "",
        arr: "",
        burn_rate: "",
        runway_months: "",
        problem: "",
        solution: "",
        tam: "",
        competitive_edge: "",
        founders: [],
        ai_rationale: "",
        website: "",
        logo_url: "",
        geo_markets: startup.geoMarkets || [],
        industry: startup.industry || "",
      },
      source_document_id: documentId,
      confidence: 0.9,
      created_by: createdBy,
    });

    if (insertErr) {
      result.errors.push(`Failed to create company "${name}": ${insertErr.message}`);
    } else {
      result.entitiesCreated++;
    }
  }

  return result;
}

/**
 * Get all entity cards (both 'company' and 'fund' types) for display.
 */
export async function getAllEntityCards(eventId: string) {
  const { data: entities, error } = await supabase
    .from("kg_entities")
    .select("id, name, entity_type, properties, event_id, created_at")
    .eq("event_id", eventId)
    .in("entity_type", ["company", "fund"])
    .order("name", { ascending: true });

  if (error || !entities) return { data: [], error };

  // Get document counts per entity (batch in chunks of 25 to avoid URL length limits)
  const entityIds = entities.map((e: any) => e.id);
  const docCountMap = new Map<string, number>();
  const BATCH_SIZE = 25;
  for (let i = 0; i < entityIds.length; i += BATCH_SIZE) {
    const batch = entityIds.slice(i, i + BATCH_SIZE);
    const { data: docLinks } = await supabase
      .from("documents")
      .select("company_entity_id")
      .in("company_entity_id", batch.length ? batch : ["__none__"]);
    (docLinks || []).forEach((d: any) => {
      const id = d.company_entity_id;
      docCountMap.set(id, (docCountMap.get(id) || 0) + 1);
    });
  }

  const rawCards = entities.map((e: any) => ({
    company_id: e.id,
    company_name: e.name,
    entity_type: e.entity_type,
    company_properties: e.properties || {},
    document_count: docCountMap.get(e.id) || 0,
    created_at: e.created_at,
  }));

  // ── Deduplicate: keep the primary card per core company name ──
  // "Chhaya", "Chhaya Technologies Limited", "CHHAYA TECHNOLOGIES PTE. LTD."
  // all share core name "chhaya" → keep the one with most documents.
  const bestByCore = new Map<string, typeof rawCards[0]>();
  for (const card of rawCards) {
    const core = extractCoreCompanyName(card.company_name);
    if (!core) continue;
    const existing = bestByCore.get(core);
    if (
      !existing ||
      card.document_count > existing.document_count ||
      (card.document_count === existing.document_count &&
        JSON.stringify(card.company_properties).length >
          JSON.stringify(existing.company_properties).length)
    ) {
      bestByCore.set(core, card);
    }
  }
  const cards = Array.from(bestByCore.values());

  return { data: cards, error: null };
}

export async function getPendingRelationshipReviews(eventId: string) {
  try {
    // First, try to get edges with joined entities
    const result = await supabase
      .from("kg_edges")
      .select(`
        id,
        relation_type,
        confidence,
        properties,
        source_document_id,
        created_at,
        source_entity_id,
        target_entity_id,
        source_entity:kg_entities!source_entity_id(name, entity_type),
        target_entity:kg_entities!target_entity_id(name, entity_type)
      `)
      .eq("event_id", eventId)
      .eq("review_status", "pending")
      .order("created_at", { ascending: false });
    
    // If the query fails (e.g., review_status column doesn't exist), return empty
    if (result.error) {
      console.warn("[getPendingRelationshipReviews] Query failed:", result.error);
      return { data: [], error: null };
    }
    
    return result;
  } catch (error) {
    console.warn("[getPendingRelationshipReviews] Exception:", error);
    return { data: [], error: null };
  }
}

// Update kg_edge review status
export async function updateKgEdgeReview(
  edgeId: string,
  reviewStatus: "approved" | "rejected" | "edited",
  reviewedBy: string
) {
  return supabase
    .from("kg_edges")
    .update({
      review_status: reviewStatus,
      reviewed_by: reviewedBy,
      reviewed_at: new Date().toISOString(),
    })
    .eq("id", edgeId)
    .select()
    .single();
}

export async function insertCompanyConnection(
  eventId: string,
  payload: {
    source_company_name: string;
    target_company_name: string;
    source_document_id?: string | null;
    target_document_id?: string | null;
    connection_type: ConnectionType;
    connection_status: ConnectionStatus;
    ai_reasoning?: string | null;
    notes?: string | null;
    created_by: string | null;
  }
) {
  return supabase
    .from("company_connections")
    .insert({ event_id: eventId, ...payload })
    .select("*")
    .single();
}

export async function updateCompanyConnection(
  connectionId: string,
  payload: Partial<{
    connection_status: ConnectionStatus;
    notes: string | null;
  }>
) {
  return supabase
    .from("company_connections")
    .update(payload)
    .eq("id", connectionId)
    .select("*")
    .single();
}

export async function deleteCompanyConnection(connectionId: string) {
  return supabase.from("company_connections").delete().eq("id", connectionId);
}

export async function insertSource(
  eventId: string,
  payload: {
    title: string | null;
    source_type: SourceRecord["source_type"];
    external_url: string | null;
    storage_path: string | null;
    tags: string[] | null;
    notes: string | null;
    status: SourceRecord["status"];
    created_by: string | null;
  }
) {
  const basePayload = { event_id: eventId, ...payload };
  const insertResult = await supabase.from("sources").insert(basePayload).select("*").single();

  // Backwards-compatible retry: older databases may not have notes/storage_path yet.
  if (insertResult.error) {
    const message = insertResult.error.message || "";
    const missingNotes = message.includes("notes") && message.includes("column");
    const missingStoragePath = message.includes("storage_path") && message.includes("column");
    if (missingNotes || missingStoragePath) {
      const fallbackPayload: Record<string, any> = { ...basePayload };
      if (missingNotes) {
        delete fallbackPayload.notes;
      }
      if (missingStoragePath) {
        delete fallbackPayload.storage_path;
      }
      return supabase.from("sources").insert(fallbackPayload).select("*").single();
    }
  }

  return insertResult;
}

export async function deleteSource(sourceId: string) {
  return supabase.from("sources").delete().eq("id", sourceId);
}

// =============================================================================
// TASKS (Dashboard task hub: MD assigns, team updates status)
// =============================================================================

export async function getTasksByEvent(eventId: string): Promise<SupabaseResult<Task[]>> {
  const { data, error } = await supabase
    .from("tasks")
    .select("*")
    .eq("event_id", eventId)
    .order("deadline", { ascending: true, nullsFirst: false })
    .order("created_at", { ascending: false });
  return { data: (data as Task[]) || [], error };
}

export async function getMyTasks(eventId: string, userId: string): Promise<SupabaseResult<Task[]>> {
  const { data, error } = await supabase
    .from("tasks")
    .select("*")
    .eq("event_id", eventId)
    .eq("assignee_user_id", userId)
    .order("deadline", { ascending: true, nullsFirst: false })
    .order("created_at", { ascending: false });
  return { data: (data as Task[]) || [], error };
}

export async function insertTask(
  eventId: string,
  payload: { assignee_user_id: string | null; title: string; description?: string | null; start_date?: string | null; deadline?: string | null; created_by: string }
): Promise<SupabaseResult<Task>> {
  const { data, error } = await supabase
    .from("tasks")
    .insert({
      event_id: eventId,
      assignee_user_id: payload.assignee_user_id ?? null,
      title: payload.title,
      description: payload.description ?? null,
      status: "not_started",
      start_date: payload.start_date ?? null,
      deadline: payload.deadline ?? null,
      created_by: payload.created_by,
    })
    .select("*")
    .single();
  return { data: data as Task, error };
}

export async function updateTask(
  taskId: string,
  updates: Partial<Pick<Task, "assignee_user_id" | "title" | "description" | "status" | "start_date" | "deadline" | "status_note">>
): Promise<SupabaseResult<Task>> {
  const { data, error } = await supabase
    .from("tasks")
    .update(updates)
    .eq("id", taskId)
    .select("*")
    .single();
  return { data: data as Task, error };
}

export async function updateTaskStatus(
  taskId: string,
  status: TaskStatus,
  statusNote?: string | null
): Promise<SupabaseResult<Task>> {
  const payload: Record<string, unknown> = { status };
  if (statusNote !== undefined) payload.status_note = statusNote;
  return updateTask(taskId, payload);
}

export async function deleteTask(taskId: string): Promise<SupabaseResult<void>> {
  const { error } = await supabase.from("tasks").delete().eq("id", taskId);
  return { data: undefined, error };
}
