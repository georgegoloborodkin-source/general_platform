-- Recency-weighted prioritization + timeline for document chunks.
-- 1) All documents are considered (no exclusion of older versions).
-- 2) Score = similarity + (alpha * recency_norm) so newer docs get a boost but old ones still appear.
-- 3) Return document_created_at and event_created_at so the app and Claude can trace timeline.

CREATE OR REPLACE FUNCTION public.match_document_chunks(
  query_embedding VECTOR(1536),
  match_count INT,
  filter_event_id UUID
)
RETURNS TABLE (
  document_id UUID,
  similarity FLOAT,
  chunk_text TEXT,
  parent_text TEXT,
  parent_index INT,
  child_index INT,
  document_created_at TIMESTAMPTZ,
  event_created_at TIMESTAMPTZ,
  combined_score FLOAT
)
LANGUAGE sql
STABLE
AS $$
  WITH event_doc_bounds AS (
    SELECT
      MIN(d.created_at) AS min_at,
      MAX(d.created_at) AS max_at
    FROM public.documents d
    WHERE d.event_id = filter_event_id
  ),
  scored AS (
    SELECT
      de.document_id,
      (1 - (de.embedding <=> query_embedding))::FLOAT AS similarity,
      de.chunk_text,
      de.parent_text,
      de.parent_index,
      de.child_index,
      d.created_at AS document_created_at,
      e.created_at AS event_created_at,
      (1 - (de.embedding <=> query_embedding))
        + (0.1 * COALESCE(
            (EXTRACT(EPOCH FROM d.created_at) - EXTRACT(EPOCH FROM (SELECT min_at FROM event_doc_bounds)))
            / NULLIF(
                EXTRACT(EPOCH FROM (SELECT max_at FROM event_doc_bounds))
                - EXTRACT(EPOCH FROM (SELECT min_at FROM event_doc_bounds)),
                0
              ),
            1
          ))::FLOAT AS combined_score
    FROM public.document_embeddings de
    JOIN public.documents d ON d.id = de.document_id
    JOIN public.events e ON e.id = d.event_id
    WHERE d.event_id = filter_event_id
  )
  SELECT
    s.document_id,
    s.similarity,
    s.chunk_text,
    s.parent_text,
    s.parent_index,
    s.child_index,
    s.document_created_at,
    s.event_created_at,
    s.combined_score
  FROM scored s
  ORDER BY s.combined_score DESC
  LIMIT match_count;
$$;

COMMENT ON FUNCTION public.match_document_chunks(VECTOR(1536), INT, UUID) IS
  'Vector similarity search with recency boost (newer docs weighted higher) and timeline columns document_created_at, event_created_at, combined_score.';
