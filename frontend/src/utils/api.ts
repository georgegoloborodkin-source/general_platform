/**
 * API client for the Orbit Platform backend.
 */

const API_BASE =
  import.meta.env.VITE_API_URL ||
  (typeof window !== "undefined" && window.location.hostname !== "localhost"
    ? "https://general-platform.onrender.com"
    : "http://localhost:10000");

export async function setupCompany(input: {
  organizationId: string;
  companyDescription: string;
  companyName?: string;
}): Promise<{ system_prompt: string; organization_id: string; message: string }> {
  const resp = await fetch(`${API_BASE}/company/setup`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      organization_id: input.organizationId,
      company_description: input.companyDescription,
      company_name: input.companyName,
    }),
  });
  if (!resp.ok) {
    const err = await resp.json().catch(() => ({}));
    throw new Error(err.detail || `HTTP ${resp.status}`);
  }
  return resp.json();
}

export async function getCompanyContext(orgId: string): Promise<{
  organization_id: string;
  company_description: string;
  system_prompt: string;
  industry_hint: string;
  company_name: string;
}> {
  const resp = await fetch(`${API_BASE}/company/${orgId}/context`);
  if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
  return resp.json();
}

export interface AskStreamInput {
  question: string;
  sources: Array<{ title?: string; file_name?: string; snippet?: string }>;
  previousMessages?: Array<{ role: string; content: string }>;
  organizationId?: string;
}

export async function askStream(
  input: AskStreamInput,
  onChunk: (text: string) => void,
  onError?: (error: Error) => void,
): Promise<void> {
  const controller = new AbortController();
  const timeout = window.setTimeout(() => controller.abort(), 90000);

  try {
    const resp = await fetch(`${API_BASE}/ask/stream`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        question: input.question,
        sources: input.sources,
        previous_messages: input.previousMessages || [],
        organization_id: input.organizationId,
      }),
      signal: controller.signal,
    });

    if (!resp.ok) {
      const err = await resp.json().catch(() => ({}));
      throw new Error(err.detail || `HTTP ${resp.status}`);
    }

    const reader = resp.body?.getReader();
    if (!reader) throw new Error("No response body");

    const decoder = new TextDecoder();
    let buffer = "";

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split("\n");
      buffer = lines.pop() || "";

      for (const line of lines) {
        if (!line.startsWith("data: ")) continue;
        const data = line.slice(6);
        if (data === "[DONE]") return;
        try {
          const parsed = JSON.parse(data);
          if (parsed.text) onChunk(parsed.text);
          if (parsed.error) throw new Error(parsed.error);
        } catch {}
      }
    }
  } catch (err: any) {
    if (err.name === "AbortError") {
      onError?.(new Error("Request timed out"));
    } else {
      onError?.(err);
    }
  } finally {
    clearTimeout(timeout);
  }
}
