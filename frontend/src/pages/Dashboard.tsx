import { useState, useEffect, useRef } from "react";
import { getCompanyContext, askStream } from "../utils/api";

interface Props {
  orgId: string;
  userId: string;
}

interface Message {
  role: "user" | "assistant";
  content: string;
}

export default function Dashboard({ orgId, userId }: Props) {
  const [companyName, setCompanyName] = useState("Your Company");
  const [companyDescription, setCompanyDescription] = useState("");
  const [systemPrompt, setSystemPrompt] = useState("");
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    getCompanyContext(orgId)
      .then((ctx) => {
        setCompanyName(ctx.company_name || "Your Company");
        setCompanyDescription(ctx.company_description);
        setSystemPrompt(ctx.system_prompt);
      })
      .catch(() => {});
  }, [orgId]);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const handleSend = async () => {
    const q = input.trim();
    if (!q || isLoading) return;

    setInput("");
    setMessages((prev) => [...prev, { role: "user", content: q }]);
    setIsLoading(true);

    let responseText = "";
    setMessages((prev) => [...prev, { role: "assistant", content: "" }]);

    await askStream(
      {
        question: q,
        sources: [],
        previousMessages: messages.slice(-10).map((m) => ({ role: m.role, content: m.content })),
        organizationId: orgId,
      },
      (chunk) => {
        responseText += chunk;
        setMessages((prev) => {
          const copy = [...prev];
          copy[copy.length - 1] = { role: "assistant", content: responseText };
          return copy;
        });
      },
      (err) => {
        responseText = `Error: ${err.message}`;
        setMessages((prev) => {
          const copy = [...prev];
          copy[copy.length - 1] = { role: "assistant", content: responseText };
          return copy;
        });
      }
    );

    setIsLoading(false);
  };

  return (
    <div className="flex h-screen bg-slate-950">
      {/* Sidebar */}
      <div className="w-72 border-r border-slate-700/60 bg-slate-900/50 flex flex-col">
        <div className="p-4 border-b border-slate-700/60">
          <div className="text-lg font-bold text-white">{companyName}</div>
          <div className="text-xs text-amber-400 font-medium">Orbit Platform</div>
        </div>

        <nav className="flex-1 p-3 space-y-1">
          <button className="w-full text-left px-3 py-2 rounded-lg bg-amber-500/10 border border-amber-500/20 text-amber-200 text-sm font-medium">
            Chat
          </button>
          <button className="w-full text-left px-3 py-2 rounded-lg text-slate-400 hover:bg-slate-800/50 hover:text-white text-sm transition-all">
            Documents
          </button>
          <button className="w-full text-left px-3 py-2 rounded-lg text-slate-400 hover:bg-slate-800/50 hover:text-white text-sm transition-all">
            Knowledge Graph
          </button>
          <button className="w-full text-left px-3 py-2 rounded-lg text-slate-400 hover:bg-slate-800/50 hover:text-white text-sm transition-all">
            Decisions
          </button>
          <button className="w-full text-left px-3 py-2 rounded-lg text-slate-400 hover:bg-slate-800/50 hover:text-white text-sm transition-all">
            Settings
          </button>
        </nav>

        {/* AI context preview */}
        {systemPrompt && (
          <div className="p-3 border-t border-slate-700/60">
            <div className="text-xs text-slate-500 font-medium uppercase tracking-wider mb-1">AI Context</div>
            <p className="text-xs text-slate-400 line-clamp-4">{systemPrompt.slice(0, 200)}...</p>
          </div>
        )}
      </div>

      {/* Chat area */}
      <div className="flex-1 flex flex-col">
        <div className="border-b border-slate-700/60 bg-slate-900/30 px-6 py-3">
          <h1 className="text-lg font-semibold text-white">Ask Orbit AI</h1>
          <p className="text-xs text-slate-400">AI customized for {companyName}</p>
        </div>

        <div className="flex-1 overflow-y-auto p-6 space-y-4">
          {messages.length === 0 && (
            <div className="flex items-center justify-center h-full">
              <div className="text-center max-w-md">
                <div className="text-4xl mb-4">🪐</div>
                <h2 className="text-xl font-bold text-white mb-2">Welcome to Orbit AI</h2>
                <p className="text-slate-400 text-sm">
                  Your AI is customized for <strong className="text-white">{companyName}</strong>.
                  Ask questions, upload documents, and get insights tailored to your business.
                </p>
              </div>
            </div>
          )}

          {messages.map((msg, i) => (
            <div key={i} className={`flex ${msg.role === "user" ? "justify-end" : "justify-start"}`}>
              <div
                className={`max-w-[70%] rounded-2xl px-4 py-3 text-sm whitespace-pre-wrap ${
                  msg.role === "user"
                    ? "bg-amber-500 text-slate-950"
                    : "bg-slate-800 text-slate-200 border border-slate-700/40"
                }`}
              >
                {msg.content || (
                  <span className="flex items-center gap-2 text-slate-400">
                    <svg className="animate-spin h-4 w-4" viewBox="0 0 24 24">
                      <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" fill="none" />
                      <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
                    </svg>
                    Thinking...
                  </span>
                )}
              </div>
            </div>
          ))}
          <div ref={messagesEndRef} />
        </div>

        {/* Input */}
        <div className="border-t border-slate-700/60 bg-slate-900/30 p-4">
          <div className="flex gap-3 max-w-4xl mx-auto">
            <input
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && !e.shiftKey && handleSend()}
              placeholder="Ask anything about your business..."
              disabled={isLoading}
              className="flex-1 rounded-xl border border-slate-600 bg-slate-800/50 text-white placeholder:text-slate-500 px-4 py-3 text-sm focus:border-amber-500/50 focus:ring-1 focus:ring-amber-500/30 outline-none disabled:opacity-50"
            />
            <button
              onClick={handleSend}
              disabled={!input.trim() || isLoading}
              className="px-6 rounded-xl bg-amber-500 text-slate-950 hover:bg-amber-400 font-semibold disabled:opacity-50 transition-all"
            >
              Send
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
