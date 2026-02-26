import React, { useMemo, useState, useCallback, useEffect, useRef } from "react";
import {
  Card,
  CardContent,
  CardHeader,
  CardTitle,
  CardDescription,
} from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import { Badge } from "@/components/ui/badge";
import { Checkbox } from "@/components/ui/checkbox";
import { Separator } from "@/components/ui/separator";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import {  
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
} from "@/components/ui/alert-dialog";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { Label } from "@/components/ui/label";
import { Slider } from "@/components/ui/slider";
import { useToast } from "@/hooks/use-toast";
import { useAuth } from "@/hooks/useAuth";
import {
  Brain,
  FileText,
  ClipboardList,
  Upload,
  Loader2,
  Download,
  Trash2,
  TrendingUp,
  Users,
  Target,
  AlertTriangle,
  CheckCircle,
  Clock,
  DollarSign,
  Sparkles,
  Square,
  Folder,
  ChevronDown,
  ChevronRight,
  FolderPlus,
  Link2,
  BarChart3,
  PieChart,
  Eye,
  MessageSquarePlus,
  Check,
  X,
  Building2,
  Globe,
  Linkedin,
  Pencil,
  Save,
  Rocket,
  TrendingDown,
  Award,
  Briefcase,
  Mail,
  Phone,
  Twitter,
  Hash,
  Zap,
  ShoppingCart,
  Repeat,
  MapPin,
  Calendar,
  Handshake,
  Trophy,
  Megaphone,
  Percent,
  Cloud,
  RefreshCw,
  LogOut,
  User,
  Plus,
  ListTodo,
  GanttChart,
  Key,
} from "lucide-react";
import {
  BarChart,
  Bar,
  LineChart,
  Line,
  PieChart as RechartsPieChart,
  Pie,
  Cell,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip as RechartsTooltip,
  Legend,
  ResponsiveContainer,
} from "recharts";
import {
  calculateDecisionStats,
  exportDecisionsToCSV,
  type Decision,
} from "@/utils/claudeConverter";
import { calculateDecisionEngineAnalytics } from "@/utils/decisionAnalytics";
import type { DocumentRecord, SourceRecord, Task, UserProfile } from "@/types";
import { TeamInvitationForm } from "@/components/TeamInvitationForm";
import { TeamMembersList } from "@/components/TeamMembersList";
import { SyncStatus } from "@/components/SyncStatus";
import { Link } from "react-router-dom";
import { Shield, CalendarIcon } from "lucide-react";
import { Calendar as CalendarPicker } from "@/components/ui/calendar";
import { Popover, PopoverContent, PopoverTrigger } from "@/components/ui/popover";
import { format } from "date-fns";
import {
  ensureActiveEventForOrg,
  ensureOrganizationForUser,
  getDecisionsByEvent,
  getDocumentsByEvent,
  getSourcesByEvent,
  getSourceFoldersByEvent,
  ensureDefaultFoldersForEvent,
  getCompanyConnectionsByEvent,
  getPendingRelationshipReviews,
  updateKgEdgeReview,
  getCompanyCards,
  updateCompanyCardProperties,
  getAllEntityCards,
  ingestStartupCSVRows,
  insertDecision,
  insertDocument,
  insertSource,
  insertSourceFolder,
  updateFolderCategory,
  deleteFolderAndContents,
  insertCompanyConnection,
  updateCompanyConnection,
  deleteCompanyConnection,
  updateDecision,
  deleteDecision,
  deleteSource,
  getDocumentById,
  getDocumentCompanyEntityId,
  getEntityProperties,
  mergeCompanyCardFromExtraction,
  normalizeCompanyNameForMatch,
  extractCoreCompanyName,
  getTasksByEvent,
  getMyTasks,
  insertTask,
  updateTask,
  updateTaskStatus,
  deleteTask,
  retrieveGraphContext,
  retrieveKpiContext,
  type ConnectionType,
  type ConnectionStatus,
  type CompanyConnection,
} from "@/utils/supabaseHelpers";
import { convertFileWithAI, convertWithAI, askClaudeAnswerStream, askAgentStream, deleteRedundantCards, deleteAllCards, embedQuery, rerankDocuments, rewriteQueryWithLLM, generateMultiQueries, suggestConnections, contextualizeChunk, agenticChunk, graphragRetrieve, analyzeQuery, logRAGEval, extractEntities, extractCompanyProperties, orchestrateQuery, criticCheck, type AIConversionResponse, type AskFundConnection, type QueryAnalysis, type VerifiableSource, type SourceDoc } from "@/utils/aiConverter";
import { getClickUpLists, ingestClickUpList, ingestGoogleDrive, listDriveFolders, listDriveFiles, downloadDriveFile, warmUpIngestion, sleep, refreshGoogleAccessToken, getGoogleAccessTokenFromBackend, type GDriveFolderEntry, type GDriveFileEntry } from "@/utils/ingestionClient";
import { getStoredGoogleRefreshToken, getStoredGoogleAccessToken, setStoredGoogleAccessToken, saveGoogleProviderTokens } from "@/utils/googleAuthStorage";
import { triggerGoogleOAuthForDrive } from "@/utils/googleOAuth";
import { gmailListMessages, gmailIngestMessage, gmailDownloadAttachment, type GmailIngestResult } from "@/utils/gmailClient";
import { supabase } from "@/integrations/supabase/client";
import { getCompanyContext, setupCompany } from "@/utils/api";

// xlsx loaded at runtime from CDN to avoid Vercel build resolution issues

// ============================================================================
// TYPES
// ============================================================================

type ScopeItem = { id: string; label: string; checked: boolean; type: "workspace" | "project" | "thread" | "global" | "folder"; category?: string };
type Message = {
  id: string;
  author: "user" | "assistant";
  text: string;
  threadId: string;
  isStreaming?: boolean;
  /** Verifiable RAG: click-to-source citations from agent */
  verifiableSources?: VerifiableSource[];
  /** Simple source docs from agent (id + title for Sources strip) */
  sourceDocs?: SourceDoc[];
  /** Non-document context used (e.g. "Knowledge Graph", "Structured KPIs") вЂ” shown under Sources */
  contextLabels?: string[];
  /** Devil's Advocate: red flags / risk critique */
  critic?: string;
};
type Thread = { id: string; title: string; parentId?: string };
type KnowledgeObject = {
  id: string;
  type: "Company" | "Person" | "Risk" | "Decision" | "Outcome";
  title: string;
  text: string;
  source: string;
  linked: string[];
};

declare global {
  interface Window {
    gapi?: any;
    google?: any;
  }
}

// ============================================================================
// INITIAL DATA
// ============================================================================

const initialScopes: ScopeItem[] = [
  { id: "my-docs", label: "My docs", checked: true, type: "workspace" },
  { id: "team-docs", label: "Team docs", checked: true, type: "global" },
  { id: "threads", label: "Saved Threads", checked: false, type: "thread" },
];

const initialThreads: Thread[] = [];
const initialMessages: Message[] = [];
const LOCAL_CHAT_CACHE_KEY = "platform_chat_cache";

type LocalChatMessage = {
  id: string;
  threadId: string;
  author: "assistant" | "user";
  text: string;
  ts: string;
};

const FOLDER_CATEGORIES = [
  "Sourcing",
  "BD",
  "Mentors / Corporates",
  "Projects",
  "Partners",
] as const;

type FolderCategory = (typeof FOLDER_CATEGORIES)[number];

type SourceFolder = {
  id: string;
  name: string;
  category?: FolderCategory | string | null;
  created_at?: string | null;
  created_by?: string | null;
};
const initialKOs: KnowledgeObject[] = [];

let googlePickerReady = false;
let googlePickerPromise: Promise<void> | null = null;

function loadGooglePicker(): Promise<void> {
  if (googlePickerReady) return Promise.resolve();
  if (googlePickerPromise) return googlePickerPromise;
  googlePickerPromise = new Promise((resolve, reject) => {
    const script = document.createElement("script");
    script.src = "https://apis.google.com/js/api.js";
    script.async = true;
    script.onload = () => {
      if (!window.gapi) {
        reject(new Error("Google API failed to load."));
        return;
      }
      window.gapi.load("picker", {
        callback: () => {
          googlePickerReady = true;
          resolve();
        },
        onerror: () => reject(new Error("Google Picker failed to load.")),
      });
    };
    script.onerror = () => reject(new Error("Google API script failed to load."));
    document.body.appendChild(script);
  });
  return googlePickerPromise;
}

// ============================================================================
// THREAD TREE COMPONENT
// ============================================================================

function ThreadTree({ threads, active, onSelect }: { threads: Thread[]; active: string; onSelect: (id: string) => void }) {
  const renderThread = (t: Thread, level = 0) => {
    return (
      <div
        key={t.id}
        className={`flex items-center gap-2 p-2 rounded-md cursor-pointer border border-slate-200 hover:border-blue-500 hover:bg-blue-600/5 transition-colors bg-white ${
          active === t.id ? "border-blue-500 bg-blue-600/10" : ""
        }`}
        style={{ paddingLeft: `${12 + level * 16}px` }}
        onClick={() => onSelect(t.id)}
      >
        <span className={`text-sm ${active === t.id ? "font-semibold" : ""}`}>{typeof t.title === "string" ? t.title.replace(/^[\s\u2022\u2023\u2043\u2219\u00A0\u2013\u2014\u0432\u0451]+/, "").trimStart() : t.title}</span>
      </div>
    );
  };
  const root = threads.filter((t) => !t.parentId);
  const children = (id: string) => threads.filter((t) => t.parentId === id);
  const walk = (t: Thread, level: number): JSX.Element[] => {
    const arr = [renderThread(t, level)];
    children(t.id).forEach((c) => arr.push(...walk(c, level + 1)));
    return arr;
  };
  return <div className="space-y-1">{root.flatMap((t) => walk(t, 0))}</div>;
}

function mapDecisionRow(row: any): Decision {
  return {
    id: row.id,
    timestamp: row.created_at,
    actor: row.actor_name,
    actionType: row.action_type,
    startupName: row.startup_name,
    context: row.context || {},
    confidenceScore: row.confidence_score ?? 0,
    outcome: row.outcome ?? undefined,
    notes: row.notes ?? undefined,
    documentId: row.document_id ?? undefined,
  };
}

// ============================================================================
// DOCUMENT CONVERTER TAB
// ============================================================================

function DocumentConverterTab({
  onDecisionDraft,
  onOpenDecisionLog,
  onAutoLogDecision,
}: {
  onDecisionDraft: (draft: { startupName: string; sector?: string; stage?: string }) => void;
  onOpenDecisionLog: () => void;
  onAutoLogDecision: (input: {
    draft: { startupName: string; sector?: string; stage?: string };
    conversion: AIConversionResponse;
    sourceType: "upload" | "paste" | "api";
    fileName: string | null;
    file: File | null;
    rawContent?: string | null;
    eventIdOverride?: string | null;
  }) => Promise<void>;
}) {
  const { toast } = useToast();
  const [documentText, setDocumentText] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [result, setResult] = useState<AIConversionResponse | null>(null);
  const MAX_PASTE_CHARS = 24000;

  const handleFileUpload = useCallback(async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    setIsLoading(true);
    setResult(null);
    try {
      const conversion = await convertFileWithAI(file);
      setResult(conversion);
      toast({ title: "Conversion complete", description: `Detected ${conversion.detectedType || "data"}` });

      const draft = conversion.startups?.[0]
        ? {
            startupName: conversion.startups[0].companyName || "Unknown Company",
            sector: conversion.startups[0].industry || undefined,
            stage: conversion.startups[0].fundingStage || undefined,
          }
        : null;
      if (draft) {
        await onAutoLogDecision({
          draft,
          conversion,
          sourceType: "upload",
          fileName: file.name || null,
          file,
        });
        toast({ title: "Decision logged", description: "Auto-created from extraction." });
      }
    } catch (error) {
      toast({
        title: "Conversion failed",
        description: error instanceof Error ? error.message : "File conversion failed",
        variant: "destructive",
      });
    } finally {
      setIsLoading(false);
    }
  }, [toast, onAutoLogDecision]);

  const handleExtract = useCallback(async () => {
    if (!documentText.trim()) {
      toast({ title: "No content", description: "Please paste or upload document text", variant: "destructive" });
      return;
    }
    setIsLoading(true);
    setResult(null);

    try {
      let input = documentText;
      if (input.length > MAX_PASTE_CHARS) {
        input = input.slice(0, MAX_PASTE_CHARS);
        toast({
          title: "Content trimmed",
          description: "Pasted text was too long; we trimmed it to fit the converter limit.",
        });
      }

      const conversion = await convertWithAI(input);
      setResult(conversion);
      toast({
        title: "Extraction complete",
        description: `Detected ${conversion.detectedType || "data"}`,
      });

      const draft = conversion.startups?.[0]
        ? {
            startupName: conversion.startups[0].companyName || "Unknown Company",
            sector: conversion.startups[0].industry || undefined,
            stage: conversion.startups[0].fundingStage || undefined,
          }
        : null;
      if (conversion.errors?.length && !draft) {
        toast({
          title: "Extraction warning",
          description: conversion.errors[0],
          variant: "destructive",
        });
      }

      if (draft) {
        await onAutoLogDecision({
          draft,
          conversion,
          sourceType: "paste",
          fileName: null,
          file: null,
        });
        toast({ title: "Decision logged", description: "Auto-created from extraction." });
      }
    } catch (error) {
      toast({
        title: "Error",
        description: error instanceof Error ? error.message : "Extraction failed",
        variant: "destructive",
      });
    } finally {
      setIsLoading(false);
    }
  }, [documentText, toast, onAutoLogDecision]);

  const downloadJSON = useCallback(() => {
    if (!result) return;
    const blob = new Blob([JSON.stringify(result, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `conversion-${new Date().toISOString().split("T")[0]}.json`;
    a.click();
    URL.revokeObjectURL(url);
  }, [result]);

  const primaryStartup = result?.startups?.[0];
  const quickLogEnabled = !!primaryStartup;

  const handleQuickLog = () => {
    if (!primaryStartup) return;
    onDecisionDraft({
      startupName: primaryStartup.companyName || "Unknown Company",
      sector: primaryStartup.industry || undefined,
      stage: primaryStartup.fundingStage || undefined,
    });
    onOpenDecisionLog();
  };

  return (
    <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
      {/* Left: Input */}
      <div className="space-y-4">
        <Card className="border border-slate-200 bg-white">
          <CardHeader className="border-b border-slate-200">
            <CardTitle className="flex items-center gap-2 text-slate-900 font-mono font-black uppercase tracking-tight">
              <Upload className="h-5 w-5 text-blue-600" />
              Document Input
            </CardTitle>
            <CardDescription className="text-slate-500 font-mono">
              Paste document text or upload a file for AI extraction
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-4 text-slate-900">
            <div>
              <Label className="text-slate-900 font-mono font-bold">Upload File</Label>
              <Input
                type="file"
                accept=".txt,.md,.pdf,.docx,.xlsx,.xls,.csv,.json"
                onChange={handleFileUpload}
                className="cursor-pointer border border-slate-200 bg-white text-slate-900 file:border-slate-300 file:bg-white file:text-slate-700"
              />
            </div>

            <div>
              <Label htmlFor="doc-text" className="text-slate-900 font-mono font-bold">Document Text</Label>
              <Textarea
                id="doc-text"
                placeholder="Paste document content, memo, or any company document here..."
                value={documentText}
                onChange={(e) => setDocumentText(e.target.value)}
                className="min-h-[300px] font-mono text-sm border border-slate-200 bg-white text-slate-900 placeholder:text-slate-400"
              />
              <p className="text-xs text-slate-500 font-mono mt-1">
                {documentText.length} characters (~{Math.ceil(documentText.length / 4)} tokens)
              </p>
            </div>

            <Button
              onClick={handleExtract}
              disabled={isLoading || !documentText.trim()}
              className="w-full bg-blue-600 text-slate-900 hover:bg-blue-600/80 font-bold border-2 border-blue-500 transition-all hover:shadow-lg hover:shadow-blue-500/20 disabled:opacity-50"
            >
              {isLoading ? (
                <>
                  <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                  Extracting with Claude...
                </>
              ) : (
                <>
                  <Sparkles className="h-4 w-4 mr-2" />
                  Extract & Detect
                </>
              )}
            </Button>
          </CardContent>
        </Card>

        {/* Cost Info */}
        <Card className="border border-slate-200 bg-white">
          <CardContent className="pt-4 text-slate-900">
            <div className="flex items-start gap-3">
              <DollarSign className="h-5 w-5 text-blue-600 mt-0.5" />
              <div className="text-sm font-mono">
                <p className="font-bold text-slate-900">Cost Transparency</p>
                <p className="text-slate-500">
                  Claude 3.5 Sonnet: ~$0.009 per 15-page deck<br/>
                  500 decks/month = ~$4.50 total
                </p>
              </div>
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Right: Results */}
      <div className="space-y-4">
        {result ? (
          <>
            <Card className="border border-slate-200 bg-white">
              <CardHeader className="border-b border-slate-200">
                <CardTitle className="flex items-center justify-between text-slate-900">
                  <span className="flex items-center gap-2 font-mono font-black uppercase tracking-tight">
                    <CheckCircle className="h-5 w-5 text-blue-600" />
                    Conversion Result
                  </span>
                  {result && (
                    <Button size="sm" variant="outline" onClick={downloadJSON} className="border border-slate-200 bg-white text-slate-900 hover:bg-blue-500/10 hover:border-blue-500 hover:text-blue-600 font-bold">
                      <Download className="h-4 w-4 mr-1" />
                      JSON
                    </Button>
                  )}
                </CardTitle>
                <CardDescription className="text-slate-500 font-mono">Detected: {result.detectedType || "unknown"}</CardDescription>
              </CardHeader>
              <CardContent className="text-slate-900">
                <div className="space-y-4">
                  {primaryStartup ? (
                    <div className="p-3 border border-slate-200 rounded-lg space-y-2 bg-white hover:border-blue-500 transition-all">
                      <div className="flex items-center justify-between">
                        <h3 className="font-mono font-black text-lg text-slate-900">{primaryStartup.companyName}</h3>
                        {primaryStartup.fundingStage && <Badge variant="outline" className="border-slate-200 text-slate-900 bg-white font-mono">{primaryStartup.fundingStage}</Badge>}
                      </div>
                      <div className="flex gap-2 flex-wrap">
                        {primaryStartup.industry && <Badge variant="outline" className="border-blue-500 text-blue-600 bg-white font-mono">{primaryStartup.industry}</Badge>}
                        {primaryStartup.geoMarkets?.length > 0 && (
                          <Badge variant="outline" className="border-slate-200 text-slate-900 bg-white font-mono">{primaryStartup.geoMarkets.join(", ")}</Badge>
                        )}
                      </div>
                    </div>
                  ) : (
                    <div className="text-sm text-slate-500 font-mono">
                      No entity detected yet. Upload a document or paste content.
                    </div>
                  )}

                  {(result.errors?.length || result.warnings?.length) && (
                    <div className="border border-slate-200 rounded-md p-3 text-xs space-y-2 bg-white">
                      {result.errors?.length ? (
                        <div>
                          <div className="font-mono font-bold text-slate-900">Errors</div>
                          <ul className="list-disc list-inside text-slate-500 font-mono">
                            {result.errors.slice(0, 3).map((err) => (
                              <li key={err}>{err}</li>
                            ))}
                          </ul>
                        </div>
                      ) : null}
                      {result.warnings?.length ? (
                        <div>
                          <div className="font-mono font-bold text-slate-900">Warnings</div>
                          <ul className="list-disc list-inside text-slate-500 font-mono">
                            {result.warnings.slice(0, 3).map((warn) => (
                              <li key={warn}>{warn}</li>
                            ))}
                          </ul>
                        </div>
                      ) : null}
                    </div>
                  )}

                  {quickLogEnabled && (
                    <Button onClick={handleQuickLog} className="w-full border border-slate-200 bg-white text-slate-900 hover:bg-blue-500/10 hover:border-blue-500 hover:text-blue-600 font-bold" variant="outline">
                      <ClipboardList className="h-4 w-4 mr-2" />
                      Open in Decision Log
                    </Button>
                  )}

                  <details className="text-sm font-mono">
                    <summary className="cursor-pointer text-slate-500 hover:text-slate-900">
                      View full JSON
                    </summary>
                    <pre className="mt-2 p-3 border border-slate-200 rounded-lg overflow-auto max-h-[300px] text-xs bg-white text-slate-900">
                      {JSON.stringify(result, null, 2)}
                    </pre>
                  </details>
                </div>
              </CardContent>
            </Card>
          </>
        ) : (
          <Card className="h-full min-h-[400px] flex items-center justify-center border border-slate-200 bg-white">
            <CardContent className="text-center text-slate-500 font-mono">
              <FileText className="h-12 w-12 mx-auto mb-4 opacity-50 text-slate-400" />
              <p className="font-bold">Paste document text and click Extract</p>
              <p className="text-sm mt-2">Results will appear here</p>
            </CardContent>
          </Card>
        )}
      </div>
    </div>
  );
}

// ============================================================================
// DECISION LOGGER TAB
// ============================================================================

function DecisionLoggerTab({
  decisions,
  setDecisions,
  activeEventId,
  actorDefault,
  draftDecision,
  onDraftConsumed,
  draftDocumentId,
  onDraftDocumentConsumed,
  documents,
  onOpenDocument,
  onOpenConverter,
  currentUserId,
}: {
  decisions: Decision[];
  setDecisions: React.Dispatch<React.SetStateAction<Decision[]>>;
  activeEventId: string | null;
  actorDefault: string;
  draftDecision: { startupName: string; sector?: string; stage?: string } | null;
  onDraftConsumed: () => void;
  draftDocumentId: string | null;
  onDraftDocumentConsumed: () => void;
  documents: Array<{ id: string; title: string | null; storage_path: string | null }>;
  onOpenDocument: (documentId: string) => void;
  onOpenConverter: () => void;
  currentUserId: string | null;
}) {
  const { toast } = useToast();
  const [showForm, setShowForm] = useState(false);
  const [isSaving, setIsSaving] = useState(false);
  const [isDeleting, setIsDeleting] = useState<string | null>(null);
  const [isUpdating, setIsUpdating] = useState<string | null>(null);
  const [deleteConfirmId, setDeleteConfirmId] = useState<string | null>(null);
  const [viewingDecision, setViewingDecision] = useState<Decision | null>(null);

  // Form state
  const [actor, setActor] = useState(() => {
    if (typeof window !== "undefined" && actorDefault) {
      const saved = localStorage.getItem("last_actor");
      return saved || actorDefault;
    }
    return actorDefault;
  });
  const [actionType, setActionType] = useState<Decision["actionType"]>("meeting");
  const [startupName, setStartupName] = useState("");
  const [sector, setSector] = useState<string>("none");
  const [stage, setStage] = useState<string>("none");
  const [geo, setGeo] = useState<string>("none");
  const [geoCustom, setGeoCustom] = useState("");
  const [confidence, setConfidence] = useState([70]);
  const [decisionReason, setDecisionReason] = useState("");
  const [decisionOutcome, setDecisionOutcome] = useState<Decision["outcome"]>("pending");
  const [attachedDocumentId, setAttachedDocumentId] = useState<string>("none");
  const [selectedDocumentId, setSelectedDocumentId] = useState<string>("all");

  const filteredDecisions = useMemo(() => {
    if (selectedDocumentId === "all") return decisions;
    return decisions.filter((d) => d.documentId === selectedDocumentId);
  }, [decisions, selectedDocumentId]);

  const stats = useMemo(() => calculateDecisionStats(filteredDecisions), [filteredDecisions]);

  useEffect(() => {
    if (actorDefault && !actor) {
      setActor(actorDefault);
    }
  }, [actorDefault, actor]);

  useEffect(() => {
    if (!draftDecision) return;
    setStartupName(draftDecision.startupName);
    setSector(draftDecision.sector || "none");
    setStage(draftDecision.stage || "none");
    setShowForm(true);
    onDraftConsumed();
  }, [draftDecision, onDraftConsumed]);

  useEffect(() => {
    if (!draftDocumentId) return;
    setAttachedDocumentId(draftDocumentId);
    setShowForm(true);
    onDraftDocumentConsumed();
  }, [draftDocumentId, onDraftDocumentConsumed]);

  const handleSaveDecision = useCallback(async () => {
    if (!activeEventId) {
      toast({ title: "No active event", description: "Please refresh and try again.", variant: "destructive" });
      return;
    }
    if (!actor.trim() || !startupName.trim()) {
      toast({ title: "Missing fields", description: "Actor and Company name are required", variant: "destructive" });
      return;
    }
    if (startupName.trim().length > 200) {
      toast({ title: "Invalid input", description: "Company name must be less than 200 characters", variant: "destructive" });
      return;
    }
    if (actor.trim().length > 100) {
      toast({ title: "Invalid input", description: "Actor name must be less than 100 characters", variant: "destructive" });
      return;
    }

    setIsSaving(true);
    const normalizedGeo = geo === "custom" ? geoCustom.trim() : geo;
    try {
      const { data, error } = await insertDecision(activeEventId, {
        actor_id: currentUserId, // Use actual user ID when available
        actor_name: actor.trim(),
        action_type: actionType,
        startup_name: startupName.trim(),
        context: {
          sector: sector !== "none" ? sector : undefined,
          stage: stage !== "none" ? stage : undefined,
          geo: normalizedGeo && normalizedGeo !== "none" ? normalizedGeo : undefined,
        },
        confidence_score: confidence[0],
        outcome: decisionOutcome || "pending",
        notes: decisionReason.trim() || null,
        document_id: attachedDocumentId === "none" ? null : attachedDocumentId,
      });

      if (error || !data) {
        toast({ 
          title: "Save failed", 
          description: error?.message || "Failed to save decision. Please try again.", 
          variant: "destructive" 
        });
        return;
      }

      // Save actor to localStorage for persistence
      if (typeof window !== "undefined") {
        localStorage.setItem("last_actor", actor.trim());
      }

      setDecisions(prev => [mapDecisionRow(data), ...prev]);
      toast({ title: "Decision logged", description: `Logged ${actionType} for ${startupName}` });

      // Reset form
      setStartupName("");
      setSector("none");
      setStage("none");
      setGeo("none");
      setGeoCustom("");
      setConfidence([70]);
      setDecisionReason("");
      setDecisionOutcome("pending");
      setAttachedDocumentId("none");
      setShowForm(false);
    } catch (err) {
      toast({ 
        title: "Unexpected error", 
        description: "An unexpected error occurred. Please try again.", 
        variant: "destructive" 
      });
    } finally {
      setIsSaving(false);
    }
  }, [
    activeEventId,
    actor,
    actionType,
    startupName,
    sector,
    stage,
    geo,
    confidence,
    decisionOutcome,
    decisionReason,
    attachedDocumentId,
    toast,
    setDecisions,
  ]);

  const handleDeleteDecision = useCallback(async (id: string) => {
    setIsDeleting(id);
    try {
      const { error } = await deleteDecision(id);
      if (error) {
        toast({ 
          title: "Delete failed", 
          description: error.message || "Failed to delete decision. Please try again.", 
          variant: "destructive" 
        });
        return;
      }
      setDecisions(prev => prev.filter(d => d.id !== id));
      toast({ title: "Deleted", description: "Decision removed" });
      setDeleteConfirmId(null);
    } catch (err) {
      toast({ 
        title: "Unexpected error", 
        description: "An unexpected error occurred. Please try again.", 
        variant: "destructive" 
      });
    } finally {
      setIsDeleting(null);
    }
  }, [toast, setDecisions]);

  const handleUpdateOutcome = useCallback(async (id: string, outcome: Decision["outcome"]) => {
    setIsUpdating(id);
    // Optimistic update
    setDecisions(prev =>
      prev.map(d => (d.id === id ? { ...d, outcome } : d))
    );
    try {
      const { error } = await updateDecision(id, { outcome });
      if (error) {
        // Revert on error
        setDecisions(prev =>
          prev.map(d => {
            if (d.id === id) {
              const originalDecision = decisions.find(od => od.id === id);
              return originalDecision || d;
            }
            return d;
          })
        );
        toast({ 
          title: "Update failed", 
          description: error.message || "Failed to update outcome. Please try again.", 
          variant: "destructive" 
        });
        return;
      }
      toast({ title: "Updated", description: "Outcome updated successfully" });
    } catch (err) {
      // Revert on error
      setDecisions(prev =>
        prev.map(d => {
          if (d.id === id) {
            const originalDecision = decisions.find(od => od.id === id);
            return originalDecision || d;
          }
          return d;
        })
      );
      toast({ 
        title: "Unexpected error", 
        description: "An unexpected error occurred. Please try again.", 
        variant: "destructive" 
      });
    } finally {
      setIsUpdating(null);
    }
  }, [toast, setDecisions, decisions]);

  const handleExport = useCallback(() => {
    const csv = exportDecisionsToCSV(decisions);
    const blob = new Blob([csv], { type: "text/csv" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `decisions-${new Date().toISOString().split("T")[0]}.csv`;
    a.click();
    URL.revokeObjectURL(url);
    toast({ title: "Exported", description: `Downloaded ${decisions.length} decisions as CSV` });
  }, [decisions, toast]);

  const documentOptions = [
    { id: "all", label: "All documents" },
    ...documents.filter((doc) => !!doc.id).map((doc) => ({ id: doc.id, label: doc.title || "Untitled document" })),
  ];

  return (
    <div className="space-y-6">
      {/* Stats Overview */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <Card className="border border-slate-200 bg-white">
          <CardContent className="pt-4 text-slate-900">
            <div className="flex items-center gap-3">
              <div className="p-2 border border-slate-200 rounded-lg bg-white">
                <ClipboardList className="h-5 w-5 text-blue-600" />
              </div>
              <div>
                <p className="text-2xl font-mono font-black">{stats.totalDecisions}</p>
                <p className="text-xs text-slate-500 font-mono uppercase tracking-wider">Total Decisions</p>
              </div>
            </div>
          </CardContent>
        </Card>

        <Card className="border border-slate-200 bg-white">
          <CardContent className="pt-4 text-slate-900">
            <div className="flex items-center gap-3">
              <div className="p-2 border-2 border-blue-500 rounded-lg bg-white">
                <TrendingUp className="h-5 w-5 text-blue-600" />
              </div>
              <div>
                <p className="text-2xl font-mono font-black">{stats.averageConfidence}%</p>
                <p className="text-xs text-slate-500 font-mono uppercase tracking-wider">Avg Confidence</p>
              </div>
            </div>
          </CardContent>
        </Card>

        <Card className="border border-slate-200 bg-white">
          <CardContent className="pt-4 text-slate-900">
            <div className="flex items-center gap-3">
              <div className="p-2 border border-slate-200 rounded-lg bg-white">
                <Target className="h-5 w-5 text-blue-600" />
              </div>
              <div>
                <p className="text-2xl font-mono font-black">{stats.byOutcome.positive || 0}</p>
                <p className="text-xs text-slate-500 font-mono uppercase tracking-wider">Positive Outcomes</p>
              </div>
            </div>
          </CardContent>
        </Card>

        <Card className="border border-slate-200 bg-white">
          <CardContent className="pt-4 text-slate-900">
            <div className="flex items-center gap-3">
              <div className="p-2 border border-slate-200 rounded-lg bg-white">
                <Users className="h-5 w-5 text-blue-600" />
              </div>
              <div>
                <p className="text-2xl font-bold">{stats.topActors.length}</p>
                <p className="text-xs text-slate-500 font-mono uppercase tracking-wider">Active Actors</p>
              </div>
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Action Buttons */}
      <div className="flex gap-2 items-center">
        <Button onClick={() => setShowForm(!showForm)} className="bg-blue-600 text-slate-900 hover:bg-blue-600/80 font-bold border-2 border-blue-500 transition-all hover:shadow-lg hover:shadow-blue-500/20">
          {showForm ? "Cancel" : "Log New Decision"}
        </Button>
        {decisions.length > 0 && (
          <Button variant="outline" onClick={handleExport} className="border border-slate-200 bg-white text-slate-900 hover:bg-blue-500/10 hover:border-blue-500 hover:text-blue-600 font-bold">
            <Download className="h-4 w-4 mr-1" />
            Export CSV
          </Button>
        )}
        <Select value={selectedDocumentId} onValueChange={setSelectedDocumentId}>
          <SelectTrigger className="w-[220px] border border-slate-200 bg-white text-slate-900 hover:bg-blue-500/10 hover:border-blue-500 font-mono font-bold">
            <SelectValue placeholder="Filter by document" className="text-slate-900" />
          </SelectTrigger>
          <SelectContent className="bg-white border border-slate-200 shadow-lg rounded-md">
            <SelectItem value="all" className="text-slate-900 font-mono hover:bg-blue-50 focus:bg-blue-50 cursor-pointer">All documents</SelectItem>
            {documents.filter((doc) => !!doc.id).map((doc) => (
              <SelectItem key={doc.id} value={doc.id} className="text-slate-900 font-mono hover:bg-blue-50 focus:bg-blue-50 cursor-pointer">
                {doc.title || doc.id}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
      </div>

      {/* New Decision Form */}
      {showForm && (
        <Card className="border border-slate-200 bg-white">
          <CardHeader className="border-b border-slate-200">
            <CardTitle className="text-slate-900 font-mono font-black uppercase tracking-tight">Log New Decision</CardTitle>
            <CardDescription className="text-slate-500 font-mono">Record a decision for pattern analysis</CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div>
                <Label className="text-slate-900 font-mono font-bold">Actor (Who made the decision) *</Label>
                <Input
                  placeholder="e.g., Partner A, John Smith"
                  value={actor}
                  onChange={(e) => setActor(e.target.value)}
                  className="border border-slate-200 bg-white text-slate-900 placeholder:text-slate-400"
                />
              </div>
              <div>
                <Label className="text-slate-900 font-mono font-bold">Action Type *</Label>
                <Select value={actionType} onValueChange={(v) => setActionType(v as Decision["actionType"])}>
                  <SelectTrigger className="border border-slate-200 bg-white text-slate-900">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent className="bg-white border border-slate-200 shadow-lg rounded-md">
                    <SelectItem value="intro" className="text-slate-900">Intro</SelectItem>
                    <SelectItem value="meeting" className="text-slate-900">Meeting</SelectItem>
                    <SelectItem value="follow_up" className="text-slate-900">Follow Up</SelectItem>
                    <SelectItem value="due_diligence" className="text-slate-900">Due Diligence</SelectItem>
                    <SelectItem value="pass" className="text-slate-900">Pass</SelectItem>
                    <SelectItem value="invest" className="text-slate-900">Invest</SelectItem>
                  </SelectContent>
                </Select>
              </div>
              <div>
                <Label className="text-slate-900 font-mono font-bold">Company Name *</Label>
                <Input
                  placeholder="e.g., Company X"
                  value={startupName}
                  onChange={(e) => setStartupName(e.target.value)}
                  className="border border-slate-200 bg-white text-slate-900 placeholder:text-slate-400"
                />
              </div>
              <div>
                <Label className="text-slate-900 font-mono font-bold">Outcome</Label>
                <Select
                  value={decisionOutcome || "pending"}
                  onValueChange={(v) => setDecisionOutcome(v as Decision["outcome"])}
                >
                  <SelectTrigger className="border border-slate-200 bg-white text-slate-900">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent className="bg-white border border-slate-200 shadow-lg rounded-md">
                    <SelectItem value="pending" className="text-slate-900">Pending</SelectItem>
                    <SelectItem value="positive" className="text-slate-900">Positive</SelectItem>
                    <SelectItem value="negative" className="text-slate-900">Negative</SelectItem>
                  </SelectContent>
                </Select>
              </div>
              <div>
                <Label className="text-slate-900 font-mono font-bold">Sector</Label>
                <Select value={sector} onValueChange={setSector}>
                  <SelectTrigger className="border border-slate-200 bg-white text-slate-900">
                    <SelectValue placeholder="Select sector" />
                  </SelectTrigger>
                  <SelectContent className="bg-white border border-slate-200 shadow-lg rounded-md">
                    <SelectItem value="none" className="text-slate-900">None</SelectItem>
                    <SelectItem value="FinTech" className="text-slate-900">FinTech</SelectItem>
                    <SelectItem value="HealthTech" className="text-slate-900">HealthTech</SelectItem>
                    <SelectItem value="SaaS" className="text-slate-900">SaaS</SelectItem>
                    <SelectItem value="AI / ML" className="text-slate-900">AI / ML</SelectItem>
                    <SelectItem value="E-commerce" className="text-slate-900">E-commerce</SelectItem>
                    <SelectItem value="EdTech" className="text-slate-900">EdTech</SelectItem>
                    <SelectItem value="PropTech" className="text-slate-900">PropTech</SelectItem>
                    <SelectItem value="AgriTech" className="text-slate-900">AgriTech</SelectItem>
                    <SelectItem value="CleanTech" className="text-slate-900">CleanTech</SelectItem>
                    <SelectItem value="Gaming" className="text-slate-900">Gaming</SelectItem>
                    <SelectItem value="Media / Content" className="text-slate-900">Media / Content</SelectItem>
                    <SelectItem value="Logistics" className="text-slate-900">Logistics</SelectItem>
                    <SelectItem value="Food & Beverage" className="text-slate-900">Food & Beverage</SelectItem>
                    <SelectItem value="Travel & Tourism" className="text-slate-900">Travel & Tourism</SelectItem>
                    <SelectItem value="HRTech" className="text-slate-900">HRTech</SelectItem>
                    <SelectItem value="LegalTech" className="text-slate-900">LegalTech</SelectItem>
                    <SelectItem value="InsurTech" className="text-slate-900">InsurTech</SelectItem>
                    <SelectItem value="Space Infrastructure" className="text-slate-900">Space Infrastructure</SelectItem>
                    <SelectItem value="Other" className="text-slate-900">Other</SelectItem>
                  </SelectContent>
                </Select>
              </div>
              <div>
                <Label className="text-slate-900 font-mono font-bold">Stage</Label>
                <Select value={stage} onValueChange={setStage}>
                  <SelectTrigger className="border border-slate-200 bg-white text-slate-900">
                    <SelectValue placeholder="Select stage" />
                  </SelectTrigger>
                  <SelectContent className="bg-white border border-slate-200 shadow-lg rounded-md">
                    <SelectItem value="none" className="text-slate-900">None</SelectItem>
                    <SelectItem value="Pre-Seed" className="text-slate-900">Pre-Seed</SelectItem>
                    <SelectItem value="Seed" className="text-slate-900">Seed</SelectItem>
                    <SelectItem value="Series A" className="text-slate-900">Series A</SelectItem>
                    <SelectItem value="Series B" className="text-slate-900">Series B</SelectItem>
                    <SelectItem value="Series C" className="text-slate-900">Series C</SelectItem>
                    <SelectItem value="Series D+" className="text-slate-900">Series D+</SelectItem>
                    <SelectItem value="Growth" className="text-slate-900">Growth</SelectItem>
                    <SelectItem value="Bridge" className="text-slate-900">Bridge</SelectItem>
                    <SelectItem value="Convertible Note" className="text-slate-900">Convertible Note</SelectItem>
                    <SelectItem value="SAFE" className="text-slate-900">SAFE</SelectItem>
                    <SelectItem value="Other" className="text-slate-900">Other</SelectItem>
                  </SelectContent>
                </Select>
              </div>
              <div>
                <Label className="text-slate-900 font-mono font-bold">Geography</Label>
                <Select value={geo} onValueChange={setGeo}>
                  <SelectTrigger className="border border-slate-200 bg-white text-slate-900">
                    <SelectValue placeholder="Select geography" />
                  </SelectTrigger>
                  <SelectContent className="bg-white border border-slate-200 shadow-lg rounded-md">
                    <SelectItem value="none" className="text-slate-900">None</SelectItem>
                    <SelectItem value="Singapore" className="text-slate-900">Singapore</SelectItem>
                    <SelectItem value="Indonesia" className="text-slate-900">Indonesia</SelectItem>
                    <SelectItem value="Malaysia" className="text-slate-900">Malaysia</SelectItem>
                    <SelectItem value="Thailand" className="text-slate-900">Thailand</SelectItem>
                    <SelectItem value="Vietnam" className="text-slate-900">Vietnam</SelectItem>
                    <SelectItem value="Philippines" className="text-slate-900">Philippines</SelectItem>
                    <SelectItem value="India" className="text-slate-900">India</SelectItem>
                    <SelectItem value="China" className="text-slate-900">China</SelectItem>
                    <SelectItem value="Hong Kong" className="text-slate-900">Hong Kong</SelectItem>
                    <SelectItem value="Taiwan" className="text-slate-900">Taiwan</SelectItem>
                    <SelectItem value="South Korea" className="text-slate-900">South Korea</SelectItem>
                    <SelectItem value="Japan" className="text-slate-900">Japan</SelectItem>
                    <SelectItem value="Australia" className="text-slate-900">Australia</SelectItem>
                    <SelectItem value="New Zealand" className="text-slate-900">New Zealand</SelectItem>
                    <SelectItem value="United States" className="text-slate-900">United States</SelectItem>
                    <SelectItem value="United Kingdom" className="text-slate-900">United Kingdom</SelectItem>
                    <SelectItem value="Europe" className="text-slate-900">Europe</SelectItem>
                    <SelectItem value="Middle East" className="text-slate-900">Middle East</SelectItem>
                    <SelectItem value="Africa" className="text-slate-900">Africa</SelectItem>
                    <SelectItem value="Latin America" className="text-slate-900">Latin America</SelectItem>
                    <SelectItem value="Other" className="text-slate-900">Other</SelectItem>
                    <SelectItem value="custom" className="text-slate-900">Add new...</SelectItem>
                  </SelectContent>
                </Select>
                {geo === "custom" && (
                  <Input
                    className="mt-2"
                    placeholder="Type a country or region"
                    value={geoCustom}
                    onChange={(e) => setGeoCustom(e.target.value)}
                  />
                )}
              </div>
            </div>

            <div>
              <Label>Confidence Score: {confidence[0]}%</Label>
              <Slider
                value={confidence}
                onValueChange={setConfidence}
                min={0}
                max={100}
                step={5}
                className="mt-2"
              />
              <p className="text-xs text-slate-500 font-mono mt-1">
                How confident are you in this decision?
              </p>
            </div>

            <div>
              <Label>Reason</Label>
              <Textarea
                placeholder="Why this decision? Market size, traction, risks..."
                value={decisionReason}
                onChange={(e) => setDecisionReason(e.target.value)}
              />
            </div>

            <div className="grid gap-3 md:grid-cols-[1fr_auto] items-end">
              <div>
                <Label className="text-slate-900 font-mono font-bold">Attach Source Document</Label>
                <Select value={attachedDocumentId} onValueChange={setAttachedDocumentId}>
                  <SelectTrigger className="border border-slate-200 bg-white text-slate-900">
                    <SelectValue placeholder="Choose a document (optional)" />
                  </SelectTrigger>
                  <SelectContent className="bg-white border border-slate-200 shadow-lg rounded-md">
                    <SelectItem value="none" className="text-slate-900">No document</SelectItem>
                    {documents.filter((doc) => !!doc.id).map((doc) => (
                      <SelectItem key={doc.id} value={doc.id} className="text-slate-900">
                        {doc.title || "Untitled document"}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>
              <Button variant="outline" onClick={onOpenConverter} className="border border-slate-200 bg-white text-slate-900 hover:bg-blue-500/10 hover:border-blue-500 hover:text-blue-600 font-bold">
                Upload new
              </Button>
            </div>

            <Button onClick={handleSaveDecision} className="w-full bg-blue-600 text-slate-900 hover:bg-blue-600/80 font-bold border-2 border-blue-500 transition-all hover:shadow-lg hover:shadow-blue-500/20 disabled:opacity-50" disabled={isSaving}>
              {isSaving ? (
                <>
                  <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                  Saving...
                </>
              ) : (
                "Save Decision"
              )}
            </Button>
          </CardContent>
        </Card>
      )}

      {/* Decision History */}
      <Card className="border border-slate-200 bg-white">
        <CardHeader className="border-b border-slate-200">
          <CardTitle className="text-slate-900 font-mono font-black uppercase tracking-tight">Decision History</CardTitle>
          <CardDescription className="text-slate-500 font-mono">
            {filteredDecisions.length} decisions shown вЂў Click outcome to update
          </CardDescription>
        </CardHeader>
        <CardContent className="text-slate-900">
          <div className="mb-4">
            <Label className="text-xs text-slate-500 font-mono font-bold">Filter by document</Label>
            <Select value={selectedDocumentId} onValueChange={setSelectedDocumentId}>
              <SelectTrigger className="mt-1 border border-slate-200 bg-white text-slate-900">
                <SelectValue />
              </SelectTrigger>
              <SelectContent className="bg-white border border-slate-200 shadow-lg rounded-md">
                {documentOptions.map((doc) => (
                  <SelectItem key={doc.id} value={doc.id} className="text-slate-900">
                    {doc.label}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>
          {filteredDecisions.length === 0 ? (
            <div className="text-center py-8 text-slate-500 font-mono">
              <ClipboardList className="h-12 w-12 mx-auto mb-4 opacity-50 text-slate-400" />
              <p className="font-bold">No decisions logged yet</p>
              <p className="text-sm">Start logging decisions to build your pattern database</p>
            </div>
          ) : (
            <div className="space-y-2 max-h-[500px] overflow-auto">
              {filteredDecisions.slice().reverse().map((d) => {
                const doc = documents.find((doc) => doc.id === d.documentId);
                return (
                <div
                  key={d.id}
                  className="flex items-center justify-between p-3 border border-slate-200 rounded-lg hover:bg-blue-600/5 hover:border-blue-500 transition-colors bg-white"
                >
                  <div className="flex items-center gap-3">
                    <Badge variant="outline" className="text-xs border-slate-200 text-slate-900 bg-white font-mono">
                      {d.actionType}
                    </Badge>
                    <div>
                      <p className="font-mono font-bold text-slate-900">{d.startupName}</p>
                      <p className="text-xs text-slate-500 font-mono">
                        {d.actor} вЂў {new Date(d.timestamp).toLocaleDateString()}
                        {d.context.sector && ` вЂў ${d.context.sector}`}
                      </p>
                      {doc && (
                        <p className="text-xs text-slate-500 font-mono">
                          Source: {doc.title || "Document"}
                        </p>
                      )}
                    </div>
                  </div>
                  <div className="flex items-center gap-2">
                    <span className="text-sm text-slate-500 font-mono">{d.confidenceScore}%</span>
                    <Button
                      size="sm"
                      variant="outline"
                      onClick={() => setViewingDecision(d)}
                      className="border border-slate-200 bg-white text-slate-900 hover:bg-blue-500/10 hover:border-blue-500 hover:text-blue-600 font-bold"
                    >
                      <Eye className="h-4 w-4 mr-1" />
                      View
                    </Button>
                    {doc?.storage_path && (
                      <Button
                        size="sm"
                        variant="outline"
                        onClick={() => onOpenDocument(doc.id)}
                        className="border border-slate-200 bg-white text-slate-900 hover:bg-blue-500/10 hover:border-blue-500 hover:text-blue-600 font-bold"
                      >
                        View source
                      </Button>
                    )}
                    <Select
                      value={d.outcome || "pending"}
                      onValueChange={(v) => handleUpdateOutcome(d.id, v as Decision["outcome"])}
                      disabled={isUpdating === d.id}
                    >
                      <SelectTrigger className="w-[100px] h-8 border border-slate-200 bg-white text-slate-900" disabled={isUpdating === d.id}>
                        <SelectValue />
                        {isUpdating === d.id && <Loader2 className="h-3 w-3 ml-1 animate-spin" />}
                      </SelectTrigger>
                      <SelectContent className="bg-white border border-slate-200 shadow-lg rounded-md">
                        <SelectItem value="pending" className="text-slate-900">
                          <span className="flex items-center gap-1">
                            <Clock className="h-3 w-3" /> Pending
                            <span className="text-xs text-slate-400 ml-1 font-mono">(No outcome yet)</span>
                          </span>
                        </SelectItem>
                        <SelectItem value="positive" className="text-slate-900">
                          <span className="flex items-center gap-1 text-blue-600">
                            <CheckCircle className="h-3 w-3" /> Positive
                            <span className="text-xs text-slate-400 ml-1 font-mono">(Success)</span>
                          </span>
                        </SelectItem>
                        <SelectItem value="negative" className="text-slate-900">
                          <span className="flex items-center gap-1 text-slate-500">
                            <AlertTriangle className="h-3 w-3" /> Negative
                            <span className="text-xs text-slate-400 ml-1 font-mono">(Passed/Declined)</span>
                          </span>
                        </SelectItem>
                      </SelectContent>
                    </Select>
                    <Button
                      size="icon"
                      variant="ghost"
                      className="h-8 w-8 text-slate-500 hover:text-slate-900 hover:bg-blue-500/10"
                      onClick={() => setDeleteConfirmId(d.id)}
                      disabled={isDeleting === d.id}
                    >
                      {isDeleting === d.id ? (
                        <Loader2 className="h-4 w-4 animate-spin" />
                      ) : (
                        <Trash2 className="h-4 w-4" />
                      )}
                    </Button>
                  </div>
                </div>
              )})}
            </div>
          )}
        </CardContent>
      </Card>

      {/* Decision View Dialog */}
      <Dialog open={!!viewingDecision} onOpenChange={(open) => !open && setViewingDecision(null)}>
        <DialogContent className="max-w-2xl max-h-[80vh] overflow-y-auto bg-white border border-slate-200 text-slate-900">
          <DialogHeader>
            <DialogTitle className="text-slate-900 font-mono font-bold">Decision Details</DialogTitle>
            <DialogDescription className="text-slate-500 font-mono">
              Full information about this decision
            </DialogDescription>
          </DialogHeader>
          {viewingDecision && (
            <div className="space-y-4">
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <Label className="text-xs text-slate-500 font-mono font-bold">Company Name</Label>
                  <p className="font-mono font-bold text-slate-900">{viewingDecision.startupName}</p>
                </div>
                <div>
                  <Label className="text-xs text-slate-500 font-mono font-bold">Actor</Label>
                  <p className="font-mono font-bold text-slate-900">{viewingDecision.actor}</p>
                </div>
                <div>
                  <Label className="text-xs text-slate-500 font-mono font-bold">Action Type</Label>
                  <Badge variant="outline" className="border-slate-200 text-slate-900 bg-white font-mono">{viewingDecision.actionType}</Badge>
                </div>
                <div>
                  <Label className="text-xs text-slate-500 font-mono font-bold">Confidence Score</Label>
                  <p className="font-mono font-bold text-slate-900">{viewingDecision.confidenceScore}%</p>
                </div>
                <div>
                  <Label className="text-xs text-slate-500 font-mono font-bold">Outcome</Label>
                  <div className="flex items-center gap-2">
                    <Badge 
                      variant="outline"
                      className={
                        viewingDecision.outcome === "positive" ? "border-blue-500 text-blue-600 bg-white font-mono" :
                        viewingDecision.outcome === "negative" ? "border-slate-200/50 text-slate-400 bg-white font-mono" :
                        "border-slate-200 text-slate-900 bg-white font-mono"
                      }
                    >
                      {viewingDecision.outcome || "Pending"}
                    </Badge>
                    <TooltipProvider>
                      <Tooltip>
                        <TooltipTrigger asChild>
                          <span className="text-xs text-slate-500 cursor-help font-mono">в„№пёЏ</span>
                        </TooltipTrigger>
                        <TooltipContent className="max-w-xs bg-white border border-slate-200 text-slate-900">
                          <p className="text-xs font-mono">
                            <strong>Pending:</strong> Decision is still in progress, no outcome yet<br/>
                            <strong>Positive:</strong> Decision led to a positive result (e.g., investment, partnership)<br/>
                            <strong>Negative:</strong> Decision led to a negative result (e.g., passed, declined)
                          </p>
                        </TooltipContent>
                      </Tooltip>
                    </TooltipProvider>
                  </div>
                </div>
                <div>
                  <Label className="text-xs text-slate-500 font-mono font-bold">Date</Label>
                  <p className="font-mono font-bold text-slate-900">{new Date(viewingDecision.timestamp).toLocaleString()}</p>
                </div>
              </div>

              {viewingDecision.context && (
                <div className="space-y-2">
                  <Label className="text-xs text-slate-500 font-mono font-bold">Context</Label>
                  <div className="grid grid-cols-3 gap-2">
                    {viewingDecision.context.sector && viewingDecision.context.sector !== "none" && (
                      <div>
                        <span className="text-xs text-slate-500 font-mono">Sector:</span>
                        <p className="font-mono font-bold text-slate-900">{viewingDecision.context.sector}</p>
                      </div>
                    )}
                    {viewingDecision.context.stage && viewingDecision.context.stage !== "none" && (
                      <div>
                        <span className="text-xs text-slate-500 font-mono">Stage:</span>
                        <p className="font-mono font-bold text-slate-900">{viewingDecision.context.stage}</p>
                      </div>
                    )}
                    {viewingDecision.context.geo && viewingDecision.context.geo !== "none" && (
                      <div>
                        <span className="text-xs text-slate-500 font-mono">Geography:</span>
                        <p className="font-mono font-bold text-slate-900">{viewingDecision.context.geo}</p>
                      </div>
                    )}
                  </div>
                </div>
              )}

              <div>
                <Label className="text-xs text-slate-500 font-mono font-bold">Reason / Notes</Label>
                {viewingDecision.notes ? (
                  <p className="mt-1 text-sm whitespace-pre-wrap text-slate-900 font-mono">{viewingDecision.notes}</p>
                ) : (
                  <p className="mt-1 text-sm text-slate-400 italic font-mono">No reason or notes provided</p>
                )}
              </div>

              {viewingDecision.documentId && (
                <div>
                  <Label className="text-xs text-slate-500 font-mono font-bold">Attached Document</Label>
                  {(() => {
                    const attachedDoc = documents.find((doc) => doc.id === viewingDecision.documentId);
                    return attachedDoc ? (
                      <div className="mt-1 flex items-center gap-2">
                        <p className="text-sm font-medium">{attachedDoc.title || "Untitled document"}</p>
                        {attachedDoc.storage_path && (
                          <Button
                            size="sm"
                            variant="outline"
                            onClick={() => {
                              onOpenDocument(attachedDoc.id);
                              setViewingDecision(null);
                            }}
                          >
                            Open
                          </Button>
                        )}
                      </div>
                    ) : (
                      <p className="text-sm text-slate-500 font-mono">Document not found</p>
                    );
                  })()}
                </div>
              )}

              <div className="flex justify-end gap-2 pt-4 border-t border-slate-300">
                <Button variant="outline" onClick={() => setViewingDecision(null)} className="border border-slate-200 bg-white text-slate-900 hover:bg-blue-500/10 hover:border-blue-500 hover:text-blue-600 font-bold">
                  Close
                </Button>
              </div>
            </div>
          )}
        </DialogContent>
      </Dialog>

      {/* Top Actors */}
      {stats.topActors.length > 0 && (
        <Card className="border border-slate-200 bg-white">
          <CardHeader className="border-b border-slate-200">
            <CardTitle className="text-slate-900 font-mono font-black uppercase tracking-tight">Top Decision Makers</CardTitle>
          </CardHeader>
          <CardContent className="text-slate-900">
            <div className="space-y-2">
              {stats.topActors.map((a, i) => (
                <div key={a.actor} className="flex items-center justify-between p-2 border border-slate-200 rounded hover:border-blue-500 hover:bg-blue-600/5 transition-all bg-white">
                  <div className="flex items-center gap-2">
                    <span className="text-sm font-mono font-bold text-slate-500">#{i + 1}</span>
                    <span className="font-mono font-bold text-slate-900">{a.actor}</span>
                  </div>
                  <div className="flex items-center gap-3 text-sm font-mono">
                    <span className="text-slate-500">{a.count} decisions</span>
                    <Badge variant="outline" className={a.winRate > 50 ? "border-blue-500 text-blue-600 bg-white font-mono" : "border-slate-200 text-slate-900 bg-white font-mono"}>
                      {a.winRate}% win rate
                    </Badge>
                  </div>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      )}

      {/* Delete Confirmation Dialog */}
      <AlertDialog open={!!deleteConfirmId} onOpenChange={(open) => !open && setDeleteConfirmId(null)}>
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>Delete Decision?</AlertDialogTitle>
            <AlertDialogDescription>
              This action cannot be undone. This will permanently delete the decision record.
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel>Cancel</AlertDialogCancel>
            <AlertDialogAction
              onClick={() => deleteConfirmId && handleDeleteDecision(deleteConfirmId)}
              className="bg-destructive text-destructive-foreground hover:bg-destructive/90"
            >
              Delete
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
    </div>
  );
}

// ============================================================================
// SOURCES TAB
// ============================================================================

function SourcesTab({
  sources,
  documents,
  sourceFolders,
  onCreateSource,
  onCreateFolder,
  onDeleteFolderAndContents,
  onFolderCategoryUpdated,
  onFoldersCategoriesSaved,
  onDriveSyncConfigChanged,
  onDeleteSource,
  getGoogleAccessToken,
  onAutoLogDecision,
  onDocumentSaved,
  activeEventId,
  ensureActiveEventId,
  currentUserId,
  indexDocumentEmbeddings,
  onRefreshCompanyCards,
  initialDriveSyncConfig,
  onSyncCategoriesFromDrive,
  onSourceFoldersRefetch,
}: {
  sources: SourceRecord[];
  documents: Array<{
    id: string;
    title: string | null;
    storage_path: string | null;
    uploader_name?: string | null;
    uploader_email?: string | null;
    folder_id?: string | null;
  }>;
  sourceFolders: SourceFolder[];
  onCreateSource: (
    payload: {
      title: string | null;
      source_type: SourceRecord["source_type"];
      external_url: string | null;
      storage_path?: string | null;
      tags: string[] | null;
      notes: string | null;
      status: SourceRecord["status"];
    },
    eventIdOverride?: string | null
  ) => Promise<void>;
  onCreateFolder: (name: string, category?: string) => Promise<SourceFolder | null>;
  onDeleteFolderAndContents?: (folderId: string) => Promise<{ docCount: number } | { error: string }>;
  onFolderCategoryUpdated?: (folderId: string, category: string) => Promise<void>;
  onSyncCategoriesFromDrive?: () => Promise<void>;
  onFoldersCategoriesSaved?: (updates: Array<{ id: string; category: string }>) => Promise<void>;
  onDriveSyncConfigChanged?: (folders: Array<{ id: string; name: string; category?: string }>) => void;
  onDeleteSource: (sourceId: string) => Promise<void>;
  getGoogleAccessToken: () => Promise<string | null>;
  onAutoLogDecision: (input: {
    draft: { startupName: string; sector?: string; stage?: string };
    conversion: AIConversionResponse;
    sourceType: "upload" | "paste" | "api";
    fileName: string | null;
    file: File | null;
    rawContent?: string | null;
    eventIdOverride?: string | null;
  }) => Promise<void>;
  onDocumentSaved: (doc: { id: string; title: string | null; storage_path: string | null; folder_id?: string | null }) => void;
  activeEventId: string | null;
  ensureActiveEventId: () => Promise<string | null>;
  currentUserId: string | null;
  indexDocumentEmbeddings: (documentId: string, rawContent?: string | null, docTitle?: string | null, pdfBase64?: string | null) => Promise<void>;
  onRefreshCompanyCards?: () => Promise<void>;
  initialDriveSyncConfig?: {
    folderId: string;
    folderName: string;
    folders: Array<{ id: string; name: string; category?: string }>;
    lastSyncAt: string | null;
  } | null;
  onSourceFoldersRefetch?: () => Promise<void>;
}) {
  /** Root folder types for Google Drive sync (each connected root can be tagged as one of these). */
  const DRIVE_ROOT_CATEGORIES = [
    "Projects",
    "BD",
    "Sourcing",
    "Partners",
    "Mentors / Corporates",
  ] as const;
  type DriveFolderEntry = { id: string; name: string; category?: string };

  const { toast } = useToast();
  const [clickUpListId, setClickUpListId] = useState(() => {
    if (typeof window === "undefined") return "";
    return localStorage.getItem("clickup_list_id") || "";
  });
  const [clickUpTeamId, setClickUpTeamId] = useState("");
  const [clickUpLists, setClickUpLists] = useState<Array<{ id: string; name: string }>>([]);
  const [selectedListId, setSelectedListId] = useState("");
  const [isLoadingLists, setIsLoadingLists] = useState(false);
  const [driveUrl, setDriveUrl] = useState("");
  const [folderToDelete, setFolderToDelete] = useState<{ id: string; name: string } | null>(null);
  const [isDeletingFolder, setIsDeletingFolder] = useState(false);
  const [isImportingClickUp, setIsImportingClickUp] = useState(false);
  const [isImportingDrive, setIsImportingDrive] = useState(false);
  const [isUploadingLocal, setIsUploadingLocal] = useState(false);
  const [uploadProgress, setUploadProgress] = useState<{ current: number; total: number; currentFile: string; results: Array<{ name: string; updated: number; conflicts: number; created: boolean }> } | null>(null);
  const [batchReviewData, setBatchReviewData] = useState<Array<{
    companyName: string;
    entityId: string;
    fields: Array<{ field: string; value: any; confidence: number; approved: boolean }>;
  }> | null>(null);
  const [autoExtract, setAutoExtract] = useState(true);
  const [selectedFolderId, setSelectedFolderId] = useState<string>("none");
  const [newFolderName, setNewFolderName] = useState("");
  const [newFolderCategory, setNewFolderCategory] = useState<string>("Projects");
  const [isCreatingFolder, setIsCreatingFolder] = useState(false);
  const [expandedFolderCategory, setExpandedFolderCategory] = useState<string | null>(null);
  const [categoryPickerOpen, setCategoryPickerOpen] = useState(false);
  const [categoryPickerFolders, setCategoryPickerFolders] = useState<Array<{ id: string; name: string; category: string }>>([]);
  const [pendingFolderDocs, setPendingFolderDocs] = useState<Array<{ id: string; title: string | null }>>([]);
  const [isDeletingCards, setIsDeletingCards] = useState(false);
  const [folderAssignmentIds, setFolderAssignmentIds] = useState<string[]>([]);
  const [isFolderDialogOpen, setIsFolderDialogOpen] = useState(false);
  const [isAssigningFolders, setIsAssigningFolders] = useState(false);
  const [folderDialogCategory, setFolderDialogCategory] = useState<string>("Projects");
  const [folderDialogNewName, setFolderDialogNewName] = useState("");

  // в”Ђв”Ђ Google Drive Folder Sync state в”Ђв”Ђ
  const [connectedDriveFolderId, setConnectedDriveFolderId] = useState<string | null>(null);
  const [connectedDriveFolderName, setConnectedDriveFolderName] = useState<string | null>(null);
  // Support multiple connected folders (each can have a root-folder type / category)
  const [connectedDriveFolders, setConnectedDriveFolders] = useState<DriveFolderEntry[]>([]);
  const [isSyncingDrive, setIsSyncingDrive] = useState(false);
  const [driveSyncProgress, setDriveSyncProgress] = useState<{ phase: string; current: number; total: number; currentItem: string } | null>(null);
  const [driveSyncResults, setDriveSyncResults] = useState<Array<{ companyName: string; newFiles: number; updatedFiles: number; skippedFiles: number }>>([]);
  const [lastDriveSyncAt, setLastDriveSyncAt] = useState<string | null>(null);
  const [isSyncingCategoriesFromDrive, setIsSyncingCategoriesFromDrive] = useState(false);
  const [selectedFolderIds, setSelectedFolderIds] = useState<Set<string>>(new Set());
  const [driveConnectCooldownUntil, setDriveConnectCooldownUntil] = useState(0);
  // Auto-sync interval (15 minutes)
  const autoSyncIntervalRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const isSyncingDriveRef = useRef(false);
  const SYNC_INTERVAL_MS = 15 * 60 * 1000; // 15 minutes

  // в”Ђв”Ђ Gmail Sync state в”Ђв”Ђ
  const [isGmailConnected, setIsGmailConnected] = useState(false);
  const [isSyncingGmail, setIsSyncingGmail] = useState(false);
  const [gmailSyncProgress, setGmailSyncProgress] = useState<{ current: number; total: number; currentItem: string } | null>(null);
  const [lastGmailSyncAt, setLastGmailSyncAt] = useState<string | null>(null);
  const [gmailQuery, setGmailQuery] = useState("");
  const [gmailMaxPerSync, setGmailMaxPerSync] = useState(50);
  const [gmailIncludeAttachments, setGmailIncludeAttachments] = useState(true);
  const [gmailSyncResults, setGmailSyncResults] = useState<{ synced: number; skipped: number; errors: number } | null>(null);

  const MAX_IMPORT_CHARS = 24000;
  const MAX_PDF_PAGES = 6;
  const canImport = Boolean(activeEventId);
  const googleApiKey = import.meta.env.VITE_GOOGLE_API_KEY as string | undefined;
  const googleClientId = import.meta.env.VITE_GOOGLE_CLIENT_ID as string | undefined;

  useEffect(() => {
    isSyncingDriveRef.current = isSyncingDrive;
  }, [isSyncingDrive]);

  useEffect(() => {
    if (driveConnectCooldownUntil <= 0) return;
    const remaining = driveConnectCooldownUntil - Date.now();
    if (remaining <= 0) {
      setDriveConnectCooldownUntil(0);
      return;
    }
    const t = window.setTimeout(() => setDriveConnectCooldownUntil(0), remaining);
    return () => clearTimeout(t);
  }, [driveConnectCooldownUntil]);

  const [driveConnectCooldownTick, setDriveConnectCooldownTick] = useState(0);
  useEffect(() => {
    if (driveConnectCooldownUntil <= 0 || Date.now() >= driveConnectCooldownUntil) return;
    const id = setInterval(() => setDriveConnectCooldownTick(Date.now()), 1000);
    return () => clearInterval(id);
  }, [driveConnectCooldownUntil]);

  const isDriveConnectOnCooldown = driveConnectCooldownUntil > 0 && Date.now() < driveConnectCooldownUntil;
  const driveConnectCooldownSeconds = isDriveConnectOnCooldown ? Math.ceil((driveConnectCooldownUntil - Date.now()) / 1000) : 0;
  
  // Debug: log env vars (remove in production)
  useEffect(() => {
    if (typeof window !== 'undefined') {
    }
  }, [googleApiKey, googleClientId]);

  // в”Ђв”Ђ Restore Drive folders from parent-loaded config (so folder list survives reload) в”Ђв”Ђ
  useEffect(() => {
    if (!initialDriveSyncConfig) return;
    setConnectedDriveFolderId(initialDriveSyncConfig.folderId);
    setConnectedDriveFolderName(initialDriveSyncConfig.folderName);
    // Only update array state if contents actually changed (avoids reference churn)
    setConnectedDriveFolders(prev => {
      const next = initialDriveSyncConfig.folders.map((f): DriveFolderEntry => ({
        id: f.id,
        name: f.name,
        category: f.category ?? "Projects",
      }));
      if (prev.length === next.length && prev.every((f, i) => f.id === next[i]?.id && f.category === next[i]?.category)) return prev;
      return next;
    });
    setLastDriveSyncAt(initialDriveSyncConfig.lastSyncAt);
  }, [initialDriveSyncConfig]);

  // в”Ђв”Ђ Load existing Drive sync configuration when activeEventId changes (fallback / tab switch) в”Ђв”Ђ
  useEffect(() => {
    if (!activeEventId || initialDriveSyncConfig != null) return;
    (async () => {
      try {
        const { data, error } = await supabase
          .from("sync_configurations")
          .select("config, last_sync_at")
          .eq("event_id", activeEventId)
          .eq("source_type", "google_drive")
          .limit(1);
        if (error) {
          console.warn("[DriveSync] sync_configurations not available:", error.message);
          return;
        }
        const row = data?.[0] as { config?: { google_drive_folder_id?: string; google_drive_folder_name?: string; folders?: Array<{ id: string; name: string; category?: string }> }; last_sync_at?: string } | undefined;
        if (row?.config?.google_drive_folder_id) {
          setConnectedDriveFolderId(row.config.google_drive_folder_id);
          setConnectedDriveFolderName(row.config.google_drive_folder_name || "Project folder");
          setLastDriveSyncAt(row.last_sync_at || null);
          const folders = row.config.folders;
          if (folders && Array.isArray(folders) && folders.length > 0) {
            setConnectedDriveFolders(folders.map((f): DriveFolderEntry => ({
              id: f.id,
              name: f.name,
              category: f.category ?? "Projects",
            })));
          } else {
            setConnectedDriveFolders([{
              id: row.config.google_drive_folder_id,
              name: row.config.google_drive_folder_name || "Project folder",
              category: "Projects",
            }]);
          }
        }
      } catch (err) {
        console.warn("[DriveSync] Failed to load sync config:", err);
      }
    })();
  }, [activeEventId, initialDriveSyncConfig]);

  // в”Ђв”Ђ Load existing Gmail sync configuration when activeEventId changes в”Ђв”Ђ
  useEffect(() => {
    if (!activeEventId) return;
    (async () => {
      try {
        const { data, error } = await supabase
          .from("sync_configurations")
          .select("config, last_sync_at")
          .eq("event_id", activeEventId)
          .eq("source_type", "gmail")
          .limit(1);
        if (error) { console.warn("[GmailSync] sync_configurations not available:", error.message); return; }
        const row = data?.[0] as { config?: { gmail_query?: string; max_emails_per_sync?: number; include_attachments?: boolean }; last_sync_at?: string } | undefined;
        if (row) {
          setIsGmailConnected(true);
          setLastGmailSyncAt(row.last_sync_at || null);
          if (row.config?.gmail_query) setGmailQuery(row.config.gmail_query);
          if (row.config?.max_emails_per_sync) setGmailMaxPerSync(row.config.max_emails_per_sync);
          if (row.config?.include_attachments !== undefined) setGmailIncludeAttachments(row.config.include_attachments);
        }
      } catch (err) {
        console.warn("[GmailSync] Failed to load sync config:", err);
      }
    })();
  }, [activeEventId]);

  useEffect(() => {
    if (typeof window === "undefined") return;
    const trimmed = clickUpListId.trim();
    if (trimmed) {
      localStorage.setItem("clickup_list_id", trimmed);
    }
  }, [clickUpListId]);

  const openFolderAssignmentDialog = useCallback(
    (docs: Array<{ id: string; title: string | null }>) => {
      if (!docs.length) return;
      setPendingFolderDocs(docs);
      const defaults = selectedFolderId !== "none" ? [selectedFolderId] : [];
      setFolderAssignmentIds(defaults);
      setFolderDialogCategory("Projects");
      setFolderDialogNewName("");
      setIsFolderDialogOpen(true);
    },
    [selectedFolderId]
  );

  const assignFoldersToDocuments = useCallback(async () => {
    if (!pendingFolderDocs.length || folderAssignmentIds.length === 0) {
      setIsFolderDialogOpen(false);
      setPendingFolderDocs([]);
      return;
    }
    setIsAssigningFolders(true);
    try {
      const docIds = pendingFolderDocs.map((d) => d.id);
      const rows = docIds.flatMap((docId) =>
        folderAssignmentIds.map((folderId) => ({
          document_id: docId,
          folder_id: folderId,
          created_by: currentUserId || null,
        }))
      );

      // Replace existing links for these documents
      await supabase.from("document_folder_links").delete().in("document_id", docIds);
      const { error: insertError } = await supabase.from("document_folder_links").insert(rows);
      if (insertError) {
        throw insertError;
      }

      // Keep a primary folder for backward compatibility
      const primaryFolderId = folderAssignmentIds[0] || null;
      await supabase.from("documents").update({ folder_id: primaryFolderId }).in("id", docIds);

      toast({
        title: "Folders assigned",
        description: `Assigned ${folderAssignmentIds.length} folder${folderAssignmentIds.length > 1 ? "s" : ""} to ${docIds.length} document${docIds.length > 1 ? "s" : ""}.`,
      });
    } catch (err) {
      toast({
        title: "Folder assignment failed",
        description: err instanceof Error ? err.message : "Could not assign folders to documents.",
        variant: "destructive",
      });
    } finally {
      setIsAssigningFolders(false);
      setIsFolderDialogOpen(false);
      setPendingFolderDocs([]);
    }
  }, [currentUserId, folderAssignmentIds, pendingFolderDocs, toast]);

  const toggleFolderAssignment = useCallback((folderId: string, checked: boolean) => {
    setFolderAssignmentIds((prev) => {
      if (checked) {
        return prev.includes(folderId) ? prev : [...prev, folderId];
      }
      return prev.filter((id) => id !== folderId);
    });
  }, []);

  // Ref to avoid "Cannot access syncGoogleDriveFolder before initialization" when Sources tab mounts
  const syncGoogleDriveFolderRef = useRef<((foldersOverride?: DriveFolderEntry[]) => Promise<void>) | null>(null);

  // в”Ђв”Ђ Connect a Google Drive root portfolio folder via Picker в”Ђв”Ђ
  const connectDrivePortfolioFolder = useCallback(async () => {
    if (!googleApiKey || !googleClientId) {
      toast({
        title: "Google Picker not configured",
        description: "Set VITE_GOOGLE_API_KEY and VITE_GOOGLE_CLIENT_ID to use Drive picker.",
        variant: "destructive",
      });
      return;
    }
    let accessToken = await getGoogleAccessToken();
    if (!accessToken) accessToken = await getGoogleAccessToken(true);
    if (!accessToken) {
      try {
        toast({
          title: "Connect Google Drive",
          description: "Redirecting to Google to grant Drive access…",
        });
        await triggerGoogleOAuthForDrive();
      } catch (e) {
        const msg = e instanceof Error ? e.message : String(e);
        setDriveConnectCooldownUntil(Date.now() + 65000);
        toast({
          title: "Could not connect Google Drive",
          description: msg,
          variant: "destructive",
        });
      }
      return;
    }
    try {
      await loadGooglePicker();
      // Folder-only view
      const folderView = new window.google.picker.DocsView()
        .setIncludeFolders(true)
        .setSelectFolderEnabled(true)
        .setMimeTypes("application/vnd.google-apps.folder")
        .setMode(window.google.picker.DocsViewMode.LIST);

      const picker = new window.google.picker.PickerBuilder()
        .setDeveloperKey(googleApiKey)
        .setOAuthToken(accessToken)
        .setAppId(googleClientId.split("-")[0])
        .addView(folderView)
        .enableFeature(window.google.picker.Feature.SUPPORT_DRIVES)
        .setTitle("Select your project root folder")
        .setCallback(async (data: any) => {
          if (data.action === window.google.picker.Action.PICKED) {
            const folder = data.docs?.[0];
            if (!folder?.id) return;
            const folderId = folder.id;
            const folderName = folder.name || "Project folder";
            
            // Add to folders list (avoid duplicates); new folder defaults to Projects
            const updatedFolders: DriveFolderEntry[] = [
              ...connectedDriveFolders.filter(f => f.id !== folderId),
              { id: folderId, name: folderName, category: "Projects" },
            ];
            setConnectedDriveFolders(updatedFolders);
            // Keep primary folder for backward compat
            if (!connectedDriveFolderId) {
              setConnectedDriveFolderId(folderId);
              setConnectedDriveFolderName(folderName);
            }

            // Persist to sync_configurations
            const eventId = activeEventId || (await ensureActiveEventId());
            if (!eventId) return;
            const { data: profile } = await supabase.auth.getUser();
            const userId = profile?.user?.id || null;
            // Look up org
            let orgId: string | null = null;
            if (userId) {
              const { data: up } = await supabase
                .from("user_profiles")
                .select("organization_id")
                .eq("id", userId)
                .limit(1);
              orgId = up?.[0]?.organization_id || null;
            }
            if (!orgId) {
              toast({ title: "Org not found", description: "Cannot save sync config without an organization.", variant: "destructive" });
              return;
            }
            const { error: upsertError } = await supabase.from("sync_configurations").upsert(
              {
                organization_id: orgId,
                event_id: eventId,
                source_type: "google_drive",
                config: {
                  google_drive_folder_id: updatedFolders[0].id,
                  google_drive_folder_name: updatedFolders[0].name,
                  folders: updatedFolders.map((f) => ({ id: f.id, name: f.name, category: f.category ?? "Projects" })),
                },
                sync_frequency: "hourly",
                is_active: true,
                created_by: userId,
              },
              { onConflict: "organization_id,event_id,source_type" }
            );
            if (upsertError) {
              console.error("[DriveSync] Failed to persist folder config:", upsertError);
              toast({
                title: "Folder not saved",
                description: "Connected folder could not be saved. It may disappear after reload. Try again.",
                variant: "destructive",
              });
              return;
            }

            toast({ title: "Folder connected", description: `Connected "${folderName}". Starting sync and extraction...` });
            // Run sync immediately so we extract and sync right away (state may not have updated yet)
            syncGoogleDriveFolderRef.current?.(updatedFolders);
          }
        })
        .build();
      picker.setVisible(true);
    } catch (err) {
      toast({ title: "Picker error", description: err instanceof Error ? err.message : "Failed to open picker.", variant: "destructive" });
    }
  }, [activeEventId, connectedDriveFolderId, connectedDriveFolders, ensureActiveEventId, getGoogleAccessToken, googleApiKey, googleClientId, toast]);

  // в”Ђв”Ђ Core folder sync logic в”Ђв”Ђ
  const syncGoogleDriveFolder = useCallback(async (foldersOverride?: Array<{ id: string; name: string }>) => {
    // Use override when e.g. we just added a folder and state may not have updated yet
    const foldersToSync = (foldersOverride && foldersOverride.length > 0)
      ? foldersOverride
      : connectedDriveFolders.length > 0
        ? connectedDriveFolders
        : connectedDriveFolderId
          ? [{ id: connectedDriveFolderId, name: connectedDriveFolderName || "Project folder" }]
          : [];
    if (foldersToSync.length === 0) {
      toast({ title: "No folder connected", description: "Connect a Google Drive folder first.", variant: "destructive" });
      return;
    }
    let accessToken = await getGoogleAccessToken();
    if (!accessToken) accessToken = await getGoogleAccessToken(true);
    if (!accessToken) {
      try {
        toast({ title: "Connect Google Drive", description: "Redirecting to Google to grant Drive access…" });
        await triggerGoogleOAuthForDrive();
      } catch (e) {
        const msg = e instanceof Error ? e.message : String(e);
        setDriveConnectCooldownUntil(Date.now() + 65000);
        toast({ title: "Could not connect Google Drive", description: msg, variant: "destructive" });
      }
      return;
    }
    const eventId = activeEventId || (await ensureActiveEventId());
    if (!eventId) {
      toast({ title: "No active event", description: "Create or activate an event before syncing.", variant: "destructive" });
      return;
    }

    setIsSyncingDrive(true);
    setDriveSyncProgress({ phase: `Discovering folders (${foldersToSync.length} root${foldersToSync.length > 1 ? "s" : ""})...`, current: 0, total: 0, currentItem: "" });
    setDriveSyncResults([]);
    const results: Array<{ companyName: string; newFiles: number; updatedFiles: number; skippedFiles: number }> = [];
    const newlyCreatedFolderIds: Array<{ id: string; name: string }> = [];

    const isDriveAuthError = (error: unknown): boolean => {
      const msg = error instanceof Error ? error.message : String(error);
      return /(invalid credentials|unauthenticated|autherror|http\s*401|code['"]?\s*:\s*401)/i.test(msg);
    };

    const refreshDriveAccessToken = async (): Promise<string> => {
      const refreshed = await getGoogleAccessToken(true);
      if (!refreshed) {
        throw new Error("Google Drive access expired during sync. Please sign in again and retry.");
      }
      accessToken = refreshed;
      return refreshed;
    };

    const withDriveAuthRetry = async <T,>(operation: () => Promise<T>): Promise<T> => {
      try {
        return await operation();
      } catch (error) {
        if (!isDriveAuthError(error)) throw error;
        console.warn("[DriveSync] Google token appears expired. Refreshing token and retrying once...");
        await refreshDriveAccessToken();
        return await operation();
      }
    };

    try {
      // 0. Wake up the Render ingestion service (free-tier cold-start can take 30-60s)
      await warmUpIngestion();

      // 1. Recursively list ALL descendant folders (sub, sub-sub, ...) up to MAX_FOLDER_DEPTH
      const MAX_FOLDER_DEPTH = 10;
      const visitedIds = new Set<string>();
      const allDescendantFolders: Array<{ id: string; name: string; path: string }> = [];

      const collectDescendants = async (
        parentId: string,
        parentPath: string,
        depth: number
      ): Promise<void> => {
        if (depth <= 0 || visitedIds.has(parentId)) return;
        visitedIds.add(parentId);
        const children = await withDriveAuthRetry(() => listDriveFolders(accessToken, parentId));
        for (const child of children) {
          if (visitedIds.has(child.id)) continue;
          const path = parentPath ? `${parentPath} / ${child.name}` : child.name;
          allDescendantFolders.push({ id: child.id, name: child.name, path });
          await sleep(300); // throttle: avoid overwhelming Render
          await collectDescendants(child.id, path, depth - 1);
        }
      };

      setDriveSyncProgress({ phase: "Discovering folders (recursive)...", current: 0, total: 0, currentItem: "" });
      for (const rootFolder of foldersToSync) {
        const rootPath = rootFolder.name;
        allDescendantFolders.push({ id: rootFolder.id, name: rootFolder.name, path: rootPath });
        await collectDescendants(rootFolder.id, rootPath, MAX_FOLDER_DEPTH - 1);
      }

      // Resolve category for a path using:
      // 1. Root folder's explicit category (if multiple roots with different categories)
      // 2. Smart keyword matching on folder name segments
      // 3. Default to Projects
      const classifyFolderName = (name: string): string | null => {
        const lower = name.toLowerCase().trim();
        // BD / Business Development
        if (/\b(bd|business\s*dev|business\s*development|partnerships?|biz\s*dev)\b/.test(lower)) return "BD";
        // Mentors / Corporates
        if (/\b(mentor|mentors?|corporate|corporates?|organizations?|advisors?|advisory)\b/.test(lower)) return "Mentors / Corporates";
        // Sourcing / Deals
        if (/\b(sourcing|deal\s*flow|pipeline|inbound|deal|deals|prospects?|market\s*research|research)\b/.test(lower)) return "Sourcing";
        // Partners / Stakeholders
        if (/\b(partners?|stakeholders?|syndicate|lps?|limited\s*partners?|co-invest|co.?invest)\b/.test(lower)) return "Partners";
        // Projects / Companies
        if (/\b(portfolio|companies|startups?|ventures?|investments?|due\s*diligence|dd|diligence|projects?)\b/.test(lower)) return "Projects";
        return null;
      };

      const normPath = (s: string) => (s || "").trim().toLowerCase().replace(/\s*\([^)]*\)\s*/g, " ").replace(/\s+/g, " ").trim();
      const normPathSegment = (s: string) => normPath(s).replace(/\s*-\s*.*$/, "").trim();
      const getCategoryForPath = (path: string): string => {
        // First: check if any root folder has an explicit category (same normalized + first-segment logic as backfill)
        for (const root of foldersToSync) {
          const rootName = root.name?.trim() || "";
          if (!rootName) continue;
          const pathNorm = normPath(path);
          const rootNorm = normPath(rootName);
          const rootNormSeg = normPathSegment(rootName);
          const firstSegNorm = normPathSegment(path.split(/\s*\/\s*/)[0] || path);
          const belongsToRoot = pathNorm === rootNorm || pathNorm.startsWith(rootNorm + " / ") || firstSegNorm === rootNormSeg;
          if (belongsToRoot) {
            const rootCategory = (root as { id: string; name: string; category?: string }).category;
            if (rootCategory != null) return rootCategory;
          }
        }

        // Second: smart classify based on folder path segments
        const segments = path.split(" / ");
        for (const segment of segments) {
          const classified = classifyFolderName(segment);
          if (classified) return classified;
        }

        // Default
        return "Projects";
      };

      // 2. Keep only folders that contain at least one file (so we sync and extract from them)
      const subFolders: Array<{ id: string; name: string; category: string }> = [];
      for (let i = 0; i < allDescendantFolders.length; i++) {
        const folder = allDescendantFolders[i];
        setDriveSyncProgress({ phase: "Checking for documents...", current: i + 1, total: allDescendantFolders.length, currentItem: folder.path });
        const files = await withDriveAuthRetry(() => listDriveFiles(accessToken, folder.id));
        if (files.length > 0) {
          const category = getCategoryForPath(folder.path);
          subFolders.push({ id: folder.id, name: folder.path, category }); // use path as company name for uniqueness
        }
        if (i < allDescendantFolders.length - 1) await sleep(300); // throttle
      }
      if (subFolders.length === 0) {
        toast({ title: "No documents found", description: "No folders with documents found in the connected folder(s)." });
        setIsSyncingDrive(false);
        setDriveSyncProgress(null);
        return;
      }

      setDriveSyncProgress({ phase: "Syncing companies...", current: 0, total: subFolders.length, currentItem: "" });

      // Helper: check if a folder name looks like a real company name
      const isLikelyCompanyName = (name: string): boolean => {
        const words = name.trim().split(/\s+/);
        if (words.length > 4) return false; // company names are short
        const generic = /^(sourcing|intern|interns|team|notes|docs|documents|shared|misc|general|archive|old|new|temp|draft|test|admin|meeting|meetings|portfolio|companies|partners?|deals?|research|diligence|dd)$/i;
        if (words.some((w) => generic.test(w))) return false;
        return true;
      };

      for (let fi = 0; fi < subFolders.length; fi++) {
        const companyFolder = subFolders[fi];
        const companyName = companyFolder.name;
        setDriveSyncProgress({ phase: "Syncing companies...", current: fi + 1, total: subFolders.length, currentItem: companyName });

        let newFiles = 0;
        let updatedFiles = 0;
        let skippedFiles = 0;

        try {
          // 2. List files in this company folder
          const files = await withDriveAuthRetry(() => listDriveFiles(accessToken, companyFolder.id));

          // 3. Ensure a source_folder exists for this company
          let platformFolderId: string | null = null;
          const normalizedCompanyName = companyName.trim();
          const folderCategory = (companyFolder as { id: string; name: string; category?: string }).category ?? "Projects";
          const existingFolder = sourceFolders.find(
            (f) => f.name.toLowerCase() === normalizedCompanyName.toLowerCase()
          );
          if (existingFolder) {
            platformFolderId = existingFolder.id;
            if (onFolderCategoryUpdated && (existingFolder.category || "Projects") !== folderCategory) {
              await onFolderCategoryUpdated(existingFolder.id, folderCategory);
            }
          } else {
            const created = await onCreateFolder(normalizedCompanyName, folderCategory);
            if (created) {
              platformFolderId = created.id;
              newlyCreatedFolderIds.push({ id: created.id, name: normalizedCompanyName });
            }
          }

          // 4. For each file: check if already synced, download if new/updated
          for (const file of files) {
            try {
              // Check existing document with same google_drive_file_id
              const { data: existingDocs } = await supabase
                .from("documents")
                .select("id, google_drive_modified_at")
                .eq("event_id", eventId)
                .eq("google_drive_file_id", file.id)
                .order("created_at", { ascending: false })
                .limit(1);
              const existingDoc = existingDocs?.[0] ?? null;

              const driveModifiedAt = file.modifiedTime || null;
              if (existingDoc && driveModifiedAt) {
                const existingModified = existingDoc.google_drive_modified_at;
                if (existingModified && new Date(existingModified).getTime() >= new Date(driveModifiedAt).getTime()) {
                  skippedFiles++;
                  continue; // Already synced and up to date
                }
              }

              // Skip unsupported binary/media types before calling the ingestion API.
              // Images are supported (backend uses Claude Vision to describe them so the model can read pictures).
              const unsupportedMimePrefixes = ["video/", "audio/", "font/"];
              const unsupportedMimeExact = new Set([
                "application/zip", "application/x-zip-compressed",
                "application/x-rar-compressed", "application/x-7z-compressed",
                "application/gzip", "application/x-tar",
                "application/octet-stream",
                "application/x-msdownload",
                "application/vnd.android.package-archive",
              ]);
              const unsupportedExtensions = new Set([
                ".mp4", ".mov", ".avi", ".mkv", ".wmv", ".flv", ".webm",
                ".mp3", ".wav", ".aac", ".ogg", ".flac", ".wma",
                ".zip", ".rar", ".7z", ".tar", ".gz",
                ".exe", ".dll", ".dmg", ".apk",
              ]);
              const mime = file.mimeType || "";
              const ext = (file.name || "").toLowerCase().match(/\.[^.]+$/)?.[0] || "";
              if (
                unsupportedMimePrefixes.some(p => mime.startsWith(p)) ||
                unsupportedMimeExact.has(mime) ||
                unsupportedExtensions.has(ext)
              ) {
                console.info(`[DriveSync] Skipping unsupported file: ${file.name} (${mime || ext})`);
                skippedFiles++;
                continue;
              }

              // 5. Download file content
              const downloaded = await withDriveAuthRetry(() =>
                downloadDriveFile(accessToken, file.id, file.mimeType, file.name)
              );
              if (!downloaded.content || downloaded.content.startsWith("[Empty")) {
                console.warn(`[DriveSync] Skipping empty file: ${file.name}`);
                skippedFiles++;
                continue;
              }

              const isUpdate = !!existingDoc;
              const fileTitle = file.name?.replace(/\.[^/.]+$/, "") || downloaded.title;

              // 6. Detect meeting notes
              const isMeetingNotes = /meeting|minutes|notes|standup|sync|recap|weekly|1-on-1|check.?in/i.test(file.name);

              // 7. Insert document row (always new row to preserve history)
              const { data: docRow, error: docError } = await supabase
                .from("documents")
                .insert({
                  event_id: eventId,
                  title: fileTitle,
                  source_type: "google_drive" as any,
                  file_name: file.name,
                  google_drive_file_id: file.id,
                  google_drive_modified_at: driveModifiedAt,
                  created_by: currentUserId,
                  folder_id: platformFolderId,
                  raw_content: downloaded.raw_content?.substring(0, 50000) || null,
                })
                .select("id")
                .single();

              if (docError || !docRow) {
                console.error(`[DriveSync] Failed to insert doc for ${file.name}:`, docError);
                continue;
              }

              // Link to folder
              if (platformFolderId) {
                await supabase.from("document_folder_links").insert({
                  document_id: docRow.id,
                  folder_id: platformFolderId,
                }).then(() => {});
              }

              // Notify parent of new doc
              onDocumentSaved({ id: docRow.id, title: fileTitle, storage_path: null, folder_id: platformFolderId || undefined });

              // 8. Run embedding + entity extraction pipeline (same as local upload)
              // This also calls extractEntities which may create company entities with proper names
              await indexDocumentEmbeddings(docRow.id, downloaded.raw_content, fileTitle, null);

              // 9. Find or create company entity
              // FIRST: check if extractEntities already linked this doc to an entity
              try {
                let companyEntityId: string | null = null;
                const { data: linkedDoc } = await supabase
                  .from("documents")
                  .select("company_entity_id")
                  .eq("id", docRow.id)
                  .single();
                if (linkedDoc?.company_entity_id) {
                  companyEntityId = linkedDoc.company_entity_id;
                }

                // If not linked yet, try folder name or file title
                if (!companyEntityId) {
                  const cleanFileTitle = (t: string) => {
                    let s = t.replace(/^copy\s+of\s+/i, "").trim();
                    s = s.replace(/\s*(due\s*diligence|dd|diligence|deck|pitch|memo|presentation|report|summary|overview|brochure|tearsheet).*$/i, "").trim();
                    s = s.replace(/\s*[-вЂ“вЂ”]\s*.*$/, "").replace(/\s*\(.*\)\s*$/, "").replace(/\.\w+$/, "").trim();
                    return s || t;
                  };
                  const entityName = isLikelyCompanyName(companyName) ? companyName : cleanFileTitle(fileTitle);
                  const normalizedName = normalizeCompanyNameForMatch(entityName);
                  const { data: entityArr } = await supabase
                    .from("kg_entities")
                    .select("id")
                    .eq("event_id", eventId)
                    .eq("normalized_name", normalizedName)
                    .eq("entity_type", "company")
                    .limit(1);
                  companyEntityId = entityArr?.[0]?.id || null;

                  // Still not found? Try matching by source_document_id (extractEntities may have created one)
                  if (!companyEntityId) {
                    const { data: entByDoc } = await supabase
                      .from("kg_entities")
                      .select("id, name")
                      .eq("event_id", eventId)
                      .eq("source_document_id", docRow.id)
                      .eq("entity_type", "company")
                      .limit(1);
                    if (entByDoc?.[0]) {
                      companyEntityId = entByDoc[0].id;
                    }
                  }

                  if (!companyEntityId) {
                    const { data: newEntity, error: createErr } = await supabase
                      .from("kg_entities")
                      .insert({
                        event_id: eventId,
                        entity_type: "company",
                        name: entityName,
                        normalized_name: normalizedName,
                        properties: {
                          auto_created: true,
                          source: "folder_based",
                          folder_name: companyName,
                          first_seen_document: docRow.id,
                          bio: "",
                          funding_stage: "",
                          amount_seeking: "",
                          valuation: "",
                          arr: "",
                          burn_rate: "",
                          runway_months: "",
                          problem: "",
                          solution: "",
                          tam: "",
                          competitive_edge: "",
                          founders: "[]",
                          ai_rationale: "",
                          website: "",
                          logo_url: "",
                        },
                        source_document_id: docRow.id,
                        confidence: 0.8,
                        created_by: currentUserId || null,
                      })
                      .select("id")
                      .single();
                    if (!createErr && newEntity) {
                      companyEntityId = newEntity.id;
                    }
                  }

                  // Link document to entity
                  if (companyEntityId) {
                    await supabase
                      .from("documents")
                      .update({ company_entity_id: companyEntityId })
                      .eq("id", docRow.id);
                  }
                }

                if (companyEntityId) {
                  // Get current entity name for logging / rename comparison
                  const { data: currentEntityRow } = await supabase
                    .from("kg_entities")
                    .select("name")
                    .eq("id", companyEntityId)
                    .single();
                  const currentEntityName = currentEntityRow?.name || fileTitle;

                  const existing = await getEntityProperties(companyEntityId);
                  const extraction = await extractCompanyProperties({
                    rawContent: downloaded.raw_content,
                    documentTitle: fileTitle,
                    existingProperties: existing?.properties || {},
                  });


                  if (Object.keys(extraction.properties).length > 0) {
                    // If AI identified a better company_name, rename the entity
                    const aiCompanyName = (extraction.properties.company_name || "").trim();
                    if (aiCompanyName && aiCompanyName.length >= 2) {
                      const aiNorm = normalizeCompanyNameForMatch(aiCompanyName);
                      const curNorm = normalizeCompanyNameForMatch(currentEntityName);
                      // Only rename if AI name differs meaningfully and isn't the doc title
                      if (aiNorm !== curNorm && !aiNorm.includes("copy of")) {
                        // Check if an entity with AI name already exists
                        const { data: existingByAiName } = await supabase
                          .from("kg_entities")
                          .select("id")
                          .eq("event_id", eventId)
                          .eq("normalized_name", aiNorm)
                          .eq("entity_type", "company")
                          .limit(1);
                        if (existingByAiName && existingByAiName.length > 0) {
                          // Merge: re-link document to the existing entity with the AI name
                          const targetId = existingByAiName[0].id;
                          await supabase.from("documents").update({ company_entity_id: targetId }).eq("id", docRow.id);
                          companyEntityId = targetId;
                        } else {
                          // Rename entity to AI-detected name
                          await supabase
                            .from("kg_entities")
                            .update({ name: aiCompanyName, normalized_name: aiNorm })
                            .eq("id", companyEntityId);
                        }
                      }
                      // Remove company_name from properties (it's stored as entity name, not a card field)
                      delete extraction.properties.company_name;
                      delete extraction.confidence.company_name;
                    }

                    const mergeResult = await mergeCompanyCardFromExtraction(
                      companyEntityId,
                      extraction.properties,
                      extraction.confidence,
                      docRow.id,
                      { isMeetingNotes },
                    );
                  } else {
                    console.warn(`[DriveSync] No properties extracted for "${currentEntityName}" from file "${file.name}"`);
                  }
                } else {
                  console.warn(`[DriveSync] No entity found/created for file "${file.name}" вЂ” skipping property extraction`);
                }
              } catch (extractErr) {
                console.warn(`[DriveSync] Property extraction for ${file.name} failed (non-fatal):`, extractErr);
              }

              if (isUpdate) {
                updatedFiles++;
              } else {
                newFiles++;
              }
            } catch (fileErr) {
              const msg = fileErr instanceof Error ? fileErr.message : String(fileErr);
              if (msg.toLowerCase().includes("unsupported file type")) {
                console.warn(`[DriveSync] Skipping unsupported file type for ${file.name}: ${msg}`);
                skippedFiles++;
                continue;
              }
              console.error(`[DriveSync] Error processing file ${file.name}:`, fileErr);
            }
          }
        } catch (folderErr) {
          if (isDriveAuthError(folderErr)) {
            throw new Error("Google Drive token expired during sync. Please reconnect Google Drive and run sync again.");
          }
          console.error(`[DriveSync] Error processing folder ${companyName}:`, folderErr);
        }

        results.push({ companyName, newFiles, updatedFiles, skippedFiles });
        setDriveSyncResults([...results]);
        if (fi < subFolders.length - 1) await sleep(500); // throttle between company folders
      }

      // 9. Update sync_configurations last_sync_at
      const now = new Date().toISOString();
      await supabase
        .from("sync_configurations")
        .update({ last_sync_at: now, last_sync_status: "success", last_sync_error: null })
        .eq("event_id", eventId)
        .eq("source_type", "google_drive");
      setLastDriveSyncAt(now);

      // Refresh company cards
      if (onRefreshCompanyCards) {
        await onRefreshCompanyCards();
      }

      // Delete redundant cards (keep one per core company name)
      try {
        const cleanup = await deleteRedundantCards(eventId);
        if (cleanup.deleted > 0 && onRefreshCompanyCards) {
          await onRefreshCompanyCards();
          toast({ title: "Cards cleaned", description: cleanup.message, variant: "default" });
        }
      } catch (e) {
        console.warn("[DriveSync] Redundant cards cleanup failed:", e);
      }

      const totalNew = results.reduce((s, r) => s + r.newFiles, 0);
      const totalUpdated = results.reduce((s, r) => s + r.updatedFiles, 0);
      const totalSkipped = results.reduce((s, r) => s + r.skippedFiles, 0);
      toast({
        title: "Drive sync complete",
        description: `${results.length} companies вЂ” ${totalNew} new, ${totalUpdated} updated, ${totalSkipped} unchanged.`,
      });

      if (onSourceFoldersRefetch) {
        await onSourceFoldersRefetch();
      }

      // Show category picker for any newly created folders
      if (newlyCreatedFolderIds.length > 0) {
        setCategoryPickerFolders(newlyCreatedFolderIds.map((f) => ({ ...f, category: "Projects" })));
        setCategoryPickerOpen(true);
      }
    } catch (err) {
      console.error("[DriveSync] Sync failed:", err);
      // Record error
      if (activeEventId) {
        await supabase
          .from("sync_configurations")
          .update({ last_sync_status: "error", last_sync_error: err instanceof Error ? err.message : "Unknown error" })
          .eq("event_id", activeEventId)
          .eq("source_type", "google_drive");
      }
      toast({
        title: "Drive sync failed",
        description: err instanceof Error ? err.message : "Could not sync Google Drive folder.",
        variant: "destructive",
      });
    } finally {
      setIsSyncingDrive(false);
      setDriveSyncProgress(null);
    }
  }, [activeEventId, connectedDriveFolderId, connectedDriveFolderName, connectedDriveFolders, currentUserId, ensureActiveEventId, getGoogleAccessToken, indexDocumentEmbeddings, onCreateFolder, onDocumentSaved, onFolderCategoryUpdated, onRefreshCompanyCards, onSourceFoldersRefetch, sourceFolders, toast]);

  // Keep ref updated so connectDrivePortfolioFolder can call sync without TDZ
  useEffect(() => {
    syncGoogleDriveFolderRef.current = syncGoogleDriveFolder;
    return () => {
      syncGoogleDriveFolderRef.current = null;
    };
  }, [syncGoogleDriveFolder]);

  // Derive a stable boolean so interval/auto-sync effects don't re-fire when
  // connectedDriveFolders gets a new array reference with the same contents.
  const hasDriveFolders = connectedDriveFolders.length > 0 || !!connectedDriveFolderId;

  // в”Ђв”Ђ Auto-sync on login: fire once per session when config + token are available в”Ђв”Ђ
  const autoSyncFiredRef = useRef(false);
  useEffect(() => {
    if (autoSyncFiredRef.current) return;
    if (!hasDriveFolders || !activeEventId || isSyncingDrive) return;
    (async () => {
      const token = await getGoogleAccessToken();
      if (!token) return;
      // Only auto-sync if last sync was > 15 min ago (or never synced)
      if (lastDriveSyncAt) {
        const elapsed = Date.now() - new Date(lastDriveSyncAt).getTime();
        if (elapsed < SYNC_INTERVAL_MS) {
          autoSyncFiredRef.current = true;
          return;
        }
      }
      autoSyncFiredRef.current = true;
      syncGoogleDriveFolder();
    })();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [hasDriveFolders, activeEventId, isSyncingDrive, getGoogleAccessToken, lastDriveSyncAt, syncGoogleDriveFolder]);

  // в”Ђв”Ђ Auto-sync interval: sync every 15 minutes while page is open в”Ђв”Ђ
  useEffect(() => {
    if (!hasDriveFolders || !activeEventId) {
      // Clear interval if no folders connected
      if (autoSyncIntervalRef.current) {
        clearInterval(autoSyncIntervalRef.current);
        autoSyncIntervalRef.current = null;
      }
      return;
    }
    // Set up interval
    if (!autoSyncIntervalRef.current) {
      autoSyncIntervalRef.current = setInterval(() => {
        if (isSyncingDriveRef.current) {
          return;
        }
        syncGoogleDriveFolderRef.current?.();
      }, SYNC_INTERVAL_MS);
    }
    return () => {
      if (autoSyncIntervalRef.current) {
        clearInterval(autoSyncIntervalRef.current);
        autoSyncIntervalRef.current = null;
      }
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [hasDriveFolders, activeEventId]);

  // в”Ђв”Ђ Connect Gmail: save sync config to Supabase в”Ђв”Ђ
  const connectGmail = useCallback(async (query?: string) => {
    const eventId = activeEventId || (await ensureActiveEventId());
    if (!eventId) { toast({ title: "No active event", variant: "destructive" }); return; }
    let token = await getGoogleAccessToken();
    if (!token) token = await getGoogleAccessToken(true);
    if (!token) { toast({ title: "Google access needed", description: "Sign in again to grant Gmail read access.", variant: "destructive" }); return; }

    // Verify token works for Gmail
    try {
      await gmailListMessages(token, { maxResults: 1 });
    } catch (err) {
      toast({ title: "Gmail access denied", description: "Please sign out and sign in again вЂ” Gmail read permission is required.", variant: "destructive" });
      return;
    }

    const profile_ = (await supabase.auth.getUser()).data.user;
    const orgId = (await supabase.from("user_profiles").select("organization_id").eq("id", profile_?.id ?? "").single()).data?.organization_id;

    const configPayload = {
      gmail_query: query || gmailQuery || "",
      max_emails_per_sync: gmailMaxPerSync,
      include_attachments: gmailIncludeAttachments,
    };

    const { error } = await supabase.from("sync_configurations").upsert({
      organization_id: orgId,
      event_id: eventId,
      source_type: "gmail",
      config: configPayload,
      sync_frequency: "daily",
      is_active: true,
      created_by: profile_?.id ?? null,
    }, { onConflict: "organization_id,event_id,source_type" });

    if (error) {
      console.error("[GmailSync] Failed to save config:", error);
      toast({ title: "Failed to save Gmail config", description: error.message, variant: "destructive" });
      return;
    }
    setIsGmailConnected(true);
    toast({ title: "Gmail connected", description: "Gmail inbox sync is now configured. Click Sync Now to start." });
  }, [activeEventId, ensureActiveEventId, getGoogleAccessToken, gmailIncludeAttachments, gmailMaxPerSync, gmailQuery, toast]);

  // в”Ђв”Ђ Sync Gmail Inbox: fetch, dedupe, ingest, store в”Ђв”Ђ
  const syncGmailInbox = useCallback(async () => {
    const eventId = activeEventId || (await ensureActiveEventId());
    if (!eventId) { toast({ title: "No active event", variant: "destructive" }); return; }
    let token = await getGoogleAccessToken();
    if (!token) token = await getGoogleAccessToken(true);
    if (!token) { toast({ title: "Google access needed", description: "Sign in again to grant Gmail access.", variant: "destructive" }); return; }

    setIsSyncingGmail(true);
    setGmailSyncResults(null);
    setGmailSyncProgress({ current: 0, total: 0, currentItem: "Fetching message listвЂ¦" });

    let synced = 0, skipped = 0, errors = 0;

    try {
      // Build Gmail query вЂ” append "after:" for incremental sync
      let q = gmailQuery || "";
      if (lastGmailSyncAt) {
        const epoch = Math.floor(new Date(lastGmailSyncAt).getTime() / 1000);
        q = q ? `${q} after:${epoch}` : `after:${epoch}`;
      }

      const listResult = await gmailListMessages(token, { query: q || undefined, maxResults: gmailMaxPerSync });
      const messageIds = listResult.messages;

      if (messageIds.length === 0) {
        toast({ title: "No new emails", description: "No new emails match your filter criteria." });
        setGmailSyncProgress(null);
        setIsSyncingGmail(false);
        return;
      }

      setGmailSyncProgress({ current: 0, total: messageIds.length, currentItem: "Processing emailsвЂ¦" });

      // Fetch existing gmail_message_ids for dedup
      const { data: existingDocs } = await supabase
        .from("documents")
        .select("gmail_message_id")
        .eq("event_id", eventId)
        .eq("source_type", "gmail")
        .not("gmail_message_id", "is", null);
      const existingIds = new Set((existingDocs ?? []).map((d: any) => d.gmail_message_id));

      for (let i = 0; i < messageIds.length; i++) {
        const msgSnippet = messageIds[i];
        setGmailSyncProgress({ current: i + 1, total: messageIds.length, currentItem: `Email ${i + 1}/${messageIds.length}` });

        if (existingIds.has(msgSnippet.id)) { skipped++; continue; }

        try {
          const ingested: GmailIngestResult = await gmailIngestMessage(token, msgSnippet.id, gmailIncludeAttachments);

          const docTitle = ingested.email_subject || ingested.title || "(no subject)";
          const { data: docRow, error: docErr } = await supabase.from("documents").insert({
            event_id: eventId,
            title: docTitle,
            source_type: "gmail",
            file_name: null,
            raw_content: ingested.content,
            detected_type: "email",
            gmail_message_id: msgSnippet.id,
            gmail_thread_id: ingested.gmail_thread_id,
            gmail_labels: ingested.gmail_labels,
            email_from: ingested.email_from,
            email_to: ingested.email_to,
            email_cc: ingested.email_cc,
            email_subject: ingested.email_subject,
            email_sent_at: ingested.email_date,
            email_has_attachments: ingested.has_attachments,
            created_by: currentUserId,
          }).select("id, title, storage_path, folder_id").single();

          if (docErr || !docRow) { console.error("[GmailSync] Insert doc error:", docErr); errors++; continue; }

          onDocumentSaved({ id: docRow.id, title: docRow.title, storage_path: docRow.storage_path, folder_id: (docRow as any).folder_id });

          // Insert attachment records + download processable ones
          if (ingested.has_attachments && ingested.attachments?.length) {
            const PROCESSABLE_MIMES = new Set([
              "application/pdf",
              "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
              "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
              "application/vnd.openxmlformats-officedocument.presentationml.presentation",
              "text/plain", "text/csv",
            ]);
            const MAX_ATTACHMENT_SIZE = 10 * 1024 * 1024; // 10 MB

            for (const att of ingested.attachments) {
              const { error: attErr } = await supabase.from("email_attachments").insert({
                document_id: docRow.id,
                gmail_attachment_id: att.id,
                filename: att.filename,
                mime_type: att.mimeType,
                size_bytes: att.size,
              });
              if (attErr) { console.warn("[GmailSync] Attachment insert error:", attErr.message); continue; }

              // Download and create a separate document for processable attachments
              if (gmailIncludeAttachments && PROCESSABLE_MIMES.has(att.mimeType) && att.size <= MAX_ATTACHMENT_SIZE) {
                try {
                  const downloaded = await gmailDownloadAttachment(token, msgSnippet.id, att.id);
                  if (downloaded.data) {
                    const isPdf = att.mimeType === "application/pdf";
                    const attTitle = `[Attachment] ${att.filename} вЂ” from "${docTitle}"`;
                    const { data: attDoc } = await supabase.from("documents").insert({
                      event_id: eventId,
                      title: attTitle,
                      source_type: "gmail",
                      file_name: att.filename,
                      detected_type: isPdf ? "pdf" : "document",
                      gmail_message_id: `${msgSnippet.id}_att_${att.id}`,
                      gmail_thread_id: ingested.gmail_thread_id,
                      email_from: ingested.email_from,
                      email_subject: ingested.email_subject,
                      email_has_attachments: false,
                      created_by: currentUserId,
                    }).select("id").single();

                    if (attDoc) {
                      // For PDFs, pass base64 directly for visual extraction; for text, decode
                      if (isPdf) {
                        indexDocumentEmbeddings(attDoc.id, att.filename, attTitle, downloaded.data);
                      } else {
                        try {
                          const textContent = atob(downloaded.data.replace(/-/g, "+").replace(/_/g, "/"));
                          await supabase.from("documents").update({ raw_content: textContent.slice(0, 50000) }).eq("id", attDoc.id);
                          indexDocumentEmbeddings(attDoc.id, textContent, attTitle);
                        } catch { /* non-text binary вЂ” skip embedding */ }
                      }
                      // Mark attachment as processed
                      await supabase.from("email_attachments")
                        .update({ processed: true })
                        .eq("document_id", docRow.id)
                        .eq("gmail_attachment_id", att.id);
                    }
                  }
                } catch (attDownloadErr) {
                  console.warn(`[GmailSync] Attachment download failed for ${att.filename}:`, attDownloadErr);
                }
              }
            }
          }

          // Upsert email_threads row
          if (ingested.gmail_thread_id) {
            await supabase.from("email_threads").upsert({
              event_id: eventId,
              gmail_thread_id: ingested.gmail_thread_id,
              subject: ingested.email_subject || null,
              participants: [...new Set([ingested.email_from, ...ingested.email_to, ...ingested.email_cc].filter(Boolean))],
              last_message_at: ingested.email_date || new Date().toISOString(),
            }, { onConflict: "event_id,gmail_thread_id" }).then(({ error: thErr }) => {
              if (thErr) console.warn("[GmailSync] Thread upsert error:", thErr.message);
            });
          }

          // Embed document content for RAG
          indexDocumentEmbeddings(docRow.id, ingested.content, docTitle);
          synced++;
        } catch (msgErr) {
          console.error(`[GmailSync] Error processing message ${msgSnippet.id}:`, msgErr);
          errors++;
        }
      }

      // Update sync timestamp
      const now = new Date().toISOString();
      await supabase.from("sync_configurations")
        .update({ last_sync_at: now, last_sync_status: "success", last_sync_error: null, updated_at: now })
        .eq("event_id", eventId)
        .eq("source_type", "gmail");
      setLastGmailSyncAt(now);

      setGmailSyncResults({ synced, skipped, errors });
      toast({ title: "Gmail sync complete", description: `${synced} new emails synced, ${skipped} already existed, ${errors} errors.` });
    } catch (err) {
      console.error("[GmailSync] Sync failed:", err);
      toast({ title: "Gmail sync failed", description: err instanceof Error ? err.message : "Unknown error", variant: "destructive" });
      if (activeEventId) {
        await supabase.from("sync_configurations")
          .update({ last_sync_status: "error", last_sync_error: err instanceof Error ? err.message : "Unknown" })
          .eq("event_id", activeEventId)
          .eq("source_type", "gmail");
      }
    } finally {
      setGmailSyncProgress(null);
      setIsSyncingGmail(false);
    }
  }, [activeEventId, currentUserId, ensureActiveEventId, getGoogleAccessToken, gmailIncludeAttachments, gmailMaxPerSync, gmailQuery, indexDocumentEmbeddings, lastGmailSyncAt, onDocumentSaved, toast]);

  // в”Ђв”Ђ Gmail auto-sync on login: fire once per session when connected в”Ђв”Ђ
  const gmailAutoSyncFiredRef = useRef(false);
  const isSyncingGmailRef = useRef(false);
  useEffect(() => { isSyncingGmailRef.current = isSyncingGmail; }, [isSyncingGmail]);
  const syncGmailInboxRef = useRef(syncGmailInbox);
  useEffect(() => { syncGmailInboxRef.current = syncGmailInbox; }, [syncGmailInbox]);

  useEffect(() => {
    if (gmailAutoSyncFiredRef.current) return;
    if (!isGmailConnected || !activeEventId || isSyncingGmail) return;
    (async () => {
      const token = await getGoogleAccessToken();
      if (!token) return;
      if (lastGmailSyncAt) {
        const elapsed = Date.now() - new Date(lastGmailSyncAt).getTime();
        if (elapsed < SYNC_INTERVAL_MS) {
          gmailAutoSyncFiredRef.current = true;
          return;
        }
      }
      gmailAutoSyncFiredRef.current = true;
      syncGmailInbox();
    })();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [isGmailConnected, activeEventId, isSyncingGmail, getGoogleAccessToken, lastGmailSyncAt, syncGmailInbox]);

  // в”Ђв”Ђ Gmail auto-sync interval (every 15 min) в”Ђв”Ђ
  const gmailAutoSyncIntervalRef = useRef<ReturnType<typeof setInterval> | null>(null);
  useEffect(() => {
    if (!isGmailConnected || !activeEventId) {
      if (gmailAutoSyncIntervalRef.current) { clearInterval(gmailAutoSyncIntervalRef.current); gmailAutoSyncIntervalRef.current = null; }
      return;
    }
    if (!gmailAutoSyncIntervalRef.current) {
      gmailAutoSyncIntervalRef.current = setInterval(() => {
        if (isSyncingGmailRef.current) return;
        syncGmailInboxRef.current?.();
      }, SYNC_INTERVAL_MS);
    }
    return () => {
      if (gmailAutoSyncIntervalRef.current) { clearInterval(gmailAutoSyncIntervalRef.current); gmailAutoSyncIntervalRef.current = null; }
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [isGmailConnected, activeEventId]);

  const handleImportClickUp = useCallback(async () => {
    const eventId = activeEventId || (await ensureActiveEventId());
    if (!eventId) {
      toast({
        title: "No active event",
        description: "Create or activate an event before importing.",
        variant: "destructive",
      });
      return;
    }
    if (!clickUpListId.trim()) {
      toast({
        title: "Missing list ID",
        description: "Enter a ClickUp list ID to import tasks.",
        variant: "destructive",
      });
      return;
    }
    setIsImportingClickUp(true);
    try {
      const response = await ingestClickUpList(clickUpListId.trim(), true);
      let created = 0;
      for (const task of response.tasks || []) {
        const tagList = ["clickup", task.status || ""]
          .concat(task.assignees || [])
          .map((t) => t.trim())
          .filter(Boolean);
        await onCreateSource({
          title: task.name || "ClickUp task",
          source_type: "syndicate",
          external_url: task.url || null,
          tags: tagList.length ? tagList : null,
        notes: null,
          status: "active",
        }, eventId);
        created += 1;
      }
      toast({ title: "Import complete", description: `Imported ${created} ClickUp tasks.` });
      setClickUpListId("");
    } catch (error) {
      toast({
        title: "ClickUp import failed",
        description: error instanceof Error ? error.message : "Could not import ClickUp tasks.",
        variant: "destructive",
      });
    } finally {
      setIsImportingClickUp(false);
    }
  }, [activeEventId, clickUpListId, ensureActiveEventId, onCreateSource, toast]);

  const handleLoadClickUpLists = useCallback(async () => {
    if (!clickUpTeamId.trim()) {
      toast({
        title: "Missing team ID",
        description: "Enter a ClickUp team ID to load lists.",
        variant: "destructive",
      });
      return;
    }
    setIsLoadingLists(true);
    try {
      const response = await getClickUpLists(clickUpTeamId.trim());
      setClickUpLists(response.lists || []);
      if (response.lists?.length) {
        setSelectedListId(response.lists[0].id);
        setClickUpListId(response.lists[0].id);
      }
    } catch (error) {
      toast({
        title: "Load lists failed",
        description: error instanceof Error ? error.message : "Could not load ClickUp lists.",
        variant: "destructive",
      });
    } finally {
      setIsLoadingLists(false);
    }
  }, [clickUpTeamId, toast]);

  const readFileText = (file: File) =>
    new Promise<string>((resolve, reject) => {
      const reader = new FileReader();
      reader.onerror = () => reject(new Error("Could not read file"));
      reader.onload = () => resolve(typeof reader.result === "string" ? reader.result : "");
      reader.readAsText(file);
    });

  const isCSVFile = (file: File) => {
    const name = file.name.toLowerCase();
    return file.type === "text/csv" || name.endsWith(".csv");
  };

  const isTextFile = (file: File) => {
    const name = file.name.toLowerCase();
    return (
      file.type.startsWith("text/") ||
      name.endsWith(".txt") ||
      name.endsWith(".md") ||
      name.endsWith(".csv") ||
      name.endsWith(".json")
    );
  };

  const extractPdfTextClientSide = async (file: File) => {
    const loadPdfJs = () =>
      new Promise<any>((resolve, reject) => {
        if ((window as any).pdfjsLib) {
          resolve((window as any).pdfjsLib);
          return;
        }
        const script = document.createElement("script");
        script.src = "https://cdnjs.cloudflare.com/ajax/libs/pdf.js/3.11.174/pdf.min.js";
        script.async = true;
        script.onload = () => resolve((window as any).pdfjsLib);
        script.onerror = () => reject(new Error("Failed to load PDF.js"));
        document.head.appendChild(script);
      });

    const pdfjs: any = await loadPdfJs();
    pdfjs.GlobalWorkerOptions.workerSrc =
      "https://cdnjs.cloudflare.com/ajax/libs/pdf.js/3.11.174/pdf.worker.min.js";
    // Avoid worker/CORS issues in production by disabling the worker
    pdfjs.disableWorker = true;
    const buffer = await file.arrayBuffer();
    const loadingTask = pdfjs.getDocument({ data: buffer });
    const pdf = await loadingTask.promise;
    const pageLimit = Math.min(pdf.numPages, MAX_PDF_PAGES);
    let text = "";
    for (let i = 1; i <= pageLimit; i += 1) {
      const page = await pdf.getPage(i);
      const content = await page.getTextContent();
      const strings = (content.items as Array<{ str?: string }>)
        .map((item) => item.str || "")
        .join(" ");
      text += `\n--- Page ${i} ---\n${strings}`;
    }
    return text.trim();
  };

  const handleLocalUpload = useCallback(
    async (e: React.ChangeEvent<HTMLInputElement>) => {
      const files = Array.from(e.target.files ?? []);
      if (!files.length) return;

      const eventId = activeEventId || (await ensureActiveEventId());
      if (!eventId) {
        toast({
          title: "No active event",
          description: "Create or activate an event before uploading.",
          variant: "destructive",
        });
        return;
      }

      setIsUploadingLocal(true);
      setUploadProgress({ current: 0, total: files.length, currentFile: "", results: [] });
      try {
        let successCount = 0;
        const uploadedDocs: Array<{ id: string; title: string | null }> = [];
        const batchResults: Array<{ name: string; updated: number; conflicts: number; created: boolean }> = [];

        for (let fileIdx = 0; fileIdx < files.length; fileIdx++) {
          const file = files[fileIdx];
          setUploadProgress((prev) => prev ? { ...prev, current: fileIdx + 1, currentFile: file.name } : null);
          // Better sanitization: replace spaces and special chars, keep extension
          const ext = file.name.includes(".") ? file.name.substring(file.name.lastIndexOf(".")) : "";
          const baseName = file.name.replace(ext, "").replace(/[^a-zA-Z0-9._-]/g, "_") || "document";
          const safeName = `${baseName}${ext}`;
          const timestamp = Date.now();
          const path = `${eventId}/${timestamp}-${safeName}`;

          // Try to extract content first (for PDFs and other files)
          let rawContent: string | null = null;
          let extractedJson: Record<string, any> = {};
          let detectedType: string | null = file.type || "file";
          let pdfBase64: string | null = null; // For Claude native PDF reading

          if (isTextFile(file)) {
            // Read text files directly
            try {
              const text = await readFileText(file);
              rawContent = text.length > MAX_IMPORT_CHARS ? `${text.slice(0, MAX_IMPORT_CHARS)}вЂ¦` : text;
            } catch (err) {
              console.error("Error reading text file:", err);
              rawContent = null;
            }

            // CSV files: ALSO send through the converter API for structured extraction
            // (investors, startups, mentors, corporates) вЂ” raw text alone doesn't give us that
            // Use shorter timeout to avoid blocking upload
            if (isCSVFile(file)) {
              try {
                const conversionPromise = convertFileWithAI(file);
                const timeoutPromise = new Promise<never>((_, reject) => 
                  setTimeout(() => reject(new Error("CSV conversion timeout")), 10000)
                );
                const conversion = await Promise.race([conversionPromise, timeoutPromise]);
                extractedJson = conversion as unknown as Record<string, any>;
                detectedType = conversion.detectedType || detectedType;
                // If converter gave us richer raw content, prefer it
                if (conversion.raw_content && (!rawContent || conversion.raw_content.length > rawContent.length)) {
                  rawContent = conversion.raw_content;
                }
              } catch (csvErr) {
                // Non-fatal вЂ” the raw text is already stored
              }
            }
          } else {
            // For PDFs: try client-side extraction FIRST (fast, no network), then AI conversion if needed
            const isPDF = file.type === "application/pdf" || file.name.toLowerCase().endsWith(".pdf");
            
            if (isPDF) {
              // Capture PDF bytes as base64 for Claude native reading (much better than text extraction)
              // Cap at 4MB base64 (~3MB binary) to keep extraction fast and within API limits
              const MAX_PDF_BASE64_BYTES = 4 * 1024 * 1024; // 4MB base64
              try {
                const buffer = await file.arrayBuffer();
                const raw = new Uint8Array(buffer).reduce((data, byte) => data + String.fromCharCode(byte), "");
                const encoded = btoa(raw);
                if (encoded.length <= MAX_PDF_BASE64_BYTES) {
                  pdfBase64 = encoded;
                } else {
                  console.warn(`[PDF] PDF too large for extraction (${Math.round(encoded.length / 1024)}KB > ${MAX_PDF_BASE64_BYTES / 1024}KB), using text-only`);
                  // Still use text extraction for large PDFs
                }
              } catch (b64Err) {
                console.warn("[PDF] Failed to capture PDF bytes:", b64Err);
              }

              // Try client-side PDF extraction as a quick text fallback (for embeddings)
              try {
                rawContent = await extractPdfTextClientSide(file);
              } catch (err) {
                console.warn("[PDF] Client-side extraction failed, will try AI conversion:", err);
              }
            }

            // For non-text files (or if PDF client-side failed), try converter API (PDF/DOCX/XLSX/etc.)
            // Use a race condition: if AI conversion takes > 10s, skip it and continue with what we have
            if (!rawContent || !isPDF) {
              try {
                const conversionPromise = convertFileWithAI(file);
                const timeoutPromise = new Promise<never>((_, reject) => 
                  setTimeout(() => reject(new Error("Conversion timeout")), 10000)
                );
                const conversion = await Promise.race([conversionPromise, timeoutPromise]);
                rawContent = conversion.raw_content ?? rawContent; // Use AI content if better
                extractedJson = conversion as unknown as Record<string, any>;
                detectedType = conversion.detectedType || detectedType;
              } catch (err) {
                console.warn("[AI] Conversion failed or timed out (non-fatal):", err);
                // Continue without AI conversion - we have client-side content or will store file reference
              }
            }
            if (!rawContent) {
              toast({
                title: "No text extracted",
                description:
                  "We couldn't extract text from this file. If it's a PDF, redeploy the converter with CORS_ALLOW_ORIGINS or try a text-based file.",
                variant: "destructive",
              });
            }
          }

          // Try to upload to storage (optional - don't fail if this fails)
          let storagePath: string | null = null;
          try {
            const { error: uploadError } = await supabase.storage
              .from("cis-documents")
              .upload(path, file, { upsert: true });
            if (!uploadError) {
              storagePath = path;
            } else {
              console.warn("Storage upload failed (non-fatal):", uploadError.message);
              // Continue without storage - document will still be saved
            }
          } catch (storageErr) {
            console.warn("Storage upload error (non-fatal):", storageErr);
            // Continue without storage
          }

          // Extract a better title from file name (remove extension, clean up)
          const getDocumentTitle = (fileName: string | null): string => {
            if (!fileName) return "Uploaded document";
            // Remove file extension
            const nameWithoutExt = fileName.replace(/\.[^/.]+$/, "");
            // Remove random IDs like "document-1tWiD79w" -> "document"
            const cleaned = nameWithoutExt.replace(/-\w{8,}$/, "").trim();
            // If it's just "document" or empty, use a better default
            if (!cleaned || cleaned.toLowerCase() === "document") {
              return "Uploaded document";
            }
            return cleaned;
          };

          // в”Ђв”Ђ Detect folder-based entity type в”Ђв”Ђ
          const currentSelectedFolder = selectedFolderId !== "none" 
            ? sourceFolders.find(f => f.id === selectedFolderId)
            : null;
          const folderName = currentSelectedFolder?.name?.toLowerCase() || "";
          const isPortfolioFolder = folderName.includes("portfolio") || folderName.includes("company") || folderName.includes("partner");
          const shouldForceCreateCard = isPortfolioFolder;
          const entityTypeHint = "company";

          // Save document record (even if storage upload failed)
          const { data: doc, error: docError } = await insertDocument(eventId, {
            title: getDocumentTitle(file.name),
            source_type: "upload",
            file_name: file.name || null,
            storage_path: storagePath,
            detected_type: detectedType,
            extracted_json: extractedJson,
            raw_content: rawContent,
            created_by: currentUserId || null,
            folder_id: selectedFolderId !== "none" ? selectedFolderId : null,
          });

          if (docError || !doc) {
            toast({
              title: "Document save failed",
              description: docError?.message || `Could not save ${file.name}`,
              variant: "destructive",
            });
            continue;
          }

          const docRecord = doc as { id?: string; title?: string | null; storage_path?: string | null } | null;
          if (!docRecord?.id) {
            toast({
              title: "Document save failed",
              description: `Could not save ${file.name} - no ID returned`,
              variant: "destructive",
            });
            continue;
          }

          onDocumentSaved({
            id: docRecord.id,
            title: docRecord.title || null,
            storage_path: docRecord.storage_path || null,
          });
          uploadedDocs.push({ id: docRecord.id, title: docRecord.title || null });

          // Create a source entry for the uploaded file
          try {
            // Use the same cleaned title logic
            const getDocumentTitle = (fileName: string | null): string => {
              if (!fileName) return "Uploaded document";
              const nameWithoutExt = fileName.replace(/\.[^/.]+$/, "");
              const cleaned = nameWithoutExt.replace(/-\w{8,}$/, "").trim();
              if (!cleaned || cleaned.toLowerCase() === "document") {
                return "Uploaded document";
              }
              return cleaned;
            };
            await onCreateSource({
              title: getDocumentTitle(file.name),
              source_type: "notes",
              external_url: null,
              storage_path: storagePath,
              tags: ["local-upload", detectedType || "file"],
              notes: rawContent ? `Content extracted: ${rawContent.length} characters` : null,
              status: "active",
            }, eventId);
          } catch (sourceErr) {
            console.error("Error creating source:", sourceErr);
            // Non-fatal - document is saved, source creation can fail
          }

          // Index embeddings if we have content (with contextual enrichment)
          // Run in background - don't block upload completion
          if ((rawContent || pdfBase64) && docRecord.id) {
            // Fire and forget - don't await, let it run in background
            indexDocumentEmbeddings(docRecord.id, rawContent, docRecord.title || file.name, pdfBase64).catch((embedErr) => {
              console.error("Error indexing embeddings (non-fatal):", embedErr);
              // Non-fatal - document is saved, embeddings can be regenerated later
            });
          }

          // в”Ђв”Ђ Structured CSV ingestion: extract rows into kg_entities в”Ђв”Ђ
          if (extractedJson && docRecord.id) {
            try {
              const convData = extractedJson as Record<string, any>;
              const startupRows = convData.startups as any[] | undefined;

              if (startupRows && startupRows.length > 0) {
                const ingResult = await ingestStartupCSVRows(
                  eventId, startupRows, docRecord.id, currentUserId || null
                );
                if (ingResult.entitiesCreated > 0 || ingResult.entitiesUpdated > 0) {
                  toast({
                    title: "Structured data processed",
                    description: `${ingResult.entitiesCreated} new + ${ingResult.entitiesUpdated} updated company entities from CSV.`,
                  });
                }
              }
            } catch (structErr) {
              // Non-fatal: the document is saved, structured extraction is a bonus
            }
          }

          // в”Ђв”Ђ Auto-extract company properties into company card в”Ђв”Ђ
          // The DB trigger auto-creates a company entity from the document title.
          // If folder-based detection is enabled, force-create entity even if title doesn't match.
          // Run in background - don't block upload completion
          if ((rawContent || pdfBase64) && docRecord.id) {
            // Fire and forget - run property extraction in background
            // Capture variables needed for async execution
            const docId = docRecord.id;
            const docTitle = docRecord.title || file.name;
            const fileContent = rawContent || "";
            const filePdfBase64 = pdfBase64; // Capture PDF bytes for Claude native reading
            const folderInfo = { shouldForceCreateCard, entityTypeHint, currentSelectedFolder };
            
            (async () => {
              try {
                // Small delay to let the DB trigger create the entity
                await new Promise((r) => setTimeout(r, 1000));

                let companyEntityId = await getDocumentCompanyEntityId(docId);
              
              // в”Ђв”Ђ Folder-based card creation в”Ђв”Ђ
              // Prefer folder name so card is "TBE" not "Copy of TBE Due Diligence"
              if (!companyEntityId && folderInfo.shouldForceCreateCard && docTitle) {
                try {
                  const rawTitle = getDocumentTitle(file.name);
                  const deriveCompanyName = (t: string) => {
                    let s = t.replace(/^copy\s+of\s+/i, "").trim();
                    s = s.replace(/\s*(due\s*diligence|dd|diligence|deck|pitch|memo|presentation|report|summary|overview|brochure|tearsheet|one[- ]?pager).*$/i, "").trim();
                    s = s.replace(/\s*[-вЂ“вЂ”]\s*.*$/, "").trim();
                    const first = s.split(/\s+/)[0];
                    return (first && first.length > 1) ? first : s || rawTitle;
                  };
                  const companyName = folderInfo.currentSelectedFolder?.name && !folderInfo.currentSelectedFolder.name.toLowerCase().match(/^(portfolio|companies|partners?)$/)
                    ? folderInfo.currentSelectedFolder.name
                    : deriveCompanyName(rawTitle);
                  const normalizedName = normalizeCompanyNameForMatch(companyName);
                  
                  // Check if entity already exists
                  const { data: existingEntity } = await supabase
                    .from("kg_entities")
                    .select("id")
                    .eq("event_id", eventId)
                    .eq("normalized_name", normalizedName)
                    .eq("entity_type", entityTypeHint)
                    .single();
                  
                  if (existingEntity) {
                    companyEntityId = existingEntity.id;
                    // Link document to existing entity
                    await supabase
                      .from("documents")
                      .update({ company_entity_id: companyEntityId })
                      .eq("id", docId);
                  } else {
                    // Create new entity
                    const { data: newEntity, error: createErr } = await supabase
                      .from("kg_entities")
                      .insert({
                        event_id: eventId,
                        entity_type: entityTypeHint,
                        name: companyName,
                        normalized_name: normalizedName,
                        properties: {
                          auto_created: true,
                          source: "folder_based",
                          folder_name: folderInfo.currentSelectedFolder?.name || null,
                          first_seen_document: docId,
                          bio: "",
                          funding_stage: "",
                          amount_seeking: "",
                          valuation: "",
                          arr: "",
                          burn_rate: "",
                          runway_months: "",
                          problem: "",
                          solution: "",
                          tam: "",
                          competitive_edge: "",
                          founders: "[]",
                          ai_rationale: "",
                          website: "",
                          logo_url: "",
                        },
                        source_document_id: docId,
                        confidence: 0.8,
                        created_by: currentUserId || null,
                      })
                      .select("id")
                      .single();
                    
                    if (!createErr && newEntity) {
                      companyEntityId = newEntity.id;
                      // Link document to new entity
                      await supabase
                        .from("documents")
                        .update({ company_entity_id: companyEntityId })
                        .eq("id", docId);
                    } else {
                      console.error("[FolderCard] Failed to create entity:", createErr);
                    }
                  }
                } catch (folderErr) {
                  console.error("[FolderCard] Failed to create entity from folder:", folderErr);
                }
              }

              // в”Ђв”Ђ Fallback: If no entity exists, create one from cleaned document title в”Ђв”Ђ
              if (!companyEntityId && docTitle) {
                try {
                  const deriveCompanyName = (t: string) => {
                    let s = t.replace(/^copy\s+of\s+/i, "").trim();
                    s = s.replace(/\s*(due\s*diligence|dd|diligence|deck|pitch|memo|presentation|report|summary|overview|brochure|tearsheet|one[- ]?pager|factsheet|whitepaper|prospectus|dataroom|data\s*room).*$/i, "").trim();
                    s = s.replace(/\s*[-вЂ“вЂ”]\s*.*$/, "").replace(/\s+\d{4,}\s*$/i, "").replace(/\s*\(.*\)\s*$/, "").replace(/\s*\[.*\]\s*$/, "").replace(/\s+v\d+(\.\d+)?$/i, "").trim();
                    const first = s.split(/\s+/)[0];
                    return (first && first.length > 1) ? first : s || t;
                  };
                  let companyName = deriveCompanyName(docTitle);
                  
                  // If title is too generic, skip
                  if (companyName && companyName.length > 2 &&
                      !companyName.toLowerCase().match(/^(document|uploaded|file|untitled)/i)) {
                    const normalizedName = normalizeCompanyNameForMatch(companyName);
                    
                    // Check if entity already exists (use .limit(1) instead of .single() to avoid 406)
                    const { data: existingArr } = await supabase
                      .from("kg_entities")
                      .select("id")
                      .eq("event_id", eventId)
                      .eq("normalized_name", normalizedName)
                      .eq("entity_type", "company")
                      .limit(1);
                    const existingEntity = existingArr?.[0] ?? null;
                    
                    if (existingEntity) {
                      companyEntityId = existingEntity.id;
                      await supabase
                        .from("documents")
                        .update({ company_entity_id: companyEntityId })
                        .eq("id", docId);
                    } else {
                      // Create new entity
                      const { data: newEntity, error: createErr } = await supabase
                        .from("kg_entities")
                        .insert({
                          event_id: eventId,
                          entity_type: "company",
                          name: companyName,
                          normalized_name: normalizedName,
                          properties: {
                            auto_created: true,
                            source: "document_title_fallback",
                            first_seen_document: docId,
                            bio: "",
                            funding_stage: "",
                            amount_seeking: "",
                            valuation: "",
                            arr: "",
                            burn_rate: "",
                            runway_months: "",
                            problem: "",
                            solution: "",
                            tam: "",
                            competitive_edge: "",
                            founders: "[]",
                            ai_rationale: "",
                            website: "",
                            logo_url: "",
                          },
                          source_document_id: docId,
                          confidence: 0.6,
                          created_by: currentUserId || null,
                        })
                        .select("id")
                        .single();
                      
                      if (!createErr && newEntity) {
                        companyEntityId = newEntity.id;
                        await supabase
                          .from("documents")
                          .update({ company_entity_id: companyEntityId })
                          .eq("id", docId);
                      } else {
                        console.error("[AutoExtract] Failed to create entity from title:", createErr);
                      }
                    }
                  }
                } catch (fallbackErr) {
                  console.warn("[AutoExtract] Failed to create entity from title (non-fatal):", fallbackErr);
                }
              }

              if (companyEntityId) {
                const existing = await getEntityProperties(companyEntityId);
                const extraction = await extractCompanyProperties({
                  rawContent: fileContent,
                  documentTitle: docTitle,
                  existingProperties: existing?.properties || {},
                  pdfBase64: filePdfBase64 || undefined,
                });


                if (Object.keys(extraction.properties).length > 0) {
                  // If AI identified a better company_name, rename the entity
                  const aiName = (extraction.properties.company_name || "").trim();
                  if (aiName && aiName.length >= 2) {
                    const aiNorm = normalizeCompanyNameForMatch(aiName);
                    const currentEntity = existing;
                    const curNorm = normalizeCompanyNameForMatch(currentEntity?.name || "");
                    if (aiNorm !== curNorm && !aiNorm.includes("copy of")) {
                      const { data: existingByAiName } = await supabase
                        .from("kg_entities")
                        .select("id")
                        .eq("event_id", activeEventId!)
                        .eq("normalized_name", aiNorm)
                        .eq("entity_type", "company")
                        .limit(1);
                      if (existingByAiName && existingByAiName.length > 0) {
                        const targetId = existingByAiName[0].id;
                        await supabase.from("documents").update({ company_entity_id: targetId }).eq("id", docId);
                        companyEntityId = targetId;
                      } else {
                        await supabase
                          .from("kg_entities")
                          .update({ name: aiName, normalized_name: aiNorm })
                          .eq("id", companyEntityId);
                      }
                    }
                    delete extraction.properties.company_name;
                    delete extraction.confidence.company_name;
                  }

                  const mergeResult = await mergeCompanyCardFromExtraction(
                    companyEntityId,
                    extraction.properties,
                    extraction.confidence,
                    docId,
                  );
                  // Refresh company cards if callback available
                  if (onRefreshCompanyCards) {
                    onRefreshCompanyCards().catch(err => console.warn("[AutoExtract] Failed to refresh cards:", err));
                  }
                } else {
                  console.warn(`[AutoExtract] вљ пёЏ No properties extracted for ${file.name} (backend may be down or document type not recognized)`);
                }
              } else {
                console.warn(`[AutoExtract] вљ пёЏ No entity found for ${file.name} - cannot extract properties`);
              }
              } catch (extractErr) {
                console.error("[AutoExtract] Property extraction failed (non-fatal):", extractErr);
              }
            })().catch((err) => {
              console.error("[AutoExtract] Background extraction error:", err);
            });
          }

          successCount += 1;
        }

        // Update progress with final results
        setUploadProgress((prev) => prev ? { ...prev, current: files.length, results: batchResults } : null);

        if (successCount > 0) {
          const totalUpdated = batchResults.reduce((s, r) => s + r.updated, 0);
          const totalConflicts = batchResults.reduce((s, r) => s + r.conflicts, 0);
          const companiesUpdated = batchResults.filter((r) => r.updated > 0).length;
          const companiesCreated = batchResults.filter((r) => r.created).length;

          let description = `Uploaded ${successCount} file${successCount > 1 ? "s" : ""}.`;
          if (totalUpdated > 0 || companiesCreated > 0) {
            description += ` Updated ${companiesUpdated} compan${companiesUpdated === 1 ? "y" : "ies"} (${totalUpdated} properties).`;
          }
          if (totalConflicts > 0) {
            description += ` ${totalConflicts} conflict${totalConflicts > 1 ? "s" : ""} to review.`;
          }

          toast({
            title: "Upload complete",
            description,
          });
        }
        if (uploadedDocs.length > 0) {
          openFolderAssignmentDialog(uploadedDocs);
        }

        // If multiple files were uploaded and we have results, open the review dialog
        if (batchResults.length > 0 && batchResults.some((r) => r.updated > 0 || r.conflicts > 0)) {
          // Build review data from the batch results by re-fetching the entities
          const reviewItems: typeof batchReviewData = [];
          for (const result of batchResults) {
            if (result.updated > 0 || result.conflicts > 0) {
              // We can't access merge details directly, so we show a summary
              reviewItems.push({
                companyName: result.name,
                entityId: "", // not critical for display
                fields: [],
              });
            }
          }
          // Only show review if meaningful results
          if (reviewItems.length > 0 && files.length > 1) {
            // The review is informational for batch uploads вЂ” toast is enough for single files
            setBatchReviewData(reviewItems);
          }
        }

        // Refresh company cards to reflect auto-extraction
        if (onRefreshCompanyCards) {
          await onRefreshCompanyCards();
        }
      } catch (err) {
        toast({
          title: "Upload error",
          description: err instanceof Error ? err.message : "An unexpected error occurred.",
          variant: "destructive",
        });
      } finally {
        setIsUploadingLocal(false);
        setUploadProgress(null);
        e.target.value = "";
      }
    },
    [activeEventId, currentUserId, ensureActiveEventId, indexDocumentEmbeddings, onCreateSource, onDocumentSaved, onRefreshCompanyCards, openFolderAssignmentDialog, selectedFolderId, sourceFolders, toast]
  );

  const importDriveUrl = useCallback(async (url: string) => {
    const eventId = activeEventId || (await ensureActiveEventId());
    if (!eventId) {
      toast({
        title: "No active event",
        description: "Create or activate an event before importing.",
        variant: "destructive",
      });
      return;
    }
    if (!url.trim()) {
      toast({
        title: "Missing Drive link",
        description: "Paste or choose a Google Drive file to import.",
        variant: "destructive",
      });
      return;
    }
    setIsImportingDrive(true);
    try {
      let accessToken = await getGoogleAccessToken();
      if (!accessToken) accessToken = await getGoogleAccessToken(true);
      if (!accessToken) {
        try {
          toast({
            title: "Connect Google Drive",
            description: "Redirecting to Google to grant Drive access…",
          });
          await triggerGoogleOAuthForDrive();
        } catch (e) {
          const msg = e instanceof Error ? e.message : String(e);
          setDriveConnectCooldownUntil(Date.now() + 65000);
          toast({ title: "Could not connect Google Drive", description: msg, variant: "destructive" });
        }
        return;
      }
      const result = await ingestGoogleDrive(url.trim(), accessToken);
      
      // Extract better title from Google Drive
      const extractTitleFromGoogleDrive = (title: string | null | undefined, content: string | null | undefined, url: string): string => {
        // First, try to extract from content (look for first heading or title)
        if (content) {
          // Look for markdown-style headings
          const headingMatch = content.match(/^#+\s+(.+)$/m);
          if (headingMatch && headingMatch[1]) {
            const extracted = headingMatch[1].trim();
            if (extracted.length > 3 && extracted.length < 100) {
              return extracted;
            }
          }
          // Look for first line that looks like a title (capitalized, short)
          const lines = content.split('\n').filter(l => l.trim().length > 0);
          for (const line of lines.slice(0, 5)) {
            const trimmed = line.trim();
            if (trimmed.length > 5 && trimmed.length < 80 && /^[A-Z]/.test(trimmed)) {
              // Check if it's not just a sentence
              if (!trimmed.includes('.') || trimmed.split('.').length <= 2) {
                return trimmed;
              }
            }
          }
        }
        
        // Try to extract from URL (Google Docs URLs sometimes have the doc name)
        const urlMatch = url.match(/\/d\/([a-zA-Z0-9_-]+)/);
        if (urlMatch) {
          // Can't get name from ID, but we can try the title
        }
        
        // Clean up the provided title
        if (title) {
          const nameWithoutExt = title.replace(/\.[^/.]+$/, "");
          const cleaned = nameWithoutExt
            .replace(/-\w{8,}$/, "") // Remove random IDs
            .replace(/^notes\s+/i, "") // Remove "notes" prefix
            .replace(/^google\s+drive\s+document$/i, "") // Remove generic text
            .trim();
          if (cleaned && cleaned.toLowerCase() !== "document" && cleaned.length > 3) {
            return cleaned;
          }
        }
        
        // Fallback: try to use URL to suggest a name
        return "Google Drive Document";
      };
      
      const cleanedTitle = extractTitleFromGoogleDrive(
        result.title,
        result.raw_content || result.content,
        url.trim()
      );
      
      await onCreateSource({
        title: cleanedTitle,
        source_type: "notes",
        external_url: url.trim(),
        tags: ["google-drive"],
        notes: null,
        status: "active",
      }, eventId);
      toast({ title: "Drive import complete", description: "Source saved to your library." });

      const rawContent = result.raw_content || result.content;
      let assignmentDoc: { id: string; title: string | null } | null = null;
      let autoLogged = false;
      let conversionResult: AIConversionResponse | null = null;
      if (autoExtract && rawContent) {
        const content = rawContent.length > MAX_IMPORT_CHARS ? rawContent.slice(0, MAX_IMPORT_CHARS) : rawContent;
        conversionResult = await convertWithAI(content);
        const primary = conversionResult.startups?.[0];
        if (primary?.companyName) {
          await onAutoLogDecision({
            draft: {
              startupName: primary.companyName || "Unknown Company",
              sector: primary.industry || undefined,
              stage: primary.fundingStage || undefined,
            },
            conversion: conversionResult,
            sourceType: "api",
            fileName: result.title || null,
            file: null,
            rawContent, // Store the raw content from Google Drive
            eventIdOverride: eventId,
          });
          autoLogged = true;
          toast({ title: "Decision logged", description: "Auto-created from Drive extraction." });
        } else if (conversionResult.errors?.length) {
          toast({ title: "Extraction warning", description: conversionResult.errors[0], variant: "destructive" });
        } else {
          toast({ title: "No entity detected", description: "Extraction completed, but no company was found." });
        }
      }

      if (!rawContent) {
        toast({
          title: "Drive import note",
          description: "Drive returned no text content. Saving the source without raw text.",
        });
      }

      // Always create a document, even if auto-logged (document might be created by onAutoLogDecision)
      // But if auto-logged didn't create one, create it here
      if (!autoLogged) {
        try {
          // Clean title for document too
          const cleanTitle = (title: string | null | undefined): string => {
            if (!title) return "Google Drive document";
            const nameWithoutExt = title.replace(/\.[^/.]+$/, "");
            const cleaned = nameWithoutExt.replace(/-\w{8,}$/, "").replace(/^notes\s+/i, "").trim();
            if (!cleaned || cleaned.toLowerCase() === "document") {
              return "Google Drive document";
            }
            return cleaned;
          };
          
          const { data: doc, error: docError } = await insertDocument(eventId, {
            title: cleanedTitle,
            source_type: "api",
            file_name: result.title || cleanedTitle || null,
            storage_path: null,
            detected_type: conversionResult?.detectedType || null,
            extracted_json: (conversionResult || {}) as Record<string, any>,
            raw_content: rawContent || null,
            created_by: currentUserId || null,
            folder_id: null,
          });
          const docRecord = doc as { id?: string; title?: string | null; storage_path?: string | null } | null;
          if (docError) {
            console.error("Document insert error:", docError);
            toast({
              title: "Document save failed",
              description: docError.message || JSON.stringify(docError),
              variant: "destructive",
            });
          } else if (!docRecord?.id) {
            console.error("Document insert returned no ID:", doc);
            toast({
              title: "Document save failed",
              description: "Insert succeeded but no document ID returned.",
              variant: "destructive",
            });
          } else {
            onDocumentSaved({
              id: docRecord.id,
              title: docRecord.title || null,
              storage_path: docRecord.storage_path || null,
            });
            toast({ title: "Document saved", description: "Raw content stored in Documents." });
            // Index embeddings in background (non-blocking)
            indexDocumentEmbeddings(docRecord.id, rawContent || null, docRecord.title || cleanedTitle).catch((err) => {
              console.error("Error indexing embeddings (non-fatal):", err);
            });
            assignmentDoc = { id: docRecord.id, title: docRecord.title || cleanedTitle };
          }
        } catch (err) {
          console.error("Exception during document insert:", err);
          toast({
            title: "Document save error",
            description: err instanceof Error ? err.message : "Unexpected error saving document.",
            variant: "destructive",
          });
        }
      }

      if (!assignmentDoc && result.title) {
        try {
          const { data: recentDocs } = await supabase
            .from("documents")
            .select("id,title")
            .eq("event_id", eventId)
            .eq("file_name", result.title)
            .order("created_at", { ascending: false })
            .limit(1);
          const candidate = Array.isArray(recentDocs) ? recentDocs[0] : null;
          if (candidate?.id) {
            assignmentDoc = { id: candidate.id, title: candidate.title || result.title || cleanedTitle };
          }
        } catch (lookupErr) {
          console.warn("Folder assignment lookup failed:", lookupErr);
        }
      }

      if (assignmentDoc) {
        openFolderAssignmentDialog([assignmentDoc]);
      }

      setDriveUrl("");
    } catch (error) {
      toast({
        title: "Drive import failed",
        description: error instanceof Error ? error.message : "Could not import Google Drive file.",
        variant: "destructive",
      });
    } finally {
      setIsImportingDrive(false);
    }
  }, [activeEventId, autoExtract, currentUserId, ensureActiveEventId, getGoogleAccessToken, indexDocumentEmbeddings, onAutoLogDecision, onCreateSource, onDocumentSaved, openFolderAssignmentDialog, toast]);

  const handleImportDrive = useCallback(async () => {
    await importDriveUrl(driveUrl.trim());
  }, [driveUrl, importDriveUrl]);

  const openDrivePicker = useCallback(async () => {
    if (!googleApiKey || !googleClientId) {
      toast({
        title: "Google Picker not configured",
        description: "Set VITE_GOOGLE_API_KEY and VITE_GOOGLE_CLIENT_ID to use Drive picker.",
        variant: "destructive",
      });
      return;
    }
    let accessToken = await getGoogleAccessToken();
    if (!accessToken) accessToken = await getGoogleAccessToken(true);
    if (!accessToken) {
      try {
        toast({
          title: "Connect Google Drive",
          description: "Redirecting to Google to grant Drive access…",
        });
        await triggerGoogleOAuthForDrive();
      } catch (e) {
        const msg = e instanceof Error ? e.message : String(e);
        setDriveConnectCooldownUntil(Date.now() + 65000);
        toast({ title: "Could not connect Google Drive", description: msg, variant: "destructive" });
      }
      return;
    }
    try {
      await loadGooglePicker();
      
      // View for all supported document types
      const allFilesView = new window.google.picker.DocsView()
        .setIncludeFolders(true)
        .setSelectFolderEnabled(false)
        .setMode(window.google.picker.DocsViewMode.LIST)
        .setMimeTypes([
          // Google native formats
          "application/vnd.google-apps.document",
          "application/vnd.google-apps.spreadsheet",
          "application/vnd.google-apps.presentation",
          // PDFs
          "application/pdf",
          // Microsoft Office formats
          "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
          "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
          "application/vnd.openxmlformats-officedocument.presentationml.presentation",
          "application/msword",
          "application/vnd.ms-excel",
          "application/vnd.ms-powerpoint",
          // Text formats
          "text/plain",
          "text/csv",
          "text/markdown",
          "application/json",
        ].join(","));
      
      // View specifically for Shared Drives (Team Drives)
      const sharedDriveView = new window.google.picker.DocsView()
        .setIncludeFolders(true)
        .setSelectFolderEnabled(false)
        .setEnableDrives(true)
        .setMode(window.google.picker.DocsViewMode.LIST);
      
      // Recently viewed files view
      const recentView = new window.google.picker.DocsView()
        .setIncludeFolders(false)
        .setSelectFolderEnabled(false)
        .setMode(window.google.picker.DocsViewMode.LIST)
        .setOwnedByMe(false);  // Include files shared with me
      
      const picker = new window.google.picker.PickerBuilder()
        .setDeveloperKey(googleApiKey)
        .setOAuthToken(accessToken)
        .setAppId(googleClientId.split("-")[0])  // Extract project number from client ID
        .addView(allFilesView)
        .addView(sharedDriveView)
        .addView(recentView)
        .enableFeature(window.google.picker.Feature.SUPPORT_DRIVES)  // Enable Shared Drives
        .enableFeature(window.google.picker.Feature.MULTISELECT_ENABLED)  // Allow multiple file selection
        .setCallback((data: any) => {
          if (data.action === window.google.picker.Action.PICKED) {
            const docs = data.docs || [];
            // Handle multiple files if selected
            for (const doc of docs) {
              const pickedUrl = doc?.url;
              if (pickedUrl) {
                setDriveUrl(pickedUrl);
                importDriveUrl(pickedUrl);
              }
            }
          }
        })
        .build();
      picker.setVisible(true);
    } catch (error) {
      toast({
        title: "Drive picker failed",
        description: error instanceof Error ? error.message : "Could not open Drive picker.",
        variant: "destructive",
      });
    }
  }, [getGoogleAccessToken, googleApiKey, googleClientId, importDriveUrl, toast]);

  return (
    <div className="space-y-6">
      {!canImport && (
        <Card className="border border-slate-200 bg-white">
          <CardContent className="pt-4 text-sm text-slate-500 font-mono">
            Loading your active event... Imports will be available in a moment.
          </CardContent>
        </Card>
      )}
      {/* ClickUp import temporarily disabled */}


      <Card className="border border-slate-200 bg-white">
        <CardHeader className="border-b border-slate-200">
          <CardTitle className="text-slate-900 font-mono font-black uppercase tracking-tight">Local Upload</CardTitle>
          <CardDescription className="text-slate-500 font-mono">
            Upload files from your computer into Sources.
            {selectedFolderId !== "none" && (
              <span className="ml-2 text-blue-600">
                в†’ Default folder: {sourceFolders.find(f => f.id === selectedFolderId)?.name}
              </span>
            )}
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-3">
          <Input
            type="file"
            multiple
            disabled={!canImport || isUploadingLocal}
            onChange={handleLocalUpload}
            accept=".txt,.md,.csv,.json,.pdf,.docx,.xlsx,.xls"
          />
          {uploadProgress && (
            <div className="space-y-2 p-3 rounded-lg border border-blue-500/30 bg-blue-600/5">
              <div className="flex items-center justify-between">
                <span className="text-xs font-mono text-slate-600">
                  Processing {uploadProgress.current}/{uploadProgress.total}: <span className="text-blue-600">{uploadProgress.currentFile}</span>
                </span>
                <span className="text-xs font-mono text-slate-400">
                  {Math.round((uploadProgress.current / uploadProgress.total) * 100)}%
                </span>
              </div>
              <div className="w-full bg-slate-100 rounded-full h-2 overflow-hidden">
                <div
                  className="bg-blue-600 h-2 rounded-full transition-all duration-500"
                  style={{ width: `${(uploadProgress.current / uploadProgress.total) * 100}%` }}
                />
              </div>
              {uploadProgress.results.length > 0 && (
                <div className="text-[10px] font-mono text-slate-400 space-y-0.5 max-h-20 overflow-auto">
                  {uploadProgress.results.map((r, i) => (
                    <div key={i} className="flex items-center gap-1">
                      <span className="text-slate-500">{r.name}:</span>
                      {r.updated > 0 && <span className="text-emerald-400">{r.updated} updated</span>}
                      {r.conflicts > 0 && <span className="text-orange-400">{r.conflicts} conflicts</span>}
                      {r.updated === 0 && r.conflicts === 0 && <span className="text-slate-400">no changes</span>}
                    </div>
                  ))}
                </div>
              )}
            </div>
          )}
          <p className="text-xs text-slate-500 font-mono">
            Supported: PDF, Word (.docx), Excel (.xlsx, .xls), Text (.txt, .md, .csv, .json) вЂ” all are indexed for AI search.
          </p>
        </CardContent>
      </Card>

      <Card className="border border-slate-200 bg-white">
        <CardHeader className="border-b border-slate-200">
          <CardTitle className="text-slate-900 font-mono font-black uppercase tracking-tight">Google Drive Import</CardTitle>
          <CardDescription className="text-slate-500 font-mono">
            Paste a Google Docs/Slides/Sheets link or choose from Drive.
            {selectedFolderId !== "none" && (
              <span className="ml-2 text-blue-600">
                в†’ Default folder: {sourceFolders.find(f => f.id === selectedFolderId)?.name}
              </span>
            )}
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-3">
          <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
            <div className="md:col-span-2">
              <Label className="text-slate-900 font-mono font-bold">Drive URL</Label>
              <Input
                value={driveUrl}
                onChange={(e) => setDriveUrl(e.target.value)}
                placeholder="https://docs.google.com/document/d/..."
                className="border border-slate-200 bg-white text-slate-900 placeholder:text-slate-400"
              />
            </div>
            <div className="flex items-end">
              <div className="flex w-full flex-col gap-2">
                <Button onClick={openDrivePicker} variant="outline" disabled={isDriveConnectOnCooldown} className="w-full border border-slate-200 bg-white text-slate-900 hover:bg-blue-500/10 hover:border-blue-500 hover:text-blue-600 font-bold disabled:opacity-50">
                  <Folder className="h-4 w-4 mr-2" />
                  {isDriveConnectOnCooldown ? `Wait ${driveConnectCooldownSeconds}s before retrying` : "Choose from Drive"}
                </Button>
                <Button onClick={handleImportDrive} disabled={isImportingDrive || isDriveConnectOnCooldown} className="w-full bg-blue-600 text-slate-900 hover:bg-blue-600/80 font-bold border-2 border-blue-500 transition-all hover:shadow-lg hover:shadow-blue-500/20 disabled:opacity-50">
                  {isImportingDrive ? <Loader2 className="h-4 w-4 mr-2 animate-spin" /> : <Upload className="h-4 w-4 mr-2" />}
                  {isDriveConnectOnCooldown ? `Wait ${driveConnectCooldownSeconds}s` : "Import Drive"}
                </Button>
              </div>
            </div>
          </div>
          <label className="flex items-center gap-2 text-xs text-slate-500 font-mono">
            <Checkbox checked={autoExtract} onCheckedChange={(val) => setAutoExtract(val === true)} className="border-slate-200 data-[state=checked]:bg-blue-600 data-[state=checked]:border-blue-500" />
            Auto-extract and log decision after import
          </label>
          <p className="text-xs text-slate-500 font-mono">
            Uses your Google Drive OAuth token. If access fails, sign out and sign in again.
          </p>
        </CardContent>
      </Card>

      {/* в”Ђв”Ђ Google Drive Portfolio Folder Sync в”Ђв”Ђ */}
      <Card className="border border-slate-200 bg-white">
        <CardHeader className="border-b border-slate-200">
          <CardTitle className="text-slate-900 font-mono font-black uppercase tracking-tight">
            <Cloud className="h-5 w-5 inline mr-2 text-blue-600" />
            Google Drive Folder Sync
          </CardTitle>
          <CardDescription className="text-slate-500 font-mono">
            Connect a root folder from Google Drive. Each sub-folder will be treated as a separate project.
            All documents are automatically fetched, embedded, and extracted.
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          {(connectedDriveFolders.length > 0 || connectedDriveFolderId) ? (
            <>
              {/* Connected folders list */}
              <div className="space-y-2">
                {(connectedDriveFolders.length > 0 ? connectedDriveFolders : [{ id: connectedDriveFolderId!, name: connectedDriveFolderName || "Project folder", category: "Projects" as const }]).map((folder) => (
                  <div key={folder.id} className="flex items-center justify-between gap-3 p-3 rounded-lg border-2 border-blue-500/30 bg-blue-600/5 flex-wrap">
                    <div className="flex items-center gap-3 min-w-0 flex-1">
                      <Folder className="h-5 w-5 text-blue-600 shrink-0" />
                      <div className="min-w-0">
                        <div className="font-mono font-bold text-slate-900 text-sm">{folder.name}</div>
                        <div className="text-[10px] text-slate-400 font-mono truncate max-w-[200px]">{folder.id}</div>
                      </div>
                    </div>
                    <div className="flex items-center gap-2 shrink-0">
                      <Label className="text-[10px] text-slate-400 font-mono whitespace-nowrap">Root folder type:</Label>
                      <Select
                        value={folder.category ?? "Projects"}
                        onValueChange={async (value) => {
                          const updated = connectedDriveFolders.map((f) =>
                            f.id === folder.id ? { ...f, category: value } : f
                          );
                          setConnectedDriveFolders(updated);
                          if (activeEventId) {
                            const foldersToSave = updated.map((f) => ({ id: f.id, name: f.name, category: f.category ?? "Projects" }));
                            await supabase.from("sync_configurations")
                              .update({
                                config: {
                                  google_drive_folder_id: updated[0]?.id || null,
                                  google_drive_folder_name: updated[0]?.name || null,
                                  folders: foldersToSave,
                                },
                              })
                              .eq("event_id", activeEventId)
                              .eq("source_type", "google_drive");

                            // Propagate category to matching source_folders so Knowledge Scope stays in sync
                            const rootName = folder.name?.trim() || "";
                            const rootNorm = rootName.toLowerCase().replace(/\s*\([^)]*\)\s*/g, " ").replace(/\s+/g, " ").trim();
                            const rootSeg = rootNorm.replace(/\s*-\s*.*$/, "").trim();
                            for (const sf of sourceFolders) {
                              const sfName = (sf.name || "").trim();
                              const sfNorm = sfName.toLowerCase().replace(/\s*\([^)]*\)\s*/g, " ").replace(/\s+/g, " ").trim();
                              const sfSeg = (sfNorm.split(/\s*\/\s*/)[0] || sfNorm).replace(/\s*-\s*.*$/, "").trim();
                              const belongs = sfNorm === rootNorm || sfNorm.startsWith(rootNorm + " / ") || sfSeg === rootSeg;
                              if (belongs && (sf.category || "Projects") !== value) {
                                await onFolderCategoryUpdated?.(sf.id, value);
                              }
                            }

                            // Notify parent so initialDriveSyncConfig (and its backfill effect) stays in sync
                            onDriveSyncConfigChanged?.(foldersToSave);

                            toast({ title: "Folder type updated", description: `"${folder.name}" is now ${value}. Source folders re-categorized.` });
                          }
                        }}
                      >
                        <SelectTrigger className="w-[190px] h-8 text-xs border border-slate-200 bg-white text-slate-900 font-mono rounded-md hover:border-blue-400 transition-colors">
                          <SelectValue />
                        </SelectTrigger>
                        <SelectContent className="bg-white border border-slate-200 shadow-lg rounded-md">
                          {DRIVE_ROOT_CATEGORIES.map((cat) => (
                            <SelectItem key={cat} value={cat} className="text-slate-900 font-mono hover:bg-blue-50 focus:bg-blue-50 cursor-pointer">
                              {cat}
                            </SelectItem>
                          ))}
                        </SelectContent>
                      </Select>
                    </div>
                    {connectedDriveFolders.length > 1 && (
                      <Button
                        variant="ghost"
                        size="sm"
                        onClick={async () => {
                          const updated = connectedDriveFolders.filter(f => f.id !== folder.id);
                          setConnectedDriveFolders(updated);
                          if (updated.length > 0) {
                            setConnectedDriveFolderId(updated[0].id);
                            setConnectedDriveFolderName(updated[0].name);
                          } else {
                            setConnectedDriveFolderId(null);
                            setConnectedDriveFolderName(null);
                          }
                          // Persist updated list (include category for each folder)
                          if (activeEventId) {
                            const foldersToSave = updated.map((f) => ({ id: f.id, name: f.name, category: f.category ?? "Projects" }));
                            await supabase.from("sync_configurations")
                              .update({
                                config: {
                                  google_drive_folder_id: updated[0]?.id || null,
                                  google_drive_folder_name: updated[0]?.name || null,
                                  folders: foldersToSave,
                                },
                              })
                              .eq("event_id", activeEventId)
                              .eq("source_type", "google_drive");
                          }
                          toast({ title: "Folder removed", description: `Removed "${folder.name}" from sync.` });
                        }}
                        className="text-slate-400 hover:text-red-400 hover:bg-red-500/10"
                      >
                        <Trash2 className="h-3.5 w-3.5" />
                      </Button>
                    )}
                  </div>
                ))}
              </div>

              {/* Sync status bar */}
              <div className="flex items-center justify-between p-2.5 rounded-lg border border-slate-200/15 bg-white">
                <div className="flex items-center gap-3">
                  <div className="flex items-center gap-2">
                    <span className="relative flex h-2 w-2">
                      <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-75" />
                      <span className="relative inline-flex rounded-full h-2 w-2 bg-emerald-400" />
                    </span>
                    <span className="text-[10px] text-emerald-400 font-mono font-semibold">AUTO-SYNC ACTIVE</span>
                  </div>
                  <span className="text-[10px] text-slate-400 font-mono">Every 15 min</span>
                  {lastDriveSyncAt && (
                    <span className="text-[10px] text-slate-400 font-mono">
                      | Last: {new Date(lastDriveSyncAt).toLocaleString()}
                    </span>
                  )}
                </div>
                <div className="flex items-center gap-2">
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={connectDrivePortfolioFolder}
                    disabled={isDriveConnectOnCooldown}
                    className="border border-slate-200 bg-white text-slate-500 hover:bg-blue-500/10 hover:border-blue-500 hover:text-blue-600 font-bold text-[10px] h-7 px-2 disabled:opacity-50"
                  >
                    <FolderPlus className="h-3.5 w-3.5 mr-1" />
                    {isDriveConnectOnCooldown ? `Wait ${driveConnectCooldownSeconds}s` : "Add Folder"}
                  </Button>
                  <Button
                    size="sm"
                    onClick={() => syncGoogleDriveFolder()}
                    disabled={isSyncingDrive || !canImport}
                    className="bg-blue-600 text-slate-900 hover:bg-blue-600/80 font-bold border-2 border-blue-500 transition-all hover:shadow-lg hover:shadow-blue-500/20 disabled:opacity-50 h-7 text-[10px] px-2"
                  >
                    {isSyncingDrive ? (
                      <>
                        <Loader2 className="h-3.5 w-3.5 mr-1 animate-spin" />
                        Syncing...
                      </>
                    ) : (
                      <>
                        <RefreshCw className="h-3.5 w-3.5 mr-1" />
                        Sync Now
                      </>
                    )}
                  </Button>
                </div>
              </div>

              {/* Sync progress */}
              {driveSyncProgress && (
                <div className="space-y-2 p-3 rounded-lg border border-blue-500/30 bg-blue-600/5">
                  <div className="flex items-center justify-between">
                    <span className="text-xs font-mono text-slate-600">
                      {driveSyncProgress.phase}{" "}
                      {driveSyncProgress.total > 0 && (
                        <>
                          {driveSyncProgress.current}/{driveSyncProgress.total}:{" "}
                          <span className="text-blue-600">{driveSyncProgress.currentItem}</span>
                        </>
                      )}
                    </span>
                    {driveSyncProgress.total > 0 && (
                      <span className="text-xs font-mono text-slate-400">
                        {Math.round((driveSyncProgress.current / driveSyncProgress.total) * 100)}%
                      </span>
                    )}
                  </div>
                  {driveSyncProgress.total > 0 && (
                    <div className="w-full bg-slate-100 rounded-full h-2 overflow-hidden">
                      <div
                        className="bg-blue-600 h-2 rounded-full transition-all duration-500"
                        style={{ width: `${(driveSyncProgress.current / driveSyncProgress.total) * 100}%` }}
                      />
                    </div>
                  )}
                </div>
              )}

              {/* Sync results */}
              {driveSyncResults.length > 0 && !driveSyncProgress && (
                <div className="space-y-1 max-h-40 overflow-y-auto">
                  {driveSyncResults.map((r, i) => (
                    <div key={i} className="flex items-center gap-2 text-[10px] font-mono text-slate-500 px-2 py-1 rounded bg-slate-50">
                      <Building2 className="h-3 w-3 text-blue-600 flex-shrink-0" />
                      <span className="font-bold text-slate-700">{r.companyName}</span>
                      {r.newFiles > 0 && <span className="text-emerald-400">+{r.newFiles} new</span>}
                      {r.updatedFiles > 0 && <span className="text-blue-600">{r.updatedFiles} updated</span>}
                      {r.skippedFiles > 0 && <span className="text-slate-400">{r.skippedFiles} unchanged</span>}
                      {r.newFiles === 0 && r.updatedFiles === 0 && r.skippedFiles === 0 && (
                        <span className="text-slate-400">empty folder</span>
                      )}
                    </div>
                  ))}
                </div>
              )}
            </>
          ) : (
            <div className="text-center py-6 space-y-3">
              <Cloud className="h-10 w-10 text-slate-300 mx-auto" />
              <p className="text-sm text-slate-400 font-mono">No Drive folder connected yet.</p>
              <Button
                onClick={connectDrivePortfolioFolder}
                disabled={!canImport || isDriveConnectOnCooldown}
                className="bg-blue-600 text-slate-900 hover:bg-blue-600/80 font-bold border-2 border-blue-500 transition-all hover:shadow-lg hover:shadow-blue-500/20 disabled:opacity-50"
              >
                <Folder className="h-4 w-4 mr-2" />
                {isDriveConnectOnCooldown ? `Wait ${driveConnectCooldownSeconds}s before retrying` : "Connect Drive Folder"}
              </Button>
              <p className="text-[10px] text-slate-400 font-mono">
                Pick a root folder from Google Drive. Each sub-folder inside it will be treated as a separate project.
              </p>
            </div>
          )}
        </CardContent>
      </Card>

      {/* в”Ђв”Ђ Gmail Inbox Sync в”Ђв”Ђ */}
      <Card className="border border-slate-200 bg-white">
        <CardHeader className="border-b border-slate-200">
          <CardTitle className="text-slate-900 font-mono font-black uppercase tracking-tight">
            <Mail className="h-5 w-5 inline mr-2 text-blue-600" />
            Gmail Inbox Sync
          </CardTitle>
          <CardDescription className="text-slate-500 font-mono">
            Read emails from your Gmail inbox. Emails are embedded and available for RAG queries.
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          {isGmailConnected ? (
            <>
              {/* Config panel */}
              <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
                <div className="md:col-span-2">
                  <Label className="text-slate-900 font-mono font-bold text-xs">Gmail Search Filter</Label>
                  <Input
                    value={gmailQuery}
                    onChange={(e) => setGmailQuery(e.target.value)}
                    placeholder='e.g. from:team@company.com, subject:"report", has:attachment'
                    className="border border-slate-200 bg-white text-slate-900 placeholder:text-slate-400 text-xs font-mono"
                  />
                  <p className="text-[10px] text-slate-400 font-mono mt-1">Standard Gmail search operators. Leave blank to sync all inbox mail.</p>
                </div>
                <div>
                  <Label className="text-slate-900 font-mono font-bold text-xs">Max emails per sync</Label>
                  <Input
                    type="number"
                    min={1}
                    value={gmailMaxPerSync}
                    onChange={(e) => {
                      const v = parseInt(e.target.value, 10);
                      if (!isNaN(v) && v >= 1) setGmailMaxPerSync(v);
                      else if (e.target.value === "") setGmailMaxPerSync(50);
                    }}
                    className="border border-slate-200 bg-white text-slate-900 text-xs font-mono"
                  />
                  <p className="text-[10px] text-slate-400 font-mono mt-1">No limit вЂ” enter any number.</p>
                </div>
              </div>
              <div className="flex items-center gap-3">
                <Checkbox
                  id="gmail-attachments"
                  checked={gmailIncludeAttachments}
                  onCheckedChange={(v) => setGmailIncludeAttachments(!!v)}
                  className="border-slate-200/50 data-[state=checked]:bg-blue-600 data-[state=checked]:border-blue-500"
                />
                <Label htmlFor="gmail-attachments" className="text-slate-500 font-mono text-xs cursor-pointer">
                  Process email attachments (PDFs, docs)
                </Label>
              </div>
              {/* Save config changes */}
              <Button
                size="sm"
                className="bg-blue-600 text-slate-900 hover:bg-blue-600/80 font-bold font-mono text-xs h-9 px-5 border-2 border-blue-500 transition-all hover:shadow-lg hover:shadow-blue-200/0.4)] uppercase tracking-wider"
                onClick={async () => {
                  if (!activeEventId) return;
                  await supabase.from("sync_configurations")
                    .update({ config: { gmail_query: gmailQuery, max_emails_per_sync: gmailMaxPerSync, include_attachments: gmailIncludeAttachments } })
                    .eq("event_id", activeEventId)
                    .eq("source_type", "gmail");
                  toast({ title: "Gmail config saved", description: "Your Gmail sync settings have been updated." });
                }}
              >
                <Save className="h-4 w-4 mr-2" /> Save Settings
              </Button>

              {/* Sync status bar */}
              <div className="flex items-center justify-between p-2.5 rounded-lg border border-slate-200/15 bg-white">
                <div className="flex items-center gap-3">
                  <div className="flex items-center gap-2">
                    <span className="relative flex h-2 w-2">
                      <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-75" />
                      <span className="relative inline-flex rounded-full h-2 w-2 bg-emerald-400" />
                    </span>
                    <span className="text-[10px] text-emerald-400 font-mono font-semibold">GMAIL CONNECTED</span>
                  </div>
                  {lastGmailSyncAt && (
                    <span className="text-[10px] text-slate-400 font-mono">
                      Last sync: {new Date(lastGmailSyncAt).toLocaleString()}
                    </span>
                  )}
                </div>
                <div className="flex items-center gap-2">
                  <Button
                    size="sm"
                    onClick={syncGmailInbox}
                    disabled={isSyncingGmail || !canImport}
                    className="bg-blue-600 text-slate-900 hover:bg-blue-600/80 font-bold border-2 border-blue-500 transition-all hover:shadow-lg hover:shadow-blue-500/20 disabled:opacity-50 h-7 text-[10px] px-2"
                  >
                    {isSyncingGmail ? (
                      <><Loader2 className="h-3.5 w-3.5 mr-1 animate-spin" />SyncingвЂ¦</>
                    ) : (
                      <><RefreshCw className="h-3.5 w-3.5 mr-1" />Sync Now</>
                    )}
                  </Button>
                </div>
              </div>

              {/* Sync progress */}
              {gmailSyncProgress && (
                <div className="space-y-2 p-3 rounded-lg border border-blue-500/30 bg-blue-600/5">
                  <div className="flex items-center justify-between">
                    <span className="text-xs font-mono text-slate-600">
                      {gmailSyncProgress.currentItem}
                    </span>
                    {gmailSyncProgress.total > 0 && (
                      <span className="text-xs font-mono text-slate-400">
                        {Math.round((gmailSyncProgress.current / gmailSyncProgress.total) * 100)}%
                      </span>
                    )}
                  </div>
                  {gmailSyncProgress.total > 0 && (
                    <div className="w-full bg-slate-100 rounded-full h-2 overflow-hidden">
                      <div
                        className="bg-blue-600 h-2 rounded-full transition-all duration-500"
                        style={{ width: `${(gmailSyncProgress.current / gmailSyncProgress.total) * 100}%` }}
                      />
                    </div>
                  )}
                </div>
              )}

              {/* Sync results */}
              {gmailSyncResults && !gmailSyncProgress && (
                <div className="flex items-center gap-4 p-3 rounded-lg border border-slate-200/15 bg-white">
                  <div className="flex items-center gap-1.5 text-[10px] font-mono">
                    <CheckCircle className="h-3.5 w-3.5 text-emerald-400" />
                    <span className="text-emerald-400">{gmailSyncResults.synced} synced</span>
                  </div>
                  <div className="flex items-center gap-1.5 text-[10px] font-mono">
                    <Clock className="h-3.5 w-3.5 text-slate-400" />
                    <span className="text-slate-400">{gmailSyncResults.skipped} skipped</span>
                  </div>
                  {gmailSyncResults.errors > 0 && (
                    <div className="flex items-center gap-1.5 text-[10px] font-mono">
                      <AlertTriangle className="h-3.5 w-3.5 text-red-400" />
                      <span className="text-red-400">{gmailSyncResults.errors} errors</span>
                    </div>
                  )}
                </div>
              )}
            </>
          ) : (
            <div className="text-center py-6 space-y-3">
              <Mail className="h-10 w-10 text-slate-300 mx-auto" />
              <p className="text-sm text-slate-400 font-mono">Gmail inbox not connected yet.</p>
              <Button
                onClick={() => connectGmail()}
                disabled={!canImport}
                className="bg-blue-600 text-slate-900 hover:bg-blue-600/80 font-bold border-2 border-blue-500 transition-all hover:shadow-lg hover:shadow-blue-500/20 disabled:opacity-50"
              >
                <Mail className="h-4 w-4 mr-2" />
                Connect Gmail
              </Button>
              <p className="text-[10px] text-slate-400 font-mono">
                Emails will be read (read-only), extracted, embedded and available in the RAG knowledge base.
              </p>
            </div>
          )}
        </CardContent>
      </Card>

      <Card className="border border-slate-200 bg-white">
        <CardHeader className="border-b border-slate-200">
          <CardTitle className="text-slate-900 font-mono font-black uppercase tracking-tight">Tracked Sources</CardTitle>
          <CardDescription className="text-slate-500 font-mono">{sources.length} items</CardDescription>
        </CardHeader>
        <CardContent className="space-y-3 text-slate-900">
          {sources.length === 0 ? (
            <div className="text-sm text-slate-500 font-mono">No sources yet. Upload documents or import from Google Drive to create sources.</div>
          ) : (
            sources.map((source) => {
              const relatedDoc = documents.find(
                (d: any) => d.storage_path === source.storage_path || d.title === source.title
              );
              const relatedFolderName = relatedDoc?.folder_id
                ? sourceFolders.find((folder) => folder.id === relatedDoc.folder_id)?.name
                : null;

              return (
                <div key={source.id} className="flex items-center justify-between gap-3 border border-slate-200 rounded-md p-3 hover:border-blue-500 hover:bg-blue-600/5 transition-all bg-white">
                  <div className="space-y-1">
                    <div className="flex items-center gap-2">
                      <Badge variant="outline" className="border-slate-200 text-slate-900 bg-white font-mono">{source.source_type}</Badge>
                      {source.status !== "active" && <Badge variant="outline" className="border-slate-200/50 text-slate-400 bg-white font-mono">{source.status}</Badge>}
                    </div>
                    <div className="font-mono font-bold text-slate-900">{source.title || "Untitled source"}</div>
                    {source.external_url && (
                      <div className="text-xs text-slate-500 font-mono truncate max-w-[420px]">{source.external_url}</div>
                    )}
                    {relatedDoc && (relatedDoc.uploader_email || relatedDoc.uploader_name) && (
                      <div className="text-xs text-slate-500 font-mono">
                        Uploaded by: {relatedDoc.uploader_name || relatedDoc.uploader_email}
                      </div>
                    )}
                    {relatedFolderName && (
                      <div className="text-xs text-slate-500 font-mono">
                        Folder: <span className="text-blue-600">{relatedFolderName}</span>
                      </div>
                    )}
                    {source.notes && (
                      <div className="text-sm text-slate-500 font-mono whitespace-pre-wrap">{source.notes}</div>
                    )}
                    {source.tags && source.tags.length > 0 && (
                      <div className="flex flex-wrap gap-1">
                        {source.tags.map((tag) => (
                          <Badge key={tag} variant="outline" className="text-xs border-slate-200 text-slate-900 bg-white font-mono">
                            {tag}
                          </Badge>
                        ))}
                      </div>
                    )}
                  </div>
                  <div className="flex items-center gap-2">
                    {source.external_url && (
                      <Button variant="outline" size="sm" asChild className="border border-slate-200 bg-white text-slate-900 hover:bg-blue-500/10 hover:border-blue-500 hover:text-blue-600 font-bold">
                        <a href={source.external_url} target="_blank" rel="noreferrer">
                          <Link2 className="h-4 w-4 mr-1" />
                          Open
                        </a>
                      </Button>
                    )}
                    <Button variant="ghost" size="icon" onClick={() => onDeleteSource(source.id)} className="text-slate-500 hover:text-slate-900 hover:bg-blue-500/10">
                      <Trash2 className="h-4 w-4" />
                    </Button>
                  </div>
                </div>
              );
            })
          )}
        </CardContent>
      </Card>

      <Dialog
        open={isFolderDialogOpen}
        onOpenChange={(open) => {
          setIsFolderDialogOpen(open);
          if (!open) {
            setPendingFolderDocs([]);
            setFolderAssignmentIds([]);
          }
        }}
      >
        <DialogContent className="bg-white border border-slate-200 text-slate-900 max-w-lg">
          <DialogHeader>
            <DialogTitle className="text-slate-900 font-mono font-black uppercase tracking-tight flex items-center gap-2">
              <FolderPlus className="h-4 w-4 text-blue-600" />
              Categorize &amp; Assign Folder
            </DialogTitle>
            <DialogDescription className="text-slate-500 font-mono text-xs">
              Pick a category, then select or create a folder for your uploaded documents.
            </DialogDescription>
          </DialogHeader>
          <div className="space-y-4">
            {/* Documents summary */}
            <div className="space-y-1 max-h-[100px] overflow-y-auto border border-slate-200 rounded-md p-2">
              <div className="text-[10px] text-slate-400 font-mono mb-1">
                {pendingFolderDocs.length} document{pendingFolderDocs.length > 1 ? "s" : ""}:
              </div>
              {pendingFolderDocs.map((doc) => (
                <div key={doc.id} className="text-xs text-slate-900 font-mono truncate">
                  вЂў {doc.title || "Untitled document"}
                </div>
              ))}
            </div>

            {/* Step 1: Category selector */}
            <div>
              <Label className="text-[10px] font-mono text-blue-600/80 uppercase tracking-wider mb-1.5 block">
                Step 1 вЂ” Choose Category
              </Label>
              <div className="flex flex-wrap gap-2">
                {FOLDER_CATEGORIES.map((cat) => {
                  const isActive = folderDialogCategory === cat;
                  return (
                    <button
                      key={cat}
                      type="button"
                      onClick={() => setFolderDialogCategory(cat)}
                      className={`px-3 py-1.5 rounded-full text-xs font-mono border-2 transition-all ${
                        isActive
                          ? "border-blue-500 bg-blue-600/15 text-blue-600 font-bold shadow-[0_0_8px_rgba(59,130,246,0.25)]"
                          : "border-slate-300 text-slate-500 hover:border-slate-200/60 hover:text-slate-900"
                      }`}
                    >
                      {cat}
                    </button>
                  );
                })}
              </div>
            </div>

            {/* Step 2: Folder selection */}
            <div>
              <Label className="text-[10px] font-mono text-blue-600/80 uppercase tracking-wider mb-1.5 block">
                Step 2 вЂ” Select or Create Folder
              </Label>

              {/* Quick create inside dialog */}
              <div className="flex items-center gap-2 mb-3">
                <Input
                  value={folderDialogNewName}
                  onChange={(e) => setFolderDialogNewName(e.target.value)}
                  placeholder={`New folder name (${folderDialogCategory})`}
                  className="flex-1 border border-slate-300 bg-white text-slate-900 text-xs placeholder:text-slate-400 h-8"
                  onKeyDown={(e) => {
                    if (e.key === "Enter" && folderDialogNewName.trim()) {
                      e.preventDefault();
                      (async () => {
                        const folder = await onCreateFolder(folderDialogNewName.trim(), folderDialogCategory);
                        if (folder) {
                          setFolderDialogNewName("");
                          setFolderAssignmentIds([folder.id]);
                          toast({ title: "Folder created", description: `"${folder.name}" in ${folderDialogCategory}` });
                        }
                      })();
                    }
                  }}
                />
                <Button
                  size="sm"
                  disabled={!folderDialogNewName.trim()}
                  className="bg-blue-600 text-slate-900 hover:bg-blue-600/80 font-bold h-8 text-xs px-3"
                  onClick={async () => {
                    if (!folderDialogNewName.trim()) return;
                    const folder = await onCreateFolder(folderDialogNewName.trim(), folderDialogCategory);
                    if (folder) {
                      setFolderDialogNewName("");
                      setFolderAssignmentIds([folder.id]);
                      toast({ title: "Folder created", description: `"${folder.name}" in ${folderDialogCategory}` });
                    }
                  }}
                >
                  <Plus className="h-3 w-3 mr-1" />
                  Create
                </Button>
              </div>

              {/* Existing folders filtered by selected category */}
              {(() => {
                const catFolders = sourceFolders.filter((f) => (f.category || "Projects") === folderDialogCategory);
                if (catFolders.length === 0) {
                  return (
                    <div className="text-xs text-slate-400 font-mono border border-slate-200/15 rounded-md p-3 text-center">
                      No folders in &ldquo;{folderDialogCategory}&rdquo; yet. Create one above.
                    </div>
                  );
                }
                return (
                  <div className="space-y-1 max-h-[200px] overflow-y-auto">
                    {catFolders.map((folder) => (
                      <label
                        key={folder.id}
                        className="flex items-center gap-2 text-xs border border-slate-300 px-2 py-1.5 rounded-md cursor-pointer hover:bg-blue-600/5 hover:border-blue-500 transition-colors text-slate-900 font-mono"
                      >
                        <Checkbox
                          checked={folderAssignmentIds.includes(folder.id)}
                          onCheckedChange={(val) => toggleFolderAssignment(folder.id, val === true)}
                          className="border-slate-200 data-[state=checked]:bg-blue-600 data-[state=checked]:border-blue-500"
                        />
                        <Folder className="h-3 w-3 text-slate-500" />
                        <span className="flex-1">{folder.name}</span>
                      </label>
                    ))}
                  </div>
                );
              })()}

              {/* Show other categories' folders collapsed */}
              {sourceFolders.filter((f) => (f.category || "Projects") !== folderDialogCategory).length > 0 && (
                <details className="mt-2">
                  <summary className="text-[10px] font-mono text-slate-400 cursor-pointer hover:text-slate-500">
                    Show other categories
                  </summary>
                  <div className="space-y-2 mt-2">
                    {FOLDER_CATEGORIES.filter((c) => c !== folderDialogCategory).map((cat) => {
                      const catFolders = sourceFolders.filter((f) => (f.category || "Projects") === cat);
                      if (catFolders.length === 0) return null;
                      return (
                        <div key={cat}>
                          <p className="text-[10px] font-mono text-slate-400 uppercase tracking-wider mb-1">{cat}</p>
                          <div className="space-y-1">
                            {catFolders.map((folder) => (
                              <label
                                key={folder.id}
                                className="flex items-center gap-2 text-xs border border-slate-200 px-2 py-1.5 rounded-md cursor-pointer hover:bg-blue-600/5 hover:border-blue-500 transition-colors text-slate-500 font-mono"
                              >
                                <Checkbox
                                  checked={folderAssignmentIds.includes(folder.id)}
                                  onCheckedChange={(val) => toggleFolderAssignment(folder.id, val === true)}
                                  className="border-slate-300 data-[state=checked]:bg-blue-600 data-[state=checked]:border-blue-500"
                                />
                                <Folder className="h-3 w-3 text-slate-400" />
                                <span className="flex-1">{folder.name}</span>
                              </label>
                            ))}
                          </div>
                        </div>
                      );
                    })}
                  </div>
                </details>
              )}
            </div>
          </div>
          <div className="flex items-center justify-end gap-2 pt-2">
            <Button
              variant="outline"
              className="border border-slate-200 bg-white text-slate-900 hover:bg-blue-500/10 hover:border-blue-500 hover:text-blue-600 font-bold"
              onClick={() => {
                setIsFolderDialogOpen(false);
                setPendingFolderDocs([]);
                setFolderAssignmentIds([]);
              }}
            >
              Skip for now
            </Button>
            <Button
              className="bg-blue-600 text-slate-900 hover:bg-blue-600/80 font-bold border-2 border-blue-500 transition-all hover:shadow-lg hover:shadow-blue-500/20 disabled:opacity-50"
              disabled={isAssigningFolders || folderAssignmentIds.length === 0}
              onClick={assignFoldersToDocuments}
            >
              {isAssigningFolders ? <Loader2 className="h-4 w-4 mr-2 animate-spin" /> : <FolderPlus className="h-4 w-4 mr-2" />}
              Assign to folder
            </Button>
          </div>
        </DialogContent>
      </Dialog>

      {/* в”Ђв”Ђ Category Picker Dialog (shown after folder creation / drive sync) в”Ђв”Ђ */}
      <Dialog
        open={categoryPickerOpen}
        onOpenChange={(open) => {
          if (!open) {
            setCategoryPickerOpen(false);
            setCategoryPickerFolders([]);
          }
        }}
      >
        <DialogContent className="bg-white border border-slate-200 text-slate-900 max-w-md">
          <DialogHeader>
            <DialogTitle className="text-slate-900 font-mono font-black uppercase tracking-tight flex items-center gap-2">
              <Folder className="h-4 w-4 text-blue-600" />
              Categorize Folders
            </DialogTitle>
            <DialogDescription className="text-slate-500 font-mono text-xs">
              Assign each folder to a category: Sourcing, Projects, Partners, BD, or Mentors/Corporates.
            </DialogDescription>
          </DialogHeader>
          <div className="space-y-3 max-h-[400px] overflow-y-auto">
            {categoryPickerFolders.map((pf, idx) => (
              <div key={pf.id} className="flex items-center gap-3 border border-slate-200 rounded-md px-3 py-2">
                <Folder className="h-4 w-4 text-blue-600 shrink-0" />
                <span className="text-xs font-mono text-slate-900 flex-1 truncate">{pf.name}</span>
                <Select
                  value={pf.category}
                  onValueChange={(val) => {
                    setCategoryPickerFolders((prev) => prev.map((f, i) => i === idx ? { ...f, category: val } : f));
                  }}
                >
                  <SelectTrigger className="w-[180px] h-7 text-[10px] border-slate-300 bg-white text-slate-900">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent className="bg-[#1a1a2e] border-slate-200">
                    {FOLDER_CATEGORIES.map((cat) => (
                      <SelectItem key={cat} value={cat} className="text-slate-900 font-mono text-xs hover:bg-blue-50 focus:bg-blue-50 cursor-pointer">
                        {cat}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>
            ))}
          </div>
          <div className="flex items-center justify-end gap-2 pt-2">
            <Button
              variant="outline"
              className="border border-slate-200 bg-white text-slate-900 hover:bg-blue-500/10 hover:border-blue-500 hover:text-blue-600 font-bold"
              onClick={() => {
                setCategoryPickerOpen(false);
                setCategoryPickerFolders([]);
              }}
            >
              Skip
            </Button>
            <Button
              className="bg-blue-600 text-slate-900 hover:bg-blue-600/80 font-bold border-2 border-blue-500 transition-all hover:shadow-lg hover:shadow-blue-500/20"
              onClick={async () => {
                await onFoldersCategoriesSaved?.(categoryPickerFolders.map((pf) => ({ id: pf.id, category: pf.category })));
                toast({ title: "Categories saved", description: `Updated ${categoryPickerFolders.length} folder(s).` });
                setCategoryPickerOpen(false);
                setCategoryPickerFolders([]);
              }}
            >
              Save Categories
            </Button>
          </div>
        </DialogContent>
      </Dialog>

      {/* в”Ђв”Ђ Batch Review Dialog в”Ђв”Ђ */}
      <Dialog
        open={batchReviewData !== null && batchReviewData.length > 0}
        onOpenChange={(open) => {
          if (!open) setBatchReviewData(null);
        }}
      >
        <DialogContent className="bg-white border border-slate-200 text-slate-900 max-w-lg">
          <DialogHeader>
            <DialogTitle className="text-slate-900 font-mono font-black uppercase tracking-tight flex items-center gap-2">
              <Sparkles className="h-4 w-4 text-blue-600" />
              Batch Extraction Summary
            </DialogTitle>
            <DialogDescription className="text-slate-500 font-mono">
              Properties auto-extracted from uploaded documents.
              Review in the Companies tab for details.
            </DialogDescription>
          </DialogHeader>
          <div className="space-y-2 max-h-[300px] overflow-y-auto">
            {batchReviewData?.map((item, i) => (
              <div key={i} className="flex items-center gap-3 p-2 rounded-md border border-slate-200 bg-slate-50">
                <Building2 className="h-4 w-4 text-blue-600 flex-shrink-0" />
                <div className="flex-1 min-w-0">
                  <div className="text-sm font-mono font-bold text-slate-900 truncate">{item.companyName}</div>
                </div>
                <div className="flex items-center gap-2 flex-shrink-0">
                  <Badge className="text-[9px] font-mono bg-emerald-500/20 text-emerald-400 border-0">
                    <Check className="h-2.5 w-2.5 mr-0.5" />
                    processed
                  </Badge>
                </div>
              </div>
            ))}
          </div>
          <div className="flex items-center justify-between pt-2">
            <p className="text-[10px] font-mono text-slate-400">
              Switch to the Companies tab to review details and resolve conflicts.
            </p>
            <Button
              className="bg-blue-600 text-slate-900 hover:bg-blue-600/80 font-bold border-2 border-blue-500 transition-all hover:shadow-lg hover:shadow-blue-500/20"
              onClick={() => setBatchReviewData(null)}
            >
              Got it
            </Button>
          </div>
        </DialogContent>
      </Dialog>
    </div>
  );
}

// ============================================================================
// DASHBOARD TAB (task hub: MD assigns tasks, team sees my tasks + Gantt + Decision Engine for MD)
// ============================================================================

type TeamMember = { id: string; email: string | null; full_name: string | null; role: string };

function DashboardTab({
  profile,
  activeEventId,
  currentUserId,
  tasks,
  onRefetchTasks,
  decisions,
  documents,
  sources,
  companyCards,
}: {
  profile: UserProfile | null;
  activeEventId: string | null;
  currentUserId: string | null;
  tasks: Task[];
  onRefetchTasks: () => Promise<void>;
  decisions: Decision[];
  documents: Array<{ id: string; title: string | null; storage_path: string | null }>;
  sources: SourceRecord[];
  companyCards: Array<{ company_id: string; company_name: string; entity_type?: string; company_properties: Record<string, any>; document_count: number }>;
}) {
  const { toast } = useToast();
  const isMD = profile?.role === "managing_partner" || profile?.role === "organizer";
  const isLP = profile?.role === "lp";
  const orgId = profile?.organization_id;

  const [addTaskOpen, setAddTaskOpen] = useState(false);
  const [addTaskTitle, setAddTaskTitle] = useState("");
  const [addTaskDescription, setAddTaskDescription] = useState("");
  const [addTaskStartDate, setAddTaskStartDate] = useState("");
  const [addTaskDeadline, setAddTaskDeadline] = useState("");
  const [addTaskAssignee, setAddTaskAssignee] = useState<string>("");
  const [addTaskPriority, setAddTaskPriority] = useState<"low" | "medium" | "high">("medium");
  const [addTaskSaving, setAddTaskSaving] = useState(false);
  const [filterAssignee, setFilterAssignee] = useState<string>("all");
  const [filterStatus, setFilterStatus] = useState<string>("all");
  const [selectedTask, setSelectedTask] = useState<Task | null>(null);
  const [teamMembers, setTeamMembers] = useState<TeamMember[]>([]);
  const [statusUpdatingId, setStatusUpdatingId] = useState<string | null>(null);
  const [viewMode, setViewMode] = useState<"list" | "gantt">("list");
  const [statusNote, setStatusNote] = useState<Record<string, string>>({});
  const [taskToDelete, setTaskToDelete] = useState<Task | null>(null);
  const [deletingTaskId, setDeletingTaskId] = useState<string | null>(null);
  const [dashboardSection, setDashboardSection] = useState<"tasks" | "analytics">("tasks");
  const [ganttRange, setGanttRange] = useState<4 | 8 | 12>(8);
  const [ganttGroupBy, setGanttGroupBy] = useState<"none" | "assignee">("none");
  const [extraProfiles, setExtraProfiles] = useState<Record<string, { full_name?: string | null; email?: string | null }>>({});

  const myTasks = useMemo(
    () => (currentUserId ? tasks.filter((t) => t.assignee_user_id === currentUserId) : []),
    [tasks, currentUserId]
  );
  const filteredTasks = useMemo(() => {
    let list = isMD ? tasks : myTasks;
    if (filterAssignee !== "all") list = list.filter((t) => t.assignee_user_id === filterAssignee);
    if (filterStatus !== "all") list = list.filter((t) => t.status === filterStatus);
    return list;
  }, [isMD, tasks, myTasks, filterAssignee, filterStatus]);

  useEffect(() => {
    if (!orgId) return;
    supabase
      .from("user_profiles")
      .select("id, email, full_name, role")
      .eq("organization_id", orgId)
      .order("created_at", { ascending: false })
      .then(({ data }) => setTeamMembers((data as TeamMember[]) || []));
  }, [orgId]);

  // Resolve "Created by" / assignee when not in teamMembers (e.g. RLS or timing)
  useEffect(() => {
    const ids = new Set<string>();
    tasks.forEach((t) => {
      if (t.created_by) ids.add(t.created_by);
      if (t.assignee_user_id) ids.add(t.assignee_user_id);
    });
    const teamIds = new Set(teamMembers.map((m) => m.id));
    const currentId = profile?.id;
    const missing = [...ids].filter((id) => !teamIds.has(id) && id !== currentId);
    if (missing.length === 0) return;
    supabase
      .from("user_profiles")
      .select("id, full_name, email")
      .in("id", missing)
      .then(({ data }) => {
        if (!data?.length) return;
        const byId: Record<string, { full_name?: string | null; email?: string | null }> = {};
        (data as { id: string; full_name?: string | null; email?: string | null }[]).forEach((row) => {
          byId[row.id] = { full_name: row.full_name, email: row.email };
        });
        setExtraProfiles((prev) => ({ ...prev, ...byId }));
      });
  }, [tasks, teamMembers, profile?.id]);

  const stats = useMemo(() => calculateDecisionStats(decisions), [decisions]);
  const latestDecision = decisions[0];
  const latestDocument = documents[0];
  const latestSource = sources[0];

  const handleCreateTask = useCallback(async () => {
    if (!activeEventId || !currentUserId || !addTaskTitle.trim()) {
      toast({ title: "Missing fields", description: "Title is required.", variant: "destructive" });
      return;
    }
    setAddTaskSaving(true);
    try {
      const { data, error } = await insertTask(activeEventId, {
        assignee_user_id: addTaskAssignee || null,
        title: addTaskTitle.trim(),
        description: addTaskDescription.trim() || null,
        start_date: addTaskStartDate ? new Date(addTaskStartDate).toISOString() : null,
        deadline: addTaskDeadline ? new Date(addTaskDeadline).toISOString() : null,
        created_by: currentUserId,
      });
      if (error) throw error;
      toast({ title: "Task created" });
      setAddTaskOpen(false);
      setAddTaskTitle("");
      setAddTaskDescription("");
      setAddTaskStartDate("");
      setAddTaskDeadline("");
      setAddTaskAssignee("");
      setAddTaskPriority("medium");
      await onRefetchTasks();
    } catch (e: any) {
      const msg = e?.message || "Unknown error";
      const isNoTable = /relation.*tasks.*does not exist|schema cache|404/i.test(msg);
      toast({
        title: isNoTable ? "Tasks table not found" : "Failed to create task",
        description: isNoTable
          ? "The tasks table is missing. Run the migration in Supabase: Dashboard в†’ SQL Editor, or run supabase db push."
          : msg,
        variant: "destructive",
      });
    } finally {
      setAddTaskSaving(false);
    }
  }, [activeEventId, currentUserId, addTaskTitle, addTaskDescription, addTaskStartDate, addTaskDeadline, addTaskAssignee, onRefetchTasks, toast]);

  const handleUpdateStatus = useCallback(
    async (taskId: string, status: Task["status"], note?: string) => {
      setStatusUpdatingId(taskId);
      try {
        const { error } = await updateTaskStatus(taskId, status, note ?? undefined);
        if (error) throw error;
        toast({ title: `Marked ${status === "done" ? "Done" : status === "cancelled" ? "Cancelled" : status === "in_progress" ? "In progress" : "Not started"}` });
        await onRefetchTasks();
      } catch (e: any) {
        toast({ title: "Update failed", description: e?.message, variant: "destructive" });
      } finally {
        setStatusUpdatingId(null);
      }
    },
    [onRefetchTasks, toast]
  );

  const handleDeleteTask = useCallback(async () => {
    if (!taskToDelete) return;
    setDeletingTaskId(taskToDelete.id);
    try {
      const { error } = await deleteTask(taskToDelete.id);
      if (error) throw error;
      toast({ title: "Task deleted" });
      setTaskToDelete(null);
      await onRefetchTasks();
    } catch (e: any) {
      toast({ title: "Delete failed", description: e?.message, variant: "destructive" });
    } finally {
      setDeletingTaskId(null);
    }
  }, [taskToDelete, onRefetchTasks, toast]);

  const displayName = (userId: string | null) => {
    if (!userId) return "Unassigned";
    const m = teamMembers.find((x) => x.id === userId);
    if (m) return m.full_name || m.email || "Team member";
    const extra = extraProfiles[userId];
    if (extra) return extra.full_name || extra.email || "Team member";
    if (profile?.id === userId) return profile.full_name || profile.email || "You";
    return "Team member";
  };

  const now = Date.now();
  const ganttWeeks = ganttRange;
  const weekMs = 7 * 24 * 60 * 60 * 1000;
  const ganttStart = new Date(now);
  ganttStart.setHours(0, 0, 0, 0);
  const ganttStartMs = ganttStart.getTime();

  // LP Dashboard: portfolio-level KPIs only
  if (isLP) {
    const portfolioCount = companyCards.filter((c) => c.entity_type === "company" || !c.entity_type).length;
    const fundCount = companyCards.filter((c) => c.entity_type === "fund").length;
    const lastUpdated = documents.length > 0 ? "Recently" : "No data yet";
    return (
      <div className="space-y-6">
        <div className="flex items-center gap-3 mb-2">
          <div className="p-2 border-2 border-blue-500 rounded-lg bg-white">
            <Building2 className="h-5 w-5 text-blue-600" />
          </div>
          <div>
            <h2 className="text-lg font-mono font-black text-slate-900 uppercase tracking-tight">Overview Dashboard</h2>
            <p className="text-xs text-slate-500 font-mono">Read-only overview for viewers</p>
          </div>
        </div>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <Card className="border border-slate-200 bg-white">
            <CardContent className="pt-4 text-slate-900">
              <div className="flex items-center gap-3">
                <div className="p-2 border border-slate-200 rounded-lg bg-white">
                  <Building2 className="h-5 w-5 text-blue-600" />
                </div>
                <div>
                  <p className="text-2xl font-mono font-black">{portfolioCount}</p>
                  <p className="text-xs text-slate-500 font-mono uppercase tracking-wider">Projects</p>
                </div>
              </div>
            </CardContent>
          </Card>
          <Card className="border border-slate-200 bg-white">
            <CardContent className="pt-4 text-slate-900">
              <div className="flex items-center gap-3">
                <div className="p-2 border border-slate-200 rounded-lg bg-white">
                  <Briefcase className="h-5 w-5 text-blue-600" />
                </div>
                <div>
                  <p className="text-2xl font-mono font-black">{fundCount}</p>
                  <p className="text-xs text-slate-500 font-mono uppercase tracking-wider">Funds</p>
                </div>
              </div>
            </CardContent>
          </Card>
          <Card className="border border-slate-200 bg-white">
            <CardContent className="pt-4 text-slate-900">
              <div className="flex items-center gap-3">
                <div className="p-2 border border-slate-200 rounded-lg bg-white">
                  <FileText className="h-5 w-5 text-blue-600" />
                </div>
                <div>
                  <p className="text-2xl font-mono font-black">{documents.length}</p>
                  <p className="text-xs text-slate-500 font-mono uppercase tracking-wider">Documents</p>
                </div>
              </div>
            </CardContent>
          </Card>
          <Card className="border border-slate-200 bg-white">
            <CardContent className="pt-4 text-slate-900">
              <div className="flex items-center gap-3">
                <div className="p-2 border-2 border-blue-500 rounded-lg bg-white">
                  <Clock className="h-5 w-5 text-blue-600" />
                </div>
                <div>
                  <p className="text-2xl font-mono font-black">{lastUpdated}</p>
                  <p className="text-xs text-slate-500 font-mono uppercase tracking-wider">Last Updated</p>
                </div>
              </div>
            </CardContent>
          </Card>
        </div>
        <Card className="border border-slate-200 bg-white">
          <CardHeader className="border-b border-slate-200">
            <CardTitle className="text-slate-900 font-mono font-black uppercase tracking-tight flex items-center gap-2">
              <Building2 className="h-5 w-5 text-blue-600" />
              Projects
            </CardTitle>
          </CardHeader>
          <CardContent className="pt-4">
            {companyCards.filter((c) => c.entity_type === "company" || !c.entity_type).length === 0 ? (
              <p className="text-slate-500 font-mono text-sm py-4">No projects yet.</p>
            ) : (
              <div className="grid gap-3">
                {companyCards.filter((c) => c.entity_type === "company" || !c.entity_type).map((card) => (
                  <div key={card.company_id} className="flex items-center justify-between p-3 border border-slate-200 rounded-lg bg-slate-50">
                    <div>
                      <div className="font-mono font-bold text-slate-900">{card.company_name}</div>
                      <div className="flex items-center gap-2 mt-1">
                        {card.company_properties?.industry && (
                          <Badge variant="outline" className="text-xs border-slate-300 text-slate-500">{card.company_properties.industry}</Badge>
                        )}
                        {card.company_properties?.funding_stage && (
                          <Badge variant="outline" className="text-xs border-blue-500/40 text-blue-600">{card.company_properties.funding_stage}</Badge>
                        )}
                      </div>
                    </div>
                    <div className="text-right">
                      {card.company_properties?.arr && (
                        <div className="text-sm font-mono text-blue-600 font-bold">{card.company_properties.arr}</div>
                      )}
                      <div className="text-xs text-slate-400 font-mono">{card.document_count} docs</div>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </CardContent>
        </Card>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Section switcher: Tasks (all) / Analytics (MD only) */}
      {isMD && (
        <div className="flex items-center gap-1 border-b border-slate-200/20 pb-2">
          <button
            onClick={() => setDashboardSection("tasks")}
            className={`px-4 py-2 rounded-t-lg text-sm font-mono font-bold transition-all ${
              dashboardSection === "tasks" ? "bg-blue-600/15 text-blue-600 border-b-2 border-blue-500" : "text-slate-500 hover:text-slate-900"
            }`}
          >
            <ListTodo className="h-4 w-4 inline mr-1.5 -mt-0.5" />
            Tasks & Overview
          </button>
          <button
            onClick={() => setDashboardSection("analytics")}
            className={`px-4 py-2 rounded-t-lg text-sm font-mono font-bold transition-all ${
              dashboardSection === "analytics" ? "bg-blue-600/15 text-blue-600 border-b-2 border-blue-500" : "text-slate-500 hover:text-slate-900"
            }`}
          >
            <BarChart3 className="h-4 w-4 inline mr-1.5 -mt-0.5" />
            Decision Analytics
          </button>
        </div>
      )}

      {/* Decision Analytics section (MD only) */}
      {isMD && dashboardSection === "analytics" && (
        <DecisionEngineDashboardTab decisions={decisions} />
      )}

      {/* Tasks & Overview section (shown for everyone when not on analytics) */}
      {(dashboardSection === "tasks" || !isMD) && (
      <>
      {/* Summary stats */}
      <div className="grid grid-cols-2 md:grid-cols-5 gap-4">
        <Card className="border border-slate-200 bg-white">
          <CardContent className="pt-4 text-slate-900">
            <div className="flex items-center gap-3">
              <div className="p-2 border border-slate-200 rounded-lg bg-white">
                <ClipboardList className="h-5 w-5 text-blue-600" />
              </div>
              <div>
                <p className="text-2xl font-mono font-black">{stats.totalDecisions}</p>
                <p className="text-xs text-slate-500 font-mono uppercase tracking-wider">Decisions</p>
              </div>
            </div>
          </CardContent>
        </Card>
        <Card className="border border-slate-200 bg-white">
          <CardContent className="pt-4 text-slate-900">
            <div className="flex items-center gap-3">
              <div className="p-2 border border-slate-200 rounded-lg bg-white">
                <FileText className="h-5 w-5 text-blue-600" />
              </div>
              <div>
                <p className="text-2xl font-mono font-black">{documents.length}</p>
                <p className="text-xs text-slate-500 font-mono uppercase tracking-wider">Documents</p>
              </div>
            </div>
          </CardContent>
        </Card>
        <Card className="border border-slate-200 bg-white">
          <CardContent className="pt-4 text-slate-900">
            <div className="flex items-center gap-3">
              <div className="p-2 border border-slate-200 rounded-lg bg-white">
                <Folder className="h-5 w-5 text-blue-600" />
              </div>
              <div>
                <p className="text-2xl font-mono font-black">{sources.length}</p>
                <p className="text-xs text-slate-500 font-mono uppercase tracking-wider">Sources</p>
              </div>
            </div>
          </CardContent>
        </Card>
        <Card className="border border-slate-200 bg-white">
          <CardContent className="pt-4 text-slate-900">
            <div className="flex items-center gap-3">
              <div className="p-2 border-2 border-blue-500 rounded-lg bg-white">
                <ListTodo className="h-5 w-5 text-blue-600" />
              </div>
              <div>
                <p className="text-2xl font-mono font-black">{isMD ? tasks.length : myTasks.length}</p>
                <p className="text-xs text-slate-500 font-mono uppercase tracking-wider">Tasks</p>
              </div>
            </div>
          </CardContent>
        </Card>
        <Card className="border border-slate-200 bg-white">
          <CardContent className="pt-4 text-slate-900">
            <div className="flex items-center gap-3">
              <div className="p-2 border border-slate-200 rounded-lg bg-white">
                <Building2 className="h-5 w-5 text-blue-600" />
              </div>
              <div>
                <p className="text-2xl font-mono font-black">{companyCards.length}</p>
                <p className="text-xs text-slate-500 font-mono uppercase tracking-wider">Companies</p>
              </div>
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Task hub */}
      <Card className="border border-slate-200 bg-white">
        <CardHeader className="border-b border-slate-200 flex flex-row items-center justify-between gap-4 py-3">
          <CardTitle className="text-slate-900 font-mono font-black uppercase tracking-tight flex items-center gap-2 text-base">
            <ListTodo className="h-5 w-5 text-blue-600" />
            {isMD ? "All tasks" : "My tasks"}
          </CardTitle>
          <div className="flex items-center gap-3">
            {/* Segmented toggle for List / Gantt */}
            <div className="flex items-center rounded-lg border border-slate-200/30 overflow-hidden">
              <button
                onClick={() => setViewMode("list")}
                className={`flex items-center gap-1.5 px-3 py-1.5 text-xs font-mono font-bold uppercase tracking-wider transition-all ${
                  viewMode === "list"
                    ? "bg-blue-600 text-slate-900"
                    : "bg-white text-slate-500 hover:text-slate-900 hover:bg-blue-500/10"
                }`}
              >
                <ListTodo className="h-3.5 w-3.5" />
                List
              </button>
              <div className="w-px h-5 bg-white/20" />
              <button
                onClick={() => setViewMode("gantt")}
                className={`flex items-center gap-1.5 px-3 py-1.5 text-xs font-mono font-bold uppercase tracking-wider transition-all ${
                  viewMode === "gantt"
                    ? "bg-blue-600 text-slate-900"
                    : "bg-white text-slate-500 hover:text-slate-900 hover:bg-blue-500/10"
                }`}
              >
                <GanttChart className="h-3.5 w-3.5" />
                Gantt
              </button>
            </div>
            {isMD && activeEventId && (
              <Button
                size="sm"
                onClick={() => setAddTaskOpen(true)}
                className="bg-blue-600 text-slate-900 hover:bg-blue-600/80 font-mono font-bold uppercase tracking-wider text-xs border-2 border-blue-500 transition-all hover:shadow-[0_0_16px_rgba(59,130,246,0.4)]"
              >
                <Plus className="h-3.5 w-3.5 mr-1.5" />
                Add task
              </Button>
            )}
          </div>
        </CardHeader>
        <CardContent className="pt-4">
          {isMD && (
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-4">
              <div>
                <Label className="text-slate-900 font-mono text-xs uppercase tracking-wider">Assignee</Label>
                <Select value={filterAssignee} onValueChange={setFilterAssignee}>
                  <SelectTrigger className="border border-slate-200 bg-white text-slate-900 mt-1">
                    <SelectValue placeholder="All" />
                  </SelectTrigger>
                  <SelectContent className="bg-white border border-slate-200 shadow-lg rounded-md">
                    <SelectItem value="all" className="text-slate-900">All</SelectItem>
                    {teamMembers.map((m) => (
                      <SelectItem key={m.id} value={m.id} className="text-slate-900">
                        {m.full_name || m.email || m.id.slice(0, 8)}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>
              <div>
                <Label className="text-slate-900 font-mono text-xs uppercase tracking-wider">Status</Label>
                <Select value={filterStatus} onValueChange={setFilterStatus}>
                  <SelectTrigger className="border border-slate-200 bg-white text-slate-900 mt-1">
                    <SelectValue placeholder="All" />
                  </SelectTrigger>
                  <SelectContent className="bg-white border border-slate-200 shadow-lg rounded-md">
                    <SelectItem value="all" className="text-slate-900">All</SelectItem>
                    <SelectItem value="not_started" className="text-slate-900">Not started</SelectItem>
                    <SelectItem value="in_progress" className="text-slate-900">In progress</SelectItem>
                    <SelectItem value="done" className="text-slate-900">Done</SelectItem>
                    <SelectItem value="cancelled" className="text-slate-900">Cancelled</SelectItem>
                  </SelectContent>
                </Select>
              </div>
            </div>
          )}

          {viewMode === "gantt" ? (
            <div className="space-y-3">
              {/* Gantt controls: time range + group by + legend */}
              <div className="flex items-center justify-between gap-3 flex-wrap">
                <div className="flex items-center gap-3 flex-wrap">
                  <div className="flex items-center gap-1.5">
                    <span className="text-xs text-slate-500 font-mono uppercase">Range:</span>
                    {([4, 8, 12] as const).map((w) => (
                      <button
                        key={w}
                        onClick={() => setGanttRange(w)}
                        className={`px-2 py-1 rounded text-xs font-mono ${ganttRange === w ? "bg-blue-600/20 text-blue-600 border border-blue-500" : "text-slate-500 border border-slate-200 hover:border-slate-300"}`}
                      >
                        {w}w
                      </button>
                    ))}
                  </div>
                  {isMD && (
                    <div className="flex items-center gap-1.5">
                      <span className="text-xs text-slate-500 font-mono uppercase">Group:</span>
                      <button
                        onClick={() => setGanttGroupBy("none")}
                        className={`px-2 py-1 rounded text-xs font-mono ${ganttGroupBy === "none" ? "bg-blue-600/20 text-blue-600 border border-blue-500" : "text-slate-500 border border-slate-200 hover:border-slate-300"}`}
                      >
                        Flat
                      </button>
                      <button
                        onClick={() => setGanttGroupBy("assignee")}
                        className={`px-2 py-1 rounded text-xs font-mono ${ganttGroupBy === "assignee" ? "bg-blue-600/20 text-blue-600 border border-blue-500" : "text-slate-500 border border-slate-200 hover:border-slate-300"}`}
                      >
                        By Assignee
                      </button>
                    </div>
                  )}
                </div>
                {/* Legend */}
                <div className="flex items-center gap-3 flex-wrap">
                  {[
                    { label: "Not started", color: "#6b7280" },
                    { label: "In progress", color: "#3b82f6" },
                    { label: "Done", color: "#22c55e" },
                    { label: "Cancelled", color: "#ef4444" },
                    { label: "Overdue", color: "#f97316" },
                  ].map((item) => (
                    <div key={item.label} className="flex items-center gap-1.5">
                      <div className="w-3 h-3 rounded-sm" style={{ backgroundColor: item.color }} />
                      <span className="text-[10px] text-slate-500 font-mono">{item.label}</span>
                    </div>
                  ))}
                </div>
              </div>
              <div className="overflow-x-auto">
                <div className="min-w-[700px]">
                  {/* Gantt header */}
                  <div className="grid font-mono text-xs text-slate-500 border-b border-slate-300 pb-2 mb-2" style={{ gridTemplateColumns: isMD ? "220px 1fr" : "180px 1fr" }}>
                    <div>{isMD ? "Task / Assignee" : "Task"}</div>
                    <div className="flex">
                      {Array.from({ length: ganttWeeks }, (_, i) => {
                        const d = new Date(ganttStartMs + i * weekMs);
                        return (
                          <div key={i} className="flex-1 text-center min-w-[56px] relative">
                            {d.toLocaleDateString(undefined, { month: "short", day: "numeric" })}
                          </div>
                        );
                      })}
                    </div>
                  </div>
                  {/* Today marker info */}
                  <div className="text-[10px] text-slate-400 font-mono mb-1">Today: {new Date().toLocaleDateString(undefined, { weekday: "short", month: "short", day: "numeric" })}</div>
                  {/* Gantt rows */}
                  {(() => {
                    const totalMs = ganttWeeks * weekMs;
                    const todayLeft = Math.max(0, Math.min(100, ((now - ganttStartMs) / totalMs) * 100));
                    const ganttStatusColor = (task: Task) => {
                      const deadlineMs = task.deadline ? new Date(task.deadline).getTime() : null;
                      const isOverdue = deadlineMs && deadlineMs < now && task.status !== "done" && task.status !== "cancelled";
                      if (task.status === "cancelled") return { bg: "#ef4444", border: "#dc2626", opacity: 0.5 };
                      if (isOverdue) return { bg: "#f97316", border: "#ea580c", opacity: 0.95 };
                      if (task.status === "done") return { bg: "#22c55e", border: "#16a34a", opacity: 0.9 };
                      if (task.status === "in_progress") return { bg: "#3b82f6", border: "#d4c800", opacity: 0.9 };
                      return { bg: "#6b7280", border: "#4b5563", opacity: 0.75 };
                    };
                    const renderRow = (task: Task) => {
                      const taskStartMs = task.start_date ? new Date(task.start_date).getTime() : (task.created_at ? new Date(task.created_at).getTime() : now);
                      const deadlineMs = task.deadline ? new Date(task.deadline).getTime() : null;
                      const effectiveEnd = deadlineMs ?? taskStartMs + 2 * weekMs;
                      const barStart = Math.max(taskStartMs, ganttStartMs);
                      const barEnd = Math.min(effectiveEnd, ganttStartMs + totalMs);
                      if (barEnd <= barStart) {
                        // Task is entirely outside the visible range вЂ” show thin marker at left or right edge
                        const edgeLeft = taskStartMs < ganttStartMs ? 0 : 99.5;
                        const { bg, border: borderColor } = ganttStatusColor(task);
                        return (
                          <div key={task.id} className="grid py-1.5 items-center border-b border-slate-200 font-mono text-sm" style={{ gridTemplateColumns: isMD ? "220px 1fr" : "180px 1fr" }}>
                            <div className="text-slate-900 truncate pr-2" title={task.title}>
                              <span className={task.status === "cancelled" ? "line-through text-slate-400" : ""}>{task.title}</span>
                              {isMD && <span className="block text-xs text-slate-400 truncate">{displayName(task.assignee_user_id)}</span>}
                            </div>
                            <div className="relative h-7">
                              <div className="absolute top-0.5 h-6 w-1 rounded" style={{ left: `${edgeLeft}%`, backgroundColor: bg, borderLeft: `2px solid ${borderColor}` }} title={`${task.title}: outside visible range`} />
                              <div className="absolute top-0 bottom-0 w-px bg-blue-400/40" style={{ left: `${todayLeft}%` }} />
                            </div>
                          </div>
                        );
                      }
                      const left = Math.max(0, ((barStart - ganttStartMs) / totalMs) * 100);
                      const width = Math.max(1.5, ((barEnd - barStart) / totalMs) * 100);
                      const { bg, border: borderColor, opacity } = ganttStatusColor(task);
                      const statusLabel = task.status === "not_started" ? "Not started" : task.status === "in_progress" ? "In progress" : task.status === "done" ? "Done" : "Cancelled";
                      const tooltipText = `${task.title}\nStatus: ${statusLabel}\n${task.start_date ? "Start: " + new Date(task.start_date).toLocaleDateString() : "No start date"}${task.deadline ? "\nDeadline: " + new Date(task.deadline).toLocaleDateString() : "\nNo deadline"}${task.assignee_user_id ? "\nAssignee: " + displayName(task.assignee_user_id) : ""}`;
                      return (
                        <div key={task.id} className="grid py-1.5 items-center border-b border-slate-200 font-mono text-sm" style={{ gridTemplateColumns: isMD ? "220px 1fr" : "180px 1fr" }}>
                          <div className="text-slate-900 truncate pr-2" title={task.title}>
                            <span className={task.status === "cancelled" ? "line-through text-slate-400" : ""}>{task.title}</span>
                            {isMD && <span className="block text-xs text-slate-400 truncate">{displayName(task.assignee_user_id)}</span>}
                          </div>
                          <div className="relative h-7">
                            {/* Today line */}
                            <div className="absolute top-0 bottom-0 w-px bg-blue-400/40 z-10" style={{ left: `${todayLeft}%` }} />
                            {/* Task bar */}
                            <div
                              className="absolute top-0.5 h-6 rounded-md transition-all cursor-default"
                              style={{ left: `${left}%`, width: `${width}%`, backgroundColor: bg, border: `1.5px solid ${borderColor}`, opacity }}
                              title={tooltipText}
                            >
                              {width > 8 && (
                                <span className="absolute inset-0 flex items-center justify-center text-[9px] font-mono font-bold truncate px-1" style={{ color: bg === "#3b82f6" ? "#000" : "#fff" }}>
                                  {task.title.length > 18 ? task.title.slice(0, 16) + ".." : task.title}
                                </span>
                              )}
                            </div>
                          </div>
                        </div>
                      );
                    };

                    if (ganttGroupBy === "assignee" && isMD) {
                      const grouped = new Map<string, Task[]>();
                      for (const t of filteredTasks) {
                        const key = t.assignee_user_id || "__unassigned__";
                        if (!grouped.has(key)) grouped.set(key, []);
                        grouped.get(key)!.push(t);
                      }
                      return Array.from(grouped.entries()).map(([userId, groupTasks]) => (
                        <div key={userId} className="mb-3">
                          <div className="text-xs font-mono text-blue-600 uppercase tracking-wider mb-1 border-b border-blue-500/30 pb-1">
                            {userId === "__unassigned__" ? "Unassigned" : displayName(userId)}
                          </div>
                          {groupTasks.map(renderRow)}
                        </div>
                      ));
                    }
                    return filteredTasks.map(renderRow);
                  })()}
                  {filteredTasks.length === 0 && (
                    <div className="text-slate-400 font-mono text-sm py-8 text-center">
                      {isMD ? "No tasks. Add one to see the timeline." : "No tasks assigned to you."}
                    </div>
                  )}
                </div>
              </div>
            </div>
          ) : (
            <div className="space-y-2">
              {filteredTasks.length === 0 ? (
                <p className="text-slate-500 font-mono text-sm">No tasks {isMD ? "" : "assigned to you"}.</p>
              ) : (
                filteredTasks.map((task) => {
                  const statusBadgeClass =
                    task.status === "done" ? "border-green-500/50 text-green-400 bg-green-500/10" :
                    task.status === "in_progress" ? "border-blue-500/50 text-blue-600 bg-blue-600/10" :
                    task.status === "cancelled" ? "border-red-500/50 text-red-400 bg-red-500/10" :
                    "border-slate-300 text-slate-500 bg-slate-50";
                  const statusLabel =
                    task.status === "not_started" ? "Not started" :
                    task.status === "in_progress" ? "In progress" :
                    task.status === "done" ? "Done" : "Cancelled";
                  const isTerminal = task.status === "done" || task.status === "cancelled";
                  const deadlineMs = task.deadline ? new Date(task.deadline).getTime() : null;
                  const isOverdue = deadlineMs && deadlineMs < now && !isTerminal;
                  const priorityDot = (task as any).priority === "high" ? "bg-red-400" : (task as any).priority === "low" ? "bg-blue-400" : "bg-yellow-400";
                  return (
                    <div
                      key={task.id}
                      onClick={() => setSelectedTask(task)}
                      className={`flex items-center gap-3 p-3 border rounded-lg cursor-pointer transition-all group ${
                        isOverdue ? "border-orange-500/40 bg-orange-500/5 hover:border-orange-500/60" :
                        task.status === "cancelled" ? "border-slate-200 bg-white opacity-60 hover:opacity-80" :
                        task.status === "done" ? "border-green-500/20 bg-green-500/[0.03] hover:border-green-500/40" :
                        "border-slate-200/15 bg-white hover:border-slate-300 hover:bg-white/[0.06]"
                      }`}
                    >
                      {/* Priority dot */}
                      <div className={`w-2 h-2 rounded-full shrink-0 ${priorityDot}`} title="Priority" />
                      {/* Main info */}
                      <div className="flex-1 min-w-0">
                        <div className="flex items-center gap-2">
                          <span className={`font-mono font-bold text-sm truncate ${task.status === "cancelled" ? "text-slate-400 line-through" : "text-slate-900"}`}>{task.title}</span>
                        </div>
                        <div className="flex items-center gap-2 mt-1">
                          <Badge variant="outline" className={`text-[10px] px-1.5 py-0 h-5 ${statusBadgeClass}`}>{statusLabel}</Badge>
                          {isMD && <span className="text-[10px] text-slate-400 font-mono">{displayName(task.assignee_user_id)}</span>}
                          {task.deadline && (
                            <span className={`text-[10px] font-mono flex items-center gap-1 ${isOverdue ? "text-orange-400 font-bold" : "text-slate-400"}`}>
                              {isOverdue && <AlertTriangle className="h-2.5 w-2.5" />}
                              <CalendarIcon className="h-2.5 w-2.5" />
                              {format(new Date(task.deadline), "MMM d")}
                            </span>
                          )}
                        </div>
                      </div>
                      {/* Quick action chevron */}
                      <ChevronRight className="h-4 w-4 text-slate-900/20 group-hover:text-slate-400 shrink-0 transition-colors" />
                    </div>
                  );
                })
              )}
            </div>
          )}
        </CardContent>
      </Card>

      {/* Add task dialog (MD only) */}
      <Dialog open={addTaskOpen} onOpenChange={setAddTaskOpen}>
        <DialogContent className="border border-slate-200 bg-white text-slate-900 max-w-lg">
          <DialogHeader>
            <DialogTitle className="font-mono font-black uppercase tracking-tight flex items-center gap-2">
              <Plus className="h-5 w-5 text-blue-600" />
              New Task
            </DialogTitle>
            <DialogDescription className="text-slate-500 font-mono text-xs">Create and assign a task to your team.</DialogDescription>
          </DialogHeader>
          <div className="space-y-4 pt-1">
            <div>
              <Label className="text-slate-600 font-mono text-xs uppercase tracking-wider">Title *</Label>
              <Input
                className="border border-slate-300 bg-slate-50 text-slate-900 mt-1.5 font-mono placeholder:text-slate-300 focus:border-blue-500 transition-colors"
                value={addTaskTitle}
                onChange={(e) => setAddTaskTitle(e.target.value)}
                placeholder="e.g. Review Q4 financials for TechCorp"
              />
            </div>
            <div className="grid grid-cols-2 gap-3">
              <div>
                <Label className="text-slate-600 font-mono text-xs uppercase tracking-wider">Assignee</Label>
                <Select value={addTaskAssignee || "unassigned"} onValueChange={(v) => setAddTaskAssignee(v === "unassigned" ? "" : v)}>
                  <SelectTrigger className="border border-slate-300 bg-slate-50 text-slate-900 mt-1.5 font-mono focus:border-blue-500">
                    <SelectValue placeholder="Select assignee" />
                  </SelectTrigger>
                  <SelectContent className="bg-[#0a0a0a] border border-slate-300">
                    <SelectItem value="unassigned" className="text-slate-500 font-mono">Unassigned</SelectItem>
                    {teamMembers.map((m) => (
                      <SelectItem key={m.id} value={m.id} className="text-slate-900 font-mono">
                        {m.full_name || m.email || m.id.slice(0, 8)}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>
              <div>
                <Label className="text-slate-600 font-mono text-xs uppercase tracking-wider">Priority</Label>
                <Select value={addTaskPriority} onValueChange={(v: "low" | "medium" | "high") => setAddTaskPriority(v)}>
                  <SelectTrigger className="border border-slate-300 bg-slate-50 text-slate-900 mt-1.5 font-mono focus:border-blue-500">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent className="bg-[#0a0a0a] border border-slate-300">
                    <SelectItem value="low" className="text-slate-900 font-mono"><span className="flex items-center gap-2"><span className="w-2 h-2 rounded-full bg-blue-400" />Low</span></SelectItem>
                    <SelectItem value="medium" className="text-slate-900 font-mono"><span className="flex items-center gap-2"><span className="w-2 h-2 rounded-full bg-yellow-400" />Medium</span></SelectItem>
                    <SelectItem value="high" className="text-slate-900 font-mono"><span className="flex items-center gap-2"><span className="w-2 h-2 rounded-full bg-red-400" />High</span></SelectItem>
                  </SelectContent>
                </Select>
              </div>
            </div>
            <div className="grid grid-cols-2 gap-3">
              <div>
                <Label className="text-slate-600 font-mono text-xs uppercase tracking-wider">Start Date</Label>
                <Popover>
                  <PopoverTrigger asChild>
                    <Button variant="outline" className={`w-full mt-1.5 justify-start text-left font-mono border border-slate-300 bg-slate-50 hover:bg-blue-500/10 hover:border-blue-500 ${addTaskStartDate ? "text-slate-900" : "text-slate-400"}`}>
                      <CalendarIcon className="mr-2 h-4 w-4 text-slate-400" />
                      {addTaskStartDate ? format(new Date(addTaskStartDate), "MMM d, yyyy") : "Pick a date"}
                    </Button>
                  </PopoverTrigger>
                  <PopoverContent className="w-auto p-0 bg-[#0a0a0a] border border-slate-300" align="start">
                    <CalendarPicker
                      mode="single"
                      selected={addTaskStartDate ? new Date(addTaskStartDate) : undefined}
                      onSelect={(date) => setAddTaskStartDate(date ? date.toISOString() : "")}
                      className="text-slate-900"
                      classNames={{
                        day_selected: "bg-blue-600 text-slate-900 hover:bg-blue-600 hover:text-black focus:bg-blue-600 focus:text-black",
                        day_today: "bg-slate-100 text-slate-900",
                        nav_button: "text-slate-500 hover:text-slate-900 border border-slate-200 hover:border-slate-300 h-7 w-7 bg-white p-0",
                        caption_label: "text-slate-900 font-mono font-bold text-sm",
                        head_cell: "text-slate-400 font-mono text-xs w-9",
                        cell: "h-9 w-9 text-center text-sm p-0",
                        day: "h-9 w-9 p-0 font-mono text-slate-600 hover:bg-blue-500/10 rounded-md",
                        day_outside: "text-slate-900/20",
                      }}
                    />
                  </PopoverContent>
                </Popover>
              </div>
              <div>
                <Label className="text-slate-600 font-mono text-xs uppercase tracking-wider">Deadline</Label>
                <Popover>
                  <PopoverTrigger asChild>
                    <Button variant="outline" className={`w-full mt-1.5 justify-start text-left font-mono border border-slate-300 bg-slate-50 hover:bg-blue-500/10 hover:border-blue-500 ${addTaskDeadline ? "text-slate-900" : "text-slate-400"}`}>
                      <CalendarIcon className="mr-2 h-4 w-4 text-slate-400" />
                      {addTaskDeadline ? format(new Date(addTaskDeadline), "MMM d, yyyy") : "Pick a date"}
                    </Button>
                  </PopoverTrigger>
                  <PopoverContent className="w-auto p-0 bg-[#0a0a0a] border border-slate-300" align="start">
                    <CalendarPicker
                      mode="single"
                      selected={addTaskDeadline ? new Date(addTaskDeadline) : undefined}
                      onSelect={(date) => setAddTaskDeadline(date ? date.toISOString() : "")}
                      className="text-slate-900"
                      classNames={{
                        day_selected: "bg-blue-600 text-slate-900 hover:bg-blue-600 hover:text-black focus:bg-blue-600 focus:text-black",
                        day_today: "bg-slate-100 text-slate-900",
                        nav_button: "text-slate-500 hover:text-slate-900 border border-slate-200 hover:border-slate-300 h-7 w-7 bg-white p-0",
                        caption_label: "text-slate-900 font-mono font-bold text-sm",
                        head_cell: "text-slate-400 font-mono text-xs w-9",
                        cell: "h-9 w-9 text-center text-sm p-0",
                        day: "h-9 w-9 p-0 font-mono text-slate-600 hover:bg-blue-500/10 rounded-md",
                        day_outside: "text-slate-900/20",
                      }}
                    />
                  </PopoverContent>
                </Popover>
              </div>
            </div>
            <div>
              <Label className="text-slate-600 font-mono text-xs uppercase tracking-wider">Description</Label>
              <Textarea
                className="border border-slate-300 bg-slate-50 text-slate-900 mt-1.5 font-mono min-h-[80px] placeholder:text-slate-300 focus:border-blue-500 transition-colors"
                value={addTaskDescription}
                onChange={(e) => setAddTaskDescription(e.target.value)}
                placeholder="What needs to be done? Add details, context, links..."
              />
            </div>
            <Separator className="bg-slate-100" />
            <div className="flex items-center justify-between">
              <Button
                variant="ghost"
                onClick={() => setAddTaskOpen(false)}
                className="text-slate-400 hover:text-slate-900 hover:bg-slate-50 font-mono text-sm"
              >
                Cancel
              </Button>
              <Button onClick={handleCreateTask} disabled={addTaskSaving || !addTaskTitle.trim()} className="bg-blue-600 text-slate-900 font-bold font-mono hover:bg-blue-600/80 transition-all hover:shadow-lg hover:shadow-blue-200/0.3)] disabled:opacity-40">
                {addTaskSaving ? <Loader2 className="h-4 w-4 animate-spin mr-2" /> : <Plus className="h-4 w-4 mr-2" />}
                Create Task
              </Button>
            </div>
          </div>
        </DialogContent>
      </Dialog>

      {/* Delete task confirm (MD only) */}
      <AlertDialog open={!!taskToDelete} onOpenChange={(open) => !open && setTaskToDelete(null)}>
        <AlertDialogContent className="border border-slate-200 bg-white text-slate-900">
          <AlertDialogHeader>
            <AlertDialogTitle className="font-mono font-black text-slate-900">Delete task?</AlertDialogTitle>
            <AlertDialogDescription className="text-slate-500 font-mono">
              {taskToDelete ? `"${taskToDelete.title}" will be permanently removed.` : ""}
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel className="border-slate-200 text-slate-900 hover:bg-blue-500/10">Cancel</AlertDialogCancel>
            <AlertDialogAction
              onClick={handleDeleteTask}
              disabled={!!deletingTaskId}
              className="bg-red-600 text-slate-900 hover:bg-red-700"
            >
              {deletingTaskId ? <Loader2 className="h-4 w-4 animate-spin" /> : "Delete"}
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>

      {/* Task detail dialog вЂ” opens when clicking a task row */}
      <Dialog open={!!selectedTask} onOpenChange={(open) => !open && setSelectedTask(null)}>
        <DialogContent className="border border-slate-200 bg-white text-slate-900 max-w-lg">
          {selectedTask && (() => {
            const t = selectedTask;
            const isTerminal = t.status === "done" || t.status === "cancelled";
            const deadlineMs = t.deadline ? new Date(t.deadline).getTime() : null;
            const isOverdue = deadlineMs && deadlineMs < now && !isTerminal;
            const statusColor =
              t.status === "done" ? "text-green-400" :
              t.status === "in_progress" ? "text-blue-600" :
              t.status === "cancelled" ? "text-red-400" : "text-slate-500";
            const statusLabel =
              t.status === "not_started" ? "Not started" :
              t.status === "in_progress" ? "In progress" :
              t.status === "done" ? "Done" : "Cancelled";
            return (
              <>
                <DialogHeader>
                  <DialogTitle className="font-mono font-black text-lg leading-tight pr-8">
                    <span className={t.status === "cancelled" ? "line-through text-slate-400" : ""}>{t.title}</span>
                  </DialogTitle>
                  <DialogDescription className="sr-only">Task details</DialogDescription>
                </DialogHeader>
                <div className="space-y-4 -mt-1">
                  {/* Status + Priority badges */}
                  <div className="flex items-center gap-2 flex-wrap">
                    <Badge className={`font-mono text-xs ${
                      t.status === "done" ? "bg-green-500/15 text-green-400 border-green-500/30" :
                      t.status === "in_progress" ? "bg-blue-600/15 text-blue-600 border-blue-500/30" :
                      t.status === "cancelled" ? "bg-red-500/15 text-red-400 border-red-500/30" :
                      "bg-slate-50 text-slate-500 border-slate-200"
                    }`}>
                      {statusLabel}
                    </Badge>
                    {isOverdue && (
                      <Badge className="bg-orange-500/15 text-orange-400 border-orange-500/30 font-mono text-xs">
                        <AlertTriangle className="h-3 w-3 mr-1" />Overdue
                      </Badge>
                    )}
                  </div>

                  {/* Info grid */}
                  <div className="grid grid-cols-2 gap-3">
                    <div className="space-y-1 p-3 rounded-lg bg-white border border-slate-200">
                      <div className="text-[10px] text-slate-400 font-mono uppercase tracking-wider">Assignee</div>
                      <div className="text-sm font-mono text-slate-900 flex items-center gap-1.5">
                        <Users className="h-3.5 w-3.5 text-slate-400" />
                        {displayName(t.assignee_user_id)}
                      </div>
                    </div>
                    <div className="space-y-1 p-3 rounded-lg bg-white border border-slate-200">
                      <div className="text-[10px] text-slate-400 font-mono uppercase tracking-wider" title="Who set this requirement">Created by</div>
                      <div className="text-sm font-mono text-slate-900 flex items-center gap-1.5">
                        <User className="h-3.5 w-3.5 text-slate-400" />
                        {displayName(t.created_by)}
                      </div>
                    </div>
                    <div className="space-y-1 p-3 rounded-lg bg-white border border-slate-200">
                      <div className="text-[10px] text-slate-400 font-mono uppercase tracking-wider">Start Date</div>
                      <div className="text-sm font-mono text-slate-900 flex items-center gap-1.5">
                        <CalendarIcon className="h-3.5 w-3.5 text-slate-400" />
                        {t.start_date ? format(new Date(t.start_date), "MMM d, yyyy") : "Not set"}
                      </div>
                    </div>
                    <div className="space-y-1 p-3 rounded-lg bg-white border border-slate-200">
                      <div className="text-[10px] text-slate-400 font-mono uppercase tracking-wider">Deadline</div>
                      <div className={`text-sm font-mono flex items-center gap-1.5 ${isOverdue ? "text-orange-400" : "text-slate-900"}`}>
                        <Clock className="h-3.5 w-3.5 text-slate-400" />
                        {t.deadline ? format(new Date(t.deadline), "MMM d, yyyy") : "Not set"}
                      </div>
                    </div>
                  </div>

                  {/* Description */}
                  {t.description && (
                    <div className="space-y-1.5">
                      <div className="text-[10px] text-slate-400 font-mono uppercase tracking-wider">Description</div>
                      <div className="text-sm font-mono text-slate-600 p-3 rounded-lg bg-white border border-slate-200 whitespace-pre-wrap">{t.description}</div>
                    </div>
                  )}

                  {/* Status note */}
                  {t.status_note && (
                    <div className="space-y-1.5">
                      <div className="text-[10px] text-slate-400 font-mono uppercase tracking-wider">Status Note</div>
                      <div className="text-sm font-mono text-slate-500 italic p-3 rounded-lg bg-white border border-slate-200">"{t.status_note}"</div>
                    </div>
                  )}

                  {/* Timestamps */}
                  <div className="flex items-center gap-4 text-[10px] text-slate-300 font-mono">
                    <span>Created: {format(new Date(t.created_at), "MMM d, yyyy 'at' h:mm a")}</span>
                    <span>Updated: {format(new Date(t.updated_at), "MMM d, yyyy 'at' h:mm a")}</span>
                  </div>

                  <Separator className="bg-slate-100" />

                  {/* Actions */}
                  {!isTerminal && (
                    <div className="space-y-3">
                      <div className="text-[10px] text-slate-400 font-mono uppercase tracking-wider">Update Status</div>
                      <div className="flex items-center gap-2">
                        <Input
                          placeholder="Add a note (optional)"
                          className="flex-1 border border-slate-200 bg-slate-50 text-slate-900 text-xs font-mono h-9 placeholder:text-slate-300"
                          value={statusNote[t.id] ?? ""}
                          onChange={(e) => setStatusNote((prev) => ({ ...prev, [t.id]: e.target.value }))}
                        />
                      </div>
                      <div className="flex items-center gap-2">
                        {t.status === "not_started" && (
                          <Button
                            size="sm"
                            variant="outline"
                            className="border-blue-500/50 text-blue-600 hover:bg-blue-600/10 font-mono text-xs flex-1"
                            disabled={statusUpdatingId === t.id}
                            onClick={() => { handleUpdateStatus(t.id, "in_progress", statusNote[t.id] || undefined); setSelectedTask(null); }}
                          >
                            {statusUpdatingId === t.id ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <><Rocket className="h-3.5 w-3.5 mr-1.5" />Start</>}
                          </Button>
                        )}
                        <Button
                          size="sm"
                          className="bg-green-600 text-slate-900 hover:bg-green-700 font-mono text-xs flex-1"
                          disabled={statusUpdatingId === t.id}
                          onClick={() => { handleUpdateStatus(t.id, "done", statusNote[t.id] || undefined); setSelectedTask(null); }}
                        >
                          {statusUpdatingId === t.id ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <><CheckCircle className="h-3.5 w-3.5 mr-1.5" />Done</>}
                        </Button>
                        <Button
                          size="sm"
                          variant="outline"
                          className="border-red-500/30 text-red-400 hover:bg-red-500/10 font-mono text-xs"
                          disabled={statusUpdatingId === t.id}
                          onClick={() => { handleUpdateStatus(t.id, "cancelled", statusNote[t.id] || undefined); setSelectedTask(null); }}
                        >
                          <X className="h-3.5 w-3.5 mr-1" />Cancel
                        </Button>
                      </div>
                    </div>
                  )}

                  {/* MD actions: delete */}
                  {isMD && (
                    <div className="flex justify-end">
                      <Button
                        size="sm"
                        variant="ghost"
                        className="text-red-400/60 hover:text-red-400 hover:bg-red-500/10 font-mono text-xs"
                        onClick={() => { setTaskToDelete(t); setSelectedTask(null); }}
                      >
                        <Trash2 className="h-3.5 w-3.5 mr-1.5" />Delete task
                      </Button>
                    </div>
                  )}
                </div>
              </>
            );
          })()}
        </DialogContent>
      </Dialog>

      {/* Latest decision / document / source вЂ” keep for context */}
      <div className="grid gap-4 md:grid-cols-3">
        <Card className="border border-slate-200 bg-white">
          <CardHeader className="border-b border-slate-200">
            <CardTitle className="text-base text-slate-900 font-mono font-black uppercase tracking-tight">Latest Decision</CardTitle>
          </CardHeader>
          <CardContent className="text-sm text-slate-500 font-mono">
            {latestDecision ? (
              <div className="space-y-1">
                <div className="font-mono font-bold text-slate-900">{latestDecision.startupName}</div>
                <div className="font-mono">{latestDecision.actionType} {latestDecision.outcome ? `(${latestDecision.outcome})` : ""}</div>
                {latestDecision.notes && <div className="font-mono">{latestDecision.notes}</div>}
              </div>
            ) : (
              "No decisions yet."
            )}
          </CardContent>
        </Card>
        <Card className="border border-slate-200 bg-white">
          <CardHeader className="border-b border-slate-200">
            <CardTitle className="text-base text-slate-900 font-mono font-black uppercase tracking-tight">Latest Document</CardTitle>
          </CardHeader>
          <CardContent className="text-sm text-slate-500 font-mono">
            {latestDocument ? (
              <div className="font-mono font-bold text-slate-900">{latestDocument.title || "Untitled document"}</div>
            ) : (
              "No documents yet."
            )}
          </CardContent>
        </Card>
        <Card className="border border-slate-200 bg-white">
          <CardHeader className="border-b border-slate-200">
            <CardTitle className="text-base text-slate-900 font-mono font-black uppercase tracking-tight">Latest Source</CardTitle>
          </CardHeader>
          <CardContent className="text-sm text-slate-500 font-mono">
            {latestSource ? (
              <div className="font-mono font-bold text-slate-900">{latestSource.title || "Untitled source"}</div>
            ) : (
              "No sources yet."
            )}
          </CardContent>
        </Card>
      </div>
      </>
      )}
    </div>
  );
}

// ============================================================================
// ONBOARDING TAB
// ============================================================================

function OnboardingTab({
  profile,
  sources,
  documents,
  decisions,
  tasks,
  onNavigate,
}: {
  profile: UserProfile | null;
  sources: SourceRecord[];
  documents: Array<{ id: string; title: string | null; storage_path: string | null }>;
  decisions: Decision[];
  tasks: Task[];
  onNavigate: (tab: string) => void;
}) {
  const orgLinked = Boolean(profile?.organization_id);
  const hasSources = sources.length > 0;
  const hasDocuments = documents.length > 0;
  const hasDecisions = decisions.length > 0;
  const hasTasks = tasks.length > 0;
  const isMD = profile?.role === "managing_partner" || profile?.role === "organizer";

  const steps = [
    {
      title: "Organization profile & access",
      status: orgLinked,
      description: "Confirm organization is linked and team access is properly configured.",
      action: () => onNavigate("overview"),
      actionLabel: "View Org Overview",
    },
    {
      title: "Sync data sources",
      status: hasSources,
      description: "Connect ClickUp or Google Drive, or upload files into Sources.",
      action: () => onNavigate("sources"),
      actionLabel: "Open Sources",
    },
    {
      title: "Index key documents",
      status: hasDocuments,
      description: "Upload reports, memos, project updates, and meeting notes.",
      action: () => onNavigate("sources"),
      actionLabel: "Add Documents",
    },
    {
      title: "Log decisions",
      status: hasDecisions,
      description: "Record key decisions with confidence, context, and rationale.",
      action: () => onNavigate("decisions"),
      actionLabel: "Open Decision Logger",
    },
    {
      title: "Dashboard & tasks",
      status: hasTasks,
      description: isMD
        ? "Assign tasks to your team from the Dashboard (assignee, deadline, status)."
        : "View your assigned tasks and update status (In progress / Done) from the Dashboard.",
      action: () => onNavigate("overview"),
      actionLabel: "Open Dashboard",
    },
    {
      title: "Review analytics",
      status: decisions.length >= 5,
      description: "Unlock Decision Engine analytics with at least 5 decisions (MD only).",
      action: () => onNavigate("dashboard"),
      actionLabel: "Open Decision Engine",
    },
  ];

  return (
    <div className="space-y-6">
      <Card className="border border-slate-200 bg-white">
        <CardHeader className="border-b border-slate-200">
          <CardTitle className="flex items-center gap-2 text-slate-900 font-mono font-black uppercase tracking-tight">
            <Sparkles className="h-5 w-5 text-blue-600" />
            Platform Onboarding
          </CardTitle>
          <CardDescription className="text-slate-500 font-mono">
            Recommended onboarding flow for your team. Complete the steps below to unlock full
            intelligence and analytics.
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4 text-slate-900">
          {steps.map((step) => (
            <div key={step.title} className="flex items-start justify-between gap-4 border border-slate-200 rounded-md p-4 bg-white hover:border-blue-500 hover:bg-blue-600/5 transition-all">
              <div className="space-y-1">
                <div className="flex items-center gap-2">
                  {step.status ? (
                    <CheckCircle className="h-4 w-4 text-blue-600" />
                  ) : (
                    <AlertTriangle className="h-4 w-4 text-slate-400" />
                  )}
                  <span className="font-mono font-bold text-slate-900">{step.title}</span>
                  <Badge variant="outline" className={step.status ? "border-blue-500 text-blue-600 bg-white font-mono text-xs" : "border-slate-200/50 text-slate-400 bg-white font-mono text-xs"}>
                    {step.status ? "Complete" : "Pending"}
                  </Badge>
                </div>
                <p className="text-sm text-slate-500 font-mono">{step.description}</p>
              </div>
              <Button variant="outline" size="sm" onClick={step.action} className="border border-slate-200 bg-white text-slate-900 hover:bg-blue-500/10 hover:border-blue-500 hover:text-blue-600 font-bold">
                {step.actionLabel}
              </Button>
            </div>
          ))}
        </CardContent>
      </Card>

      {(profile?.role === "managing_partner" || profile?.role === "organizer" || profile?.role === "admin") && (
        <>
          <TeamInvitationForm />
          {profile?.organization_id && <TeamMembersList />}
          {profile?.organization_id && <SyncStatus />}
        </>
      )}

      <div className="grid gap-4 md:grid-cols-2">
        <Card className="border border-slate-200 bg-white">
          <CardHeader className="border-b border-slate-200">
            <CardTitle className="text-base text-slate-900 font-mono font-black uppercase tracking-tight">Recommended Data Sources</CardTitle>
            <CardDescription className="text-slate-500 font-mono">Prioritize these sources for strong answers.</CardDescription>
          </CardHeader>
          <CardContent className="space-y-2 text-sm text-slate-500 font-mono">
            <ul className="list-disc pl-4 space-y-1">
              <li>Strategy memos, analysis notes, and key reports</li>
              <li>Project updates, KPIs, and presentations</li>
              <li>Organizational documents and guidelines</li>
              <li>Meeting notes and email summaries</li>
              <li>CRM exports, pipeline logs, and workflow snapshots</li>
            </ul>
          </CardContent>
        </Card>

        <Card className="border border-slate-200 bg-white">
          <CardHeader className="border-b border-slate-200">
            <CardTitle className="text-base text-slate-900 font-mono font-black uppercase tracking-tight">Sync Guidance</CardTitle>
            <CardDescription className="text-slate-500 font-mono">Fastest path to a live knowledge base.</CardDescription>
          </CardHeader>
          <CardContent className="space-y-2 text-sm text-slate-500 font-mono">
            <div className="space-y-1">
              <div className="font-mono font-bold text-slate-900">Google Drive</div>
              <p>Import key docs directly from Drive to keep investment materials current.</p>
            </div>
            <div className="space-y-1">
              <div className="font-mono font-bold text-slate-900">ClickUp</div>
              <p>Sync pipeline tasks and IC checklists for real-time deal visibility.</p>
            </div>
            <div className="space-y-1">
              <div className="font-mono font-bold text-slate-900">Manual Uploads</div>
              <p>Upload PDFs, spreadsheets, and memos for immediate indexing.</p>
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  );
}

// ============================================================================
// DECISION ENGINE DASHBOARD TAB
// ============================================================================

function DecisionEngineDashboardTab({ decisions }: { decisions: Decision[] }) {
  const [selectedSector, setSelectedSector] = useState<string>("all");
  const [selectedStage, setSelectedStage] = useState<string>("all");
  const [selectedPartner, setSelectedPartner] = useState<string>("all");

  // Filter decisions based on selected filters
  const filteredDecisions = useMemo(() => {
    return decisions.filter((d) => {
      if (selectedSector !== "all" && d.context?.sector !== selectedSector) return false;
      if (selectedStage !== "all" && d.context?.stage !== selectedStage) return false;
      if (selectedPartner !== "all" && d.actor !== selectedPartner) return false;
      return true;
    });
  }, [decisions, selectedSector, selectedStage, selectedPartner]);

  const analytics = useMemo(() => calculateDecisionEngineAnalytics(filteredDecisions), [filteredDecisions]);
  const hasEnoughData = filteredDecisions.length >= 5;

  const partnerOutcomeSeries = useMemo(
    () =>
      analytics.partnerStats.map((p) => ({
        partner: p.partner,
        positive: p.positiveOutcomes,
        negative: p.negativeOutcomes,
        pending: p.pendingOutcomes,
        avgDecisionVelocity: p.avgDecisionVelocity,
      })),
    [analytics.partnerStats]
  );

  const actionConversionSeries = useMemo(
    () =>
      analytics.actionTypeStats.map((a) => ({
        action: a.action,
        conversionRate: a.total ? Math.round((a.positive / a.total) * 100) : 0,
        total: a.total,
      })),
    [analytics.actionTypeStats]
  );

  const confidenceRateSeries = useMemo(
    () =>
      analytics.confidenceBuckets.map((b) => ({
        range: b.range,
        positiveRate: b.count ? Math.round((b.positive / b.count) * 100) : 0,
        total: b.count,
      })),
    [analytics.confidenceBuckets]
  );

  // Get unique values for filters
  const sectors = useMemo(() => {
    const unique = new Set(
      decisions
        .map((d) => d.context?.sector)
        .filter((value): value is string => !!value && value.trim().length > 0)
        .map((value) => value.trim())
    );
    return Array.from(unique).sort();
  }, [decisions]);

  const stages = useMemo(() => {
    const unique = new Set(
      decisions
        .map((d) => d.context?.stage)
        .filter((value): value is string => !!value && value.trim().length > 0)
        .map((value) => value.trim())
    );
    return Array.from(unique).sort();
  }, [decisions]);

  const partners = useMemo(() => {
    const unique = new Set(
      decisions
        .map((d) => d.actor)
        .filter((value): value is string => !!value && value.trim().length > 0)
        .map((value) => value.trim())
    );
    return Array.from(unique).sort();
  }, [decisions]);

  const COLORS = ["#0088FE", "#00C49F", "#FFBB28", "#FF8042", "#8884d8", "#82ca9d"];

  return (
    <div className="space-y-6">
      {/* Filters */}
      <Card className="border border-slate-200 bg-white">
        <CardHeader className="border-b border-slate-200">
          <CardTitle className="text-slate-900 font-mono font-black uppercase tracking-tight">Filters</CardTitle>
          <CardDescription className="text-slate-500 font-mono">Filter decisions by sector, stage, or partner</CardDescription>
        </CardHeader>
        <CardContent className="text-slate-900">
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div>
              <Label className="text-slate-900 font-mono font-bold">Sector</Label>
              <Select value={selectedSector} onValueChange={setSelectedSector}>
                <SelectTrigger className="border border-slate-200 bg-white text-slate-900">
                  <SelectValue placeholder="All sectors" />
                </SelectTrigger>
                <SelectContent className="bg-white border border-slate-200 shadow-lg rounded-md">
                  <SelectItem value="all" className="text-slate-900">All sectors</SelectItem>
                  {sectors.map((sector) => (
                    <SelectItem key={sector} value={sector} className="text-slate-900">
                      {sector}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>
            <div>
              <Label className="text-slate-900 font-mono font-bold">Stage</Label>
              <Select value={selectedStage} onValueChange={setSelectedStage}>
                <SelectTrigger className="border border-slate-200 bg-white text-slate-900">
                  <SelectValue placeholder="All stages" />
                </SelectTrigger>
                <SelectContent className="bg-white border border-slate-200 shadow-lg rounded-md">
                  <SelectItem value="all" className="text-slate-900">All stages</SelectItem>
                  {stages.map((stage) => (
                    <SelectItem key={stage} value={stage} className="text-slate-900">
                      {stage}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>
            <div>
              <Label className="text-slate-900 font-mono font-bold">Partner</Label>
              <Select value={selectedPartner} onValueChange={setSelectedPartner}>
                <SelectTrigger className="border border-slate-200 bg-white text-slate-900">
                  <SelectValue placeholder="All partners" />
                </SelectTrigger>
                <SelectContent className="bg-white border border-slate-200 shadow-lg rounded-md">
                  <SelectItem value="all" className="text-slate-900">All partners</SelectItem>
                  {partners.map((partner) => (
                    <SelectItem key={partner} value={partner} className="text-slate-900">
                      {partner}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>
          </div>
          {(selectedSector !== "all" || selectedStage !== "all" || selectedPartner !== "all") && (
            <div className="mt-4">
              <Button
                variant="outline"
                size="sm"
                onClick={() => {
                  setSelectedSector("all");
                  setSelectedStage("all");
                  setSelectedPartner("all");
                }}
                className="border border-slate-200 bg-white text-slate-900 hover:bg-blue-500/10 hover:border-blue-500 hover:text-blue-600 font-bold"
              >
                Clear filters ({filteredDecisions.length} decisions)
              </Button>
            </div>
          )}
        </CardContent>
      </Card>

      {/* Key Metrics */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <Card className="border border-slate-200 bg-white">
          <CardContent className="pt-4 text-slate-900">
            <div className="flex items-center gap-3">
              <div className="p-2 border border-slate-200 rounded-lg bg-white">
                <Target className="h-5 w-5 text-blue-600" />
              </div>
              <div>
                <p className="text-2xl font-mono font-black">{analytics.totalDecisions}</p>
                <p className="text-xs text-slate-500 font-mono uppercase tracking-wider">Decisions Logged</p>
              </div>
            </div>
          </CardContent>
        </Card>
        <Card className="border border-slate-200 bg-white">
          <CardContent className="pt-4 text-slate-900">
            <div className="flex items-center gap-3">
              <div className="p-2 border-2 border-blue-500 rounded-lg bg-white">
                <TrendingUp className="h-5 w-5 text-blue-600" />
              </div>
              <div>
                <p className="text-2xl font-mono font-black">{analytics.positiveRate}%</p>
                <p className="text-xs text-slate-500 font-mono uppercase tracking-wider">Positive Rate</p>
              </div>
            </div>
          </CardContent>
        </Card>
        <Card className="border border-slate-200 bg-white">
          <CardContent className="pt-4 text-slate-900">
            <div className="flex items-center gap-3">
              <div className="p-2 border border-slate-200 rounded-lg bg-white">
                <BarChart3 className="h-5 w-5 text-blue-600" />
              </div>
              <div className="flex-1">
                <p className="text-2xl font-mono font-black">{analytics.avgConfidence}%</p>
                <TooltipProvider>
                  <Tooltip>
                    <TooltipTrigger asChild>
                      <p className="text-xs text-slate-500 font-mono cursor-help">
                        Avg Confidence
                        <span className="ml-1">в„№пёЏ</span>
                      </p>
                    </TooltipTrigger>
                    <TooltipContent className="bg-white border border-slate-200 text-slate-900">
                      <p className="max-w-xs font-mono">
                        Average confidence score (0-100) you assigned when logging decisions.
                        <br />
                        Higher = more certain about the decision.
                      </p>
                    </TooltipContent>
                  </Tooltip>
                </TooltipProvider>
              </div>
            </div>
          </CardContent>
        </Card>
        <Card className="border border-slate-200 bg-white">
          <CardContent className="pt-4 text-slate-900">
            <div className="flex items-center gap-3">
              <div className="p-2 border border-slate-200 rounded-lg bg-white">
                <Clock className="h-5 w-5 text-blue-600" />
              </div>
              <div>
                <p className="text-2xl font-mono font-black">{analytics.avgDecisionVelocity}</p>
                <p className="text-xs text-slate-500 font-mono uppercase tracking-wider">Avg Velocity (days)</p>
              </div>
            </div>
          </CardContent>
        </Card>
      </div>

      {hasEnoughData && (
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <Card className="border border-slate-200 bg-white">
            <CardContent className="pt-4 text-slate-900">
              <div className="flex items-center gap-3">
                <div className="p-2 border border-slate-200 rounded-lg bg-white">
                  <Clock className="h-5 w-5 text-blue-600" />
                </div>
                <div>
                  <p className="text-2xl font-mono font-black">{analytics.recencyStats.last7}</p>
                  <p className="text-xs text-slate-500 font-mono uppercase tracking-wider">Decisions (7d)</p>
                </div>
              </div>
            </CardContent>
          </Card>
          <Card className="border border-slate-200 bg-white">
            <CardContent className="pt-4 text-slate-900">
              <div className="flex items-center gap-3">
                <div className="p-2 border border-slate-200 rounded-lg bg-white">
                  <TrendingUp className="h-5 w-5 text-blue-600" />
                </div>
                <div>
                  <p className="text-2xl font-mono font-black">{analytics.recencyStats.last30}</p>
                  <p className="text-xs text-slate-500 font-mono uppercase tracking-wider">Decisions (30d)</p>
                </div>
              </div>
            </CardContent>
          </Card>
          <Card className="border border-slate-200 bg-white">
            <CardContent className="pt-4 text-slate-900">
              <div className="flex items-center gap-3">
                <div className="p-2 border border-slate-200 rounded-lg bg-white">
                  <BarChart3 className="h-5 w-5 text-blue-600" />
                </div>
                <div>
                  <p className="text-2xl font-mono font-black">{analytics.recencyStats.last90}</p>
                  <p className="text-xs text-slate-500 font-mono uppercase tracking-wider">Decisions (90d)</p>
                </div>
              </div>
            </CardContent>
          </Card>
          <Card className="border border-slate-200 bg-white">
            <CardContent className="pt-4 text-slate-900">
              <div className="flex items-center gap-3">
                <div className="p-2 border border-slate-200 rounded-lg bg-white">
                  <TrendingUp className="h-5 w-5 text-blue-600" />
                </div>
                <div>
                  <p className="text-2xl font-mono font-black">{analytics.recencyStats.momentumPct}%</p>
                  <p className="text-xs text-slate-500 font-mono uppercase tracking-wider">30d Momentum</p>
                </div>
              </div>
            </CardContent>
          </Card>
        </div>
      )}

      {!hasEnoughData ? (
        <Card className="border border-slate-200 bg-white">
          <CardContent className="pt-6 text-slate-900">
            <div className="text-center py-8">
              <AlertTriangle className="h-12 w-12 text-slate-400 mx-auto mb-4" />
              <p className="text-lg font-mono font-bold mb-2">Not Enough Data</p>
              <p className="text-sm text-slate-500 font-mono">
                You need at least 5 decisions to see analytics. Start logging decisions to unlock insights.
              </p>
            </div>
          </CardContent>
        </Card>
      ) : (() => {
        const adv = analytics?.advancedInsights;
        return (
        <>
          {/* ========== ADVANCED DECISION ANALYTICS ========== */}
          {adv && (
          <Card className="border-2 border-blue-500/50 bg-white">
            <CardHeader className="border-b border-slate-200">
              <CardTitle className="flex items-center gap-2 text-slate-900 font-mono font-black uppercase tracking-tight">
                <Sparkles className="h-5 w-5 text-blue-600" />
                Advanced Decision Analytics
              </CardTitle>
              <CardDescription className="text-slate-500 font-mono">Insights, calibration, and focus recommendations</CardDescription>
            </CardHeader>
            <CardContent className="pt-4 text-slate-900 space-y-6">
              {/* Advanced Insights KPIs */}
              <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-6 gap-3">
                {adv.bestSectorByRate && (
                  <div className="p-3 rounded-lg border border-blue-500/30 bg-blue-600/5">
                    <p className="text-[10px] font-mono uppercase tracking-wider text-slate-500">Best sector (rate)</p>
                    <p className="font-mono font-bold text-blue-600">{adv.bestSectorByRate.sector}</p>
                    <p className="text-xs font-mono text-slate-600">{adv.bestSectorByRate.rate}% ({adv.bestSectorByRate.total} decisions)</p>
                  </div>
                )}
                {adv.worstSectorByRate && adv.worstSectorByRate.sector !== adv.bestSectorByRate?.sector && (
                  <div className="p-3 rounded-lg border border-slate-200 bg-slate-50">
                    <p className="text-[10px] font-mono uppercase tracking-wider text-slate-500">Lowest sector (rate)</p>
                    <p className="font-mono font-bold text-slate-900">{adv.worstSectorByRate.sector}</p>
                    <p className="text-xs font-mono text-slate-500">{adv.worstSectorByRate.rate}% ({adv.worstSectorByRate.total})</p>
                  </div>
                )}
                {adv.topSectorByVolume && (
                  <div className="p-3 rounded-lg border border-slate-200 bg-slate-50">
                    <p className="text-[10px] font-mono uppercase tracking-wider text-slate-500">Top sector (volume)</p>
                    <p className="font-mono font-bold text-slate-900">{adv.topSectorByVolume.sector}</p>
                    <p className="text-xs font-mono text-slate-500">{adv.topSectorByVolume.total} decisions</p>
                  </div>
                )}
                <div className="p-3 rounded-lg border border-slate-200 bg-slate-50">
                  <p className="text-[10px] font-mono uppercase tracking-wider text-slate-500">Concentration (top 3)</p>
                  <p className="font-mono font-bold text-slate-900">{adv.concentrationTop3Pct ?? 0}%</p>
                  <p className="text-xs font-mono text-slate-500 truncate" title={(adv.concentrationTop3Sectors ?? []).join(", ")}>
                    {(adv.concentrationTop3Sectors ?? []).join(", ") || "вЂ”"}
                  </p>
                </div>
                {adv.momDecisionsPct != null && (
                  <div className="p-3 rounded-lg border border-slate-200 bg-slate-50">
                    <p className="text-[10px] font-mono uppercase tracking-wider text-slate-500">MoM volume change</p>
                    <p className={`font-mono font-bold ${adv.momDecisionsPct >= 0 ? "text-blue-600" : "text-orange-400"}`}>
                      {adv.momDecisionsPct >= 0 ? "+" : ""}{adv.momDecisionsPct}%
                    </p>
                    <p className="text-xs font-mono text-slate-500">vs previous month</p>
                  </div>
                )}
                {adv.momPositiveRatePct != null && (
                  <div className="p-3 rounded-lg border border-slate-200 bg-slate-50">
                    <p className="text-[10px] font-mono uppercase tracking-wider text-slate-500">MoM positive rate</p>
                    <p className={`font-mono font-bold ${adv.momPositiveRatePct >= 0 ? "text-blue-600" : "text-orange-400"}`}>
                      {adv.momPositiveRatePct >= 0 ? "+" : ""}{adv.momPositiveRatePct}pp
                    </p>
                    <p className="text-xs font-mono text-slate-500">vs previous month</p>
                  </div>
                )}
              </div>

              {/* Calibration: High vs Low confidence */}
              <div className="grid md:grid-cols-2 gap-4">
                <div className="p-4 rounded-lg border border-slate-200 bg-slate-50">
                  <p className="text-xs font-mono uppercase tracking-wider text-slate-500 mb-2">Calibration вЂ” High confidence (81вЂ“100)</p>
                  <p className="font-mono font-bold text-slate-900 text-lg">{(adv.calibrationHighConfidence?.positiveRate ?? 0)}% positive rate</p>
                  <p className="text-xs font-mono text-slate-500">{(adv.calibrationHighConfidence?.total ?? 0)} decisions in this band</p>
                  <p className="text-[10px] font-mono text-slate-400 mt-1">When you were very confident, how often were you right?</p>
                </div>
                <div className="p-4 rounded-lg border border-slate-200 bg-slate-50">
                  <p className="text-xs font-mono uppercase tracking-wider text-slate-500 mb-2">Calibration вЂ” Low confidence (0вЂ“40)</p>
                  <p className="font-mono font-bold text-slate-900 text-lg">{(adv.calibrationLowConfidence?.positiveRate ?? 0)}% positive rate</p>
                  <p className="text-xs font-mono text-slate-500">{(adv.calibrationLowConfidence?.total ?? 0)} decisions in this band</p>
                  <p className="text-[10px] font-mono text-slate-400 mt-1">When you were uncertain, how often did it still go positive?</p>
                </div>
              </div>

              {/* Confidence by outcome */}
              <div className="flex flex-wrap gap-4 p-3 rounded-lg border border-slate-200 bg-slate-50">
                <span className="font-mono text-sm"><span className="text-slate-500">Avg confidence when positive:</span> <strong className="text-blue-600">{adv.confidenceWhenPositive ?? 0}%</strong></span>
                <span className="font-mono text-sm"><span className="text-slate-500">Avg confidence when negative:</span> <strong className="text-slate-900">{adv.confidenceWhenNegative ?? 0}%</strong></span>
                <span className="font-mono text-sm"><span className="text-slate-500">Pending:</span> <strong className="text-slate-900">{adv.pendingPct ?? 0}%</strong> of decisions</span>
              </div>

              {/* Suggested focus */}
              {adv.suggestedFocus && (
                <div className="p-3 rounded-lg border border-blue-500/40 bg-blue-600/10">
                  <p className="text-[10px] font-mono uppercase tracking-wider text-slate-500 mb-1">Suggested focus</p>
                  <p className="font-mono font-bold text-blue-600">{adv.suggestedFocus}</p>
                </div>
              )}

              {/* Peak month */}
              {adv.peakMonth && (
                <p className="text-xs font-mono text-slate-400">Peak month: <strong className="text-slate-900">{adv.peakMonth.date}</strong> ({adv.peakMonth.decisions} decisions)</p>
              )}
            </CardContent>
          </Card>
          )}

          {/* Sector Г— Stage heatmap */}
          {analytics.sectorStageMatrix.length > 0 && (() => {
            const stages = Array.from(new Set(analytics.sectorStageMatrix.map((c) => c.stage))).sort();
            const sectors = Array.from(new Set(analytics.sectorStageMatrix.map((c) => c.sector))).sort();
            return (
              <Card className="border border-slate-200 bg-white">
                <CardHeader className="border-b border-slate-200">
                  <CardTitle className="flex items-center gap-2 text-slate-900 font-mono font-black uppercase tracking-tight">
                    <BarChart3 className="h-5 w-5 text-blue-600" />
                    Sector Г— Stage Heatmap
                  </CardTitle>
                  <CardDescription className="text-slate-500 font-mono">Decision volume and positive rate by sector and stage</CardDescription>
                </CardHeader>
                <CardContent className="text-slate-900">
                  <div className="overflow-x-auto">
                    <table className="w-full font-mono text-xs border-collapse">
                      <thead>
                        <tr className="border-b border-slate-300">
                          <th className="text-left p-2 text-slate-600 font-bold">Sector \ Stage</th>
                          {stages.map((stage) => (
                            <th key={stage} className="p-2 text-center text-slate-600 font-bold truncate max-w-[80px]" title={stage}>{stage}</th>
                          ))}
                        </tr>
                      </thead>
                      <tbody>
                        {sectors.map((sector) => (
                          <tr key={sector} className="border-b border-slate-200 hover:bg-slate-50">
                            <td className="p-2 font-bold text-slate-900 truncate max-w-[120px]" title={sector}>{sector}</td>
                            {stages.map((stage) => {
                              const cell = analytics.sectorStageMatrix.find((c) => c.sector === sector && c.stage === stage);
                              const total = cell?.total ?? 0;
                              const rate = cell?.positiveRate ?? 0;
                              const intensity = total > 0 ? Math.min(1, total / 10) : 0;
                              return (
                                <td
                                  key={`${sector}-${stage}`}
                                  className="p-2 text-center rounded"
                                  style={{ backgroundColor: total > 0 ? `rgba(255, 237, 0, ${0.12 + intensity * 0.5})` : "rgba(255,255,255,0.03)" }}
                                  title={`${sector} Г— ${stage}: ${total} decisions, ${rate}% positive`}
                                >
                                  <span className="font-mono font-bold text-slate-900">{total}</span>
                                  {total > 0 && <span className="block text-[10px] text-slate-600">{rate}%</span>}
                                </td>
                              );
                            })}
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </CardContent>
              </Card>
            );
          })()}

          {/* Sector Performance */}
          {analytics.sectorStats.length > 0 && (
            <Card className="border border-slate-200 bg-white">
              <CardHeader className="border-b border-slate-200">
                <CardTitle className="flex items-center gap-2 text-slate-900 font-mono font-black uppercase tracking-tight">
                  <BarChart3 className="h-5 w-5 text-blue-600" />
                  Sector Performance
                </CardTitle>
                <CardDescription className="text-slate-500 font-mono">Decision breakdown by sector</CardDescription>
              </CardHeader>
              <CardContent className="text-slate-900">
                <ResponsiveContainer width="100%" height={300}>
                  <BarChart data={analytics.sectorStats.slice(0, 10)}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#FFFFFF" opacity={0.2} />
                    <XAxis dataKey="sector" angle={-45} textAnchor="end" height={100} stroke="#FFFFFF" />
                    <YAxis stroke="#FFFFFF" />
                    <RechartsTooltip contentStyle={{ backgroundColor: "#050505", border: "2px solid #FFFFFF", color: "#FFFFFF" }} />
                    <Legend wrapperStyle={{ color: "#FFFFFF" }} />
                    <Bar dataKey="positive" fill="#3b82f6" name="Positive" />
                    <Bar dataKey="negative" fill="#FFFFFF" name="Negative" />
                    <Bar dataKey="pending" fill="#FFFFFF" name="Pending" />
                  </BarChart>
                </ResponsiveContainer>
              </CardContent>
            </Card>
          )}

          {/* Stage Performance */}
          {analytics.stageStats.length > 0 && (
            <div className="grid gap-4 md:grid-cols-2">
              <Card className="border border-slate-200 bg-white">
                <CardHeader className="border-b border-slate-200">
                  <CardTitle className="flex items-center gap-2 text-slate-900 font-mono font-black uppercase tracking-tight">
                    <PieChart className="h-5 w-5 text-blue-600" />
                    Stage Distribution
                  </CardTitle>
                  <CardDescription className="text-slate-500 font-mono">Decisions by funding stage</CardDescription>
                </CardHeader>
                <CardContent>
                  <ResponsiveContainer width="100%" height={250}>
                    <RechartsPieChart>
                      <Pie
                        data={analytics.stageStats.map((s) => ({
                          name: s.stage,
                          value: s.total,
                        }))}
                        cx="50%"
                        cy="50%"
                        label={({ name, percent }) => `${name}: ${(percent * 100).toFixed(0)}%`}
                        outerRadius={80}
                        fill="#8884d8"
                        dataKey="value"
                      >
                        {analytics.stageStats.map((entry, index) => (
                          <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                        ))}
                      </Pie>
                      <RechartsTooltip contentStyle={{ backgroundColor: "#050505", border: "2px solid #FFFFFF", color: "#FFFFFF" }} />
                      <Legend wrapperStyle={{ color: "#FFFFFF" }} />
                    </RechartsPieChart>
                  </ResponsiveContainer>
                </CardContent>
              </Card>

              <Card className="border border-slate-200 bg-white">
                <CardHeader className="border-b border-slate-200">
                  <CardTitle className="flex items-center gap-2 text-slate-900 font-mono font-black uppercase tracking-tight">
                    <BarChart3 className="h-5 w-5 text-blue-600" />
                    Stage Conversion Rates
                  </CardTitle>
                  <CardDescription className="text-slate-500 font-mono">
                    <TooltipProvider>
                      <Tooltip>
                        <TooltipTrigger asChild>
                          <span className="cursor-help">
                            Positive rate by stage
                            <span className="ml-1">в„№пёЏ</span>
                          </span>
                        </TooltipTrigger>
                        <TooltipContent className="bg-white border border-slate-200 text-slate-900">
                          <p className="max-w-xs font-mono">
                            Conversion Rate = (Positive Decisions / Total Decisions) Г— 100%
                            <br />
                            Shows what % of decisions in each stage resulted in positive outcomes.
                          </p>
                        </TooltipContent>
                      </Tooltip>
                    </TooltipProvider>
                  </CardDescription>
                </CardHeader>
                <CardContent className="text-slate-900">
                  <ResponsiveContainer width="100%" height={250}>
                    <BarChart data={analytics.stageStats}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#FFFFFF" opacity={0.2} />
                      <XAxis dataKey="stage" stroke="#FFFFFF" />
                      <YAxis stroke="#FFFFFF" />
                      <RechartsTooltip contentStyle={{ backgroundColor: "#050505", border: "2px solid #FFFFFF", color: "#FFFFFF" }} />
                      <Legend wrapperStyle={{ color: "#FFFFFF" }} />
                      <Bar dataKey="conversionRate" fill="#3b82f6" name="Conversion Rate %" />
                    </BarChart>
                  </ResponsiveContainer>
                </CardContent>
              </Card>
            </div>
          )}

          {/* Partner Performance */}
          {analytics.partnerStats.length > 0 && (
            <Card className="border border-slate-200 bg-white">
              <CardHeader className="border-b border-slate-200">
                <CardTitle className="flex items-center gap-2 text-slate-900 font-mono font-black uppercase tracking-tight">
                  <Users className="h-5 w-5 text-blue-600" />
                  Partner Performance
                </CardTitle>
                <CardDescription className="text-slate-500 font-mono">Decision metrics by partner</CardDescription>
              </CardHeader>
              <CardContent className="text-slate-900">
                <ResponsiveContainer width="100%" height={300}>
                  <BarChart data={analytics.partnerStats.slice(0, 10)}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#FFFFFF" opacity={0.2} />
                    <XAxis dataKey="partner" angle={-45} textAnchor="end" height={100} stroke="#FFFFFF" />
                    <YAxis yAxisId="left" stroke="#FFFFFF" />
                    <YAxis yAxisId="right" orientation="right" stroke="#FFFFFF" />
                    <RechartsTooltip contentStyle={{ backgroundColor: "#050505", border: "2px solid #FFFFFF", color: "#FFFFFF" }} />
                    <Legend wrapperStyle={{ color: "#FFFFFF" }} />
                    <Bar yAxisId="left" dataKey="totalDecisions" fill="#3b82f6" name="Total Decisions" />
                    <Bar yAxisId="right" dataKey="winRate" fill="#FFFFFF" name="Win Rate %" />
                  </BarChart>
                </ResponsiveContainer>
              </CardContent>
            </Card>
          )}

          {/* Partner Outcome Mix */}
          {partnerOutcomeSeries.length > 0 && (
            <Card className="border border-slate-200 bg-white">
              <CardHeader className="border-b border-slate-200">
                <CardTitle className="flex items-center gap-2 text-slate-900 font-mono font-black uppercase tracking-tight">
                  <BarChart3 className="h-5 w-5 text-blue-600" />
                  Partner Outcome Mix
                </CardTitle>
                <CardDescription className="text-slate-500 font-mono">Outcome breakdown by partner (top 10)</CardDescription>
              </CardHeader>
              <CardContent className="text-slate-900">
                <ResponsiveContainer width="100%" height={300}>
                  <BarChart data={partnerOutcomeSeries.slice(0, 10)}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#FFFFFF" opacity={0.2} />
                    <XAxis dataKey="partner" angle={-25} textAnchor="end" height={70} stroke="#FFFFFF" />
                    <YAxis stroke="#FFFFFF" />
                    <RechartsTooltip contentStyle={{ backgroundColor: "#050505", border: "2px solid #FFFFFF", color: "#FFFFFF" }} />
                    <Legend wrapperStyle={{ color: "#FFFFFF" }} />
                    <Bar dataKey="positive" stackId="a" fill="#3b82f6" name="Positive" />
                    <Bar dataKey="negative" stackId="a" fill="#FFFFFF" name="Negative" />
                    <Bar dataKey="pending" stackId="a" fill="#FFFFFF" name="Pending" />
                  </BarChart>
                </ResponsiveContainer>
              </CardContent>
            </Card>
          )}

          {/* Decision Velocity by Partner */}
          {partnerOutcomeSeries.length > 0 && (
            <Card className="border border-slate-200 bg-white">
              <CardHeader className="border-b border-slate-200">
                <CardTitle className="flex items-center gap-2 text-slate-900 font-mono font-black uppercase tracking-tight">
                  <Clock className="h-5 w-5 text-blue-600" />
                  Decision Velocity by Partner
                </CardTitle>
                <CardDescription className="text-slate-500 font-mono">Average decision cycle length (days)</CardDescription>
              </CardHeader>
              <CardContent className="text-slate-900">
                <ResponsiveContainer width="100%" height={260}>
                  <BarChart data={partnerOutcomeSeries.slice(0, 10)}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#FFFFFF" opacity={0.2} />
                    <XAxis dataKey="partner" angle={-25} textAnchor="end" height={70} stroke="#FFFFFF" />
                    <YAxis stroke="#FFFFFF" />
                    <RechartsTooltip contentStyle={{ backgroundColor: "#050505", border: "2px solid #FFFFFF", color: "#FFFFFF" }} />
                    <Legend wrapperStyle={{ color: "#FFFFFF" }} />
                    <Bar dataKey="avgDecisionVelocity" fill="#3b82f6" name="Avg Days" />
                  </BarChart>
                </ResponsiveContainer>
              </CardContent>
            </Card>
          )}

          {/* Outcome & Confidence */}
          {analytics.outcomeStats.length > 0 && (
            <div className="grid gap-4 md:grid-cols-2">
              <Card className="border border-slate-200 bg-white">
                <CardHeader className="border-b border-slate-200">
                  <CardTitle className="flex items-center gap-2 text-slate-900 font-mono font-black uppercase tracking-tight">
                    <PieChart className="h-5 w-5 text-blue-600" />
                    Outcome Mix
                  </CardTitle>
                  <CardDescription className="text-slate-500 font-mono">Overall outcome distribution</CardDescription>
                </CardHeader>
                <CardContent className="text-slate-900">
                  <ResponsiveContainer width="100%" height={260}>
                    <RechartsPieChart>
                      <Pie
                        data={analytics.outcomeStats.map((o) => ({
                          name: o.outcome,
                          value: o.total,
                        }))}
                        cx="50%"
                        cy="50%"
                        label={({ name, percent }) => `${name}: ${(percent * 100).toFixed(0)}%`}
                        outerRadius={90}
                        fill="#3b82f6"
                        dataKey="value"
                      >
                        {analytics.outcomeStats.map((entry, index) => (
                          <Cell key={`outcome-${index}`} fill={COLORS[index % COLORS.length]} />
                        ))}
                      </Pie>
                      <RechartsTooltip contentStyle={{ backgroundColor: "#050505", border: "2px solid #FFFFFF", color: "#FFFFFF" }} />
                      <Legend wrapperStyle={{ color: "#FFFFFF" }} />
                    </RechartsPieChart>
                  </ResponsiveContainer>
                </CardContent>
              </Card>

              <Card className="border border-slate-200 bg-white">
                <CardHeader className="border-b border-slate-200">
                  <CardTitle className="flex items-center gap-2 text-slate-900 font-mono font-black uppercase tracking-tight">
                    <BarChart3 className="h-5 w-5 text-blue-600" />
                    Confidence by Outcome
                  </CardTitle>
                  <CardDescription className="text-slate-500 font-mono">Average confidence score per outcome</CardDescription>
                </CardHeader>
                <CardContent className="text-slate-900">
                  <ResponsiveContainer width="100%" height={260}>
                    <BarChart data={analytics.outcomeStats}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#FFFFFF" opacity={0.2} />
                      <XAxis dataKey="outcome" stroke="#FFFFFF" />
                      <YAxis stroke="#FFFFFF" />
                      <RechartsTooltip contentStyle={{ backgroundColor: "#050505", border: "2px solid #FFFFFF", color: "#FFFFFF" }} />
                      <Legend wrapperStyle={{ color: "#FFFFFF" }} />
                      <Bar dataKey="avgConfidence" fill="#3b82f6" name="Avg Confidence %" />
                    </BarChart>
                  </ResponsiveContainer>
                </CardContent>
              </Card>
            </div>
          )}

          {/* Confidence Distribution */}
          {analytics.confidenceBuckets.length > 0 && (
            <Card className="border border-slate-200 bg-white">
              <CardHeader className="border-b border-slate-200">
                <CardTitle className="flex items-center gap-2 text-slate-900 font-mono font-black uppercase tracking-tight">
                  <BarChart3 className="h-5 w-5 text-blue-600" />
                  Confidence Distribution
                </CardTitle>
                <CardDescription className="text-slate-500 font-mono">Decision volume by confidence band</CardDescription>
              </CardHeader>
              <CardContent className="text-slate-900">
                <ResponsiveContainer width="100%" height={280}>
                  <BarChart data={analytics.confidenceBuckets}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#FFFFFF" opacity={0.2} />
                    <XAxis dataKey="range" stroke="#FFFFFF" />
                    <YAxis stroke="#FFFFFF" />
                    <RechartsTooltip contentStyle={{ backgroundColor: "#050505", border: "2px solid #FFFFFF", color: "#FFFFFF" }} />
                    <Legend wrapperStyle={{ color: "#FFFFFF" }} />
                    <Bar dataKey="positive" stackId="a" fill="#3b82f6" name="Positive" />
                    <Bar dataKey="negative" stackId="a" fill="#FFFFFF" name="Negative" />
                    <Bar dataKey="pending" stackId="a" fill="#FFFFFF" name="Pending" />
                  </BarChart>
                </ResponsiveContainer>
              </CardContent>
            </Card>
          )}

          {/* Decision Age Distribution */}
          {analytics.ageBuckets.length > 0 && (
            <Card className="border border-slate-200 bg-white">
              <CardHeader className="border-b border-slate-200">
                <CardTitle className="flex items-center gap-2 text-slate-900 font-mono font-black uppercase tracking-tight">
                  <BarChart3 className="h-5 w-5 text-blue-600" />
                  Decision Age Distribution
                </CardTitle>
                <CardDescription className="text-slate-500 font-mono">Volume and outcome mix by decision age</CardDescription>
              </CardHeader>
              <CardContent className="text-slate-900">
                <ResponsiveContainer width="100%" height={280}>
                  <BarChart data={analytics.ageBuckets}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#FFFFFF" opacity={0.2} />
                    <XAxis dataKey="range" stroke="#FFFFFF" />
                    <YAxis stroke="#FFFFFF" />
                    <RechartsTooltip contentStyle={{ backgroundColor: "#050505", border: "2px solid #FFFFFF", color: "#FFFFFF" }} />
                    <Legend wrapperStyle={{ color: "#FFFFFF" }} />
                    <Bar dataKey="positive" stackId="a" fill="#3b82f6" name="Positive" />
                    <Bar dataKey="negative" stackId="a" fill="#FFFFFF" name="Negative" />
                    <Bar dataKey="pending" stackId="a" fill="#FFFFFF" name="Pending" />
                  </BarChart>
                </ResponsiveContainer>
              </CardContent>
            </Card>
          )}

          {/* Outcome by Stage */}
          {analytics.outcomeByStage.length > 0 && (
            <Card className="border border-slate-200 bg-white">
              <CardHeader className="border-b border-slate-200">
                <CardTitle className="flex items-center gap-2 text-slate-900 font-mono font-black uppercase tracking-tight">
                  <BarChart3 className="h-5 w-5 text-blue-600" />
                  Outcome by Stage
                </CardTitle>
                <CardDescription className="text-slate-500 font-mono">Stage-level outcome mix</CardDescription>
              </CardHeader>
              <CardContent className="text-slate-900">
                <ResponsiveContainer width="100%" height={280}>
                  <BarChart data={analytics.outcomeByStage}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#FFFFFF" opacity={0.2} />
                    <XAxis dataKey="stage" angle={-20} textAnchor="end" height={70} stroke="#FFFFFF" />
                    <YAxis stroke="#FFFFFF" />
                    <RechartsTooltip contentStyle={{ backgroundColor: "#050505", border: "2px solid #FFFFFF", color: "#FFFFFF" }} />
                    <Legend wrapperStyle={{ color: "#FFFFFF" }} />
                    <Bar dataKey="positive" stackId="a" fill="#3b82f6" name="Positive" />
                    <Bar dataKey="negative" stackId="a" fill="#FFFFFF" name="Negative" />
                    <Bar dataKey="pending" stackId="a" fill="#FFFFFF" name="Pending" />
                  </BarChart>
                </ResponsiveContainer>
              </CardContent>
            </Card>
          )}

          {/* Geo Focus */}
          {analytics.geoStats.length > 0 && (
            <Card className="border border-slate-200 bg-white">
              <CardHeader className="border-b border-slate-200">
                <CardTitle className="flex items-center gap-2 text-slate-900 font-mono font-black uppercase tracking-tight">
                  <BarChart3 className="h-5 w-5 text-blue-600" />
                  Geo Focus
                </CardTitle>
                <CardDescription className="text-slate-500 font-mono">Decision volume by geography</CardDescription>
              </CardHeader>
              <CardContent className="text-slate-900">
                <ResponsiveContainer width="100%" height={280}>
                  <BarChart data={analytics.geoStats.slice(0, 12)}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#FFFFFF" opacity={0.2} />
                    <XAxis dataKey="geo" angle={-25} textAnchor="end" height={70} stroke="#FFFFFF" />
                    <YAxis stroke="#FFFFFF" />
                    <RechartsTooltip contentStyle={{ backgroundColor: "#050505", border: "2px solid #FFFFFF", color: "#FFFFFF" }} />
                    <Legend wrapperStyle={{ color: "#FFFFFF" }} />
                    <Bar dataKey="total" fill="#3b82f6" name="Total Decisions" />
                    <Bar dataKey="avgConfidence" fill="#FFFFFF" name="Avg Confidence" />
                  </BarChart>
                </ResponsiveContainer>
              </CardContent>
            </Card>
          )}

          {/* Time Series */}
          {analytics.timeSeries.length > 0 && (
            <Card className="border border-slate-200 bg-white">
              <CardHeader className="border-b border-slate-200">
                <CardTitle className="flex items-center gap-2 text-slate-900 font-mono font-black uppercase tracking-tight">
                  <TrendingUp className="h-5 w-5 text-blue-600" />
                  Decision Trends Over Time
                </CardTitle>
                <CardDescription className="text-slate-500 font-mono">Monthly decision volume and outcomes</CardDescription>
              </CardHeader>
              <CardContent className="text-slate-900">
                <ResponsiveContainer width="100%" height={300}>
                  <LineChart data={analytics.timeSeries}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#FFFFFF" opacity={0.2} />
                    <XAxis dataKey="date" stroke="#FFFFFF" />
                    <YAxis stroke="#FFFFFF" />
                    <RechartsTooltip contentStyle={{ backgroundColor: "#050505", border: "2px solid #FFFFFF", color: "#FFFFFF" }} />
                    <Legend wrapperStyle={{ color: "#FFFFFF" }} />
                    <Line type="monotone" dataKey="decisions" stroke="#3b82f6" name="Total Decisions" />
                    <Line type="monotone" dataKey="positive" stroke="#FFFFFF" name="Positive" />
                    <Line type="monotone" dataKey="negative" stroke="#FFFFFF" name="Negative" />
                    <Line type="monotone" dataKey="pending" stroke="#FFFFFF" name="Pending" />
                  </LineChart>
                </ResponsiveContainer>
              </CardContent>
            </Card>
          )}

          {/* Action Type Mix */}
          {analytics.actionTypeStats.length > 0 && (
            <Card className="border border-slate-200 bg-white">
              <CardHeader className="border-b border-slate-200">
                <CardTitle className="flex items-center gap-2 text-slate-900 font-mono font-black uppercase tracking-tight">
                  <BarChart3 className="h-5 w-5 text-blue-600" />
                  Action Type Mix
                </CardTitle>
                <CardDescription className="text-slate-500 font-mono">Outcomes by decision action</CardDescription>
              </CardHeader>
              <CardContent className="text-slate-900">
                <ResponsiveContainer width="100%" height={300}>
                  <BarChart data={analytics.actionTypeStats.slice(0, 10)}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#FFFFFF" opacity={0.2} />
                    <XAxis dataKey="action" angle={-25} textAnchor="end" height={70} stroke="#FFFFFF" />
                    <YAxis stroke="#FFFFFF" />
                    <RechartsTooltip contentStyle={{ backgroundColor: "#050505", border: "2px solid #FFFFFF", color: "#FFFFFF" }} />
                    <Legend wrapperStyle={{ color: "#FFFFFF" }} />
                    <Bar dataKey="positive" stackId="a" fill="#3b82f6" name="Positive" />
                    <Bar dataKey="negative" stackId="a" fill="#FFFFFF" name="Negative" />
                    <Bar dataKey="pending" stackId="a" fill="#FFFFFF" name="Pending" />
                  </BarChart>
                </ResponsiveContainer>
              </CardContent>
            </Card>
          )}

          {/* Action Type Conversion Rate */}
          {actionConversionSeries.length > 0 && (
            <Card className="border border-slate-200 bg-white">
              <CardHeader className="border-b border-slate-200">
                <CardTitle className="flex items-center gap-2 text-slate-900 font-mono font-black uppercase tracking-tight">
                  <BarChart3 className="h-5 w-5 text-blue-600" />
                  Action Type Conversion Rate
                </CardTitle>
                <CardDescription className="text-slate-500 font-mono">Positive rate by action type</CardDescription>
              </CardHeader>
              <CardContent className="text-slate-900">
                <ResponsiveContainer width="100%" height={260}>
                  <BarChart data={actionConversionSeries.slice(0, 10)}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#FFFFFF" opacity={0.2} />
                    <XAxis dataKey="action" angle={-25} textAnchor="end" height={70} stroke="#FFFFFF" />
                    <YAxis stroke="#FFFFFF" />
                    <RechartsTooltip contentStyle={{ backgroundColor: "#050505", border: "2px solid #FFFFFF", color: "#FFFFFF" }} />
                    <Legend wrapperStyle={{ color: "#FFFFFF" }} />
                    <Bar dataKey="conversionRate" fill="#3b82f6" name="Positive Rate %" />
                    <Bar dataKey="total" fill="#FFFFFF" name="Decisions" />
                  </BarChart>
                </ResponsiveContainer>
              </CardContent>
            </Card>
          )}

          {/* Partner Win Rate */}
          {analytics.partnerStats.length > 0 && (
            <Card className="border border-slate-200 bg-white">
              <CardHeader className="border-b border-slate-200">
                <CardTitle className="flex items-center gap-2 text-slate-900 font-mono font-black uppercase tracking-tight">
                  <BarChart3 className="h-5 w-5 text-blue-600" />
                  Partner Win Rate
                </CardTitle>
                <CardDescription className="text-slate-500 font-mono">Win rate by partner (top 10 by volume)</CardDescription>
              </CardHeader>
              <CardContent className="text-slate-900">
                <ResponsiveContainer width="100%" height={260}>
                  <BarChart data={analytics.partnerStats.slice(0, 10)}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#FFFFFF" opacity={0.2} />
                    <XAxis dataKey="partner" angle={-25} textAnchor="end" height={70} stroke="#FFFFFF" />
                    <YAxis stroke="#FFFFFF" />
                    <RechartsTooltip contentStyle={{ backgroundColor: "#050505", border: "2px solid #FFFFFF", color: "#FFFFFF" }} />
                    <Legend wrapperStyle={{ color: "#FFFFFF" }} />
                    <Bar dataKey="winRate" fill="#3b82f6" name="Win Rate %" />
                    <Bar dataKey="totalDecisions" fill="#FFFFFF" name="Decisions" />
                  </BarChart>
                </ResponsiveContainer>
              </CardContent>
            </Card>
          )}

          {/* Decision Velocity */}
          {analytics.decisionVelocity.length > 0 && (
            <Card className="border border-slate-200 bg-white">
              <CardHeader className="border-b border-slate-200">
                <CardTitle className="flex items-center gap-2 text-slate-900 font-mono font-black uppercase tracking-tight">
                  <Clock className="h-5 w-5 text-blue-600" />
                  Decision Velocity Trend
                </CardTitle>
                <CardDescription className="text-slate-500 font-mono">Average decision time over time</CardDescription>
              </CardHeader>
              <CardContent className="text-slate-900">
                <ResponsiveContainer width="100%" height={250}>
                  <LineChart data={analytics.decisionVelocity}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#FFFFFF" opacity={0.2} />
                    <XAxis dataKey="date" stroke="#FFFFFF" />
                    <YAxis stroke="#FFFFFF" />
                    <RechartsTooltip contentStyle={{ backgroundColor: "#050505", border: "2px solid #FFFFFF", color: "#FFFFFF" }} />
                    <Legend wrapperStyle={{ color: "#FFFFFF" }} />
                    <Line type="monotone" dataKey="avgDays" stroke="#3b82f6" name="Avg Days" />
                  </LineChart>
                </ResponsiveContainer>
              </CardContent>
            </Card>
          )}

          {/* Outcome Rate Trend */}
          {analytics.outcomeRateSeries.length > 0 && (
            <Card className="border border-slate-200 bg-white">
              <CardHeader className="border-b border-slate-200">
                <CardTitle className="flex items-center gap-2 text-slate-900 font-mono font-black uppercase tracking-tight">
                  <TrendingUp className="h-5 w-5 text-blue-600" />
                  Positive Rate Trend
                </CardTitle>
                <CardDescription className="text-slate-500 font-mono">Monthly positive rate across decisions</CardDescription>
              </CardHeader>
              <CardContent className="text-slate-900">
                <ResponsiveContainer width="100%" height={250}>
                  <LineChart data={analytics.outcomeRateSeries}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#FFFFFF" opacity={0.2} />
                    <XAxis dataKey="date" stroke="#FFFFFF" />
                    <YAxis stroke="#FFFFFF" />
                    <RechartsTooltip contentStyle={{ backgroundColor: "#050505", border: "2px solid #FFFFFF", color: "#FFFFFF" }} />
                    <Legend wrapperStyle={{ color: "#FFFFFF" }} />
                    <Line type="monotone" dataKey="positiveRate" stroke="#3b82f6" name="Positive Rate %" />
                  </LineChart>
                </ResponsiveContainer>
              </CardContent>
            </Card>
          )}

          {/* Cumulative Decisions */}
          {analytics.cumulativeSeries.length > 0 && (
            <Card className="border border-slate-200 bg-white">
              <CardHeader className="border-b border-slate-200">
                <CardTitle className="flex items-center gap-2 text-slate-900 font-mono font-black uppercase tracking-tight">
                  <TrendingUp className="h-5 w-5 text-blue-600" />
                  Cumulative Decisions
                </CardTitle>
                <CardDescription className="text-slate-500 font-mono">Total decisions over time</CardDescription>
              </CardHeader>
              <CardContent className="text-slate-900">
                <ResponsiveContainer width="100%" height={250}>
                  <LineChart data={analytics.cumulativeSeries}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#FFFFFF" opacity={0.2} />
                    <XAxis dataKey="date" stroke="#FFFFFF" />
                    <YAxis stroke="#FFFFFF" />
                    <RechartsTooltip contentStyle={{ backgroundColor: "#050505", border: "2px solid #FFFFFF", color: "#FFFFFF" }} />
                    <Legend wrapperStyle={{ color: "#FFFFFF" }} />
                    <Line type="monotone" dataKey="cumulativeDecisions" stroke="#3b82f6" name="Cumulative Decisions" />
                  </LineChart>
                </ResponsiveContainer>
              </CardContent>
            </Card>
          )}

          {/* Confidence vs Positive Rate */}
          {confidenceRateSeries.length > 0 && (
            <Card className="border border-slate-200 bg-white">
              <CardHeader className="border-b border-slate-200">
                <CardTitle className="flex items-center gap-2 text-slate-900 font-mono font-black uppercase tracking-tight">
                  <TrendingUp className="h-5 w-5 text-blue-600" />
                  Confidence vs Positive Rate
                </CardTitle>
                <CardDescription className="text-slate-500 font-mono">How confidence bands correlate with outcomes</CardDescription>
              </CardHeader>
              <CardContent>
                <ResponsiveContainer width="100%" height={250}>
                  <LineChart data={confidenceRateSeries}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#FFFFFF" opacity={0.2} />
                    <XAxis dataKey="range" stroke="#FFFFFF" />
                    <YAxis stroke="#FFFFFF" />
                    <RechartsTooltip contentStyle={{ backgroundColor: "#050505", border: "2px solid #FFFFFF", color: "#FFFFFF" }} />
                    <Legend wrapperStyle={{ color: "#FFFFFF" }} />
                    <Line type="monotone" dataKey="positiveRate" stroke="#3b82f6" name="Positive Rate %" />
                    <Line type="monotone" dataKey="total" stroke="#FFFFFF" name="Decisions" />
                  </LineChart>
                </ResponsiveContainer>
              </CardContent>
            </Card>
          )}

          {/* Top Startups */}
          {analytics.startupStats.length > 0 && (
            <Card className="border border-slate-200 bg-white">
              <CardHeader className="border-b border-slate-200">
                <CardTitle className="text-slate-900 font-mono font-black uppercase tracking-tight">Top Startups by Decision Volume</CardTitle>
                <CardDescription className="text-slate-500 font-mono">Most discussed companies and outcomes</CardDescription>
              </CardHeader>
              <CardContent className="text-slate-900">
                <div className="overflow-x-auto">
                  <table className="w-full text-sm font-mono">
                    <thead>
                      <tr className="border-b border-slate-200">
                        <th className="text-left p-2 text-slate-900 font-bold">Startup</th>
                        <th className="text-right p-2 text-slate-900 font-bold">Total</th>
                        <th className="text-right p-2 text-slate-900 font-bold">Positive</th>
                        <th className="text-right p-2 text-slate-900 font-bold">Negative</th>
                        <th className="text-right p-2 text-slate-900 font-bold">Pending</th>
                        <th className="text-right p-2 text-slate-900 font-bold">Avg Confidence</th>
                      </tr>
                    </thead>
                    <tbody>
                      {analytics.startupStats.slice(0, 10).map((startup) => (
                        <tr key={startup.startupName} className="border-b border-slate-300 hover:bg-blue-600/5">
                          <td className="p-2 font-bold text-slate-900">{startup.startupName}</td>
                          <td className="text-right p-2 text-slate-900">{startup.total}</td>
                          <td className="text-right p-2 text-blue-600">{startup.positive}</td>
                          <td className="text-right p-2 text-slate-400">{startup.negative}</td>
                          <td className="text-right p-2 text-slate-500">{startup.pending}</td>
                          <td className="text-right p-2 text-slate-900">{startup.avgConfidence}%</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </CardContent>
            </Card>
          )}

          {/* Sector Conversion Rates Table */}
          {analytics.sectorStats.length > 0 && (
            <Card className="border border-slate-200 bg-white">
              <CardHeader className="border-b border-slate-200">
                <CardTitle className="text-slate-900 font-mono font-black uppercase tracking-tight">Sector Conversion Rates</CardTitle>
                <CardDescription className="text-slate-500 font-mono">
                  <TooltipProvider>
                    <Tooltip>
                      <TooltipTrigger asChild>
                        <span className="cursor-help">
                          Detailed sector performance metrics
                          <span className="ml-1">в„№пёЏ</span>
                        </span>
                      </TooltipTrigger>
                      <TooltipContent className="bg-white border border-slate-200 text-slate-900">
                        <p className="max-w-xs font-mono">
                          Conversion Rate = (Positive Decisions / Total Decisions) Г— 100%
                          <br />
                          Shows what % of decisions in each sector resulted in positive outcomes.
                        </p>
                      </TooltipContent>
                    </Tooltip>
                  </TooltipProvider>
                </CardDescription>
              </CardHeader>
              <CardContent className="text-slate-900">
                <div className="overflow-x-auto">
                  <table className="w-full text-sm font-mono">
                    <thead>
                      <tr className="border-b border-slate-200">
                        <th className="text-left p-2 text-slate-900 font-bold">Sector</th>
                        <th className="text-right p-2 text-slate-900 font-bold">Total</th>
                        <th className="text-right p-2 text-slate-900 font-bold">Positive</th>
                        <th className="text-right p-2 text-slate-900 font-bold">Negative</th>
                        <th className="text-right p-2 text-slate-900 font-bold">Pending</th>
                        <th className="text-right p-2 text-slate-900 font-bold">Conversion %</th>
                        <th className="text-right p-2 text-slate-900 font-bold">Avg Confidence</th>
                      </tr>
                    </thead>
                    <tbody>
                      {analytics.sectorStats.map((sector) => (
                        <tr key={sector.sector} className="border-b border-slate-300 hover:bg-blue-600/5">
                          <td className="p-2 font-bold text-slate-900">{sector.sector}</td>
                          <td className="text-right p-2 text-slate-900">{sector.total}</td>
                          <td className="text-right p-2 text-blue-600">{sector.positive}</td>
                          <td className="text-right p-2 text-slate-400">{sector.negative}</td>
                          <td className="text-right p-2 text-slate-500">{sector.pending}</td>
                          <td className="text-right p-2 font-bold text-slate-900">{sector.conversionRate}%</td>
                          <td className="text-right p-2 text-slate-900">{sector.avgConfidence}%</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </CardContent>
            </Card>
          )}
        </>
        );})()}
    </div>
  );
}

// ============================================================================
// MAIN CIS COMPONENT
// ============================================================================

export default function Dashboard() {
  const { user, profile, signOut } = useAuth();
  const { toast } = useToast();
  const [scopes, setScopes] = useState<ScopeItem[]>(initialScopes);
  const [expandedScopeGroups, setExpandedScopeGroups] = useState<Set<string>>(new Set());
  const [threads, setThreads] = useState<Thread[]>(initialThreads);
  const [activeThread, setActiveThread] = useState<string>(initialThreads[0]?.id ?? "");
  const [messages, setMessages] = useState<Message[]>(initialMessages);
  const messagesRef = useRef<Message[]>(initialMessages);
  useEffect(() => { messagesRef.current = messages; }, [messages]);
  const [input, setInput] = useState("");
  const [chatIsLoading, setChatIsLoading] = useState(false);
  const [isClaudeLoading, setIsClaudeLoading] = useState(false);
  const [chatLoadingStage, setChatLoadingStage] = useState<string>("Analyzing your question...");
  const [chatLoaded, setChatLoaded] = useState(false);
  const [isInitialLoad, setIsInitialLoad] = useState(true);
  const [costLog, setCostLog] = useState<
    Array<{
      ts: string;
      question: string;
      estInputTokens: number;
      estOutputTokens: number;
      estCostUsd: number;
    }>
  >([]);
  const [lastEvidence, setLastEvidence] = useState<{
    question: string;
    docs: Array<{
      id: string;
      title: string | null;
      file_name: string | null;
      raw_content: string | null;
      extracted_json?: Record<string, any> | null;
      created_at: string;
      storage_path: string | null;
    }>;
    decisions: Decision[];
  } | null>(null);
  const [lastEvidenceThreadId, setLastEvidenceThreadId] = useState<string | null>(null);
  const [activeTab, setActiveTab] = useState("chat");
  const [webSearchEnabled, setWebSearchEnabled] = useState(false);
  const [multiAgentEnabled, setMultiAgentEnabled] = useState(false);
  const [editingMessageId, setEditingMessageId] = useState<string | null>(null);
  
  // Company Connections state for visual graph and decision logging
  const [companyConnections, setCompanyConnections] = useState<Array<{
    id: string;
    source_company_name: string;
    target_company_name: string;
    source_document_id?: string | null;
    target_document_id?: string | null;
    connection_type: "BD" | "INV" | "Knowledge" | "Partnership" | "Project";
    connection_status: "To Connect" | "Connected" | "Rejected" | "In Progress" | "Completed";
    ai_reasoning?: string | null;
    notes?: string | null;
    created_at: string;
  }>>([]);
  
  // Pending relationship reviews from knowledge graph
  const [pendingReviews, setPendingReviews] = useState<Array<{
    id: string;
    relation_type: string;
    confidence: number;
    properties: Record<string, any>;
    source_document_id: string | null;
    created_at: string;
    source_entity: { name: string; entity_type: string } | null;
    target_entity: { name: string; entity_type: string } | null;
  }>>([]);
  
  // Company Cards вЂ” unified view of companies with documents, connections, KPIs
  const [companyCards, setCompanyCards] = useState<Array<{
    company_id: string;
    company_name: string;
    entity_type?: string;
    company_properties: Record<string, any>;
    document_count: number;
    document_ids?: string[];
    connection_count?: number;
    connection_ids?: string[];
    kpi_count?: number;
    kpi_summary?: Record<string, any>;
    relationship_count?: number;
    related_companies?: string[];
    created_at?: string;
  }>>([]);
  const [logDecisionDialogOpen, setLogDecisionDialogOpen] = useState(false);
  const [pendingDecisionContext, setPendingDecisionContext] = useState<{
    aiReasoning: string;
    sourceDocIds?: string[];
  } | null>(null);

  // My Account — editable company info (used by Account tab)
  const [companyAccountName, setCompanyAccountName] = useState("");
  const [companyAccountDescription, setCompanyAccountDescription] = useState("");
  const [companyAccountLoading, setCompanyAccountLoading] = useState(false);
  const [companyAccountSaving, setCompanyAccountSaving] = useState(false);
  
  const embeddingsDisabledRef = useRef(false);
  const chatContainerRef = useRef<HTMLDivElement>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const chatAbortRef = useRef<AbortController | null>(null);
  const sessionRefreshInFlightRef = useRef<Promise<void> | null>(null);

  // Helper function to scroll chat container to bottom
  const scrollChatToBottom = useCallback(() => {
    if (chatContainerRef.current) {
      const container = chatContainerRef.current;
      container.scrollTop = container.scrollHeight;
    }
  }, []);
  const [activeEventId, setActiveEventId] = useState<string | null>(null);
  const [initialDriveSyncConfig, setInitialDriveSyncConfig] = useState<{
    folderId: string;
    folderName: string;
    folders: Array<{ id: string; name: string; category?: string }>;
    lastSyncAt: string | null;
  } | null>(null);
  const [decisions, setDecisions] = useState<Decision[]>([]);
  const [documents, setDocuments] = useState<
    Array<{ id: string; title: string | null; storage_path: string | null; folder_id?: string | null }>
  >([]);
  const documentsRef = useRef<Array<{ id: string; title: string | null; storage_path: string | null; folder_id?: string | null }>>([]);
  useEffect(() => {
    documentsRef.current = documents;
  }, [documents]);

  const readLocalChatCache = useCallback((): LocalChatMessage[] => {
    if (typeof window === "undefined") return [];
    try {
      const raw = localStorage.getItem(LOCAL_CHAT_CACHE_KEY);
      if (!raw) return [];
      const parsed = JSON.parse(raw);
      return Array.isArray(parsed) ? parsed : [];
    } catch {
      return [];
    }
  }, []);

  const writeLocalChatCache = useCallback((items: LocalChatMessage[]) => {
    if (typeof window === "undefined") return;
    localStorage.setItem(LOCAL_CHAT_CACHE_KEY, JSON.stringify(items));
  }, []);
  const [sources, setSources] = useState<SourceRecord[]>([]);
  const [tasks, setTasks] = useState<Task[]>([]);
  const [sourceFolders, setSourceFolders] = useState<SourceFolder[]>([]);
  const [foldersExpanded, setFoldersExpanded] = useState(false);
  const [draftDecision, setDraftDecision] = useState<{
    startupName: string;
    sector?: string;
    stage?: string;
  } | null>(null);

  // LP can only see Dashboard and Account
  useEffect(() => {
    if (profile && profile.role === "lp" && activeTab !== "overview" && activeTab !== "account") {
      setActiveTab("overview");
    }
  }, [profile?.role, activeTab]);

  // Fetch company context when Account tab is open (for editable company name/description)
  useEffect(() => {
    if (activeTab !== "account" || !profile?.organization_id) return;
    let cancelled = false;
    setCompanyAccountLoading(true);
    getCompanyContext(profile.organization_id)
      .then((ctx) => {
        if (!cancelled) {
          setCompanyAccountName(ctx.company_name ?? "");
          setCompanyAccountDescription(ctx.company_description ?? "");
        }
      })
      .catch(() => {
        if (!cancelled) {
          setCompanyAccountName("");
          setCompanyAccountDescription("");
        }
      })
      .finally(() => {
        if (!cancelled) setCompanyAccountLoading(false);
      });
    return () => { cancelled = true; };
  }, [activeTab, profile?.organization_id]);

  // Normalize folder/root name for matching: trim, lower, strip parentheticals, collapse spaces.
  const normalizeFolderMatch = useCallback((s: string): string => {
    return (s || "")
      .trim()
      .toLowerCase()
      .replace(/\s*\([^)]*\)\s*/g, " ")
      .replace(/\s+/g, " ")
      .trim();
  }, []);
  // For root/first-segment comparison only: also strip trailing " - suffix" so "Root - Pipeline" matches "Root".
  const normRootOrSegment = useCallback((s: string): string => {
    return normalizeFolderMatch(s).replace(/\s*-\s*.*$/, "").trim();
  }, [normalizeFolderMatch]);

  // Backfill source_folder categories from Drive sync config.
  // Three-pass matching: exact/prefix -> segment -> contains.
  useEffect(() => {
    if (!activeEventId || !initialDriveSyncConfig?.folders?.length || !sourceFolders.length) return;
    const driveFolders = initialDriveSyncConfig.folders;
    const updates: Array<{ id: string; category: string }> = [];
    const assigned = new Set<string>();

    // Pass 1: exact name and prefix matches (high confidence)
    for (const df of driveFolders) {
      const rootName = (df.name || "").trim();
      if (!rootName) continue;
      const wantCategory = df.category ?? "Projects";
      const rootNorm = normalizeFolderMatch(rootName);
      const prefixNorm = rootNorm + " / ";
      for (const sf of sourceFolders) {
        if (assigned.has(sf.id)) continue;
        const name = (sf.name || "").trim();
        const nameNorm = normalizeFolderMatch(name);
        const exactOrPrefix = nameNorm === rootNorm || nameNorm.startsWith(prefixNorm);
        if (!exactOrPrefix) continue;
        assigned.add(sf.id);
        if ((sf.category || "Projects") !== wantCategory) {
          updates.push({ id: sf.id, category: wantCategory });
        }
      }
    }

    // Pass 2: segment-level matches for remaining unassigned source folders
    for (const df of driveFolders) {
      const rootName = (df.name || "").trim();
      if (!rootName) continue;
      const wantCategory = df.category ?? "Projects";
      const rootNormSegment = normRootOrSegment(rootName);
      for (const sf of sourceFolders) {
        if (assigned.has(sf.id)) continue;
        const name = (sf.name || "").trim();
        const firstSegmentNorm = normRootOrSegment(name.split(/\s*\/\s*/)[0] || name);
        if (firstSegmentNorm !== rootNormSegment) continue;
        assigned.add(sf.id);
        if ((sf.category || "Projects") !== wantCategory) {
          updates.push({ id: sf.id, category: wantCategory });
        }
      }
    }

    // Pass 3: contains вЂ” broadest match for any remaining unassigned
    for (const df of driveFolders) {
      const rootName = (df.name || "").trim();
      if (!rootName) continue;
      const wantCategory = df.category ?? "Projects";
      const rootNorm = normalizeFolderMatch(rootName);
      for (const sf of sourceFolders) {
        if (assigned.has(sf.id)) continue;
        const segments = (sf.name || "").trim().split(/\s*\/\s*/);
        const matchesAny = segments.some((seg: string) => {
          const segNorm = normalizeFolderMatch(seg);
          return segNorm === rootNorm || segNorm.includes(rootNorm) || rootNorm.includes(segNorm);
        });
        if (!matchesAny) continue;
        assigned.add(sf.id);
        if ((sf.category || "Projects") !== wantCategory) {
          updates.push({ id: sf.id, category: wantCategory });
        }
      }
    }

    if (updates.length === 0) return;
    let cancelled = false;
    (async () => {
      let errors = 0;
      for (let i = 0; i < updates.length; i++) {
        if (cancelled) return;
        const { id, category } = updates[i];
        const { error } = await updateFolderCategory(id, category);
        if (error) {
          console.warn("[CIS] Folder category update failed:", id, error.message);
          errors++;
          continue;
        }
      }
      if (cancelled) return;
      const { data } = await getSourceFoldersByEvent(activeEventId);
      setSourceFolders((data || []) as SourceFolder[]);
    })();
    return () => { cancelled = true; };
  }, [activeEventId, initialDriveSyncConfig, sourceFolders, normalizeFolderMatch, normRootOrSegment]);

  // Auto-expand folder list if any folder is selected
  useEffect(() => {
    if (scopes.some((scope) => scope.type === "folder" && scope.checked)) {
      setFoldersExpanded(true);
    }
  }, [scopes]);
  const [draftDocumentId, setDraftDocumentId] = useState<string | null>(null);
  const [viewingDocument, setViewingDocument] = useState<{
    id: string;
    title: string | null;
    raw_content: string | null;
    extracted_json: Record<string, any> | null;
    file_name: string | null;
    storage_path: string | null;
  } | null>(null);

  const handleOpenDocument = useCallback(
    async (documentId: string) => {
      const { data: doc, error } = await getDocumentById(documentId);
      if (error || !doc) {
        toast({
          title: "Document not found",
          description: "Could not load document details.",
          variant: "destructive",
        });
        return;
      }
      const docData = doc as any;
      setViewingDocument({
        id: docData.id,
        title: docData.title,
        raw_content: docData.raw_content || null,
        extracted_json: docData.extracted_json || null,
        file_name: docData.file_name || null,
        storage_path: docData.storage_path || null,
      });
    },
    [toast]
  );

  const handleLogDecisionFromDocument = useCallback(() => {
    if (!viewingDocument) return;
    setDraftDecision({
      startupName: viewingDocument.title || viewingDocument.file_name || "Decision from document",
    });
    setDraftDocumentId(viewingDocument.id);
    setViewingDocument(null);
    setActiveTab("decisions");
  }, [viewingDocument]);

  const handleCreateSource = useCallback(
    async (
      payload: {
        title: string | null;
        source_type: SourceRecord["source_type"];
        external_url: string | null;
        storage_path?: string | null;
        tags: string[] | null;
        notes: string | null;
        status: SourceRecord["status"];
      },
      eventIdOverride?: string | null
    ) => {
      const eventId = eventIdOverride ?? activeEventId;
      if (!eventId) {
        throw new Error("No active event available.");
      }
      const userId = user?.id || profile?.id || null;
      const { data, error } = await insertSource(eventId, {
        ...payload,
        storage_path: payload.storage_path || null,
        created_by: userId,
      });
      if (error || !data) {
        throw new Error("Supabase rejected the source.");
      }
      setSources((prev) => [data as SourceRecord, ...prev]);
    },
    [activeEventId, profile, user]
  );

  const handleCreateFolder = useCallback(
    async (name: string, category?: string): Promise<SourceFolder | null> => {
      const eventId = activeEventId;
      if (!eventId) {
        toast({ title: "No active event", description: "Cannot create folder.", variant: "destructive" });
        return null;
      }
      const userId = user?.id || profile?.id || null;
      const { data, error } = await insertSourceFolder(eventId, {
        name,
        created_by: userId,
        category: category || "Projects",
      });
      if (error || !data) {
        toast({ title: "Folder creation failed", description: error?.message || "Unknown error", variant: "destructive" });
        return null;
      }
      const folder = data as SourceFolder;
      setSourceFolders((prev) => [folder, ...prev]);
      return folder;
    },
    [activeEventId, profile, user, toast]
  );

  const handleDeleteFolderAndContents = useCallback(
    async (folderId: string): Promise<{ docCount: number } | { error: string }> => {
      const result = await deleteFolderAndContents(folderId);
      if (result.error) return { error: result.error };
      if (!activeEventId) return result;
      const { data: refreshedFolders } = await getSourceFoldersByEvent(activeEventId);
      const { data: refreshedDocs } = await getDocumentsByEvent(activeEventId);
      setSourceFolders((refreshedFolders || []) as SourceFolder[]);
      setDocuments(
        (refreshedDocs || []).map((doc: any) => ({
          id: doc.id,
          title: doc.title,
          storage_path: doc.storage_path || null,
          folder_id: doc.folder_id || null,
        }))
      );
      return { docCount: result.docCount };
    },
    [activeEventId]
  );

  const handleFolderCategoryUpdated = useCallback(
    async (folderId: string, category: string) => {
      const { error } = await updateFolderCategory(folderId, category);
      if (!error) {
        setSourceFolders((prev) =>
          prev.map((sf) => (sf.id === folderId ? { ...sf, category } : sf))
        );
      }
    },
    []
  );

  /** Three-pass matching to sync Drive root categories to source_folders.
   *  Pass 1: exact/prefix. Pass 2: first-segment. Pass 3: contains (broadest). */
  const handleSyncCategoriesFromDrive = useCallback(async () => {
    if (!activeEventId || !initialDriveSyncConfig?.folders?.length || !sourceFolders.length) {
      console.warn("[SyncCategories] Skipped: missing event, drive folders, or source folders");
      return;
    }
    const driveFolders = initialDriveSyncConfig.folders;
    const updates: Array<{ id: string; category: string }> = [];
    const assigned = new Set<string>();

    // Pass 1: exact name or path-prefix
    for (const df of driveFolders) {
      const rootName = (df.name || "").trim();
      if (!rootName) continue;
      const wantCategory = df.category ?? "Projects";
      const rootNorm = normalizeFolderMatch(rootName);
      const prefixNorm = rootNorm + " / ";
      for (const sf of sourceFolders) {
        if (assigned.has(sf.id)) continue;
        const nameNorm = normalizeFolderMatch((sf.name || "").trim());
        if (nameNorm !== rootNorm && !nameNorm.startsWith(prefixNorm)) continue;
        assigned.add(sf.id);
        if ((sf.category || "Projects") !== wantCategory) {
          updates.push({ id: sf.id, category: wantCategory });
        }
      }
    }
    // Pass 2: first path-segment
    for (const df of driveFolders) {
      const rootName = (df.name || "").trim();
      if (!rootName) continue;
      const wantCategory = df.category ?? "Projects";
      const rootNormSegment = normRootOrSegment(rootName);
      for (const sf of sourceFolders) {
        if (assigned.has(sf.id)) continue;
        const firstSegmentNorm = normRootOrSegment(((sf.name || "").trim().split(/\s*\/\s*/)[0] || sf.name || ""));
        if (firstSegmentNorm !== rootNormSegment) continue;
        assigned.add(sf.id);
        if ((sf.category || "Projects") !== wantCategory) {
          updates.push({ id: sf.id, category: wantCategory });
        }
      }
    }
    // Pass 3: contains вЂ” any path segment of source folder matches drive root
    for (const df of driveFolders) {
      const rootName = (df.name || "").trim();
      if (!rootName) continue;
      const wantCategory = df.category ?? "Projects";
      const rootNorm = normalizeFolderMatch(rootName);
      for (const sf of sourceFolders) {
        if (assigned.has(sf.id)) continue;
        const segments = (sf.name || "").trim().split(/\s*\/\s*/);
        const matchesAny = segments.some((seg: string) => {
          const segNorm = normalizeFolderMatch(seg);
          return segNorm === rootNorm || segNorm.includes(rootNorm) || rootNorm.includes(segNorm);
        });
        if (!matchesAny) continue;
        assigned.add(sf.id);
        if ((sf.category || "Projects") !== wantCategory) {
          updates.push({ id: sf.id, category: wantCategory });
        }
      }
    }

    let errors = 0;
    for (const { id, category } of updates) {
      const { error } = await updateFolderCategory(id, category);
      if (error) { console.error("[SyncCategories] Failed:", id, error.message); errors++; }
    }
    const { data } = await getSourceFoldersByEvent(activeEventId);
    setSourceFolders((data || []) as SourceFolder[]);
    if (errors > 0) throw new Error(`${errors} of ${updates.length} updates failed`);
  }, [activeEventId, initialDriveSyncConfig, sourceFolders, normalizeFolderMatch, normRootOrSegment]);

  const handleFoldersCategoriesSaved = useCallback(
    async (updates: Array<{ id: string; category: string }>) => {
      for (const { id, category } of updates) {
        await updateFolderCategory(id, category);
      }
      setSourceFolders((prev) =>
        prev.map((f) => {
          const u = updates.find((x) => x.id === f.id);
          return u ? { ...f, category: u.category } : f;
        })
      );
    },
    []
  );

  const ensureActiveEventId = useCallback(async () => {
    if (!profile) {
      console.error("ensureActiveEventId: No profile");
      return null;
    }
    const { data: orgData, error: orgError } = await ensureOrganizationForUser(profile);
    if (orgError || !orgData?.organization) {
      console.error("ensureActiveEventId: Organization error:", orgError);
      toast({
        title: "Organization missing",
        description: orgError?.message || "We could not load your organization.",
        variant: "destructive",
      });
      return null;
    }
    const { data: event, error: eventError } = await ensureActiveEventForOrg(orgData.organization.id);
    if (eventError) {
      const isRlsOrAuth = eventError?.code === "42501" || eventError?.code === "PGRST301" || (eventError?.message || "").includes("row-level security");
      toast({
        title: "Event creation failed",
        description: isRlsOrAuth
          ? "Your session may have expired. Try signing out and signing back in, or wait a moment and try again."
          : (eventError.message || "Could not create an active event. Please refresh."),
        variant: "destructive",
      });
      return null;
    }
    if (!event) {
      console.error("ensureActiveEventId: No event returned");
      toast({
        title: "No active event",
        description: "Could not create an active event. Please refresh.",
        variant: "destructive",
      });
      return null;
    }
    setActiveEventId(event.id);
    return event.id;
  }, [profile, toast]);

  const handleDeleteSource = useCallback(async (sourceId: string) => {
    const { error } = await deleteSource(sourceId);
    if (error) {
      return;
    }
    setSources((prev) => prev.filter((source) => source.id !== sourceId));
  }, []);

  // Load chat history on initial mount and when switching threads
  useEffect(() => {
    const loadChatHistory = async () => {
      if (!profile) return;
      // Never replace messages while the AI is still streaming a response
      if (isClaudeLoading || chatIsLoading) return;
      const eventId = activeEventId || (await ensureActiveEventId());
      if (!eventId) return;

      const mapSourceDocsFromIds = (sourceDocIds: unknown): SourceDoc[] | undefined => {
        const ids = Array.isArray(sourceDocIds)
          ? sourceDocIds.filter((id): id is string => typeof id === "string" && id.trim().length > 0)
          : [];
        if (ids.length === 0) return undefined;
        const titleById = new Map(
          documentsRef.current.map((doc) => [doc.id, doc.title || doc.storage_path?.split("/").pop() || "Document"] as const)
        );
        return ids.map((id) => ({ id, title: titleById.get(id) || "Document" }));
      };

      const mapMessageRow = (m: any): Message => ({
        id: m.id,
        author: (m.role === "assistant" ? "assistant" : "user") as "assistant" | "user",
        text: m.content,
        threadId: m.thread_id,
        sourceDocs: mapSourceDocsFromIds(m.source_doc_ids),
      });

      // Preserve in-memory enrichment (sourceDocs, contextLabels, etc.) across DB reloads
      const preserveEnrichment = (loaded: Message[]): Message[] => {
        const currentMsgs = messagesRef.current;
        if (!currentMsgs.length) return loaded;
        const enrichById = new Map<string, Partial<Message>>();
        const enrichBySignature = new Map<string, Partial<Message>>();
        for (const m of currentMsgs) {
          const extras: Partial<Message> = {};
          if (m.sourceDocs?.length) extras.sourceDocs = m.sourceDocs;
          if (m.contextLabels?.length) extras.contextLabels = m.contextLabels;
          if (m.verifiableSources?.length) extras.verifiableSources = m.verifiableSources;
          if (m.critic) extras.critic = m.critic;
          if (Object.keys(extras).length) {
            enrichById.set(m.id, extras);
            enrichBySignature.set(`${m.threadId}|${m.author}|${m.text}`, extras);
          }
        }
        if (!enrichById.size && !enrichBySignature.size) return loaded;
        return loaded.map((m) => {
          const signature = `${m.threadId}|${m.author}|${m.text}`;
          const extras = enrichById.get(m.id) || enrichBySignature.get(signature);
          if (!extras) return m;
          return {
            ...m,
            sourceDocs: m.sourceDocs?.length ? m.sourceDocs : extras.sourceDocs,
            contextLabels: m.contextLabels?.length ? m.contextLabels : extras.contextLabels,
            verifiableSources: m.verifiableSources?.length ? m.verifiableSources : extras.verifiableSources,
            critic: m.critic || extras.critic,
          };
        });
      };
      
      const mergeLocalMessages = (threadId: string, loadedMessages: Message[]) => {
        const cache = readLocalChatCache();
        if (!cache.length) return preserveEnrichment(loadedMessages);
        const otherThreads = cache.filter((m) => m.threadId !== threadId);
        const threadCache = cache.filter((m) => m.threadId === threadId);
        const existingKeys = new Set(loadedMessages.map((m) => `${m.author}|${m.text}`));
        const merged = [...loadedMessages];
        const remaining: LocalChatMessage[] = [];
        for (const localMsg of threadCache) {
          const key = `${localMsg.author}|${localMsg.text}`;
          if (!existingKeys.has(key)) {
            merged.push({
              id: localMsg.id,
              author: localMsg.author,
              text: localMsg.text,
              threadId: localMsg.threadId,
            });
            remaining.push(localMsg);
          }
        }
        writeLocalChatCache([...otherThreads, ...remaining]);
        return preserveEnrichment(merged);
      };

      try {
        const { data: threadRows } = await supabase
          .from("chat_threads")
          .select("*")
          .eq("event_id", eventId)
          .order("created_at", { ascending: true });
        const { data: messageRows } = await supabase
          .from("chat_messages")
          .select("*")
          .eq("event_id", eventId)
          .order("created_at", { ascending: true });

        if (threadRows?.length) {
          const mappedThreads = threadRows.map((t: any) => ({
            id: t.id,
            title: t.title,
            parentId: t.parent_id || undefined,
          }));
          setThreads(mappedThreads);
          
          // Determine which thread to use: activeThread if set, otherwise first thread
          let targetThreadId = activeThread || mappedThreads[0]?.id;
          
          // On initial load, set activeThread to first thread if not set
          if (isInitialLoad && !activeThread && mappedThreads[0]?.id) {
            targetThreadId = mappedThreads[0].id;
            setActiveThread(targetThreadId);
            setIsInitialLoad(false);
          }
          
          // Load messages for the target thread
          if (targetThreadId && messageRows?.length) {
            const threadMessages = messageRows
              .filter((m: any) => m.thread_id === targetThreadId)
              .map(mapMessageRow);
            setMessages(mergeLocalMessages(targetThreadId, threadMessages));
          } else if (messageRows?.length && !targetThreadId) {
            const mappedMessages = messageRows.map(mapMessageRow);
            setMessages(preserveEnrichment(mappedMessages));
          } else {
            // No messages found for this thread - clear messages array
            if (targetThreadId) {
              setMessages(mergeLocalMessages(targetThreadId, []));
            } else {
              setMessages([]);
            }
          }
        } else if (messageRows?.length) {
          const mappedMessages = messageRows.map(mapMessageRow);
          setMessages(preserveEnrichment(mappedMessages));
        } else {
          // No threads and no messages - ensure empty state
          setMessages([]);
        }
        setChatLoaded(true);
        if (isInitialLoad) {
          setIsInitialLoad(false);
        }
      } catch (error) {
        console.error("Failed to load chat history:", error);
        setChatLoaded(true); // Set to true even on error to prevent retries
        setIsInitialLoad(false);
      }
    };

    void loadChatHistory();
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [profile, activeEventId, activeThread, ensureActiveEventId, isInitialLoad, readLocalChatCache, writeLocalChatCache]);

  const getGoogleAccessToken = useCallback(async (forceRefresh = false): Promise<string | null> => {
    const tryResolve = async (doRefreshSession: boolean): Promise<string | null> => {
      let session: import("@supabase/supabase-js").Session | null = null;

      if (doRefreshSession) {
        try {
          const { data, error } = await supabase.auth.refreshSession();
          if (!error && data?.session) {
            session = data.session;
          }
        } catch {
          // 429 or network error — fall through to getSession cache
        }
      }

      if (!session) {
        const { data } = await supabase.auth.getSession();
        session = data.session;
      }

      const providerToken = session?.provider_token ?? null;
      const providerRefreshToken = session?.provider_refresh_token ?? null;
      if (providerToken || providerRefreshToken) {
        saveGoogleProviderTokens(providerToken, providerRefreshToken);
      }

      // 1) Backend-stored token (from backend OAuth flow)
      if (session?.access_token) {
        const backendToken = await getGoogleAccessTokenFromBackend(session.access_token);
        if (backendToken) return backendToken;
      }

      // 2) Session or stored provider token
      if (providerToken && !forceRefresh) return providerToken;
      const storedAccess = getStoredGoogleAccessToken();
      if (storedAccess && !forceRefresh) return storedAccess;

      // 3) Refresh Google token via backend (doesn't need Supabase session)
      const refreshToken = providerRefreshToken || getStoredGoogleRefreshToken();
      if (refreshToken) {
        const newToken = await refreshGoogleAccessToken(refreshToken);
        if (newToken) {
          setStoredGoogleAccessToken(newToken);
          return newToken;
        }
      }

      return providerToken ?? storedAccess ?? null;
    };

    let token = await tryResolve(forceRefresh);
    if (!token) {
      token = await tryResolve(true);
    }
    return token;
  }, []);

  const handleAutoLogDecision = useCallback(
    async (input: {
      draft: { startupName: string; sector?: string; stage?: string };
      conversion: AIConversionResponse;
      sourceType: "upload" | "paste" | "api";
      fileName: string | null;
      file: File | null;
      rawContent?: string | null;
      eventIdOverride?: string | null;
    }) => {
      const eventId = input.eventIdOverride ?? activeEventId;
      if (!eventId) {
        return;
      }
      let storagePath: string | null = null;
      if (input.file) {
        const safeName = input.fileName?.replace(/[^a-zA-Z0-9._-]/g, "_") || "document";
        const path = `${eventId}/${Date.now()}-${safeName}`;
        const { error: uploadError } = await supabase.storage
          .from("cis-documents")
          .upload(path, input.file, { upsert: true });
        if (!uploadError) {
          storagePath = path;
        }
      }

      const userId = profile?.id || user?.id || null;
      const { data: doc, error: docError } = await insertDocument(eventId, {
        title: input.draft.startupName,
        source_type: input.sourceType,
        file_name: input.fileName,
        storage_path: storagePath,
        detected_type: input.conversion.detectedType || "unknown",
        extracted_json: input.conversion as unknown as Record<string, any>,
        raw_content: input.rawContent || null,
        created_by: userId,
      });

      const docRecord = doc as { id?: string; title?: string | null; storage_path?: string | null; folder_id?: string | null } | null;
      const docId = docRecord?.id;
      if (docError) {
        console.error("Document insert error in auto-log:", docError);
        toast({
          title: "Document save failed",
          description: docError.message || "Could not save document.",
          variant: "destructive",
        });
        return;
      }
      if (!docId) {
        console.error("Document insert returned no ID:", doc);
        toast({
          title: "Document save failed",
          description: "Insert succeeded but no document ID returned.",
          variant: "destructive",
        });
        return;
      }

      // Index embeddings in background (non-blocking)
      indexDocumentEmbeddings(docId, input.rawContent || null, docRecord?.title || input.title || null).catch((err) => {
        console.error("Error indexing embeddings (non-fatal):", err);
      });
      setDocuments((prev) => [
        {
          id: docId,
          title: docRecord?.title || null,
          storage_path: docRecord?.storage_path || null,
          folder_id: docRecord?.folder_id || null,
        },
        ...prev,
      ]);

      const { data: decision, error } = await insertDecision(eventId, {
        actor_id: userId,
        actor_name: profile?.full_name || profile?.email || user?.email || "Unknown",
        action_type: "meeting",
        startup_name: input.draft.startupName,
        context: {
          sector: input.draft.sector || undefined,
          stage: input.draft.stage || undefined,
        },
        confidence_score: 70,
        outcome: "pending",
        notes: null,
        document_id: docId,
      });

      if (error || !decision) {
        return;
      }
      setDecisions((prev) => [mapDecisionRow(decision), ...prev]);
    },
    [activeEventId, profile, user]
  );

  const scopedMessages = useMemo(() => messages.filter((m) => m.threadId === activeThread), [messages, activeThread]);

  // Cycle through loading stage messages while chat is processing
  useEffect(() => {
    if (!chatIsLoading) {
      setChatLoadingStage("Analyzing your question...");
      return;
    }
    const stages = multiAgentEnabled ? [
      "Orchestrator routing query...",
      "Dispatching retrieval agents...",
      "Vector + Graph + KPI search...",
      "ColBERT reranking documents...",
      "Synthesizing multi-agent context...",
      "Generating initial answer...",
      "System 2: Reflecting on answer...",
      "System 2: Searching for gaps...",
      "System 2: Refining answer...",
      "Critic verifying...",
    ] : [
      "Analyzing your question...",
      "Searching documents...",
      "Retrieving relevant context...",
      "Thinking deeply...",
      "Synthesizing answer...",
      "Almost there...",
    ];
    let idx = 0;
    setChatLoadingStage(stages[0]);
    const interval = setInterval(() => {
      idx = Math.min(idx + 1, stages.length - 1);
      setChatLoadingStage(stages[idx]);
    }, 3000);
    return () => clearInterval(interval);
  }, [chatIsLoading, multiAgentEnabled]);

  useEffect(() => {
    let cancelled = false;
    let documentsChannel: ReturnType<typeof supabase.channel> | null = null;
    let decisionsChannel: ReturnType<typeof supabase.channel> | null = null;
    let sourcesChannel: ReturnType<typeof supabase.channel> | null = null;

    const loadDecisions = async () => {
      if (!profile) return;
      // Sync decisions from Supabase

      const { data: orgData, error: orgError } = await ensureOrganizationForUser(profile);
      if (orgError || !orgData?.organization) {
        console.error("Failed to ensure organization:", orgError);
        toast({
          title: "Organization error",
          description: orgError?.message || "Could not load your organization. Please refresh.",
          variant: "destructive",
        });
        return;
      }

      const { data: event, error: eventError } = await ensureActiveEventForOrg(orgData.organization.id);
      if (eventError) {
        console.error("Failed to ensure active event:", eventError);
        toast({
          title: "Event creation failed",
          description: eventError.message || "Could not create an active event. Please refresh or contact support.",
          variant: "destructive",
        });
        return;
      }
      if (!event) {
        console.error("No event returned from ensureActiveEventForOrg");
        toast({
          title: "No active event",
          description: "Could not create an active event. Please refresh.",
          variant: "destructive",
        });
        return;
      }

      if (cancelled) return;
      setActiveEventId(event.id);

      // Load Drive sync config in same flow so Sources tab restores folders after reload
      const { data: syncRows } = await supabase
        .from("sync_configurations" as any)
        .select("config, last_sync_at")
        .eq("event_id", event.id)
        .eq("source_type", "google_drive")
        .limit(1);
      if (!cancelled && (syncRows as any)?.[0]?.config?.google_drive_folder_id) {
        const row = (syncRows as any)[0];
        const rawFolders = Array.isArray(row.config.folders) && row.config.folders.length > 0
          ? row.config.folders
          : [{ id: row.config.google_drive_folder_id, name: row.config.google_drive_folder_name || "Project folder" }];
        const folders = rawFolders.map((f: { id: string; name: string; category?: string }) => ({
          id: f.id,
          name: f.name,
          category: f.category ?? "Projects",
        }));
        setInitialDriveSyncConfig({
          folderId: row.config.google_drive_folder_id,
          folderName: row.config.google_drive_folder_name || "Project folder",
          folders,
          lastSyncAt: row.last_sync_at || null,
        });
      } else {
        setInitialDriveSyncConfig(null);
      }

      const [decisionsRes, documentsRes, sourcesRes, foldersRes, connectionsRes, pendingReviewsRes, companyCardsRes, tasksRes] = await Promise.all([
        getDecisionsByEvent(event.id),
        getDocumentsByEvent(event.id),
        getSourcesByEvent(event.id),
        getSourceFoldersByEvent(event.id),
        getCompanyConnectionsByEvent(event.id),
        getPendingRelationshipReviews(event.id),
        getAllEntityCards(event.id),
        getTasksByEvent(event.id).catch(() => ({ data: [] as Task[], error: null })),
      ]);
      if (cancelled) return;
      const mapped = (decisionsRes.data || []).map(mapDecisionRow);
      setDecisions(mapped);
      setTasks((tasksRes.data || []) as Task[]);
      
      // Check for documents with NULL event_id and fix them
      if (documentsRes.error) {
        console.error("[DOCUMENTS] Query error:", documentsRes.error);
        toast({
          title: "Documents load error",
          description: documentsRes.error.message || "Could not load documents. Check RLS policies.",
          variant: "destructive",
        });
      }
      
      // Also query documents with NULL event_id (orphaned documents)
      const { data: orphanedDocs } = await supabase
        .from("documents")
        .select("id, title, event_id, created_by")
        .is("event_id", null)
        .eq("created_by", profile?.id || user?.id || "")
        .limit(100);
      
      // If we found orphaned documents, link them to the current event
      if (orphanedDocs && orphanedDocs.length > 0 && event.id) {
        const { error: updateError } = await supabase
          .from("documents")
          .update({ event_id: event.id })
          .in("id", orphanedDocs.map((d) => d.id));
        
        if (updateError) {
          console.warn("[DOCUMENTS] Failed to link orphaned documents:", updateError);
        } else {
          toast({
            title: "Documents linked",
            description: `Linked ${orphanedDocs.length} orphaned document(s) to current event.`,
          });
        }
        
        // Reload documents after linking
        const { data: reloadedDocs } = await getDocumentsByEvent(event.id);
        setDocuments(
          (reloadedDocs || []).map((doc: any) => ({
            id: doc.id,
            title: doc.title,
            storage_path: doc.storage_path || null,
            folder_id: doc.folder_id || null,
          }))
        );
      } else {
        setDocuments(
          (documentsRes.data || []).map((doc: any) => ({
            id: doc.id,
            title: doc.title,
            storage_path: doc.storage_path || null,
            folder_id: doc.folder_id || null,
          }))
        );
      }
      // Check for sources with NULL event_id and fix them
      if (sourcesRes.error) {
        console.error("[SOURCES] Query error:", sourcesRes.error);
        toast({
          title: "Sources load error",
          description: sourcesRes.error.message || "Could not load sources. Check RLS policies.",
          variant: "destructive",
        });
      }
      
      // Also query sources with NULL event_id (orphaned sources)
      const userId = profile?.id || user?.id;
      if (userId) {
        const { data: orphanedSources } = await supabase
          .from("sources")
          .select("id, title, event_id, created_by")
          .is("event_id", null)
          .eq("created_by", userId)
          .limit(100);
        
        // If we found orphaned sources, link them to the current event
        if (orphanedSources && orphanedSources.length > 0 && event.id) {
          const orphanedIds = orphanedSources.map((s) => s.id);
          const { error: updateError } = await supabase
            .from("sources")
            .update({ event_id: event.id })
            .in("id", orphanedIds);
          
          if (updateError) {
            console.warn("[SOURCES] Failed to link orphaned sources:", updateError);
          } else {
            toast({
              title: "Sources linked",
              description: `Linked ${orphanedSources.length} orphaned source(s) to current event.`,
            });
            
            // Reload sources after linking
            const { data: reloadedSources } = await getSourcesByEvent(event.id);
            if (reloadedSources) {
              const normalized = reloadedSources.map((source: any) => {
                const tags = Array.isArray(source.tags)
                  ? source.tags
                  : typeof source.tags === "string"
                  ? source.tags.split(",").map((t: string) => t.trim()).filter(Boolean)
                  : null;
                return { ...source, tags };
              });
              setSources(normalized as SourceRecord[]);
              return; // Early return after reload
            }
          }
        }
      }
      
      // Set sources from the original query
      const normalizedSources = (sourcesRes.data || []).map((source: any) => {
        const tags = Array.isArray(source.tags)
          ? source.tags
          : typeof source.tags === "string"
          ? source.tags.split(",").map((t: string) => t.trim()).filter(Boolean)
          : null;
        return { ...source, tags };
      });
      setSources(normalizedSources as SourceRecord[]);
      // Load source folders and ensure default folders exist
      try {
        await ensureDefaultFoldersForEvent(event.id);
        // Reload folders after ensuring defaults
        const { data: refreshedFolders } = await getSourceFoldersByEvent(event.id);
        setSourceFolders((refreshedFolders || []) as SourceFolder[]);
      } catch (folderErr) {
        console.warn("[FOLDERS] Failed to ensure default folders:", folderErr);
        // Fallback to original folders
        setSourceFolders((foldersRes.data || []) as SourceFolder[]);
      }
      
      // Load company connections for graph view
      setCompanyConnections((connectionsRes.data || []) as typeof companyConnections);
      
      // Load pending relationship reviews (handle errors gracefully if migration not run)
      if (pendingReviewsRes.error) {
        console.warn("[PENDING REVIEWS] Query failed (migration may not be run):", pendingReviewsRes.error);
        setPendingReviews([]);
      } else {
        const pendingData = (pendingReviewsRes.data || []).map((r: any) => ({
          id: r.id,
          relation_type: r.relation_type,
          confidence: r.confidence || 0.5,
          properties: r.properties || {},
          source_document_id: r.source_document_id,
          created_at: r.created_at,
          source_entity: r.source_entity || null,
          target_entity: r.target_entity || null,
        }));
        setPendingReviews(pendingData);
      }
      
      // Load company cards (unified view of companies)
      if (companyCardsRes.error) {
        console.warn("[COMPANY CARDS] Query failed:", companyCardsRes.error);
        setCompanyCards([]);
      } else {
        setCompanyCards((companyCardsRes.data || []) as typeof companyCards);
      }

      // Set up real-time subscriptions for documents
      documentsChannel = supabase
        .channel(`documents:${event.id}`)
        .on(
          "postgres_changes",
          {
            event: "*",
            schema: "public",
            table: "documents",
            filter: `event_id=eq.${event.id}`,
          },
          (payload) => {
            if (cancelled) return;
            
            if (payload.eventType === "INSERT" && payload.new) {
              const newDoc = payload.new as any;
              // Clean up title if it looks like a storage path
              const cleanTitle = (title: string | null): string | null => {
                if (!title) return null;
                // Remove file extension and random IDs
                const cleaned = title.replace(/\.[^/.]+$/, "").replace(/-\w{8,}$/, "").trim();
                if (!cleaned || cleaned.toLowerCase() === "document") {
                  return "Uploaded document";
                }
                return cleaned;
              };
              setDocuments((prev) => {
                // Check if already exists to avoid duplicates
                if (prev.some((d) => d.id === newDoc.id)) return prev;
                return [
                  {
                    id: newDoc.id,
                    title: cleanTitle(newDoc.title) || newDoc.title || "Untitled document",
                    storage_path: newDoc.storage_path || null,
                    folder_id: newDoc.folder_id || null,
                  },
                  ...prev,
                ];
              });
              toast({
                title: "New document added",
                description: `${cleanTitle(newDoc.title) || newDoc.title || "Untitled"} was added by a team member.`,
              });
            } else if (payload.eventType === "UPDATE" && payload.new) {
              const updatedDoc = payload.new as any;
              setDocuments((prev) =>
                prev.map((d) =>
                  d.id === updatedDoc.id
                    ? {
                        id: updatedDoc.id,
                        title: updatedDoc.title,
                        storage_path: updatedDoc.storage_path || null,
                        folder_id: updatedDoc.folder_id || null,
                      }
                    : d
                )
              );
            } else if (payload.eventType === "DELETE" && payload.old) {
              const deletedDoc = payload.old as any;
              setDocuments((prev) => prev.filter((d) => d.id !== deletedDoc.id));
              toast({
                title: "Document removed",
                description: "A document was deleted by a team member.",
              });
            }
          }
        )
        .subscribe();

      // Set up real-time subscriptions for decisions
      decisionsChannel = supabase
        .channel(`decisions:${event.id}`)
        .on(
          "postgres_changes",
          {
            event: "*",
            schema: "public",
            table: "decisions",
            filter: `event_id=eq.${event.id}`,
          },
          (payload) => {
            if (cancelled) return;
            
            if (payload.eventType === "INSERT" && payload.new) {
              const newDecision = payload.new as any;
              setDecisions((prev) => {
                // Check if already exists to avoid duplicates
                if (prev.some((d) => d.id === newDecision.id)) return prev;
                return [mapDecisionRow(newDecision), ...prev];
              });
              toast({
                title: "New decision logged",
                description: `${newDecision.startup_name || "Unknown"} decision was logged by a team member.`,
              });
            } else if (payload.eventType === "UPDATE" && payload.new) {
              const updatedDecision = payload.new as any;
              setDecisions((prev) =>
                prev.map((d) => (d.id === updatedDecision.id ? mapDecisionRow(updatedDecision) : d))
              );
            } else if (payload.eventType === "DELETE" && payload.old) {
              const deletedDecision = payload.old as any;
              setDecisions((prev) => prev.filter((d) => d.id !== deletedDecision.id));
              toast({
                title: "Decision removed",
                description: "A decision was deleted by a team member.",
              });
            }
          }
        )
        .subscribe();

      // Set up real-time subscription for sources
      sourcesChannel = supabase
        .channel(`sources:${event.id}`)
        .on(
          "postgres_changes",
          {
            event: "*",
            schema: "public",
            table: "sources",
            filter: `event_id=eq.${event.id}`,
          },
          (payload) => {
            if (cancelled) return;
            
            if (payload.eventType === "INSERT" && payload.new) {
              const newSource = payload.new as any;
              const tags = Array.isArray(newSource.tags)
                ? newSource.tags
                : typeof newSource.tags === "string"
                ? newSource.tags.split(",").map((t: string) => t.trim()).filter(Boolean)
                : null;
              setSources((prev) => {
                if (prev.some((s) => s.id === newSource.id)) return prev;
                return [{ ...newSource, tags } as SourceRecord, ...prev];
              });
              toast({
                title: "New source added",
                description: `${newSource.title || "Untitled"} was added by a team member.`,
              });
            } else if (payload.eventType === "UPDATE" && payload.new) {
              const updatedSource = payload.new as any;
              const tags = Array.isArray(updatedSource.tags)
                ? updatedSource.tags
                : typeof updatedSource.tags === "string"
                ? updatedSource.tags.split(",").map((t: string) => t.trim()).filter(Boolean)
                : null;
              setSources((prev) =>
                prev.map((s) => (s.id === updatedSource.id ? { ...updatedSource, tags } as SourceRecord : s))
              );
            } else if (payload.eventType === "DELETE" && payload.old) {
              const deletedSource = payload.old as any;
              setSources((prev) => prev.filter((s) => s.id !== deletedSource.id));
              toast({
                title: "Source removed",
                description: "A source was deleted by a team member.",
              });
            }
          }
        )
        .subscribe();
    };

    loadDecisions();
    return () => {
      cancelled = true;
      if (documentsChannel) {
        supabase.removeChannel(documentsChannel);
      }
      if (decisionsChannel) {
        supabase.removeChannel(decisionsChannel);
      }
      if (sourcesChannel) {
        supabase.removeChannel(sourcesChannel);
      }
    };
  }, [profile, toast]);

  const buildSnippet = useCallback((text: string | null) => {
    if (!text) return "No preview available.";
    const normalized = text.replace(/\s+/g, " ").trim();
    return normalized.length > 240 ? `${normalized.slice(0, 240)}вЂ¦` : normalized;
  }, []);

  const formatTabularContent = useCallback((text: string) => {
    const rawLines = text.split(/\n/).map((line) => line.replace(/\r/g, ""));
    const nonEmpty = rawLines.filter((line) => line.trim().length > 0);
    if (nonEmpty.length < 3) return text;

    const detectSeparator = (line: string) => {
      const commaCount = (line.match(/,/g) || []).length;
      const semicolonCount = (line.match(/;/g) || []).length;
      const tabCount = (line.match(/\t/g) || []).length;
      if (tabCount > commaCount && tabCount > semicolonCount) return "\t";
      if (semicolonCount > commaCount) return ";";
      return ",";
    };

    const parseCsvLine = (line: string, separator: string) => {
      const cells: string[] = [];
      let current = "";
      let inQuotes = false;
      for (let i = 0; i < line.length; i += 1) {
        const char = line[i];
        if (char === '"') {
          if (inQuotes && line[i + 1] === '"') {
            current += '"';
            i += 1;
          } else {
            inQuotes = !inQuotes;
          }
          continue;
        }
        if (char === separator && !inQuotes) {
          cells.push(current.trim());
          current = "";
          continue;
        }
        current += char;
      }
      cells.push(current.trim());
      return cells;
    };

    const separator = detectSeparator(nonEmpty[0]);
    const parsed = nonEmpty.map((line) => parseCsvLine(line, separator));
    const counts = parsed.map((row) => row.length).filter((count) => count > 1);
    if (counts.length < 3) return text;

    const frequency = new Map<number, number>();
    counts.forEach((count) => frequency.set(count, (frequency.get(count) || 0) + 1));
    const [targetCols, targetCount] = [...frequency.entries()].sort((a, b) => b[1] - a[1])[0];
    if (targetCols < 2 || targetCount < 3) return text;

    const tableRows = parsed.filter((row) => row.length === targetCols);
    if (tableRows.length < 3) return text;

    const maxRows = 25;
    const rows = tableRows.slice(0, maxRows);
    const headerRow = rows[0].map((cell, index) => cell || `Column ${index + 1}`);
    const renderRow = (cells: string[]) => `| ${cells.map((cell) => cell || " ").join(" | ")} |`;
    const tableLines = [
      "TABLE (formatted):",
      renderRow(headerRow),
      `| ${headerRow.map(() => "---").join(" | ")} |`,
      ...rows.slice(1).map(renderRow),
    ];
    if (tableRows.length > maxRows) {
      tableLines.push("вЂ¦(table truncated)");
    }
    return tableLines.join("\n");
  }, []);

  const buildNormalizedDocText = useCallback(
    (doc: { raw_content: string | null; extracted_json?: Record<string, any> | null }) => {
      const raw = doc.raw_content?.trim() ? formatTabularContent(doc.raw_content) : "";
      const json = doc.extracted_json ? JSON.stringify(doc.extracted_json) : "";
      return [raw, json].filter(Boolean).join("\n").replace(/\r/g, "").trim();
    },
    [formatTabularContent]
  );

  const buildDocSnippet = useCallback(
    (doc: { raw_content: string | null; extracted_json?: Record<string, any> | null }) => {
      const combined = buildNormalizedDocText(doc);
      if (!combined) return "No preview available.";
      return buildSnippet(combined);
    },
    [buildSnippet, buildNormalizedDocText]
  );

  const buildRelevantSnippet = useCallback(
    (doc: { raw_content: string | null; extracted_json?: Record<string, any> | null }, tokens: string[]) => {
      const combined = buildNormalizedDocText(doc).replace(/\s+/g, " ").trim();
      if (!combined) return "No preview available.";
      const haystack = combined.toLowerCase();
      const match = tokens.find((t) => haystack.includes(t));
      if (!match) return buildDocSnippet(doc);
      const idx = haystack.indexOf(match);
      const start = Math.max(0, idx - 140);
      const end = Math.min(combined.length, idx + match.length + 160);
      const snippet = combined.slice(start, end).trim();
      return snippet.length > 0 ? `${start > 0 ? "вЂ¦" : ""}${snippet}${end < combined.length ? "вЂ¦" : ""}` : buildDocSnippet(doc);
    },
    [buildDocSnippet, buildNormalizedDocText]
  );

  const buildClaudeContext = useCallback(
    (
      doc: { raw_content: string | null; extracted_json?: Record<string, any> | null },
      tokens: string[],
      isComprehensive: boolean = false,
      snippetOverride?: string | null
    ) => {
      if (snippetOverride?.trim()) {
        const limit = isComprehensive ? 2500 : 1000;
        const trimmed = snippetOverride.trim();
        return trimmed.length > limit ? `${trimmed.slice(0, limit)}вЂ¦` : trimmed;
      }

      const combined = buildNormalizedDocText(doc);
      if (!combined) return "No preview available.";

      const lowerTokens = tokens.map((t) => t.toLowerCase());
      const lines = combined.split("\n").map((line) => line.trim()).filter(Boolean);
      const startIdx = lines.findIndex((line) =>
        lowerTokens.some((t) => line.toLowerCase().includes(t))
      );

      if (startIdx >= 0) {
        const slice = lines.slice(startIdx, startIdx + (isComprehensive ? 80 : 40));
        const joined = slice.join("\n");
        const limit = isComprehensive ? 2500 : 1000;
        return joined.length > limit ? `${joined.slice(0, limit)}вЂ¦` : joined;
      }

      // Fallback: return the first chunk of the document
      const limit = isComprehensive ? 2500 : 1000;
      return combined.length > limit ? `${combined.slice(0, limit)}вЂ¦` : combined;
    },
    [buildNormalizedDocText]
  );

  const formatDecisionMatches = useCallback((matchedDecisions: Decision[]) => {
    return (
      "Here are the matching decisions:\n" +
      matchedDecisions
        .map(
          (d, index) =>
            `${index + 1}. ${d.startupName} вЂ” ${d.actionType}${d.outcome ? ` (${d.outcome})` : ""}${
              d.notes ? ` вЂ” ${d.notes}` : ""
            }`
        )
        .join("\n")
    );
  }, []);

  const docContainsTokens = useCallback(
    (
      doc: {
        raw_content: string | null;
        extracted_json?: Record<string, any> | null;
        title?: string | null;
        file_name?: string | null;
      },
      tokens: string[]
    ) => {
      if (!tokens.length) return false; // No tokens = no match
      const haystack = [
        doc.raw_content || "",
        doc.extracted_json ? JSON.stringify(doc.extracted_json) : "",
        doc.title || "",
        doc.file_name || "",
      ]
        .join(" ")
        .toLowerCase();
      // Require at least 60% of tokens to match.
      // For short queries (1-2 tokens), allow a single match.
      const minMatches = tokens.length <= 2 ? 1 : Math.max(2, Math.ceil(tokens.length * 0.6));
      const matches = tokens.filter((t) => haystack.includes(t)).length;
      return matches >= minMatches;
    },
    []
  );

  // Removed buildStructuredAnswer - it was causing irrelevant "Responsibilities" sections
  // We now trust Claude's answers completely. If Claude says no info, we respect that.

  const isDeveloper =
    (import.meta.env.VITE_DEV_MODE as string | undefined) === "true" ||
    (profile?.email && (import.meta.env.VITE_DEV_EMAIL as string | undefined) === profile.email);

  const persistCostLog = useCallback((entry: typeof costLog[number]) => {
    const updated = [entry, ...costLog].slice(0, 100);
    setCostLog(updated);
    if (typeof window !== "undefined") {
      localStorage.setItem("platform_cost_log", JSON.stringify(updated));
    }
  }, [costLog]);

  const createChatThread = useCallback(
    async (title: string) => {
      const eventId = activeEventId || (await ensureActiveEventId());
      if (!eventId) return null;
      const userId = profile?.id || user?.id || null;
      const { data, error } = await supabase
        .from("chat_threads")
        .insert({
          event_id: eventId,
          title,
          created_by: userId,
        })
        .select("id")
        .single();
      if (error || !data?.id) {
        console.error("Failed to create chat thread:", error);
        return null;
      }
      return data.id as string;
    },
    [activeEventId, ensureActiveEventId, profile, user]
  );

  const persistChatMessage = useCallback(
    async (payload: {
      threadId: string;
      role: "user" | "assistant";
      content: string;
      model?: string | null;
      sourceDocIds?: string[] | null;
    }) => {
      try {
        const eventId = activeEventId || (await ensureActiveEventId());
        if (!eventId) return;
        const userId = profile?.id || user?.id || null;
        
        // Ensure thread exists and is a real UUID (DB thread_id is UUID type)
        let threadId = payload.threadId;
        const isUuid = /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i.test(threadId || "");
        if (!threadId || !isUuid) {
          const newThreadId = await createChatThread("Chat");
          if (newThreadId) {
            threadId = newThreadId;
          } else {
            // Cannot persist without a real thread UUID — skip DB insert (message stays in state only)
            return;
          }
        }

        // Retry logic for network failures
        let lastError: any = null;
        for (let attempt = 0; attempt < 3; attempt++) {
          try {
            const { error, data } = await supabase.from("chat_messages").insert({
              event_id: eventId,
              thread_id: threadId,
              role: payload.role,
              content: payload.content,
              model: payload.model || null,
              source_doc_ids: payload.sourceDocIds || null,
              created_by: userId,
            }).select();
            if (!error) {
              return; // Success
            } else {
              console.error("[DEBUG] вќЊ Failed to save chat message:", error);
            }
            lastError = error;
            // Don't retry on RLS/auth errors
            if (error.code === '42501' || error.code === 'PGRST116') {
              break;
            }
          } catch (err) {
            lastError = err;
            // Retry on network errors
            if (attempt < 2 && (err instanceof TypeError || err instanceof Error)) {
              await new Promise(resolve => setTimeout(resolve, 1000 * (attempt + 1)));
              continue;
            }
            break;
          }
        }
        
        if (lastError) {
          console.error("Failed to save chat message after retries:", lastError);
          const cached = readLocalChatCache();
          const localMessage: LocalChatMessage = {
            id: `local-${Date.now()}-${Math.random().toString(36).slice(2, 6)}`,
            threadId,
            author: payload.role,
            text: payload.content,
            ts: new Date().toISOString(),
          };
          writeLocalChatCache([localMessage, ...cached].slice(0, 200));
        }
      } catch (err) {
        console.error("Failed to save chat message:", err);
        // Silently fail - don't block chat functionality
      }
    },
    [activeEventId, ensureActiveEventId, profile, user, createChatThread, readLocalChatCache, writeLocalChatCache]
  );

  useEffect(() => {
    if (typeof window === "undefined") return;
    // Clear legacy permanent disable flag вЂ” we now use session-only failure tracking
    localStorage.removeItem("disable_embeddings");
    embeddingsDisabledRef.current = false;
    const existing = localStorage.getItem("platform_cost_log");
    if (existing) {
      try {
        const parsed = JSON.parse(existing);
        if (Array.isArray(parsed)) {
          setCostLog(parsed);
        }
      } catch {
        // ignore invalid JSON
      }
    }
  }, []);

  const estimateClaudeCost = useCallback((question: string) => {
    const ASK_MAX_TOKENS = 700;
    const inputChars = question.length + (lastEvidence?.docs?.length || 0) * 500;
    const estInputTokens = Math.max(1, Math.ceil(inputChars / 4));
    const estOutputTokens = ASK_MAX_TOKENS;
    const inputCost = (estInputTokens / 1_000_000) * 3.0;
    const outputCost = (estOutputTokens / 1_000_000) * 15.0;
    const estCostUsd = Number((inputCost + outputCost).toFixed(5));
    return { estInputTokens, estOutputTokens, estCostUsd };
  }, [lastEvidence]);

  // в”Ђв”Ђ Semantic chunking helpers в”Ђв”Ђ

  const chunkTextWithOverlap = useCallback((text: string, size: number, overlap: number) => {
    const chunks: string[] = [];
    let start = 0;
    while (start < text.length) {
      const end = Math.min(text.length, start + size);
      const chunk = text.slice(start, end).trim();
      if (chunk) {
        chunks.push(chunk);
      }
      if (end === text.length) break;
      start = Math.max(0, end - overlap);
    }
    return chunks;
  }, []);

  const splitIntoParagraphs = useCallback((text: string): string[] => {
    const raw = text.split(/\n\s*\n/);
    const paragraphs: string[] = [];
    for (const p of raw) {
      const trimmed = p.trim();
      if (trimmed) paragraphs.push(trimmed);
    }
    return paragraphs;
  }, []);

  const splitIntoSentences = useCallback((text: string): string[] => {
    const parts = text.split(/(?<=[.!?])\s+/);
    const sentences: string[] = [];
    for (const s of parts) {
      const trimmed = s.trim();
      if (trimmed) sentences.push(trimmed);
    }
    return sentences;
  }, []);

  const mergeUnitsIntoChunks = useCallback(
    (units: string[], maxSize: number, fallbackOverlap: number): string[] => {
      if (units.length === 0) return [];
      const chunks: string[] = [];
      let current = "";
      for (const unit of units) {
        if (!current) {
          current = unit;
        } else if (current.length + 1 + unit.length <= maxSize) {
          current += "\n" + unit;
        } else {
          chunks.push(current);
          current = unit;
        }
      }
      if (current.trim()) chunks.push(current);

      // If any chunk exceeds maxSize (e.g. single giant paragraph/sentence), split it with fixed-size fallback
      const result: string[] = [];
      for (const chunk of chunks) {
        if (chunk.length > maxSize * 1.2) {
          result.push(...chunkTextWithOverlap(chunk, maxSize, fallbackOverlap));
        } else {
          result.push(chunk);
        }
      }
      return result;
    },
    [chunkTextWithOverlap]
  );

  const buildSemanticParentChildChunks = useCallback(
    (text: string) => {
      const PARENT_SIZE = 2000;
      const CHILD_SIZE = 500;
      const FALLBACK_OVERLAP = 100;
      const MAX_PARENT_CHUNKS = 8;
      const MAX_CHILD_PER_PARENT = 4;

      const paragraphs = splitIntoParagraphs(text);
      let parentTexts: string[];
      if (paragraphs.length <= 1) {
        const sentences = splitIntoSentences(text);
        parentTexts = mergeUnitsIntoChunks(sentences, PARENT_SIZE, FALLBACK_OVERLAP);
      } else {
        parentTexts = mergeUnitsIntoChunks(paragraphs, PARENT_SIZE, FALLBACK_OVERLAP);
      }
      parentTexts = parentTexts.slice(0, MAX_PARENT_CHUNKS);

      const pairs: Array<{ parentText: string; childText: string; parentIndex: number; childIndex: number }> = [];
      parentTexts.forEach((parentText, parentIndex) => {
        const sentences = splitIntoSentences(parentText);
        let children: string[];
        if (sentences.length <= 1) {
          children = chunkTextWithOverlap(parentText, CHILD_SIZE, FALLBACK_OVERLAP);
        } else {
          children = mergeUnitsIntoChunks(sentences, CHILD_SIZE, FALLBACK_OVERLAP);
        }
        children = children.slice(0, MAX_CHILD_PER_PARENT);
        children.forEach((childText, childIndex) => {
          pairs.push({ parentText, childText, parentIndex, childIndex });
        });
      });
      return pairs;
    },
    [splitIntoParagraphs, splitIntoSentences, mergeUnitsIntoChunks, chunkTextWithOverlap]
  );

  const buildParentChildChunks = useCallback(
    async (text: string, docTitle?: string) => {
      const CHILD_SIZE = 500;
      const FALLBACK_OVERLAP = 100;
      const MAX_CHILD_PER_PARENT = 4;

      // в”Ђв”Ђ Agentic chunking: ask the LLM to split the document by topic в”Ђв”Ђ
      try {
        const agenticResult = await agenticChunk({
          document_title: docTitle || "Untitled",
          document_text: text,
          max_sections: 8,
        });

        if (!agenticResult.fallback && agenticResult.sections.length > 0) {
          const pairs: Array<{ parentText: string; childText: string; parentIndex: number; childIndex: number }> = [];

          agenticResult.sections.forEach((section, parentIndex) => {
            const parentText = section.text;
            const sentences = splitIntoSentences(parentText);
            let children: string[];
            if (sentences.length <= 1) {
              children = chunkTextWithOverlap(parentText, CHILD_SIZE, FALLBACK_OVERLAP);
            } else {
              children = mergeUnitsIntoChunks(sentences, CHILD_SIZE, FALLBACK_OVERLAP);
            }
            children = children.slice(0, MAX_CHILD_PER_PARENT);
            children.forEach((childText, childIndex) => {
              pairs.push({ parentText, childText, parentIndex, childIndex });
            });
          });

          if (pairs.length > 0) {
            return {
              pairs,
              mode: "agentic" as const,
              parentCount: agenticResult.sections.length,
              modelUsed: agenticResult.model_used || "",
            };
          }
        }
      } catch (err) {
        console.warn("[CHUNK] Agentic chunking failed, falling back to semantic:", err);
      }

      // в”Ђв”Ђ Fallback: semantic chunking (paragraph/sentence boundaries) в”Ђв”Ђ
      const semanticPairs = buildSemanticParentChildChunks(text);
      const parentCount = new Set(semanticPairs.map((p) => p.parentIndex)).size;
      return {
        pairs: semanticPairs,
        mode: "semantic_fallback" as const,
        parentCount,
        modelUsed: "",
      };
    },
    [splitIntoSentences, mergeUnitsIntoChunks, chunkTextWithOverlap, buildSemanticParentChildChunks]
  );

  // Track transient failures per session (NOT permanently in localStorage)
  const embeddingFailCountRef = useRef(0);
  const MAX_EMBEDDING_FAILURES = 5; // disable only after 5 consecutive failures in this session

  const disableEmbeddings = useCallback((reason?: string) => {
    embeddingFailCountRef.current++;
    if (reason) {
      console.warn(`[EMBED] Failure #${embeddingFailCountRef.current}: ${reason}`);
    }
    // Only disable for this session after repeated failures вЂ” never persist to localStorage
    if (embeddingFailCountRef.current >= MAX_EMBEDDING_FAILURES) {
      embeddingsDisabledRef.current = true;
      console.error(`[EMBED] Disabled for this session after ${MAX_EMBEDDING_FAILURES} failures. Refresh page to retry.`);
    }
  }, []);

  // в”Ђв”Ђ Entity extraction helper: populate knowledge graph + KPIs from documents в”Ђв”Ђ
  const extractAndStoreEntities = useCallback(
    async (documentId: string, rawContent: string, docTitle: string, eventId: string, pdfBase64ForExtraction?: string | null) => {
      if ((!rawContent?.trim() && !pdfBase64ForExtraction) || !eventId) return;
      
      (async () => {
        try {
          const hasPdf = !!pdfBase64ForExtraction;
          // Detect document type from content/title for better extraction
          const lowerContent = (rawContent || "").slice(0, 500).toLowerCase();
          const lowerTitle = (docTitle || "").toLowerCase();
          let detectedDocType = "pitch_deck";
          if (lowerContent.startsWith("from:") || lowerContent.includes("\nto:") || lowerTitle.includes("email") || lowerContent.includes("\nsubject:")) {
            detectedDocType = "email";
          } else if (lowerTitle.includes("memo") || lowerTitle.includes("note")) {
            detectedDocType = "memo";
          } else if (lowerTitle.includes("report") || lowerTitle.includes("analysis")) {
            detectedDocType = "report";
          }
          const extraction = await extractEntities({
            document_title: docTitle,
            document_text: rawContent?.slice(0, 12000) || "",
            document_type: detectedDocType,
            pdf_base64: pdfBase64ForExtraction || undefined,
          });

          if (extraction.entities.length === 0 && extraction.relationships.length === 0 && extraction.kpis.length === 0) {
            console.warn("[EXTRACT] No entities/relationships/KPIs found вЂ” check backend logs for errors");
            return;
          }

          const userId = profile?.id || user?.id;
          if (!userId) {
            console.warn("[EXTRACT] No user ID, skipping entity storage");
            return;
          }

          // в”Ђв”Ђ Step 1: Insert entities (dedupe by normalized_name) в”Ђв”Ђ
          const entityMap = new Map<string, string>(); // normalized_name в†’ entity_id
          
          for (const entity of extraction.entities) {
            const normalized = (entity.type === "company" || entity.type === "fund")
              ? normalizeCompanyNameForMatch(entity.name)
              : entity.name.toLowerCase().trim();
            // Check if entity already exists
            const { data: existing } = await supabase
              .from("kg_entities")
              .select("id")
              .eq("event_id", eventId)
              .eq("normalized_name", normalized)
              .eq("entity_type", entity.type)
              .limit(1);

            let entityId: string;
            if (existing && existing.length > 0) {
              entityId = existing[0].id;
            } else {
              const { data: newEntity, error: insertErr } = await supabase
                .from("kg_entities")
                .insert({
                  event_id: eventId,
                  entity_type: entity.type,
                  name: entity.name,
                  normalized_name: normalized,
                  properties: entity.properties || {},
                  confidence: entity.confidence,
                  source_document_id: documentId,
                  created_by: userId,
                })
                .select("id")
                .single();
              
              if (insertErr || !newEntity) {
                console.warn(`[EXTRACT] Failed to insert entity ${entity.name}:`, insertErr);
                continue;
              }
              entityId = newEntity.id;
            }
            entityMap.set(normalized, entityId);
          }

          // в”Ђв”Ђ Step 2: Insert relationships в”Ђв”Ђ
          // Allowed relation types per DB CHECK constraint on kg_edges
          const ALLOWED_RELATION_TYPES = new Set([
            'founded', 'works_at', 'invested_in', 'raised', 'led_round',
            'partner_of', 'competitor_of', 'acquired', 'operates_in',
            'located_in', 'board_member', 'advisor', 'portfolio_company',
          ]);
          // Map common AI-generated variants to allowed types
          const RELATION_TYPE_MAP: Record<string, string> = {
            'founder': 'founded', 'co_founded': 'founded', 'co-founded': 'founded',
            'employs': 'works_at', 'employed_at': 'works_at', 'works_for': 'works_at',
            'member_of': 'works_at', 'team_member': 'works_at',
            'invested': 'invested_in', 'investor': 'invested_in', 'invests_in': 'invested_in',
            'backed_by': 'invested_in', 'funded_by': 'invested_in',
            'partners_with': 'partner_of', 'partnership': 'partner_of', 'partnered_with': 'partner_of',
            'competes_with': 'competitor_of', 'competition': 'competitor_of',
            'acquired_by': 'acquired', 'acquisition': 'acquired',
            'sector': 'operates_in', 'industry': 'operates_in', 'in_sector': 'operates_in',
            'headquartered_in': 'located_in', 'based_in': 'located_in', 'hq': 'located_in',
            'advises': 'advisor', 'advisor_to': 'advisor', 'advising': 'advisor',
            'board': 'board_member', 'on_board': 'board_member',
            'portfolio': 'portfolio_company', 'in_portfolio': 'portfolio_company',
            'uses': 'partner_of', 'client_of': 'partner_of', 'customer_of': 'partner_of',
            'provides_to': 'partner_of', 'supplies': 'partner_of',
          };

          for (const rel of extraction.relationships) {
            const sourceKey = rel.source_name.toLowerCase().trim();
            const targetKey = rel.target_name.toLowerCase().trim();
            const sourceCanon = normalizeCompanyNameForMatch(rel.source_name);
            const targetCanon = normalizeCompanyNameForMatch(rel.target_name);
            const sourceId = entityMap.get(sourceKey) ?? entityMap.get(sourceCanon);
            const targetId = entityMap.get(targetKey) ?? entityMap.get(targetCanon);

            if (!sourceId || !targetId) {
              console.warn(`[EXTRACT] Missing entity for relationship ${rel.source_name} в†’ ${rel.target_name}`);
              continue;
            }

            // Normalize relation_type: map AI variants to allowed DB types
            let relationType = (rel.relation_type || '').toLowerCase().trim();
            if (!ALLOWED_RELATION_TYPES.has(relationType)) {
              const mapped = RELATION_TYPE_MAP[relationType];
              if (mapped) {
                relationType = mapped;
              } else {
                // Default to partner_of for unknown types rather than failing
                console.warn(`[EXTRACT] Unknown relation_type "${rel.relation_type}", defaulting to "partner_of"`);
                relationType = 'partner_of';
              }
            }

            // Check if edge already exists
            const { data: existingEdge } = await supabase
              .from("kg_edges")
              .select("id")
              .eq("source_entity_id", sourceId)
              .eq("target_entity_id", targetId)
              .eq("relation_type", relationType)
              .limit(1);

            if (!existingEdge || existingEdge.length === 0) {
              // Auto-approve high-confidence extractions (в‰Ґ85%) so fewer items need manual review
              const reviewStatus = rel.confidence >= 0.85 ? 'approved' : 'pending';
              
              const { error: edgeErr } = await supabase.from("kg_edges").insert({
                event_id: eventId,
                source_entity_id: sourceId,
                target_entity_id: targetId,
                relation_type: relationType,
                properties: rel.properties || {},
                confidence: rel.confidence,
                source_document_id: documentId,
                created_by: userId,
                review_status: reviewStatus,
                // Auto-approve high-confidence by setting reviewed_by to creator
                ...(reviewStatus === 'approved' ? { reviewed_by: userId, reviewed_at: new Date().toISOString() } : {}),
              });
              if (edgeErr) {
                console.warn(`[EXTRACT] Failed to insert edge:`, edgeErr);
              } else if (reviewStatus === 'pending') {
              }
            }
          }

          // в”Ђв”Ђ Step 3: Insert KPIs в”Ђв”Ђ
          for (const kpi of extraction.kpis) {
            // Check if KPI already exists (same company + metric + period)
            const { data: existingKpi } = await supabase
              .from("company_kpis")
              .select("id")
              .eq("event_id", eventId)
              .eq("company_name", kpi.company_name)
              .eq("metric_name", kpi.metric_name)
              .eq("period", kpi.period || "")
              .limit(1);

            if (!existingKpi || existingKpi.length === 0) {
              const { error: kpiErr } = await supabase.from("company_kpis").insert({
                event_id: eventId,
                company_name: kpi.company_name,
                metric_name: kpi.metric_name,
                value: kpi.value,
                unit: kpi.unit,
                period: kpi.period || null,
                metric_category: kpi.category,
                confidence: kpi.confidence,
                source_document_id: documentId,
                extraction_method: "claude_extraction",
                created_by: userId,
              });
              if (kpiErr) {
                console.warn(`[EXTRACT] Failed to insert KPI:`, kpiErr);
              }
            }
          }

        } catch (err) {
          console.error("[EXTRACT] Entity extraction failed:", err);
          // Non-fatal вЂ” document is saved, embeddings work
        }
      })();
    },
    [profile, user]
  );

  const indexDocumentEmbeddings = useCallback(
    async (documentId: string, rawContent?: string | null, docTitle?: string | null, pdfBase64ForExtraction?: string | null) => {
      if (embeddingsDisabledRef.current) return;
      if (!rawContent?.trim()) return;
      (async () => {
        try {
          const { data: existing } = await supabase
            .from("document_embeddings")
            .select("id")
            .eq("document_id", documentId)
            .limit(1);
          if (existing && existing.length > 0) return;

          const MAX_EMBED_CHARS = 12000; // Increased for better coverage
          const truncated = rawContent.slice(0, MAX_EMBED_CHARS);
          const title = docTitle || "Untitled document";
          const chunkBuild = await buildParentChildChunks(truncated, title);
          const pairs = chunkBuild.pairs;

          // Build a short document summary for contextual headers (first 500 chars)
          const docSummary = rawContent.slice(0, 500);
          let chunksAttempted = 0;
          let chunksEmbedded = 0;
          let chunksEmbeddingFailed = 0;
          let chunksInsertFailed = 0;
          let contextualSkipped = 0;

          for (let i = 0; i < pairs.length; i++) {
            const pair = pairs[i];
            chunksAttempted++;
            try {
              // в”Ђв”Ђ Contextual Retrieval: enrich chunk with a Claude-generated header в”Ђв”Ђ
              // This dramatically improves embedding quality (per Anthropic's paper)
              // BUT: Use fast timeout (3s) and skip if backend is slow to prevent blocking
              let textToEmbed = pair.childText;
              let contextualHeader = "";
              try {
                // Fast timeout: if contextual enrichment takes > 3s, skip it
                const ctxPromise = contextualizeChunk({
                  document_title: title,
                  document_summary: docSummary,
                  chunk_text: pair.childText,
                  chunk_index: i,
                  total_chunks: pairs.length,
                });
                const timeoutPromise = new Promise<never>((_, reject) => 
                  setTimeout(() => reject(new Error("Contextual enrichment timeout")), 6000)
                );
                const ctx = await Promise.race([ctxPromise, timeoutPromise]);
                if (ctx.enriched_chunk) {
                  textToEmbed = ctx.enriched_chunk;
                  contextualHeader = ctx.contextual_header || "";
                }
              } catch {
                // Contextual enrichment failed or timed out вЂ” embed raw chunk (still works, just less precise)
                // This is non-fatal and shouldn't block the upload
                contextualSkipped++;
                const firstLine = pair.childText.trim().split(/\n/)[0]?.trim().slice(0, 100) || pair.childText.trim().slice(0, 100);
                if (firstLine) contextualHeader = `${title}: Chunk ${i + 1}/${pairs.length}. ${firstLine}`;
              }

              // Generate embedding with timeout
              let embedding: number[] | null = null;
              try {
                const embeddingPromise = embedQuery(textToEmbed, "document");
                const embeddingTimeout = new Promise<never>((_, reject) => 
                  setTimeout(() => reject(new Error("Embedding timeout")), 10000)
                );
                embedding = await Promise.race([embeddingPromise, embeddingTimeout]);
              } catch (embedErr) {
                console.warn(`[EMBED] Embedding failed for chunk ${i + 1}/${pairs.length}:`, embedErr);
                chunksEmbeddingFailed++;
                continue; // Skip this chunk
              }
              
              if (!embedding || embedding.length === 0) {
                chunksEmbeddingFailed++;
                continue;
              }

              const { error } = await supabase.from("document_embeddings").insert({
                document_id: documentId,
                chunk_text: pair.childText,
                parent_text: pair.parentText,
                parent_index: pair.parentIndex,
                child_index: pair.childIndex,
                embedding,
                contextual_header: contextualHeader || null,
              });
              if (error) {
                // If contextual_header column doesn't exist yet, retry without it
                if (error.message?.includes("contextual_header")) {
                  const { error: retryError } = await supabase.from("document_embeddings").insert({
                    document_id: documentId,
                    chunk_text: pair.childText,
                    parent_text: pair.parentText,
                    parent_index: pair.parentIndex,
                    child_index: pair.childIndex,
                    embedding,
                  });
                  if (retryError) {
                    disableEmbeddings(retryError.message || "Embedding insert failed");
                    chunksInsertFailed++;
                    // Skip this chunk but continue with others
                    continue;
                  }
                } else {
                  disableEmbeddings(error.message || "Embedding insert failed");
                  chunksInsertFailed++;
                  // Skip this chunk but continue with others
                  continue;
                }
              }
              chunksEmbedded++;
            } catch (chunkErr) {
              disableEmbeddings(chunkErr instanceof Error ? chunkErr.message : "Embedding error");
              chunksEmbeddingFailed++;
              // Skip this chunk but continue with others
              continue;
            }
          }
          const failedTotal = chunksEmbeddingFailed + chunksInsertFailed;
          const statusEmoji = chunksEmbedded > 0 ? "вњ…" : "вљ пёЏ";
          
          // в”Ђв”Ђ Trigger entity extraction after embeddings are done в”Ђв”Ђ
          const eventId = activeEventId || (await ensureActiveEventId());
          if (eventId && (rawContent || pdfBase64ForExtraction) && docTitle) {
            void extractAndStoreEntities(documentId, rawContent || "", docTitle, eventId, pdfBase64ForExtraction);
          }
        } catch (err) {
          disableEmbeddings(err instanceof Error ? err.message : "Embedding setup failed");
        }
      })();
    },
    [buildParentChildChunks, disableEmbeddings, extractAndStoreEntities, activeEventId, ensureActiveEventId]
  );

  const createAssistantMessage = useCallback(
    (text: string, threadId: string, sourceDocIds?: string[] | null) => {
      const id = `m-${Date.now()}-${Math.random().toString(36).slice(2, 6)}`;
      setMessages((prev) => [...prev, { id, author: "assistant", text, threadId }]);
      void persistChatMessage({
        threadId,
        role: "assistant",
        content: text,
        model: "claude",
        sourceDocIds: sourceDocIds || null,
      });
      // Auto-scroll to bottom after message is added
      setTimeout(() => {
        scrollChatToBottom();
      }, 100);
    },
    [persistChatMessage, scrollChatToBottom]
  );

  const createStreamingAssistantMessage = useCallback(
    (threadId: string, sourceDocIds?: string[] | null) => {
      const id = `m-${Date.now()}-${Math.random().toString(36).slice(2, 6)}`;
      let currentText = "";

      // Keep reference to sourceDocs/contextLabels so finalize can re-apply them
      let _sourceDocs: SourceDoc[] = [];
      let _contextLabels: string[] = [];

      // Create placeholder message with thinking indicator (dots)
      setMessages((prev) => [
        ...prev,
        { id, author: "assistant" as const, text: "...", threadId, isStreaming: true },
      ]);

      // Auto-scroll when thinking
      setTimeout(() => {
        scrollChatToBottom();
      }, 100);

      // Helper: find the streaming message by its stable id (never stale)
      const patchById = (prev: Message[], patch: Partial<Message>): Message[] => {
        const idx = prev.findIndex((m) => m.id === id);
        if (idx === -1) return prev; // message was removed вЂ” no-op
        const updated = [...prev];
        updated[idx] = { ...updated[idx], ...patch };
        return updated;
      };

      return {
        appendChunk: (chunk: string) => {
          currentText += chunk;
          setMessages((prev) => patchById(prev, { text: currentText, isStreaming: true }));
          // Auto-scroll as text streams
          setTimeout(() => {
            scrollChatToBottom();
          }, 50);
        },
        finalize: () => {
          // Strip intermediate status lines (e.g. *Searching: ...*)  from persisted text
          const cleanText = currentText
            .replace(/\n?\*(?:Analyzing|Searching|Generating)[^*]*\*\n?/g, "")
            .replace(/^\s+/, "");
          // Re-apply sourceDocs and contextLabels so they survive the finalize patch
          setMessages((prev) => patchById(prev, {
            text: cleanText || currentText,
            isStreaming: false,
            sourceDocs: _sourceDocs.length > 0 ? _sourceDocs : undefined,
            contextLabels: _contextLabels.length > 0 ? _contextLabels : undefined,
          }));
          void persistChatMessage({
            threadId,
            role: "assistant",
            content: cleanText || currentText,
            model: "claude",
            sourceDocIds: sourceDocIds || null,
          });
        },
        setError: (error: string) => {
          setMessages((prev) => patchById(prev, { text: error, isStreaming: false }));
          void persistChatMessage({
            threadId,
            role: "assistant",
            content: error,
            model: "claude",
            sourceDocIds: sourceDocIds || null,
          });
        },
        setVerifiableSources: (sources: VerifiableSource[]) => {
          setMessages((prev) => patchById(prev, { verifiableSources: sources }));
        },
        setSourceDocs: (docs: SourceDoc[]) => {
          _sourceDocs = docs;
          setMessages((prev) => patchById(prev, { sourceDocs: docs }));
        },
        setCritic: (text: string) => {
          setMessages((prev) => patchById(prev, { critic: text }));
        },
        setContextLabels: (labels: string[]) => {
          _contextLabels = labels;
          setMessages((prev) => patchById(prev, { contextLabels: labels }));
        },
      };
    },
    [persistChatMessage, scrollChatToBottom]
  );

  // Scroll to bottom when chat tab is first opened or when messages are loaded
  useEffect(() => {
    if (activeTab === "chat" && chatContainerRef.current) {
      // Scroll to bottom when tab opens or when messages change
      const container = chatContainerRef.current;
      // Use multiple timeouts to ensure DOM is fully rendered and messages are displayed
      const scrollToBottom = () => {
        if (container) {
          container.scrollTop = container.scrollHeight;
        }
      };
      // Immediate scroll
      scrollToBottom();
      // Delayed scrolls to catch any async rendering
      setTimeout(scrollToBottom, 50);
      setTimeout(scrollToBottom, 200);
      setTimeout(scrollToBottom, 500);
    }
  }, [activeTab, scopedMessages.length]);

  // Auto-scroll when new messages arrive - only if near bottom
  useEffect(() => {
    if (chatContainerRef.current && scopedMessages.length > 0) {
      const container = chatContainerRef.current;
      // Only auto-scroll if we're near the bottom (within 150px)
      const isNearBottom = container.scrollHeight - container.scrollTop - container.clientHeight < 150;
      
      if (isNearBottom) {
        // Use requestAnimationFrame for smooth scrolling
        requestAnimationFrame(() => {
          if (chatContainerRef.current) {
            chatContainerRef.current.scrollTop = chatContainerRef.current.scrollHeight;
          }
        });
      }
    }
  }, [scopedMessages]);

  // Helper function to get thread messages (from state or DB)
  const getThreadMessages = useCallback(async (threadId: string, limit: number = 10): Promise<Array<{ role: "user" | "assistant"; content: string }>> => {
    let threadMessages: Array<{ role: "user" | "assistant"; content: string }> = [];
    const isUuid = /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i.test(threadId || "");
    if (!threadId || !isUuid) {
      return messages
        .filter((m) => m.threadId === threadId)
        .slice(-limit)
        .map((m) => ({
          role: (m.author === "assistant" ? "assistant" : "user") as "user" | "assistant",
          content: m.text,
        }));
    }
    try {
      const eventId = activeEventId || (await ensureActiveEventId());
      if (eventId) {
        const { data: dbMessages, error } = await supabase
          .from("chat_messages")
          .select("role, content, created_at")
          .eq("event_id", eventId)
          .eq("thread_id", threadId)
          .order("created_at", { ascending: true })
          .limit(limit);
          
          if (error) {
            console.error("[DEBUG] Error fetching chat history from DB:", error);
          } else if (dbMessages && dbMessages.length > 0) {
            threadMessages = dbMessages.map((m: any) => ({
              role: (m.role === "assistant" ? "assistant" : "user") as "assistant" | "user",
              content: m.content || "",
            }));
          } else {
          }
        }
      } catch (fetchError) {
        console.error("[DEBUG] вќЊ Failed to fetch chat history from DB:", fetchError);
        // Fallback to state messages if DB fetch fails
        threadMessages = messages
          .filter((m) => m.threadId === threadId)
          .slice(-limit)
          .map((m) => ({
            role: (m.author === "assistant" ? "assistant" : "user") as "assistant" | "user",
            content: m.text,
          }));
      }

    // If still no messages, try state as last resort
    if (threadMessages.length === 0) {
      threadMessages = messages
        .filter((m) => m.threadId === threadId)
        .slice(-limit)
        .map((m) => ({
          role: (m.author === "assistant" ? "assistant" : "user") as "assistant" | "user",
          content: m.text,
        }));
    }
    
    return threadMessages;
  }, [messages, activeEventId, ensureActiveEventId]);

  // в”Ђв”Ђ Convert companyConnections state to the format expected by askClaudeAnswerStream в”Ђв”Ђ
  const connectionsForChat: AskFundConnection[] = useMemo(() => {
    return companyConnections.map((c) => ({
      source_company_name: c.source_company_name,
      target_company_name: c.target_company_name,
      connection_type: c.connection_type,
      connection_status: c.connection_status,
      ai_reasoning: c.ai_reasoning ?? null,
      notes: c.notes ?? null,
    }));
  }, [companyConnections]);

  // в”Ђв”Ђ Build a single card snippet (reusable) в”Ђв”Ђ
  const buildCardSnippet = useCallback(
    (card: { company_name: string; company_properties: Record<string, any> }, slim = false): string => {
      const p = card.company_properties || {};
      const parts: string[] = [`Company: ${card.company_name}`];
      if (p.industry) parts.push(`Industry: ${p.industry}`);
      const stage = p.funding_stage || p.stage || p.round || p.funding_round;
      if (stage) parts.push(`Stage: ${stage}`);
      if (p.business_model) parts.push(`Model: ${p.business_model}`);
      if (p.headquarters) parts.push(`HQ: ${p.headquarters}`);
      if (p.country) parts.push(`Country: ${p.country}`);
      if (p.location) parts.push(`Location: ${p.location}`);
      const geoList = p.geo_focus || p.geo_markets || p.geography || p.region || p.regions;
      if (Array.isArray(geoList) && geoList.length) parts.push(`Geo: ${geoList.join(", ")}`);
      else if (typeof geoList === "string") parts.push(`Geo: ${geoList}`);
      if (!slim) {
        if (p.bio) parts.push(`Bio: ${String(p.bio).slice(0, 200)}${String(p.bio).length > 200 ? "..." : ""}`);
        if (p.website) parts.push(`Website: ${p.website}`);
        if (p.linkedin_url) parts.push(`LinkedIn: ${p.linkedin_url}`);
        if (p.email) parts.push(`Email: ${p.email}`);
        if (p.phone) parts.push(`Phone: ${p.phone}`);
        if (p.twitter_url) parts.push(`Twitter/X: ${p.twitter_url}`);
      } else {
        if (p.bio) parts.push(`Bio: ${String(p.bio).slice(0, 80)}${String(p.bio).length > 80 ? "..." : ""}`);
      }
      return parts.join("\n");
    },
    []
  );

  // в”Ђв”Ђ Extract country/sector from question for filtering (e.g. "how many in Bangladesh") в”Ђв”Ђ
  const extractFilterFromQuestion = useCallback((question: string): { country?: string; sector?: string; stage?: string } => {
    const q = question.toLowerCase();
    const result: { country?: string; sector?: string; stage?: string } = {};
    // Common countries (expand as needed)
    const countries = [
      "bangladesh", "morocco", "egypt", "india", "pakistan", "nigeria", "kenya", "ghana",
      "indonesia", "vietnam", "philippines", "brazil", "mexico", "colombia", "argentina",
      "tunisia", "algeria", "saudi arabia", "uae", "turkey", "south africa", "ethiopia",
      "jordan", "lebanon", "senegal", "rwanda", "tanzania", "uganda", "cameroon",
      "ivory coast", "cote d'ivoire", "myanmar", "cambodia", "thailand", "malaysia",
      "singapore", "china", "japan", "korea", "usa", "united states", "uk", "united kingdom",
      "france", "germany", "spain", "canada", "australia",
    ];
    for (const c of countries) {
      if (q.includes(c)) {
        result.country = c;
        break;
      }
    }
    // Sectors
    const sectors = ["fintech", "saas", "b2b", "b2c", "healthtech", "edtech", "agritech", "logistics", "ecommerce", "proptech", "insurtech", "cleantech", "deeptech", "biotech", "medtech", "legaltech", "regtech", "hrtech", "martech"];
    for (const s of sectors) {
      if (q.includes(s)) {
        result.sector = s;
        break;
      }
    }
    // Funding stages
    const stages: Array<{ match: string; label: string }> = [
      { match: "pre-seed", label: "pre-seed" },
      { match: "preseed", label: "pre-seed" },
      { match: "pre seed", label: "pre-seed" },
      { match: "seed", label: "seed" },
      { match: "series a", label: "series a" },
      { match: "series b", label: "series b" },
      { match: "series c", label: "series c" },
      { match: "series d", label: "series d" },
    ];
    for (const s of stages) {
      if (q.includes(s.match)) {
        result.stage = s.label;
        break;
      }
    }
    return result;
  }, []);

  // в”Ђв”Ђ Smart company card sources: filter by country/sector, cap, put matches first в”Ђв”Ђ
  const buildCompanyCardSources = useCallback(
    (question: string, cards: Array<{ company_name: string; company_properties: Record<string, any> }>, detectedNames: string[]): Array<{ title: string | null; file_name: string | null; snippet: string | null }> => {
      if (!cards.length) return [];

      // Deduplicate by core company name вЂ” "Chhaya Technologies PTE. LTD." and "Chhaya" merge
      const deduped = new Map<string, typeof cards[0]>();
      for (const card of cards) {
        const core = extractCoreCompanyName(card.company_name);
        if (!core) continue;
        const existing = deduped.get(core);
        if (!existing || JSON.stringify(card.company_properties).length > JSON.stringify(existing.company_properties).length) {
          deduped.set(core, card);
        }
      }
      cards = Array.from(deduped.values());

      const qLower = question.toLowerCase();
      const nameLowers = detectedNames.map(n => n.toLowerCase());
      const filter = extractFilterFromQuestion(question);

      // Determine if this is a "list all" / portfolio-wide question
      const isPortfolioWide = /\b(how many|list|all compan|portfolio|every company|which compan)\b/i.test(question);
      const isFilterQuestion = /\b(in\s+\w+|from\s+\w+|based in|headquartered|country|sector|stage|preseed|pre-seed|seed|series)\b/i.test(question) ||
        filter.country || filter.sector || filter.stage;

      const result: Array<{ title: string | null; file_name: string | null; snippet: string | null }> = [];

      const cardMatchesFilter = (card: { company_name: string; company_properties: Record<string, any> }): boolean => {
        const p = card.company_properties || {};
        const cardText = [
          p.headquarters || "", p.country || "", p.location || "",
          p.industry || "", p.bio || "",
          ...(Array.isArray(p.geo_focus) ? p.geo_focus : []),
          ...(Array.isArray(p.geo_markets) ? p.geo_markets : []),
        ].join(" ").toLowerCase();
        const stageText = [
          p.funding_stage || "", p.stage || "", p.round || "",
        ].join(" ").toLowerCase();
        let matches = true;
        let hasFilter = false;
        if (filter.country) { hasFilter = true; if (!cardText.includes(filter.country)) matches = false; }
        if (filter.sector) { hasFilter = true; if (!cardText.includes(filter.sector)) matches = false; }
        if (filter.stage) { hasFilter = true; if (!stageText.includes(filter.stage) && !cardText.includes(filter.stage)) matches = false; }
        return hasFilter && matches;
      };

      const STOP_WORDS = new Set([
        "about", "after", "also", "been", "before", "between", "both", "call",
        "came", "come", "could", "each", "find", "from", "give", "have", "help",
        "here", "high", "into", "just", "know", "last", "like", "long", "look",
        "made", "make", "many", "more", "most", "much", "must", "need", "only",
        "over", "said", "same", "some", "such", "take", "tell", "than", "that",
        "the", "them", "then", "there", "these", "they", "this", "time", "very",
        "want", "well", "were", "what", "when", "which", "will", "with", "work",
        "would", "your", "company", "companies", "portfolio", "startup", "startups",
        "information", "details", "about", "everything", "anything", "something",
        "right", "currently", "please", "should", "think", "really", "today",
      ]);
      const cardMatchesNameOrContent = (card: { company_name: string; company_properties: Record<string, any> }): boolean => {
        const nameLower = (card.company_name || "").toLowerCase();
        const p = card.company_properties || {};
        if (nameLowers.some(n => nameLower.includes(n) || n.includes(nameLower))) return true;
        if (qLower.includes(nameLower) && nameLower.length >= 3) return true;
        if (nameLower.includes(qLower.split(/[\s,?!]+/)[0]) && qLower.split(/[\s,?!]+/)[0].length >= 3) return true;
        const meaningfulTokens = qLower.split(/[\s,?!.;:]+/).filter(w => w.length >= 4 && !STOP_WORDS.has(w));
        if (!meaningfulTokens.length) return false;
        return meaningfulTokens.some(tok => nameLower.includes(tok));
      };

      // For "how many in Bangladesh" or "all B2B SaaS preseed": matching cards FULL, rest SLIM
      if ((isPortfolioWide || isFilterQuestion) && (filter.country || filter.sector || filter.stage)) {
        const matching = cards.filter(c => (c.company_name || "").trim() && cardMatchesFilter(c));
        const nonMatching = cards.filter(c => (c.company_name || "").trim() && !cardMatchesFilter(c));
        // Matching cards first (full snippet)
        for (const card of matching) {
          result.push({ title: `Company card: ${card.company_name}`, file_name: null, snippet: buildCardSnippet(card, false) });
        }
        // ALL non-matching cards get slim snippets so the model knows about the entire portfolio
        for (const card of nonMatching) {
          result.push({ title: `Company card: ${card.company_name}`, file_name: null, snippet: buildCardSnippet(card, true) });
        }
        return result;
      }

      // General case: relevant cards (name/content match) full, others slim, cap total
      const relevantCards: typeof cards = [];
      const otherCards: typeof cards = [];
      for (const card of cards) {
        if (!(card.company_name || "").trim()) continue;
        if (cardMatchesNameOrContent(card)) relevantCards.push(card);
        else otherCards.push(card);
      }

      for (const card of relevantCards) {
        result.push({ title: `Company card: ${card.company_name}`, file_name: null, snippet: buildCardSnippet(card, false) });
      }
      // For portfolio-wide or filter questions, send matching cards + slim list of others (capped)
      if (isPortfolioWide || isFilterQuestion) {
        // Cap non-matching cards to avoid token waste вЂ” 20 slim cards is enough for context
        for (const card of otherCards.slice(0, 20)) {
          result.push({ title: `Company card: ${card.company_name}`, file_name: null, snippet: buildCardSnippet(card, true) });
        }
        if (otherCards.length > 20) {
          const remaining = otherCards.slice(20).map(c => c.company_name).join(", ");
          result.push({ title: "Other portfolio companies", file_name: null, snippet: `Also in portfolio: ${remaining}` });
        }
      } else {
        // For specific-company questions, only include a few extra for cross-reference
        const isSpecificCompany = relevantCards.length > 0 && relevantCards.length <= 3;
        const otherCap = isSpecificCompany ? 3 : 10;
        for (const card of otherCards.slice(0, otherCap)) {
          result.push({ title: `Company card: ${card.company_name}`, file_name: null, snippet: buildCardSnippet(card, true) });
        }
      }
      return result;
    },
    [buildCardSnippet, extractFilterFromQuestion]
  );

  const askFund = useCallback(
    async (question: string, threadId: string) => {
      if (!scopes.some((s) => s.checked)) {
        createAssistantMessage("Select at least one scope to search fund memory.", threadId);
        return;
      }

      const eventId = activeEventId || (await ensureActiveEventId());
      if (!eventId) {
        createAssistantMessage("I can't access documents yet. Please try again in a moment.", threadId);
        return;
      }

      // в•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђ
      // AGENTIC RAG вЂ” Backend-driven retrieval with Claude tool use
      // When enabled, the backend handles ALL retrieval (SQL, vector search,
      // knowledge graph) via Claude's tool_use loop. The frontend just streams.
      // в•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђ
      // When multi-agent is ON, use the legacy pipeline which has orchestrator + graph + KPI agents.
      // When OFF, use the faster backend-driven agent RAG.
      const USE_AGENT_RAG = !multiAgentEnabled;

      if (USE_AGENT_RAG) {
        setChatIsLoading(false);
        setIsClaudeLoading(true);
        setLastEvidence(null);

        const streamer = createStreamingAssistantMessage(threadId);
        let streamCompleted = false;
        const agentTimeoutMs = webSearchEnabled ? 120000 : 90000;
        const agentTimeout = window.setTimeout(() => {
          if (!streamCompleted) {
            streamCompleted = true;
            streamer.setError("Request timed out. Please try again with a simpler question.");
            setIsClaudeLoading(false);
          }
        }, agentTimeoutMs);

        try {
          const threadMsgs = await getThreadMessages(threadId, 10);

          // Compute folder scope for agentic RAG
          const agentFolderIds = scopes
            .filter((s) => s.type === "folder" && s.checked)
            .map((s) => s.id.replace("folder:", ""));

          await askAgentStream(
            {
              question,
              eventId,
              previousMessages: threadMsgs,
              webSearchEnabled: webSearchEnabled,
              folderIds: agentFolderIds.length > 0 ? agentFolderIds : undefined,
            },
            (chunk) => {
              if (!streamCompleted) streamer.appendChunk(chunk);
            },
            (status) => {
            },
            (err) => {
              if (!streamCompleted) {
                streamCompleted = true;
                clearTimeout(agentTimeout);
                streamer.setError(err.message || "Failed. Please try again.");
                setIsClaudeLoading(false);
              }
            },
            chatAbortRef.current?.signal,
            (sources) => {
              if (!streamCompleted) streamer.setVerifiableSources(sources);
            },
            (criticText) => {
              if (!streamCompleted) streamer.setCritic(criticText);
            },
            (docs) => {
              if (!streamCompleted) streamer.setSourceDocs(docs);
            }
          );

          if (!streamCompleted) {
            streamCompleted = true;
            clearTimeout(agentTimeout);
            streamer.finalize();
          }
        } catch (err) {
          if (!streamCompleted) {
            streamCompleted = true;
            clearTimeout(agentTimeout);
            streamer.setError(err instanceof Error ? err.message : "Failed. Please try again.");
          }
        } finally {
          clearTimeout(agentTimeout);
          setIsClaudeLoading(false);
        }
        return; // Skip entire old retrieval pipeline
      }

      // в•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђ
      // LEGACY PIPELINE (below) вЂ” only runs when USE_AGENT_RAG = false
      // в•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђ

      const previousEvidence = lastEvidence;
      const previousEvidenceThreadId = lastEvidenceThreadId;
      setChatIsLoading(true);
      // Reset evidence for new prompt to avoid showing previous sources
      setLastEvidence(null);
      let timedOut = false;
      let searchTimeoutId: number | null = null;
      // Increased timeout: 60s for document search (90s when web search enabled)
      // This timeout is cleared once documents are found or Claude starts processing
      const searchTimeoutMs = webSearchEnabled ? 90000 : 60000;
      const searchTimeoutId_temp = window.setTimeout(() => {
        timedOut = true;
        setChatIsLoading(false);
        createAssistantMessage(
          `Search is taking too long (${Math.round(searchTimeoutMs / 1000)}s timeout). Please try a more specific query or check your connection.`,
          threadId
        );
      }, searchTimeoutMs);
      searchTimeoutId = searchTimeoutId_temp;
      const myDocsSelected = scopes.find((s) => s.id === "my-docs")?.checked ?? false;
      const teamDocsSelected = scopes.find((s) => s.id === "team-docs")?.checked ?? false;
      const currentUserId = profile?.id || user?.id || null;
      const selectedFolderIds = scopes
        .filter((s) => s.type === "folder" && s.checked)
        .map((s) => s.id.replace("folder:", ""));

      const filterDocsByFolderScope = async <T extends { id: string; folder_id?: string | null }>(
        docList: T[]
      ): Promise<T[]> => {
        if (selectedFolderIds.length === 0 || docList.length === 0) return docList;
        const docIds = docList.map((doc) => doc.id);
        try {
          // Check document_folder_links table
          const { data: links } = await supabase
            .from("document_folder_links")
            .select("document_id, folder_id")
            .in("document_id", docIds)
            .in("folder_id", selectedFolderIds);
          const allowed = new Set((links || []).map((row: any) => row.document_id));
          
          // Also check if document's direct folder_id matches any selected folder
          const filtered = docList.filter(
            (doc) =>
              allowed.has(doc.id) ||
              (doc.folder_id && selectedFolderIds.includes(doc.folder_id))
          );
          
          
          // IMPORTANT: If folder filter removes ALL documents, return the original list
          // This prevents the case where a document IS in the folder but the link table
          // is out of sync. Better to show too many results than none.
          if (filtered.length === 0 && docList.length > 0) {
            console.warn("[DEBUG] Folder scope filter removed ALL docs вЂ” returning unfiltered to avoid empty results");
            return docList;
          }
          
          return filtered;
        } catch (err) {
          console.warn("Folder scope filter failed:", err);
          return docList;
        }
      };
      
      // REWRITE QUERY BEFORE SEARCHING (ChatGPT-style "Condense" step)
      // Use backend LLM-based rewriting for robust pronoun resolution
      let searchQuestion = question;
      
      // Get thread messages (from state or DB)
      const threadMessages = await getThreadMessages(threadId, 10); // Get more messages for better context
      
      if (threadMessages.length > 0) {
      }
      
      // Use backend LLM rewriting if we have chat history and the question might need rewriting
      const qLower = question.toLowerCase();
      const hasPronouns = /\b(it|its|him|his|her|she|they|them|their|this|that|these|those)\b/i.test(question);
      const hasVaguePattern = /\b(tell me more|tell me all|what about|and what|how about|what else|tell more|more about|more details|more info|more complete|more comprehensive|more profound|give more|give more info|expand|elaborate|all you know|everything|full|complete|comprehensive|detailed|another|other|different|alternative|someone else|something else|any other|next one|one more)\b/i.test(qLower);
      const followUpCueInQuestion = /\b(what about|and what|tell me|more about|more info|more complete|more comprehensive|more profound|give more|give more info|elaborate|explain|requirements|responsibilities|limitations|cannot|can't|couldn't|allowed|forbidden|answer|profound|comprehensive|detail|full|complete|detailed|another|other|different|alternative|someone else|something else|any other|next one|one more)\b/i.test(qLower);
      
      // CRITICAL: Detect "another/different/other" queries вЂ” user wants NEW results, not repeats
      const wantsAlternative = /\b(another|other|different|alternative|someone else|something else|any other|next one|one more|what else|who else|which else)\b/i.test(qLower);
      const isShort = question.split(/\s+/).length <= 15;
      
      // CRITICAL: Extract names from chat history for fallback pronoun replacement
      const extractNamesFromHistory = (msgs: Array<{ role: string; content: string }>): string[] => {
        // ONLY look at USER messages вЂ” assistant responses have tons of capitalized
        // section headers ("Due Diligence", "Information Accuracy", "Burn Rate") that
        // are NOT company/person names and pollute pronoun resolution
        const userText = msgs.filter(m => m.role === "user").map(m => m.content).join(" ");
        const namePattern = /\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b/g;
        const rawNames = userText.match(namePattern) || [];
        // Filter out common VC/business terms that look like names but aren't
        const FAKE_NAME_TERMS = new Set([
          "Due Diligence", "Information Accuracy", "Burn Rate", "Monthly Expenses",
          "Monthly Revenues", "Runway Calculation", "Current Cash", "Monthly Burn",
          "Qualified Financing", "Product Due", "Diligence Concepts", "Aha Moment",
          "Legal Due", "Diligence Framework", "Grant Agreement", "Corporate Standing",
          "Financial Representations", "Legal Compliance", "Document Analysis",
          "Structured Information", "Reference Analysis", "Information Available",
          "Financial Metrics", "Analysis Framework", "Series A", "Series B",
          "Series C", "Pre Seed", "Total Addressable", "Market Size", "Business Model",
          "Revenue Model", "Key Metrics", "Go To", "Market Strategy", "Team Size",
          "Gross Margin", "Net Margin", "Churn Rate", "Customer Acquisition",
          "Value Proposition", "Competitive Edge", "Market Growth",
        ]);
        // Cross-reference with known company cards if available
        const knownCompanyNames = new Set(
          (companyCards || []).map(c => (c.company_name || "").toLowerCase())
        );
        return [...new Set(rawNames)].filter(n => {
          if (FAKE_NAME_TERMS.has(n)) return false;
          // Keep if it matches a known company name
          if (knownCompanyNames.has(n.toLowerCase())) return true;
          // Keep if it's a single capitalized word (likely a company name like "Payd", "Chari")
          if (/^[A-Z][a-z]+$/.test(n) && n.length >= 3) return true;
          // Keep if it looks like a person name (2 words, each 2-15 chars)
          if (/^[A-Z][a-z]{1,14}\s+[A-Z][a-z]{1,14}$/.test(n)) {
            const words = n.split(/\s+/);
            return words.every(w => w.length >= 2 && w.length <= 15);
          }
          return false;
        });
      };
      
      // Extract company/entity names that were already mentioned in ASSISTANT responses
      // Used to tell Claude to exclude these when user asks for "another" option
      const extractMentionedCompaniesFromAssistant = (msgs: Array<{ role: string; content: string }>): string[] => {
        const assistantTexts = msgs.filter(m => m.role === "assistant").map(m => m.content).join("\n");
        const companies: string[] = [];
        // Match bold headers like **CoreTechX** or **Company Name**
        const boldPattern = /\*\*([A-Z][A-Za-z0-9\s&\-\.]+?)\*\*/g;
        let match;
        while ((match = boldPattern.exec(assistantTexts)) !== null) {
          const name = match[1].trim();
          // Filter out common headings that aren't company names
          const skipWords = new Set(['Key', 'Note', 'Summary', 'Recommendation', 'Connection Type', 'Potential Synergies', 'Rationale', 'Sources Used', 'MENA Portfolio', 'Why', 'How', 'What', 'Location', 'Business', 'Pricing Model', 'Core Business', 'Key Value Proposition']);
          if (name.length > 1 && name.length < 50 && !skipWords.has(name) && !/^(The |A |An |This |That |Here |Our |Your )/.test(name)) {
            companies.push(name);
          }
        }
        // Also match "# Name" or "## Name" markdown headers  
        const headerPattern = /^#{1,3}\s+([A-Z][A-Za-z0-9\s&\-\.]+)/gm;
        while ((match = headerPattern.exec(assistantTexts)) !== null) {
          const name = match[1].trim();
          if (name.length > 1 && name.length < 50) {
            companies.push(name);
          }
        }
        return [...new Set(companies)];
      };
      
      const namesInHistory = extractNamesFromHistory(threadMessages);
      
      if ((hasPronouns || hasVaguePattern || isShort) && threadMessages.length > 0) {
        try {
          // Call backend LLM to rewrite the query (much more robust than frontend heuristics)
          searchQuestion = await rewriteQueryWithLLM(question, threadMessages);
          
          // VALIDATION: Only fall back to name injection if LLM rewrite failed AND
          // we have VERIFIED company names (not AI response headers)
          if (hasPronouns && namesInHistory.length > 0) {
            const rewrittenLower = searchQuestion.toLowerCase();
            const hasNameInRewritten = namesInHistory.some(name => rewrittenLower.includes(name.toLowerCase()));
            if (!hasNameInRewritten) {
              // Only inject if the name is a KNOWN company (cross-checked against cards)
              const knownCompanyLower = new Set((companyCards || []).map(c => (c.company_name || "").toLowerCase()));
              const verifiedName = namesInHistory.reverse().find(n => knownCompanyLower.has(n.toLowerCase()));
              if (verifiedName) {
                searchQuestion = question;
                for (const pronoun of ["him", "her", "it", "they", "them", "his", "her", "their", "this", "that"]) {
                  const regex = new RegExp(`\\b${pronoun}\\b`, "gi");
                  searchQuestion = searchQuestion.replace(regex, verifiedName);
                }
              } else {
              }
            }
          }
          
          // CRITICAL: Extract company name from the LAST USER QUESTION (most important context)
          // This ensures we filter out documents about other companies
          const lastUserQuestion = threadMessages.filter(m => m.role === "user").slice(-1)[0]?.content || "";
          const companyNameFromLastQ = (() => {
            // Look for capitalized words that might be company names
            const matches = lastUserQuestion.match(/\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\b/g) || [];
            // Filter out common words
            const commonWords = new Set(['The', 'This', 'That', 'Here', 'There', 'What', 'When', 'Where', 'Which', 'Could', 'Would', 'Should', 'Based', 'Found', 'Sorry', 'Please', 'User', 'Assistant', 'How', 'Help', 'Make', 'Go', 'On', 'Given', 'Resources', 'Have', 'Right', 'Now']);
            return matches.filter(m => !commonWords.has(m) && m.length > 2)[0] || null;
          })();
          
          if (companyNameFromLastQ) {
            // Ensure the rewritten query includes this company name
            const searchLower = searchQuestion.toLowerCase();
            if (!searchLower.includes(companyNameFromLastQ.toLowerCase())) {
              searchQuestion = `${companyNameFromLastQ} ${searchQuestion}`.replace(/\s+/g, " ").trim();
            }
          }
        } catch (rewriteError) {
          console.warn("[DEBUG] Query rewriting failed:", rewriteError);
          // FALLBACK: If we have pronouns and names in history, replace pronouns with the most recent name
          if (hasPronouns && namesInHistory.length > 0) {
            const mostRecentName = namesInHistory[namesInHistory.length - 1];
            searchQuestion = question;
            for (const pronoun of ["him", "her", "it", "they", "them", "his", "her", "their", "this", "that"]) {
              const regex = new RegExp(`\\b${pronoun}\\b`, "gi");
              searchQuestion = searchQuestion.replace(regex, mostRecentName);
            }
          } else {
            searchQuestion = question;
          }
        }
      } else if ((hasPronouns || hasVaguePattern || followUpCueInQuestion) && namesInHistory.length > 0) {
        // Even if no LLM rewriting triggered, still resolve pronouns if we have names
        const mostRecentName = namesInHistory[namesInHistory.length - 1];
        for (const pronoun of ["him", "her", "it", "they", "them", "his", "her", "their", "this", "that"]) {
          const regex = new RegExp(`\\b${pronoun}\\b`, "gi");
          searchQuestion = searchQuestion.replace(regex, mostRecentName);
        }
        // If no pronouns were present, append subject to the query
        if (!hasPronouns && !searchQuestion.toLowerCase().includes(mostRecentName.toLowerCase())) {
          searchQuestion = `${searchQuestion} about ${mostRecentName}`.replace(/\s+/g, " ").trim();
        }
      }
      
      
      // PHASE 1: Extract proper nouns (names) BEFORE cleaning to preserve them
      const extractProperNouns = (text: string): string[] => {
        // Find capitalized words (potential names)
        const pattern = /\b[A-Z][a-z]+\b/g;
        const matches = text.match(pattern) || [];
        const commonCaps = new Set(['The', 'A', 'An', 'And', 'Or', 'But', 'In', 'On', 'At', 'To', 'For', 'Of', 'With', 'By']);
        return matches.filter(m => !commonCaps.has(m) && m.length > 2);
      };
      
      const properNouns = extractProperNouns(searchQuestion);
      const properNounsLower = properNouns.map(pn => pn.toLowerCase());
      
      // Detect if query contains names (Phase 1) - IMPROVED for typo tolerance
      const detectNameInQuery = (query: string): [boolean, string[]] => {
        const nouns = extractProperNouns(query);
        // Pattern for "FirstName LastName" - capital letters at start
        const namePattern = /\b[A-Z][a-z]+\s+[A-Z][a-z]+\b/g;
        const nameMatches = (query.match(namePattern) || []).map(m => m.trim());
        
        // Also detect potential names with typos - capitalized words of 4+ letters
        const potentialNames = /\b[A-Z][a-z]{3,}\b/g;
        const potentialMatches = (query.match(potentialNames) || []).filter(m => 
          !['What', 'Where', 'When', 'Which', 'About', 'Tell', 'Give', 'Find', 'Search', 'Show', 'Explain', 'Describe', 'More', 'Complete', 'Comprehensive'].includes(m)
        );
        
        // All-caps words (e.g. BIANCA, Weego) so chat can match company cards
        const allCapsPattern = /\b[A-Z]{2,}\b/g;
        const allCapsMatches = (query.match(allCapsPattern) || []).filter(m => 
          !['AI', 'LLC', 'USA', 'UK', 'CEO', 'CFO', 'CTO', 'VP', 'HR', 'IT', 'ID', 'UI', 'API'].includes(m)
        );
        
        // Check for common name-like patterns even without capitals (handles "george goloborodkin" in lowercase)
        const lowerQuery = query.toLowerCase();
        const hasNameLikeWords = /\b[a-z]{4,}\s+[a-z]{6,}\b/.test(lowerQuery); // First + Last name pattern
        
        const allNames = Array.from(new Set([...nouns, ...nameMatches, ...potentialMatches, ...allCapsMatches]));
        
        // If query looks like a "who is X" or "tell me about X" pattern, assume it's a name query
        const isNameQuery = allNames.length > 0 || 
          /\b(who is|about|tell me about|search for)\s+\w{4,}/i.test(query) ||
          hasNameLikeWords;
        
        return [isNameQuery, allNames];
      };
      
      const [hasName, detectedNames] = detectNameInQuery(searchQuestion);
      
      // QUERY CLEANING: Remove instruction words to focus on entities/keywords
      // PRESERVES proper nouns (Phase 1)
      const instructionWords = [
        "summarize", "summarise", "tell me about", "tell me", "find", "search for",
        "what is", "what are", "what does", "explain", "describe", "show me",
        "get", "fetch", "retrieve", "give me", "provide", "list", "show",
      ];
      let cleanedSearchQuery = searchQuestion;
      for (const instruction of instructionWords) {
        const regex = new RegExp(`\\b${instruction}\\b`, "gi");
        cleanedSearchQuery = cleanedSearchQuery.replace(regex, "");
      }
      // Clean up extra spaces
      cleanedSearchQuery = cleanedSearchQuery.replace(/\s+/g, " ").trim();
      
      // Ensure proper nouns are preserved (Phase 1)
      if (properNouns.length > 0) {
        const cleanedLower = cleanedSearchQuery.toLowerCase();
        for (const pn of properNouns) {
          if (!cleanedLower.includes(pn.toLowerCase())) {
            cleanedSearchQuery = `${pn} ${cleanedSearchQuery}`.trim();
          }
        }
      }
      
      // Use cleaned query if it's not empty, otherwise use original
      let finalSearchQuery = cleanedSearchQuery || searchQuestion;
      
      // PHASE 2: Query intent classification
      const classifyQueryIntent = (query: string): string => {
        const qLower = query.toLowerCase();
        if (/\b(find|search|locate|get|fetch|retrieve|show me)\b/.test(qLower)) return "FIND";
        if (/\b(summarize|summarise|summary|overview|brief|sum up)\b/.test(qLower)) return "SUMMARIZE";
        if (/\b(explain|why|how does|how do|what is|what are)\b/.test(qLower)) return "EXPLAIN";
        if (/\b(compare|difference|versus|vs|contrast)\b/.test(qLower)) return "COMPARE";
        return "FIND"; // Default
      };
      
      const queryIntent = classifyQueryIntent(question);
      
      const normalizedQuestion = finalSearchQuery.toLowerCase();
      // Unicode-aware tokenization (supports non-English)
      const tokens = normalizedQuestion
        .split(/[\s\p{P}]+/u)
        .map((t) => t.trim())
        .filter((t) => t.length > 2);
      const contentStopwords = new Set([
        "what",
        "about",
        "know",
        "tell",
        "me",
        "the",
        "and",
        "for",
        "with",
        "his",
        "her",
        "their",
        "there",
        "this",
        "that",
        "these",
        "those",
        "who",
        "when",
        "where",
        "why",
        "how",
        "company",
        "startup",
        "business",
      ]);
      const contentTokens = tokens.filter((t) => !contentStopwords.has(t));
      const isComprehensiveQuestion =
        /\b(all you know|everything|comprehensive|detailed|full|complete|tell me all|what do you know|what can you tell me|summarize|overview)\b/i.test(
          question
        );
      const followUpHasPronoun = /\b(it|its|they|them|their|he|him|his|she|her|hers|there|that|those|these)\b/i.test(normalizedQuestion);
      const isFollowUpQuery = (() => {
        const q = normalizedQuestion;
        const isShort = q.split(/\s+/).length <= 15; // Increased from 12
        return (followUpHasPronoun || followUpCueInQuestion) && isShort;
      })();
      let docs: Array<{
        id: string;
        title: string | null;
        file_name: string | null;
        raw_content: string | null;
        extracted_json?: Record<string, any> | null;
        created_at: string;
        storage_path: string | null;
        folder_id?: string | null;
      }> = [];
      const snippetByDocId = new Map<string, string>();
      let error: { message?: string } | null = null;
      let semanticFailed = false;
      let semanticMatches: Array<{ document_id: string; similarity: number; chunk_text?: string | null; parent_text?: string | null }> = [];
      let keywordMatches: Array<{ document_id: string; rank: number; snippet?: string | null }> = [];

      const canSemantic = tokens.length >= 1;

      // в”Ђв”Ђ STEP 1: Query Router вЂ” analyze intent, entities, complexity, routing strategy в”Ђв”Ђ
      let queryAnalysis: QueryAnalysis | null = null;
      try {
        queryAnalysis = await analyzeQuery(question, threadMessages);
        // Use rewritten query from router if available, BUT ensure it includes company name from conversation
        if (queryAnalysis.rewritten_query && queryAnalysis.rewritten_query !== question) {
          // Extract company name from last user question
          const lastUserQuestion = threadMessages.filter(m => m.role === "user").slice(-1)[0]?.content || "";
          const companyNameFromLastQ = (() => {
            const matches = lastUserQuestion.match(/\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\b/g) || [];
            const commonWords = new Set(['The', 'This', 'That', 'Here', 'There', 'What', 'When', 'Where', 'Which', 'Could', 'Would', 'Should', 'Based', 'Found', 'Sorry', 'Please', 'User', 'Assistant', 'How', 'Help', 'Make', 'Go', 'On', 'Given', 'Resources', 'Have', 'Right', 'Now', 'About', 'Tell', 'Me']);
            return matches.filter(m => !commonWords.has(m) && m.length > 2)[0] || null;
          })();
          
          if (companyNameFromLastQ) {
            const rewrittenLower = queryAnalysis.rewritten_query.toLowerCase();
            if (!rewrittenLower.includes(companyNameFromLastQ.toLowerCase())) {
              // Inject company name into router's rewritten query
              finalSearchQuery = `${companyNameFromLastQ} ${queryAnalysis.rewritten_query}`.replace(/\s+/g, " ").trim();
            } else {
              finalSearchQuery = queryAnalysis.rewritten_query;
            }
          } else {
            finalSearchQuery = queryAnalysis.rewritten_query;
          }
        }
      } catch (routerErr) {
        console.warn("[ROUTER] Analysis failed, using fallback:", routerErr);
        queryAnalysis = null;
      }

      // Clear search timeout as soon as we start document search (search is in progress)
      if (searchTimeoutId !== null) {
        window.clearTimeout(searchTimeoutId);
        searchTimeoutId = null;
      }
      
      // в•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђ
      // FAST PATH: Portfolio / company-index questions в†’ skip heavy RAG
      // Questions like "how many companies in Morocco?", "list my fintech companies",
      // "which companies could connect with X?" can be answered from cards alone.
      // в•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђ
      const isPortfolioIndexQuestion = (() => {
        const q = normalizedQuestion;
        return (
          /\b(how many|list|show|tell me|all compan|portfolio|every company|which compan|companies in|companies from|headquartered|based in|preseed|pre-seed|seed stage|series [a-d]|b2b|b2c|saas|fintech|healthtech|edtech|agritech)\b/i.test(question) &&
          !isComprehensiveQuestion &&
          !(/\b(pitch|deck|memo|document|meeting notes|revenue|mrr|arr|kpi|financial)\b/i.test(q))
        );
      })();

      if (isPortfolioIndexQuestion && companyCards.length > 0) {
        if (searchTimeoutId !== null) {
          window.clearTimeout(searchTimeoutId);
        }
        setChatIsLoading(false);
        setIsClaudeLoading(true);
        const streamer = createStreamingAssistantMessage(threadId);
        let streamCompleted = false;
        const streamTimeout = setTimeout(() => {
          if (!streamCompleted) {
            streamer.setError("Request timed out. Please try again.");
            setIsClaudeLoading(false);
          }
        }, 45000);
        try {
          const cardSources = buildCompanyCardSources(question, companyCards, detectedNames || []);
          const threadMsgs = await getThreadMessages(threadId, 10);
          await askClaudeAnswerStream(
            {
              question,
              sources: cardSources,
              decisions: [],
              connections: connectionsForChat,
              previousMessages: threadMsgs,
            },
            (chunk) => { if (!streamCompleted) streamer.appendChunk(chunk); },
            (err) => {
              if (!streamCompleted) {
                streamCompleted = true;
                clearTimeout(streamTimeout);
                streamer.setError(err.message || "Failed. Please try again.");
                setIsClaudeLoading(false);
              }
            },
            chatAbortRef.current?.signal
          );
          if (!streamCompleted) {
            streamCompleted = true;
            clearTimeout(streamTimeout);
            streamer.finalize();
          }
        } catch (err) {
          streamCompleted = true;
          clearTimeout(streamTimeout);
          streamer.setError(err instanceof Error ? err.message : "Failed. Please try again.");
        } finally {
          setIsClaudeLoading(false);
        }
        return;
      }

      // в•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђ
      // MULTI-AGENT RAG (only when toggle is ON)
      // When OFF в†’ classic single-path RAG (vector + optional web search)
      // When ON  в†’ Orchestrator в†’ parallel Graph/KPI/Vector в†’ Critic
      // в•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђ
      let routingPlan = { use_vector: true, use_graph: false, use_kpis: false, use_web: false, reasoning: "", sub_queries: {} as Record<string, string> };
      let graphContext = "";
      let kpiContext = "";

      if (multiAgentEnabled) {
        // в”Ђв”Ђ Step 1: Orchestrator (Router Agent) в”Ђв”Ђ
        try {
          const threadMsgsForRouter = await getThreadMessages(threadId, 5);
          routingPlan = await orchestrateQuery({
            question: finalSearchQuery,
            previousMessages: threadMsgsForRouter,
          });

          if (routingPlan.use_web && !webSearchEnabled) {
          }
        } catch (routerErr) {
          console.warn("[MULTI-AGENT] Orchestrator failed, falling back to vector-only:", routerErr);
        }

        // Client-side override: force graph ON for connection/relationship questions
        const connectionKeywords = /\b(connect|connections?|relationship|partner|who\s+invest|portfolio|expand|expansion|introduce|introductions?|linked|network|graph)\b/i;
        if (connectionKeywords.test(question) && !routingPlan.use_graph) {
          routingPlan.use_graph = true;
        }
        // Force KPIs ON for metric questions
        const metricKeywords = /\b(arr|revenue|valuation|burn|runway|growth|metric|kpi|financials?|numbers?)\b/i;
        if (metricKeywords.test(question) && !routingPlan.use_kpis) {
          routingPlan.use_kpis = true;
        }
        // Force graph + KPIs ON for multi-company comparison / differs / business model questions
        const comparisonKeywords = /\b(compare|comparison|differs?|difference|versus|vs\.?|business\s*model|between\s+\w+\s+and)\b/i;
        if (comparisonKeywords.test(question)) {
          if (!routingPlan.use_graph) {
            routingPlan.use_graph = true;
          }
          if (!routingPlan.use_kpis) {
            routingPlan.use_kpis = true;
          }
        }
      } else {
      }

      // в”Ђв”Ђ Step 2: Parallel Graph/KPI Retrieval (multi-agent only) в”Ђв”Ђ
      const graphPromise = (multiAgentEnabled && routingPlan.use_graph && eventId) ? (async () => {
        try {
          const entityNames = queryAnalysis?.entities?.map((e: any) => e.name) || [];
          const searchWords = finalSearchQuery.split(/\s+/).filter((w: string) => w.length > 3);
          const result = await retrieveGraphContext(eventId, searchWords, entityNames);
          return result.summary;
        } catch { return ""; }
      })() : Promise.resolve("");

      const kpiPromise = (multiAgentEnabled && routingPlan.use_kpis && eventId) ? (async () => {
        try {
          const companyNames = queryAnalysis?.entities
            ?.filter((e: any) => e.type === "company" || e.type === "fund")
            ?.map((e: any) => e.name) || [];
          const metricNames = queryAnalysis?.entities
            ?.filter((e: any) => e.type === "metric")
            ?.map((e: any) => e.name) || [];
          const result = await retrieveKpiContext(eventId, companyNames.length > 0 ? companyNames : undefined, metricNames.length > 0 ? metricNames : undefined);
          return result.summary;
        } catch { return ""; }
      })() : Promise.resolve("");

      // в•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђ
      // PARALLEL RETRIEVAL with global time budget (8s)
      // Fire semantic + keyword search simultaneously, merge with RRF.
      // Only run heavier fallbacks if both came back empty.
      // в•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђ
      const RETRIEVAL_BUDGET_MS = 8000;
      const retrievalDeadline = Date.now() + RETRIEVAL_BUDGET_MS;
      const isRetrievalBudgetExhausted = () => Date.now() >= retrievalDeadline;

      // в”Ђв”Ђ Multi-Query + Semantic search promise в”Ђв”Ђ
      const semanticSearchPromise = (async (): Promise<typeof semanticMatches> => {
        if (!canSemantic) return [];
        try {
          // Step 1: Generate multi-query variants (fast, with fallback to original)
          let queryVariants: string[] = [finalSearchQuery];
          try {
            const mqResult = await Promise.race([
              generateMultiQueries(finalSearchQuery, 3),
              new Promise<{ queries: string[]; model_used: string }>((resolve) =>
                setTimeout(() => resolve({ queries: [finalSearchQuery], model_used: "" }), 3000)
              ),
            ]);
            if (mqResult.queries.length > 1) {
              queryVariants = mqResult.queries;
            }
          } catch {
          }

          // Step 2: Embed all variants in parallel
          const embeddingTimeout = Math.min(8000, RETRIEVAL_BUDGET_MS);
          const embedPromises = queryVariants.map((q) =>
            Promise.race([
              embedQuery(q, "query"),
              new Promise<number[]>((_, reject) =>
                setTimeout(() => reject(new Error("Embedding timeout")), embeddingTimeout)
              ),
            ]).catch(() => null as number[] | null)
          );
          const embeddings = await Promise.all(embedPromises);
          const validEmbeddings = embeddings.filter((e): e is number[] => !!e && e.length > 0);
          if (validEmbeddings.length === 0) return [];

          // Step 3: Run match_document_chunks for each embedding in parallel
          const searchPromises = validEmbeddings.map(async (emb) => {
            try {
              const { data, error: err } = await supabase.rpc("match_document_chunks", {
                query_embedding: emb,
                match_count: 20,
                filter_event_id: eventId,
              });
              return err || !data?.length ? [] : (data as any[]);
            } catch {
              return [] as any[];
            }
          });
          const allResults = await Promise.all(searchPromises);

          // Step 4: Merge and dedupe вЂ” keep best similarity per document_id
          const bestByDoc = new Map<string, any>();
          for (const results of allResults) {
            for (const m of results) {
              const existing = bestByDoc.get(m.document_id);
              if (!existing || m.similarity > existing.similarity) {
                bestByDoc.set(m.document_id, m);
              }
            }
          }
          const mergedMatches = Array.from(bestByDoc.values())
            .sort((a, b) => b.similarity - a.similarity)
            .slice(0, 25);

          if (mergedMatches.length === 0) return [];

          const totalRaw = allResults.reduce((acc, r) => acc + r.length, 0);

          // GraphRAG expansion (optional, with tight timeout)
          const useGraphRAG = queryAnalysis?.retrieval_strategy?.includes("graph") ?? false;
          let finalChunks: Array<{ id: string; text: string; score?: number; metadata?: Record<string, unknown> }> = mergedMatches.map((m: any) => ({
            id: m.document_id,
            text: (m.parent_text || m.chunk_text || "").slice(0, 1500),
            score: m.similarity,
            metadata: { chunk_text: m.chunk_text, parent_text: m.parent_text },
          }));

          if (useGraphRAG && finalChunks.length > 0 && !isRetrievalBudgetExhausted()) {
            try {
              const graphragResult = await Promise.race([
                graphragRetrieve({
                  query: finalSearchQuery,
                  initial_chunks: finalChunks,
                  min_relevant_chunks: queryAnalysis?.complexity && queryAnalysis.complexity > 0.6 ? 3 : 2,
                }),
                new Promise<never>((_, reject) => setTimeout(() => reject(new Error("GraphRAG timeout")), 4000)),
              ]);
              finalChunks = graphragResult.relevant_chunks;
            } catch {
              console.warn("[PARALLEL] GraphRAG skipped (timeout or error)");
            }
          }

          const chunkMap = new Map(finalChunks.map((c) => [c.id, c]));
          let filteredMatches = mergedMatches
            .filter((m: any) => chunkMap.has(m.document_id))
            .map((m: any) => ({
              document_id: m.document_id,
              similarity: chunkMap.get(m.document_id)?.score ?? m.similarity,
              chunk_text: m.chunk_text,
              parent_text: m.parent_text,
            }));

          if (!useGraphRAG) {
            const SIMILARITY_THRESHOLD = hasName ? 0.15 : 0.35;
            filteredMatches = filteredMatches.filter((m) => m.similarity >= SIMILARITY_THRESHOLD);
          }
              
          return filteredMatches;
        } catch (err) {
          console.warn("[PARALLEL] Semantic search error:", err instanceof Error ? err.message : String(err));
          return [];
        }
      })();

      // в”Ђв”Ђ Keyword search promise (runs in PARALLEL with semantic) в”Ђв”Ђ
      const keywordSearchPromise = (async (): Promise<typeof keywordMatches> => {
        try {
          const keywordQueryText = finalSearchQuery.replace(/[^\w\s-]/g, " ").trim();
          if (keywordQueryText.length <= 1) return [];
          const { data: keywordRows, error: keywordError } = await supabase.rpc("match_documents_keyword", {
            query_text: keywordQueryText,
            match_count: 20,
            filter_event_id: eventId,
          });
          if (keywordError || !keywordRows?.length) return [];
          return keywordRows as typeof keywordMatches;
        } catch {
          return [];
        }
      })();

      // в”Ђв”Ђ Direct title search promise (for name queries, also in parallel) в”Ђв”Ђ
      const directTitleSearchPromise = (async (): Promise<typeof keywordMatches> => {
        if (!hasName) return [];
        try {
          const searchTerms = new Set<string>();
          detectedNames.forEach(name => {
            searchTerms.add(name.toLowerCase());
            name.split(/\s+/).forEach(part => { if (part.length > 3) searchTerms.add(part.toLowerCase()); });
          });
          finalSearchQuery.split(/\s+/).forEach(word => {
            if (word.length > 4 && /^[A-Za-z]+$/.test(word)) searchTerms.add(word.toLowerCase());
          });
          const { data: titleMatches, error: titleError } = await supabase
            .from("documents")
            .select("id,title,file_name,raw_content,extracted_json,created_at,storage_path,created_by")
            .eq("event_id", eventId)
            .limit(30);
          if (titleError || !titleMatches?.length) return [];
          const directMatches = titleMatches.filter((doc: any) => {
            const fullText = `${doc.title || ""} ${doc.file_name || ""} ${(doc.raw_content || "").substring(0, 5000)}`.toLowerCase();
            return Array.from(searchTerms).some(term => fullText.includes(term));
          });
          return directMatches.map((doc: any) => ({
            document_id: doc.id, rank: 0.5, snippet: (doc.raw_content || "").substring(0, 200)
          }));
        } catch { return []; }
      })();

      // в”Ђв”Ђ AWAIT ALL IN PARALLEL with global budget в”Ђв”Ђ
      const allSearches = Promise.all([semanticSearchPromise, keywordSearchPromise, directTitleSearchPromise]);
      const budgetTimeout = new Promise<[typeof semanticMatches, typeof keywordMatches, typeof keywordMatches]>((resolve) =>
        setTimeout(() => resolve([[], [], []]), RETRIEVAL_BUDGET_MS)
      );
      const [semResults, kwResults, directResults] = await Promise.race([allSearches, budgetTimeout]);
      if (timedOut) return;

      // Merge results
      semanticMatches = semResults;
      semanticFailed = semResults.length === 0 && canSemantic;
      keywordMatches = kwResults;
      // Merge direct title results into keyword matches (dedup)
      const kwDocIds = new Set(keywordMatches.map(m => m.document_id));
      for (const d of directResults) {
        if (!kwDocIds.has(d.document_id)) keywordMatches.push(d);
      }
      // Populate snippet map from semantic results
      semanticMatches.forEach((m) => {
        if (m.parent_text?.trim()) snippetByDocId.set(m.document_id, m.parent_text);
        else if (m.chunk_text?.trim()) snippetByDocId.set(m.document_id, m.chunk_text);
      });
      keywordMatches.forEach((m) => {
        if (!snippetByDocId.has(m.document_id) && m.snippet?.trim()) snippetByDocId.set(m.document_id, m.snippet!);
      });


      if (!docs.length && !error) {
        // в”Ђв”Ђ RRF merge в”Ђв”Ђ
        const RRF_K = 60;
        const scoreMap = new Map<string, number>();
        semanticMatches.forEach((m, idx) => {
          scoreMap.set(m.document_id, (scoreMap.get(m.document_id) || 0) + 1 / (RRF_K + idx + 1));
        });
        keywordMatches.forEach((m, idx) => {
          scoreMap.set(m.document_id, (scoreMap.get(m.document_id) || 0) + 1 / (RRF_K + idx + 1));
        });

        const rankedIds = Array.from(scoreMap.entries())
          .sort((a, b) => b[1] - a[1])
          .map(([id]) => id)
          .slice(0, 15);
        

        // CRITICAL FALLBACK: If all searches fail but it's a name query, try direct document query
        if (rankedIds.length === 0 && hasName) {
          try {
            // Query ALL documents for this event and filter manually
            let fallbackQuery = supabase
              .from("documents")
              .select("id,title,file_name,raw_content,extracted_json,created_at,storage_path,created_by")
              .eq("event_id", eventId)
              .limit(100);
            
            if (myDocsSelected && !teamDocsSelected && currentUserId) {
              fallbackQuery = fallbackQuery.eq("created_by", currentUserId);
            } else if (!myDocsSelected && teamDocsSelected && currentUserId) {
              fallbackQuery = fallbackQuery.neq("created_by", currentUserId);
            }
            
            const { data: allDocs, error: fallbackError } = await fallbackQuery;
            
            if (!fallbackError && allDocs?.length) {
              
              // Filter documents that contain any name or query token
              const searchTermsLower = new Set<string>();
              detectedNames.forEach(name => {
                searchTermsLower.add(name.toLowerCase());
                name.split(/\s+/).forEach(part => {
                  if (part.length > 3) searchTermsLower.add(part.toLowerCase());
                });
              });
              contentTokens.forEach(token => {
                if (token.length > 3) searchTermsLower.add(token);
              });
              
              const matchedDocs = allDocs.filter((doc: any) => {
                const fullText = `${doc.title || ""} ${doc.file_name || ""} ${doc.raw_content || ""}`.toLowerCase();
                return Array.from(searchTermsLower).some(term => fullText.includes(term));
              });
              
              
              if (matchedDocs.length > 0) {
                docs = matchedDocs.slice(0, 10);
              }
            }
          } catch (fallbackErr) {
          }
        }

        if (rankedIds.length > 0) {
          let docQuery = supabase
            .from("documents")
            .select("id,title,file_name,raw_content,extracted_json,created_at,storage_path,created_by,folder_id")
            .in("id", rankedIds);
          if (myDocsSelected && !teamDocsSelected && currentUserId) {
            docQuery = docQuery.eq("created_by", currentUserId);
          } else if (!myDocsSelected && teamDocsSelected && currentUserId) {
            docQuery = docQuery.neq("created_by", currentUserId);
          }
          const { data: docRows, error: docError } = await docQuery;
          if (timedOut) return;
          if (docError) {
            error = docError as { message?: string };
          } else if (docRows?.length) {
            const docMap = new Map(docRows.map((d: any) => [d.id, d]));
            let fetchedDocs = rankedIds.map((id) => docMap.get(id)).filter(Boolean);
            
            // PHASE 2: Document title boosting - boost documents where query terms appear in title
            const titleBoost = (docTitle: string | null, docFileName: string | null): number => {
              if (!docTitle && !docFileName) return 0;
              const titleText = `${docTitle || ""} ${docFileName || ""}`.toLowerCase();
              const queryLower = finalSearchQuery.toLowerCase();
              const queryWords = queryLower.split(/\s+/).filter(w => w.length > 2);
              let boost = 0;
              for (const word of queryWords) {
                if (titleText.includes(word)) {
                  boost += 0.5; // Boost for each matching word in title
                }
              }
              // Extra boost if name appears in title
              if (hasName && detectedNames.length > 0) {
                for (const name of detectedNames) {
                  if (titleText.includes(name.toLowerCase())) {
                    boost += 1.0; // Strong boost for name in title
                  }
                }
              }
              return boost;
            };
            
            // PHASE 1: Fuzzy name matching - check if document names match query names with typos
            const fuzzyMatchName = (queryName: string, docName: string, maxDistance: number = 2): boolean => {
              const queryLower = queryName.toLowerCase().trim();
              const docLower = docName.toLowerCase().trim();
              
              // Exact match
              if (queryLower === docLower) return true;
              
              // Contains match
              if (queryLower.includes(docLower) || docLower.includes(queryLower)) return true;
              
              // Levenshtein distance (simple version)
              const distance = (s1: string, s2: string): number => {
                if (s1.length === 0) return s2.length;
                if (s2.length === 0) return s1.length;
                const matrix: number[][] = [];
                for (let i = 0; i <= s2.length; i++) {
                  matrix[i] = [i];
                }
                for (let j = 0; j <= s1.length; j++) {
                  matrix[0][j] = j;
                }
                for (let i = 1; i <= s2.length; i++) {
                  for (let j = 1; j <= s1.length; j++) {
                    if (s2.charAt(i - 1) === s1.charAt(j - 1)) {
                      matrix[i][j] = matrix[i - 1][j - 1];
                    } else {
                      matrix[i][j] = Math.min(
                        matrix[i - 1][j - 1] + 1,
                        matrix[i][j - 1] + 1,
                        matrix[i - 1][j] + 1
                      );
                    }
                  }
                }
                return matrix[s2.length][s1.length];
              };
              
              const dist = distance(queryLower, docLower);
              const maxAllowed = Math.min(maxDistance, Math.floor(queryLower.length / 3));
              return dist <= maxAllowed;
            };
            
            // Apply fuzzy matching and title boosting
            if (hasName && detectedNames.length > 0) {
              fetchedDocs = fetchedDocs.map(doc => {
                let boost = titleBoost(doc.title, doc.file_name);
                // Check if any detected name fuzzy matches document title/content
                const docText = `${doc.title || ""} ${doc.file_name || ""} ${doc.raw_content || ""}`.toLowerCase();
                for (const name of detectedNames) {
                  const nameParts = name.split(/\s+/);
                  for (const part of nameParts) {
                    if (part.length > 3) { // Only check significant name parts
                      // Check title
                      if (doc.title && fuzzyMatchName(part, doc.title)) {
                        boost += 1.5; // Strong boost for fuzzy name match in title
                      }
                      // Check filename
                      if (doc.file_name && fuzzyMatchName(part, doc.file_name)) {
                        boost += 1.5;
                      }
                      // Check content (weaker boost)
                      if (doc.raw_content && doc.raw_content.toLowerCase().includes(part.toLowerCase())) {
                        boost += 0.3;
                      }
                    }
                  }
                }
                return { ...doc, _boost: boost };
              });
              
              // Re-sort by boost + original score
              fetchedDocs.sort((a, b) => {
                const boostA = (a as any)._boost || 0;
                const boostB = (b as any)._boost || 0;
                return boostB - boostA;
              });
              
              // Remove boost property
              docs = fetchedDocs.map(({ _boost, ...doc }) => doc);
            } else {
              // Just apply title boosting without fuzzy matching
              fetchedDocs = fetchedDocs.map(doc => {
                const boost = titleBoost(doc.title, doc.file_name);
                return { ...doc, _boost: boost };
              });
              fetchedDocs.sort((a, b) => {
                const boostA = (a as any)._boost || 0;
                const boostB = (b as any)._boost || 0;
                return boostB - boostA;
              });
              docs = fetchedDocs.map(({ _boost, ...doc }) => doc);
            }
          }
        }
      }

      if (!docs.length && !error) {
        // More aggressive search: try multiple search strategies
        // First, try full-text search with the question
        let responseQuery = supabase
          .from("documents")
          .select("id,title,file_name,raw_content,extracted_json,created_at,storage_path,created_by,folder_id")
          .eq("event_id", eventId)
          .textSearch("raw_content", question.replace(/[^\w\s-]/g, ' ').trim(), { type: "websearch", config: "english" })
          .order("created_at", { ascending: false })
          .limit(6);
        
        // Apply scope filters
        if (myDocsSelected && !teamDocsSelected && currentUserId) {
          responseQuery = responseQuery.eq("created_by", currentUserId);
        } else if (!myDocsSelected && teamDocsSelected && currentUserId) {
          responseQuery = responseQuery.neq("created_by", currentUserId);
        }
        
        let response;
        try {
          response = await responseQuery;
        } catch (queryErr) {
          console.warn("Document query failed:", queryErr);
          response = { data: [], error: queryErr };
        }
        if (timedOut) return;
        docs = (response.data || []) as typeof docs;
        
        // If still no results, try keyword search with individual terms
        if (!docs.length && !response.error) {
          const keywords = question
            .toLowerCase()
            .split(/\W+/)
            .filter((w) => w.length > 3)
            .slice(0, 3); // Use top 3 keywords
          
          if (keywords.length > 0) {
            try {
              let keywordQuery = supabase
                .from("documents")
                .select("id,title,file_name,raw_content,extracted_json,created_at,storage_path,created_by,folder_id")
                .eq("event_id", eventId);
              
              // Build OR conditions safely
              const orConditions = keywords.map((k) => `raw_content.ilike.%${k}%`).join(",");
              if (orConditions) {
                keywordQuery = keywordQuery.or(orConditions);
              }
              
              keywordQuery = keywordQuery.order("created_at", { ascending: false }).limit(6);
              
              if (myDocsSelected && !teamDocsSelected && currentUserId) {
                keywordQuery = keywordQuery.eq("created_by", currentUserId);
              } else if (!myDocsSelected && teamDocsSelected && currentUserId) {
                keywordQuery = keywordQuery.neq("created_by", currentUserId);
              }
              
              const keywordResponse = await keywordQuery;
              if (timedOut) return;
              if (keywordResponse.data?.length) {
                docs = (keywordResponse.data || []) as typeof docs;
              }
            } catch (keywordErr) {
              console.warn("Keyword search failed:", keywordErr);
              // Continue without keyword results
            }
          }
        }
        
        if (response.error) {
          error = response.error as { message?: string };
        }

        // Final fallback: client-side scan of recent docs using raw + extracted JSON
        if (!docs.length && !error && contentTokens.length > 0) {
          let recentQuery = supabase
            .from("documents")
            .select("id,title,file_name,raw_content,extracted_json,created_at,storage_path,created_by,folder_id")
            .eq("event_id", eventId)
            .order("created_at", { ascending: false })
            .limit(50);
          if (myDocsSelected && !teamDocsSelected && currentUserId) {
            recentQuery = recentQuery.eq("created_by", currentUserId);
          } else if (!myDocsSelected && teamDocsSelected && currentUserId) {
            recentQuery = recentQuery.neq("created_by", currentUserId);
          }
          const recentResponse = await recentQuery;
          if (timedOut) return;
          if (!recentResponse.error && recentResponse.data?.length) {
            const filtered = (recentResponse.data as typeof docs).filter((doc) => {
              if (!contentTokens.length) return false;
              const haystack = [
                doc.raw_content || "",
                doc.extracted_json ? JSON.stringify(doc.extracted_json) : "",
                doc.title || "",
                doc.file_name || "",
              ]
                .join(" ")
                .toLowerCase();
              // Require at least 60% of tokens to match (or at least 1 if short)
              const minMatches = contentTokens.length <= 2
                ? 1
                : Math.max(2, Math.ceil(contentTokens.length * 0.6));
              const matches = contentTokens.filter((t) => haystack.includes(t)).length;
              const hasStrongMatch =
                contentTokens.length <= 6 &&
                contentTokens.some((t) => t.length >= 4 && haystack.includes(t));
              return matches >= minMatches || hasStrongMatch;
            });
            if (filtered.length) {
              docs = filtered.slice(0, 6);
            }
          }
        }
      }

      if (selectedFolderIds.length > 0 && docs.length > 0) {
        docs = await filterDocsByFolderScope(docs);
      }

      const decisionIntent =
        /\b(decision|decisions|outcome|log|logged|approve|approved|reject|rejected)\b/i.test(question);
      const decisionStopwords = new Set([
        "the",
        "and",
        "for",
        "with",
        "about",
        "tell",
        "what",
        "which",
        "that",
        "this",
        "from",
        "into",
        "your",
        "you",
        "have",
        "does",
        "did",
        "are",
        "can",
        "will",
        "should",
        "could",
        "please",
        "company",
        "companies",
        "decision",
        "decisions",
        "meeting",
        "notes",
        "table",
        "document",
      ]);
      const decisionTokens = tokens.filter((t) => !decisionStopwords.has(t));
      const minDecisionMatches = Math.max(
        1,
        decisionTokens.length >= 3 ? Math.ceil(decisionTokens.length * 0.5) : 1
      );
      
      const decisionMatches = decisionIntent
        ? decisions
            .filter((d) => {
              const haystack = [
                d.startupName,
                d.actionType,
                d.outcome ?? "",
                d.notes ?? "",
                d.actor ?? "",
              ]
                .join(" ")
                .toLowerCase();
              if (!decisionTokens.length) return false;
              const matches = decisionTokens.filter((t) => haystack.includes(t)).length;
              return matches >= minDecisionMatches;
            })
            .slice(0, 5)
        : [];

      if (error) {
        if (searchTimeoutId !== null) {
          window.clearTimeout(searchTimeoutId);
        }
        createAssistantMessage(
          `Search failed: ${error.message || "Could not query documents."}`,
          threadId
        );
        setChatIsLoading(false);
        return;
      }

      // STRICT FILTERING: Only keep documents that are actually relevant
      // Allow strong matches for short, entity-like queries
      // CRITICAL: For name queries, be VERY lenient - names are the signal
      const minTokenMatches = hasName 
        ? 1 // Name queries: just 1 token match is enough
        : contentTokens.length <= 2
          ? 1
          : Math.max(2, Math.ceil(contentTokens.length * 0.6));
      
      
      const filteredDocs = (docs || []).filter((doc) => {
        if (!contentTokens.length && !hasName) return false; // No tokens = no match (unless name query)
        
        // For name queries with no tokens but detected names, check directly
        if (hasName && contentTokens.length === 0 && detectedNames.length > 0) {
          const haystack = `${doc.title || ""} ${doc.file_name || ""} ${doc.raw_content || ""}`.toLowerCase();
          return detectedNames.some(name => haystack.includes(name.toLowerCase()));
        }
        
        const haystack = [
          doc.raw_content || "",
          doc.extracted_json ? JSON.stringify(doc.extracted_json) : "",
          doc.title || "",
          doc.file_name || "",
        ]
          .join(" ")
          .toLowerCase();
        const matches = contentTokens.filter((t) => haystack.includes(t)).length;
        const hasStrongMatch =
          contentTokens.length <= 6 &&
          contentTokens.some((t) => t.length >= 4 && haystack.includes(t));
        
        // For name queries, also check if detected names appear in doc
        const hasNameMatch = hasName && detectedNames.some(name => {
          const nameLower = name.toLowerCase();
          const nameParts = nameLower.split(/\s+/);
          // Match if full name OR any part of name (first/last) appears
          return haystack.includes(nameLower) || 
            nameParts.some(part => part.length > 3 && haystack.includes(part));
        });
        
        return matches >= minTokenMatches || hasStrongMatch || hasNameMatch;
      });
      

      let rankedDocs = filteredDocs;
      if (rankedDocs.length > 1) {
        try {
          const rerankPayload = rankedDocs.map((doc) => {
            const snippet = snippetByDocId.get(doc.id);
            const baseText = snippet || buildNormalizedDocText(doc) || "";
            return {
              id: doc.id,
              text: baseText.slice(0, 1500),
            };
          });
          const rerankResults = await rerankDocuments({
            query: question,
            documents: rerankPayload,
            topN: Math.min(10, rerankPayload.length),
          });
          if (rerankResults.length > 0) {
            // Filter by reranking score threshold (0.1 = minimum relevance)
            // Cohere scores are typically 0-1, but can be negative for very poor matches
            const RERANK_SCORE_THRESHOLD = 0.1;
            const filteredRerankResults = rerankResults.filter(
              (r) => r.score >= RERANK_SCORE_THRESHOLD
            );
            if (filteredRerankResults.length > 0) {
              const docMap = new Map(rankedDocs.map((d) => [d.id, d]));
              const reranked = filteredRerankResults.map((r) => docMap.get(r.id)).filter(Boolean);
              rankedDocs = reranked as typeof rankedDocs;
            } else {
              // If all reranked results are below threshold, keep original order but limit to top 3
              rankedDocs = rankedDocs.slice(0, 3);
            }
          }
        } catch (rerankErr) {
          // If rerank fails, keep existing order
        }
      }


      // в”Ђв”Ђ ColBERT Late Interaction Reranking (multi-agent only) в”Ђв”Ђ
      // Uses token-level MaxSim scoring for fine-grained relevance matching
      if (multiAgentEnabled && rankedDocs.length > 1) {
        try {
          const colbertDocs = rankedDocs.map((doc) => {
            const snippet = snippetByDocId.get(doc.id);
            return (snippet || buildNormalizedDocText(doc) || "").slice(0, 1500);
          });
          const colbertResult = await colbertRerank({
            query: question,
            documents: colbertDocs,
            docIds: rankedDocs.map((d) => d.id),
            topK: Math.min(10, rankedDocs.length),
          });
          if (colbertResult.results.length > 0) {
            const docMap = new Map(rankedDocs.map((d) => [d.id, d]));
            const colbertReranked = colbertResult.results
              .map((r) => docMap.get(r.doc_id))
              .filter(Boolean) as typeof rankedDocs;
            if (colbertReranked.length > 0) {
              rankedDocs = colbertReranked;
            }
          }
        } catch (colbertErr) {
          console.warn("[MULTI-AGENT] ColBERT reranking failed (non-fatal):", colbertErr);
        }
      }

      // Check if this is a meta-question (about capabilities/system)
      const isMetaQuestion = (() => {
        const q = normalizedQuestion;
        const metaPatterns = [
          "what can you do",
          "what could you do",
          "what are you",
          "what do you do",
          "how do you work",
          "what is your purpose",
          "what are your capabilities",
          "what can you help",
          "how can you help",
          "what features",
          "what functionality",
          "what is this platform",
          "who are you",
          "introduce yourself",
          "what is this",
          "what is this system",
          "what is this platform",
        ];
        return metaPatterns.some(pattern => q.includes(pattern));
      })();
      
      // в”Ђв”Ђ CONNECTION-INTENT DETECTION в”Ђв”Ђ
      // When user asks about connections, partnerships, or "who to connect with",
      // we need to send ALL portfolio context so Claude knows what companies exist.
      const isConnectionIntent = (() => {
        const q = normalizedQuestion;
        return /\b(connect|connected|connection|connections|partner|partnership|partnerships|introduce|introduction|link|linked|linking|network|networking|relationship|relationships|relate|who.*help|help.*them|could.*help|could.*connect|should.*connect|could.*partner|suggest.*compan|recommend.*compan|match.*with|pair.*with|synerg|collaborate|collaboration|another.*compan|other.*compan|different.*compan|what.*compan.*help|which.*compan)\b/i.test(q);
      })();

      // For meta-questions, answer even without sources
      if (isMetaQuestion && (!rankedDocs || rankedDocs.length === 0)) {
        if (searchTimeoutId !== null) {
          window.clearTimeout(searchTimeoutId);
        }
        setChatIsLoading(false);
        setIsClaudeLoading(true);
        const streamer = createStreamingAssistantMessage(threadId);
        let streamCompleted = false;
        const streamTimeout = setTimeout(() => {
          if (!streamCompleted) {
            console.error("Meta-question stream timeout");
            streamer.setError("Request timed out. Please try again.");
            setIsClaudeLoading(false);
          }
        }, 120000);
        try {
          // Answer meta-questions with general knowledge (streaming)
          // Get thread messages for context (from state or DB)
          const threadMessages = await getThreadMessages(threadId, 10);
          
          await askClaudeAnswerStream(
            {
              question,
              sources: [],
              decisions: [],
              connections: connectionsForChat,
              previousMessages: threadMessages,
            },
            (chunk) => {
              streamer.appendChunk(chunk);
            },
            (error) => {
              streamCompleted = true;
              clearTimeout(streamTimeout);
              streamer.setError(error.message || "Failed to answer. Please try again.");
              setIsClaudeLoading(false);
            },
            chatAbortRef.current?.signal
          );
          streamCompleted = true;
          clearTimeout(streamTimeout);
          streamer.finalize();
        } catch (err) {
          streamCompleted = true;
          clearTimeout(streamTimeout);
          streamer.setError(err instanceof Error ? err.message : "Failed to answer. Please try again.");
        } finally {
          setIsClaudeLoading(false);
        }
        return;
      }

      // CRITICAL: If this is a pronoun-based follow-up, reuse previous evidence directly.
      // This avoids searching for "him" and failing to find new docs.
      // BUT: If user asks for "another/different/other", do NOT reuse вЂ” they want NEW results.
      if (
        isFollowUpQuery &&
        followUpHasPronoun &&
        !wantsAlternative &&
        previousEvidence &&
        previousEvidence.docs.length > 0 &&
        previousEvidenceThreadId === threadId
      ) {
        const maxDocs = isComprehensiveQuestion ? 5 : 3;
        const answerDocs = previousEvidence.docs.slice(0, maxDocs);
        setLastEvidence({ question: searchQuestion, docs: answerDocs, decisions: decisionMatches });
        setLastEvidenceThreadId(threadId);
        setChatIsLoading(false);
        if (searchTimeoutId !== null) {
          window.clearTimeout(searchTimeoutId);
        }
        setIsClaudeLoading(true);
        const streamer = createStreamingAssistantMessage(threadId);
        let streamCompleted = false;
        const streamTimeout = setTimeout(() => {
          if (!streamCompleted) {
            console.error("Follow-up stream timeout");
            streamer.setError("Request timed out. Please try again.");
            setIsClaudeLoading(false);
          }
        }, 120000);
        try {
          const claudeTokens = searchQuestion
            .toLowerCase()
            .split(/\W+/)
            .map((t) => t.trim())
            .filter((t) => t.length > 3);
          const companyCardSourcesPronoun = buildCompanyCardSources(searchQuestion, companyCards, []);
          const sources = [
            ...companyCardSourcesPronoun,
            ...answerDocs.map((doc) => ({
              title: doc.title,
              file_name: doc.file_name,
              snippet: buildClaudeContext(doc, claudeTokens, isComprehensiveQuestion, snippetByDocId.get(doc.id)),
            })),
          ];
          const decisionsForClaude = decisionIntent
            ? decisionMatches.map((d) => ({
                startup_name: d.startupName,
                action_type: d.actionType,
                outcome: d.outcome ?? null,
                notes: d.notes ?? null,
              }))
            : [];
          const threadMessages = await getThreadMessages(threadId, 10);
          await askClaudeAnswerStream(
            {
              question: searchQuestion,
              sources,
              decisions: decisionsForClaude,
              connections: connectionsForChat,
              previousMessages: threadMessages,
            },
            (chunk) => {
              if (!streamCompleted) {
                streamer.appendChunk(chunk);
              }
            },
            (error) => {
              if (!streamCompleted) {
                streamCompleted = true;
                clearTimeout(streamTimeout);
                streamer.setError(error.message || "Claude answer failed. Please try again.");
                setIsClaudeLoading(false);
              }
            },
            chatAbortRef.current?.signal
          );
          if (!streamCompleted) {
            streamCompleted = true;
            clearTimeout(streamTimeout);
            streamer.finalize();
          }
        } catch (err) {
          streamCompleted = true;
          clearTimeout(streamTimeout);
          streamer.setError(err instanceof Error ? err.message : "Claude answer failed. Please try again.");
        } finally {
          setIsClaudeLoading(false);
        }
        return;
      }

      const lowSignalFollowUp =
        isFollowUpQuery && contentTokens.length <= 1;


      if (!rankedDocs || rankedDocs.length === 0 || lowSignalFollowUp) {
        // CRITICAL: If search fails but we have context (pronouns OR follow-up cues), use previous evidence
        // Be MORE lenient here - if user is asking for "more info/complete/profound", use previous docs
        const hasPronounInQuestion = /\b(him|her|it|they|them|his|hers|their|this|that)\b/i.test(question);
        const hasFollowupCueInOriginal = /\b(more about|more info|more complete|more comprehensive|more profound|give more|give more info|tell me more|elaborate|explain|full|complete|comprehensive|detailed)\b/i.test(question.toLowerCase());
        const shouldUsePreviousEvidence = (
          (isFollowUpQuery || hasPronounInQuestion || hasFollowupCueInOriginal) &&
          !wantsAlternative && // Don't reuse when user asks for "another/different"
          previousEvidence &&
          previousEvidence.docs.length > 0
          // Removed: && previousEvidenceThreadId === threadId (too strict!)
        );
        
        
        if (shouldUsePreviousEvidence) {
          const answerDocs = previousEvidence.docs.slice(0, 3);
          setLastEvidence({ question, docs: answerDocs, decisions: decisionMatches });
          setLastEvidenceThreadId(threadId);
          setChatIsLoading(false);
          // Clear search timeout - Claude has its own 70s timeout
          if (searchTimeoutId !== null) {
            window.clearTimeout(searchTimeoutId);
          }
          // Use Claude with the prior sources
          setIsClaudeLoading(true);
          const streamer = createStreamingAssistantMessage(threadId);
          let streamCompleted = false;
          const streamTimeout = setTimeout(() => {
            if (!streamCompleted) {
              console.error("Follow-up stream timeout");
              streamer.setError("Request timed out. Please try again.");
              setIsClaudeLoading(false);
            }
          }, 120000);
          try {
            const claudeTokens = question
              .toLowerCase()
              .split(/\W+/)
              .map((t) => t.trim())
              .filter((t) => t.length > 3);
            const sources = answerDocs.map((doc) => ({
              title: doc.title,
              file_name: doc.file_name,
              snippet: buildClaudeContext(doc, claudeTokens, isComprehensiveQuestion, snippetByDocId.get(doc.id)),
            }));
            const decisionsForClaude = decisionIntent
              ? decisionMatches.map((d) => ({
                  startup_name: d.startupName,
                  action_type: d.actionType,
                  outcome: d.outcome ?? null,
                  notes: d.notes ?? null,
                }))
              : [];
            
            // Get previous messages from this thread for context (from state or DB)
            const threadMessages = await getThreadMessages(threadId, 10);
            
            await askClaudeAnswerStream(
              {
                question,
                sources,
                decisions: decisionsForClaude,
                connections: connectionsForChat,
                previousMessages: threadMessages,
              },
              (chunk) => {
                if (!streamCompleted) {
                  streamer.appendChunk(chunk);
                }
              },
              (error) => {
                if (!streamCompleted) {
                  streamCompleted = true;
                  clearTimeout(streamTimeout);
                  streamer.setError(error.message || "Claude answer failed. Please try again.");
                  setIsClaudeLoading(false);
                }
              },
              chatAbortRef.current?.signal
            );
            // Only finalize if stream completed successfully
            if (!streamCompleted) {
              streamCompleted = true;
              clearTimeout(streamTimeout);
              streamer.finalize();
            }
          } catch (err) {
            streamCompleted = true;
            clearTimeout(streamTimeout);
            streamer.setError(err instanceof Error ? err.message : "Claude answer failed. Please try again.");
          } finally {
            setIsClaudeLoading(false);
          }
          return;
        }
        // FALLBACK: If we have chat history and the question has pronouns, try to answer from context
        // This is a last resort - Claude can reference previous conversation even without new sources
        const hasPronounInOriginal = /\b(him|her|it|they|them|his|hers|their|this|that)\b/i.test(question);
        const threadMessagesForFallback = await getThreadMessages(threadId, 10);
        
        if (hasPronounInOriginal && threadMessagesForFallback.length > 0) {
          if (searchTimeoutId !== null) {
            window.clearTimeout(searchTimeoutId);
          }
          setChatIsLoading(false);
          setIsClaudeLoading(true);
          const streamer = createStreamingAssistantMessage(threadId);
          let streamCompleted = false;
          const streamTimeout = setTimeout(() => {
            if (!streamCompleted) {
              console.error("Fallback stream timeout");
              streamer.setError("Request timed out. Please try again.");
              setIsClaudeLoading(false);
            }
          }, 120000);
          try {
            await askClaudeAnswerStream(
              {
                question,
                sources: [], // No sources, but chat history should help
                decisions: [],
                connections: connectionsForChat,
                previousMessages: threadMessagesForFallback,
              },
              (chunk) => {
                if (!streamCompleted) {
                  streamer.appendChunk(chunk);
                }
              },
              (error) => {
                if (!streamCompleted) {
                  streamCompleted = true;
                  clearTimeout(streamTimeout);
                  streamer.setError(error.message || "Failed to answer. Please try again.");
                  setIsClaudeLoading(false);
                }
              },
              chatAbortRef.current?.signal
            );
            if (!streamCompleted) {
              streamCompleted = true;
              clearTimeout(streamTimeout);
              streamer.finalize();
            }
          } catch (err) {
            streamCompleted = true;
            clearTimeout(streamTimeout);
            streamer.setError(err instanceof Error ? err.message : "Failed to answer. Please try again.");
          } finally {
            setIsClaudeLoading(false);
          }
          return;
        }
        
        // Show searchQuestion (rewritten) if different from original, for better debugging
        const queryToShow = searchQuestion !== question ? `${searchQuestion} (original: ${question})` : question;
        // If we have decision matches, show them
        if (decisionIntent && decisionMatches.length) {
          const fallback = `${formatDecisionMatches(decisionMatches)}\n\nIf you want deeper answers, upload or link supporting documents in the Sources tab.`;
          if (searchTimeoutId !== null) {
            window.clearTimeout(searchTimeoutId);
          }
          createAssistantMessage(fallback, threadId);
          setLastEvidence(null);
          setChatIsLoading(false);
          return;
        }

        // NO DOCUMENTS FOUND вЂ” but instead of showing an error, forward to Claude
        // so it can still answer general questions, greetings, or use conversation context.
        // This fixes the problem where "hello" or document-specific questions get blocked.
        
        if (searchTimeoutId !== null) {
          window.clearTimeout(searchTimeoutId);
        }
        
        // в”Ђв”Ђ CONNECTION-INTENT: Build portfolio context from ALL documents в”Ђв”Ђ
        // When user asks about connections/partnerships, send all doc titles
        // so Claude knows what companies are in the portfolio even though
        // the search didn't match the specific company name.
        let portfolioSources: Array<{ title: string | null; file_name: string | null; snippet: string | null }> = [];
        if (isConnectionIntent && documents.length > 0) {
          portfolioSources = documents.slice(0, 15).map((doc) => ({
            title: doc.title || "Untitled",
            file_name: null,
            snippet: `[Portfolio company/document: ${doc.title || "Untitled"}]`,
          }));
        }
        
        // Set evidence (even with empty docs) and call Claude directly
        setLastEvidence({ question, docs: [], decisions: decisionMatches });
        setLastEvidenceThreadId(threadId);
        setChatIsLoading(false);
        setIsClaudeLoading(true);
        
        // Create streaming message
        const streamer = createStreamingAssistantMessage(threadId, []);
        let fullAnswer = "";
        let streamCompleted = false;
        
        const streamTimeout = setTimeout(() => {
          if (!streamCompleted) {
            console.error("Stream timeout - no response after 75 seconds");
            streamer.setError("Request timed out. The response is taking too long. Please try again with a simpler question.");
            setIsClaudeLoading(false);
          }
        }, 120000);
        
        try {
          // Get previous messages from this thread for context
          const threadMessages = await getThreadMessages(threadId, 10);
          
          // CRITICAL: When user asks for "another/different", add exclusion context
          let questionForClaudeFallback = question;
          if (wantsAlternative && threadMessages.length > 0) {
            const alreadyMentioned = extractMentionedCompaniesFromAssistant(threadMessages);
            if (alreadyMentioned.length > 0) {
              const exclusionNote = `\n\n[IMPORTANT: The user is asking for a DIFFERENT option. Do NOT mention or recommend these companies/entities that were already discussed: ${alreadyMentioned.join(", ")}. Suggest only NEW companies that have NOT been mentioned yet.]`;
              questionForClaudeFallback = question + exclusionNote;
            }
          }
          
          // Call Claude with portfolio context (web search is handled natively by Anthropic when enabled)
          const companyCardSourcesFallback = buildCompanyCardSources(question, companyCards, detectedNames || []);
          const noDocSources = [
            ...companyCardSourcesFallback,
            ...(portfolioSources.length > 0 ? portfolioSources : []),
          ] as Array<{ title: string | null; file_name: string | null; snippet: string | null }>;
          
          await askClaudeAnswerStream(
            {
              question: questionForClaudeFallback,
              sources: noDocSources,
              webSearchEnabled,
              decisions: decisionIntent
                ? decisionMatches.map((d) => ({
                    startup_name: d.startupName,
                    action_type: d.actionType,
                    outcome: d.outcome ?? null,
                    notes: d.notes ?? null,
                  }))
                : [],
              connections: connectionsForChat,
              previousMessages: threadMessages,
            },
            (chunk) => {
              if (!streamCompleted) {
                fullAnswer += chunk;
                streamer.appendChunk(chunk);
              }
            },
            (error) => {
              if (!streamCompleted) {
                streamCompleted = true;
                clearTimeout(streamTimeout);
                const errorMsg = error.message || "Claude answer failed. Please try again.";
                console.error("Stream error:", errorMsg);
                streamer.setError(errorMsg);
                setIsClaudeLoading(false);
              }
            },
            chatAbortRef.current?.signal
          );
          
          if (!streamCompleted && fullAnswer.length > 0) {
            streamCompleted = true;
            clearTimeout(streamTimeout);
            streamer.finalize();
          } else if (!streamCompleted) {
            streamCompleted = true;
            clearTimeout(streamTimeout);
            setIsClaudeLoading(false);
          }
          
          const estimate = estimateClaudeCost(question);
          persistCostLog({
            ts: new Date().toISOString(),
            question: question.slice(0, 120),
            estInputTokens: estimate.estInputTokens,
            estOutputTokens: estimate.estOutputTokens,
            estCostUsd: estimate.estCostUsd,
          });
        } catch (error: any) {
          streamCompleted = true;
          clearTimeout(streamTimeout);
          const errorMsg = error?.message || "Could not generate an answer.";
          streamer.setError(errorMsg);
          setIsClaudeLoading(false);
        }
        return;
      }

      const decisionBlock = decisionIntent && decisionMatches.length
        ? `\n\nRelated decisions:\n${decisionMatches
            .map(
              (d, index) =>
                `${index + 1}. ${d.startupName} вЂ” ${d.actionType}${
                  d.outcome ? ` (${d.outcome})` : ""
                }${d.notes ? ` вЂ” ${d.notes}` : ""}`
            )
            .join("\n")}`
        : "";

      // Semantic note completely removed вЂ” never show this to users
      const semanticNote = "";

      // For comprehensive questions, use more sources (up to 5)
      const maxDocs = isComprehensiveQuestion ? 5 : 3;
      // в”Ђв”Ђ SOURCE DIVERSITY: Don't let one company dominate all results в”Ђв”Ђ
      // If multiple docs share the same title prefix (e.g. all about Chari), cap per-company to 2
      const diversifyDocs = (allDocs: typeof rankedDocs, cap: number): typeof rankedDocs => {
        const result: typeof rankedDocs = [];
        const titleSeen = new Map<string, number>();
        for (const doc of allDocs) {
          const key = (doc.title || doc.file_name || "").toLowerCase().split(/[\s\-_:]+/).slice(0, 2).join(" ").trim() || doc.id;
          const count = titleSeen.get(key) || 0;
          if (count < 2) {
            result.push(doc);
            titleSeen.set(key, count + 1);
            if (result.length >= cap) break;
          }
        }
        // If diversity filtering gave fewer than cap, fill from remaining
        if (result.length < cap) {
          const usedIds = new Set(result.map(d => d.id));
          for (const doc of allDocs) {
            if (!usedIds.has(doc.id)) {
              result.push(doc);
              if (result.length >= cap) break;
            }
          }
        }
        return result;
      };
      const answerDocs = diversifyDocs(rankedDocs, maxDocs);
      setLastEvidence({ question, docs: answerDocs, decisions: decisionMatches });
      setLastEvidenceThreadId(threadId);
      setChatIsLoading(false);
      // Clear search timeout - Claude has its own 70s timeout
      if (searchTimeoutId !== null) {
        window.clearTimeout(searchTimeoutId);
      }

      // Always use Claude for the final answer once sources exist
      setIsClaudeLoading(true);
      const streamer = createStreamingAssistantMessage(threadId, answerDocs.map((doc) => doc.id));
      // Ensure Sources strip appears under the answer: set doc list from retrieved docs (backend does not send source_docs for /ask/stream)
      streamer.setSourceDocs(
        answerDocs.map((doc) => ({ id: doc.id, title: doc.title || doc.file_name || "Document" }))
      );
      let fullAnswer = "";
      let streamCompleted = false;
      
      // Add timeout to prevent infinite hanging
      const streamTimeout = setTimeout(() => {
        if (!streamCompleted) {
          console.error("Stream timeout - no response after 120 seconds");
          streamer.setError("Request timed out. The response is taking too long. Please try again with a simpler question.");
          setIsClaudeLoading(false);
        }
      }, 120000);
      
      try {
        const docsForClaude = answerDocs;
        const claudeTokens = question
          .toLowerCase()
          .split(/\W+/)
          .map((t) => t.trim())
          .filter((t) => t.length > 3);
        const companyCardSources = buildCompanyCardSources(question, companyCards, detectedNames || []);
        let sources = [
          ...companyCardSources,
          ...docsForClaude.map((doc) => ({
            title: doc.title,
            file_name: doc.file_name,
            snippet: buildClaudeContext(doc, claudeTokens, isComprehensiveQuestion, snippetByDocId.get(doc.id)),
          })),
        ];
        
        // в”Ђв”Ђ CONNECTION-INTENT: Inject additional portfolio context в”Ђв”Ђ
        // When user asks about connections, also include titles of OTHER docs
        // not already in sources so Claude knows the full portfolio.
        if (isConnectionIntent) {
          const existingDocIds = new Set(docsForClaude.map((d) => d.id));
          const extraPortfolio = documents
            .filter((d) => !existingDocIds.has(d.id))
            .slice(0, 10)
            .map((doc) => ({
              title: doc.title || "Untitled",
              file_name: null as string | null,
              snippet: `[Portfolio company/document: ${doc.title || "Untitled"}]`,
            }));
          if (extraPortfolio.length > 0) {
            sources = [...sources, ...extraPortfolio];
          }
        }
        // Web search is now handled natively by Anthropic's web_search tool (no manual DuckDuckGo needed)
        // Web search: use orchestrator recommendation (multi-agent) OR user toggle
        const effectiveWebSearch = multiAgentEnabled
          ? (webSearchEnabled || routingPlan.use_web)
          : webSearchEnabled;

        // в”Ђв”Ђ MULTI-AGENT RAG вЂ” Await graph & KPI agents (only when enabled) в”Ђв”Ђ
        if (multiAgentEnabled) {
          try {
            const [graphResult, kpiResult] = await Promise.all([graphPromise, kpiPromise]);
            graphContext = graphResult || "";
            kpiContext = kpiResult || "";

            if (graphContext && !graphContext.startsWith("No entities found")) {
              sources.push({
                title: "[GRAPH] Knowledge Graph вЂ” Entities & Relationships",
                file_name: null as string | null,
                snippet: graphContext,
              });
            }

            if (kpiContext && kpiContext !== "No KPI data found.") {
              sources.push({
                title: "[KPIs] Structured Metrics & Numbers",
                file_name: null as string | null,
                snippet: kpiContext,
              });
            }
            // Set context labels so user can see multi-agent sources (Knowledge Graph, KPIs) under the answer
            const labels: string[] = [];
            if (graphContext && !graphContext.startsWith("No entities found")) labels.push("Knowledge Graph");
            if (kpiContext && kpiContext !== "No KPI data found.") labels.push("Structured KPIs");
            if (labels.length > 0) streamer.setContextLabels(labels);
          } catch (maErr) {
            console.warn("[MULTI-AGENT] Graph/KPI await failed (non-fatal):", maErr);
          }
        }

        const decisionsForClaude = decisionIntent
          ? decisionMatches.map((d) => ({
              startup_name: d.startupName,
              action_type: d.actionType,
              outcome: d.outcome ?? null,
              notes: d.notes ?? null,
            }))
          : [];
        
        // Get previous messages from this thread for context (from state or DB)
        const threadMessages = await getThreadMessages(threadId, 10);
        
        // CRITICAL: When user asks for "another/different", extract already-mentioned companies
        // and inject exclusion context so Claude avoids repeating them
        let questionForClaude = question;
        if (wantsAlternative && threadMessages.length > 0) {
          const alreadyMentioned = extractMentionedCompaniesFromAssistant(threadMessages);
          if (alreadyMentioned.length > 0) {
            const exclusionNote = `\n\n[IMPORTANT: The user is asking for a DIFFERENT option. Do NOT mention or recommend these companies/entities that were already discussed: ${alreadyMentioned.join(", ")}. Suggest only NEW companies that have NOT been mentioned yet.]`;
            questionForClaude = question + exclusionNote;
          }
        }
        
        // Debug logging
        
        await askClaudeAnswerStream(
          {
            question: questionForClaude,
            sources,
            decisions: decisionsForClaude,
            connections: connectionsForChat,
            previousMessages: threadMessages,
            webSearchEnabled: effectiveWebSearch,
          },
          (chunk) => {
            if (!streamCompleted) {
              fullAnswer += chunk;
              streamer.appendChunk(chunk);
            }
          },
          (error) => {
            if (!streamCompleted) {
              streamCompleted = true;
              clearTimeout(streamTimeout);
              const errorMsg = error.message || "Claude answer failed. Please try again.";
              console.error("Stream error:", errorMsg);
              streamer.setError(errorMsg);
              setIsClaudeLoading(false);
            }
          },
          chatAbortRef.current?.signal
        );
        // Only finalize if stream completed successfully (no error was called)
        if (!streamCompleted && fullAnswer.length > 0) {
          streamCompleted = true;
          clearTimeout(streamTimeout);
          // Append decision block and semantic note after streaming completes
          streamer.appendChunk(decisionBlock + semanticNote);
          streamer.finalize();
        } else if (!streamCompleted) {
          // Stream completed but no data received - ensure timeout is cleared
          streamCompleted = true;
          clearTimeout(streamTimeout);
          setIsClaudeLoading(false);
        }
        const estimate = estimateClaudeCost(question);
        persistCostLog({
          ts: new Date().toISOString(),
          question: question.slice(0, 120),
          estInputTokens: estimate.estInputTokens,
          estOutputTokens: estimate.estOutputTokens,
          estCostUsd: estimate.estCostUsd,
        });

        // в”Ђв”Ђ MULTI-AGENT RAG вЂ” Step 4: Critic (only when multi-agent is ON) в”Ђв”Ђ
        if (multiAgentEnabled && fullAnswer.length > 50) {
          criticCheck({
            question: questionForClaude,
            answer: fullAnswer,
            contextVector: sources.map((s) => `${s.title}: ${(s.snippet || "").slice(0, 500)}`).join("\n"),
            contextGraph: graphContext,
            contextKpis: kpiContext,
          }).then((criticResult) => {
            if (criticResult.issues.length > 0) {
              console.warn("[MULTI-AGENT] Critic found issues:", criticResult.issues);
            } else {
            }
          }).catch(() => { /* non-fatal */ });
        }

        // в”Ђв”Ђ MULTI-AGENT RAG вЂ” System 2 RAG: Test-Time Compute (Reflect в†’ Search в†’ Refine) в”Ђв”Ђ
        // After initial answer, reflect on gaps, search again, and produce refined answer
        if (multiAgentEnabled && fullAnswer.length > 100) {
          try {
            const vectorContextStr = sources.map((s) => `${s.title}: ${(s.snippet || "").slice(0, 500)}`).join("\n");
            let draftForReflection = fullAnswer;
            let totalIterations = 0;
            const MAX_SYSTEM2_ITERATIONS = 2;

            for (let iter = 0; iter < MAX_SYSTEM2_ITERATIONS; iter++) {
              setChatLoadingStage?.(`System 2: Reflecting (iteration ${iter + 1})...`);

              const reflection = await system2Reflect({
                question: questionForClaude,
                draftAnswer: draftForReflection,
                vectorContext: vectorContextStr,
                graphContext,
                kpiContext,
                iteration: iter,
                maxIterations: MAX_SYSTEM2_ITERATIONS,
              });


              if (!reflection.needs_more_data || reflection.confidence >= 0.85) {
                break;
              }

              totalIterations++;
              setChatLoadingStage?.(`System 2: Searching for missing data...`);

              // Execute follow-up searches based on reflection
              let additionalContext = "";
              for (const fq of reflection.follow_up_queries.slice(0, 2)) {
                try {
                  const fqEmbedding = await embedQuery(fq);
                  if (fqEmbedding && eventId) {
                    const { data: extraChunks } = await supabase.rpc("match_document_chunks", {
                      query_embedding: fqEmbedding,
                      match_threshold: 0.3,
                      match_count: 5,
                      filter_event_id: eventId,
                    });
                    if (extraChunks?.length) {
                      additionalContext += extraChunks
                        .map((ch: any) => `[Follow-up: ${fq}] ${ch.content?.slice(0, 600) || ""}`)
                        .join("\n\n");
                    }
                  }
                } catch { /* non-fatal */ }
              }

              if (!additionalContext) {
                break;
              }

              // Stream refined answer
              setChatLoadingStage?.("System 2: Refining answer...");
              try {
                const refineResp = await system2RefineStream({
                  question: questionForClaude,
                  draftAnswer: draftForReflection,
                  originalContext: vectorContextStr,
                  additionalContext,
                  reflectionReasoning: reflection.reasoning,
                  previousMessages: threadMessages,
                });

                if (refineResp.ok && refineResp.body) {
                  let refinedText = "";
                  const reader = refineResp.body.getReader();
                  const decoder = new TextDecoder();
                  // Append refinement separator and stream refined version
                  streamer.appendChunk("\n\n---\n\n**Refined answer (System 2):**\n\n");
                  
                  while (true) {
                    const { done, value } = await reader.read();
                    if (done) break;
                    const chunk = decoder.decode(value, { stream: true });
                    const sseLines = chunk.split("\n");
                    for (const line of sseLines) {
                      if (!line.startsWith("data: ")) continue;
                      const payload = line.slice(6).trim();
                      if (payload === "[DONE]") continue;
                      try {
                        const parsed = JSON.parse(payload);
                        if (parsed.text) {
                          refinedText += parsed.text;
                          streamer.appendChunk(parsed.text);
                        }
                      } catch { /* skip */ }
                    }
                  }
                  
                  if (refinedText.length > 50) {
                    fullAnswer = refinedText;
                    draftForReflection = refinedText;
                  }
                }
              } catch (refineErr) {
                console.warn("[SYSTEM2] Refinement streaming failed:", refineErr);
              }
            }
            if (totalIterations > 0) {
            }
          } catch (s2err) {
            console.warn("[SYSTEM2] System 2 RAG failed (non-fatal):", s2err);
          }
        }

      } catch (error: any) {
        streamCompleted = true;
        clearTimeout(streamTimeout);
        const errorMsg = error?.message || "Could not generate an answer.";
        // Provide more helpful error messages
        let userMessage = `Claude answer failed: ${errorMsg}`;
        if (errorMsg.includes("timeout") || errorMsg.includes("timed out")) {
          userMessage = `The request timed out after 70 seconds. This can happen with:\n\n` +
            `вЂў Complex questions requiring deep analysis\n` +
            `вЂў Large documents with lots of context\n` +
            `вЂў Slow API responses\n\n` +
            `рџ’Ў **Try:**\n` +
            `вЂў Rephrasing your question to be more specific\n` +
            `вЂў Breaking complex questions into smaller parts\n` +
            `вЂў Asking about specific companies/topics (e.g., "Giga Energy intern responsibilities")\n` +
            `вЂў Checking if your documents contain the information\n` +
            `вЂў Trying again in a moment`;
        } else if (errorMsg.includes("HTTP error") || errorMsg.includes("Failed to fetch")) {
          userMessage = `Network error: ${errorMsg}\n\n` +
            `рџ’Ў **Check:**\n` +
            `вЂў Your internet connection\n` +
            `вЂў If the API service is available\n` +
            `вЂў Try again in a moment`;
        } else if (errorMsg.includes("AbortError") || errorMsg.includes("aborted")) {
          userMessage = `Request was cancelled. Please try again.`;
        }
        streamer.setError(userMessage);
      } finally {
        setIsClaudeLoading(false);
      }
    },
    [
      activeEventId,
      ensureActiveEventId,
      buildSnippet,
      buildClaudeContext,
      docContainsTokens,
      createAssistantMessage,
      decisions,
      scopes,
      profile,
      user,
      askClaudeAnswerStream,
      connectionsForChat,
      documents,
      persistCostLog,
      getThreadMessages,
      webSearchEnabled,
      multiAgentEnabled,
    ]
  );

  const stopGenerating = useCallback(() => {
    if (chatAbortRef.current) {
      chatAbortRef.current.abort();
      chatAbortRef.current = null;
    }
    setChatIsLoading(false);
    setIsClaudeLoading(false);
  }, []);

  const addMessage = async () => {
    if (chatIsLoading || isClaudeLoading) return;
    if (!input.trim()) return;
    const question = input.trim();
    // If editing a previous message: remove that message and all following messages in this thread, then resend
    if (editingMessageId) {
      setMessages((prev) => {
        const scoped = prev.filter((m) => m.threadId === activeThread);
        const idx = scoped.findIndex((m) => m.id === editingMessageId);
        if (idx < 0) return prev;
        const toRemoveIds = new Set(scoped.slice(idx).map((m) => m.id));
        return prev.filter((m) => !toRemoveIds.has(m.id));
      });
      setEditingMessageId(null);
    }
    let threadId = activeThread;
    const isUuid = /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i.test(threadId || "");
    if (!threadId || !isUuid) {
      const createdId = await createChatThread("Main thread");
      if (createdId) {
        threadId = createdId;
        setThreads((prev) => [...prev, { id: createdId, title: "Main thread" }]);
        setActiveThread(createdId);
      } else {
        console.error("[Chat] Failed to create chat thread — cannot persist messages.");
        toast({ title: "Chat Error", description: "Could not create a chat thread. Please reload and try again.", variant: "destructive" });
        return;
      }
    }
    const id = `m-${Date.now()}`;
    setMessages((prev) => [...prev, { id, author: "user", text: question, threadId }]);
    void persistChatMessage({
      threadId,
      role: "user",
      content: question,
      model: null,
      sourceDocIds: null,
    });
    setInput("");
    // Create a fresh AbortController for this request
    chatAbortRef.current = new AbortController();
    setChatIsLoading(true);
    try {
      await askFund(question, threadId);
    } catch (err) {
      if (err instanceof DOMException && err.name === "AbortError") {
        // User cancelled вЂ” don't show error
        return;
      }
      console.error("Chat error:", err);
      const errorMsg = err instanceof Error ? err.message : "Chat failed unexpectedly. Please try again.";
      createAssistantMessage(
        `вќЊ Error: ${errorMsg}\n\nPlease try again or check the console for details.`,
        threadId
      );
    } finally {
      chatAbortRef.current = null;
      setChatIsLoading(false);
      setIsClaudeLoading(false);
    }
  };

  // Removed createBranch - no longer needed

  const toggleScope = (id: string, checked: boolean) => {
    setScopes((prev) => prev.map((s) => (s.id === id ? { ...s, checked } : s)));
  };

  // Handler to open Log Decision dialog after AI response
  const handleLogDecisionFromChat = useCallback((aiReasoning: string, sourceDocIds?: string[]) => {
    setPendingDecisionContext({ aiReasoning, sourceDocIds });
    setLogDecisionDialogOpen(true);
  }, []);

  // Handler to create a company connection
  const handleCreateConnection = useCallback(async (connectionData: {
    source_company_name: string;
    target_company_name: string;
    source_document_id?: string | null;
    target_document_id?: string | null;
    connection_type: ConnectionType;
    connection_status: ConnectionStatus;
    ai_reasoning?: string | null;
    notes?: string | null;
  }) => {
    const eventId = activeEventId;
    if (!eventId) {
      toast({
        title: "No active event",
        description: "Please wait for the event to load.",
        variant: "destructive",
      });
      return;
    }

    try {
      const { data, error } = await insertCompanyConnection(eventId, {
        ...connectionData,
        created_by: profile?.id || null,
      });

      if (error) throw error;

      // Add to local state
      if (data) {
        setCompanyConnections((prev) => [data as typeof prev[0], ...prev]);
      }

      toast({
        title: "Connection logged",
        description: `Created ${connectionData.connection_type} connection: ${connectionData.source_company_name} в†’ ${connectionData.target_company_name}`,
      });

      setLogDecisionDialogOpen(false);
      setPendingDecisionContext(null);
    } catch (err) {
      console.error("Failed to create connection:", err);
      toast({
        title: "Failed to log decision",
        description: err instanceof Error ? err.message : "Could not create connection.",
        variant: "destructive",
      });
    }
  }, [activeEventId, profile?.id, toast]);

  // Handler to update connection status
  const handleUpdateConnectionStatus = useCallback(async (
    connectionId: string, 
    newStatus: ConnectionStatus
  ) => {
    try {
      const { error } = await updateCompanyConnection(connectionId, { connection_status: newStatus });
      if (error) throw error;

      setCompanyConnections((prev) =>
        prev.map((c) => c.id === connectionId ? { ...c, connection_status: newStatus } : c)
      );

      toast({
        title: "Status updated",
        description: `Connection status changed to "${newStatus}"`,
      });
    } catch (err) {
      console.error("Failed to update connection:", err);
      toast({
        title: "Update failed",
        description: err instanceof Error ? err.message : "Could not update status.",
        variant: "destructive",
      });
    }
  }, [toast]);

  // Handler: AI-powered connection suggestions
  const [aiSuggestions, setAiSuggestions] = useState<Array<{
    source_company: string;
    target_company: string;
    connection_type: string;
    reasoning: string;
    confidence: number;
  }>>([]);
  const [suggestionsLoading, setSuggestionsLoading] = useState(false);

  const handleSuggestConnections = useCallback(async () => {
    setSuggestionsLoading(true);
    try {
      const docSources = documents.slice(0, 10).map((doc) => ({
        title: doc.title,
        file_name: null as string | null,
        snippet: null as string | null,
      }));

      const result = await suggestConnections({
        sources: docSources,
        existingConnections: connectionsForChat,
        maxSuggestions: 5,
      });

      setAiSuggestions(result.suggestions);

      if (result.suggestions.length === 0) {
        // Only show error toast if it's an actual error, not just "no suggestions found"
        const isError = result.contextSummary?.includes("require") || result.contextSummary?.includes("unavailable");
        toast({
          title: isError ? "Suggestion unavailable" : "No suggestions",
          description: result.contextSummary || "Upload more documents to get AI suggestions.",
          variant: isError ? "destructive" : "default",
        });
      } else {
        toast({
          title: `${result.suggestions.length} connection(s) suggested`,
          description: result.contextSummary || "Review and add them to your graph.",
        });
      }
    } catch (err) {
      console.error("Suggest connections failed:", err);
      toast({
        title: "Suggestion failed",
        description: err instanceof Error ? err.message : "Could not generate suggestions.",
        variant: "destructive",
      });
    } finally {
      setSuggestionsLoading(false);
    }
  }, [documents, connectionsForChat, toast]);

  const handleImportConnections = useCallback(async (file: File) => {
    if (!activeEventId) {
      toast({ title: "No event", description: "Select an event first.", variant: "destructive" });
      return;
    }
    const userId = profile?.id || user?.id || null;
    const CONNECTION_TYPES: ConnectionType[] = ["BD", "INV", "Knowledge", "Partnership", "Project"];
    const CONNECTION_STATUSES: ConnectionStatus[] = ["To Connect", "In Progress", "Connected", "Rejected", "Completed"];
    const norm = (s: string) => s.trim().toLowerCase().replace(/\s+/g, "_");
    const findCol = (row: Record<string, unknown>, keys: string[]) => {
      const rowKeys = Object.keys(row);
      for (const k of keys) {
        const n = norm(k);
        const found = rowKeys.find((rk) => norm(rk) === n || norm(rk).replace(/_/g, "") === n.replace(/_/g, ""));
        if (found) {
          const v = row[found];
          return (v != null ? String(v).trim() : "") || "";
        }
      }
      return "";
    };
    let rows: Array<Record<string, unknown>> = [];
    const name = file.name.toLowerCase();
    try {
      if (name.endsWith(".csv")) {
        const text = await file.text();
        const lines = text.split(/\r?\n/).filter((l) => l.trim());
        if (lines.length < 2) {
          toast({ title: "Invalid CSV", description: "File must have a header row and at least one data row.", variant: "destructive" });
          return;
        }
        const sep = lines[0].includes("\t") ? "\t" : lines[0].includes(";") ? ";" : ",";
        const parseLine = (line: string) => {
          const out: string[] = [];
          let cur = "";
          let inQ = false;
          for (let i = 0; i < line.length; i++) {
            const c = line[i];
            if (c === '"') {
              if (inQ && line[i + 1] === '"') {
                cur += '"';
                i++;
              } else inQ = !inQ;
              continue;
            }
            if (c === sep && !inQ) {
              out.push(cur.trim());
              cur = "";
              continue;
            }
            cur += c;
          }
          out.push(cur.trim());
          return out;
        };
        const header = parseLine(lines[0]);
        for (let i = 1; i < lines.length; i++) {
          const cells = parseLine(lines[i]);
          const row: Record<string, unknown> = {};
          header.forEach((h, j) => {
            row[h] = cells[j] ?? "";
          });
          rows.push(row);
        }
      } else if (name.endsWith(".xlsx") || name.endsWith(".xls")) {
        const buf = await file.arrayBuffer();
        const XLSX = await import("https://esm.sh/xlsx@0.18.5");
        const read = XLSX.read ?? (XLSX.default && XLSX.default.read);
        const utils = XLSX.utils ?? (XLSX.default && XLSX.default.utils);
        if (!read || !utils) {
          toast({ title: "XLSX load failed", description: "Could not load spreadsheet library.", variant: "destructive" });
          return;
        }
        const wb = read(buf, { type: "array" });
        const first = wb.SheetNames[0];
        if (!first) {
          toast({ title: "Invalid XLSX", description: "No sheet found.", variant: "destructive" });
          return;
        }
        const sheet = wb.Sheets[first];
        rows = utils.sheet_to_json(sheet, { defval: "" }) as Array<Record<string, unknown>>;
      } else {
        toast({ title: "Unsupported file", description: "Use .csv or .xlsx", variant: "destructive" });
        return;
      }
      let added = 0;
      let skipped = 0;
      const errors: string[] = [];
      for (const row of rows) {
        const source = findCol(row, ["source_company_name", "source_company", "source", "Source Company", "From", "Company A"]) || String((row as any)["Source"] ?? "").trim() || String((row as any)["source"] ?? "").trim();
        const target = findCol(row, ["target_company_name", "target_company", "target", "Target Company", "To", "Company B"]) || String((row as any)["Target"] ?? "").trim() || String((row as any)["target"] ?? "").trim();
        if (!source || !target) {
          skipped++;
          continue;
        }
        const typeVal = findCol(row, ["connection_type", "type", "Connection Type", "Type"]);
        const statusVal = findCol(row, ["connection_status", "status", "Status"]);
        const notesVal = findCol(row, ["notes", "Notes", "ai_reasoning", "reasoning", "Reasoning"]);
        const connectionType: ConnectionType = CONNECTION_TYPES.find((t) => norm(t) === norm(typeVal)) ?? "BD";
        const connectionStatus: ConnectionStatus = CONNECTION_STATUSES.find((s) => norm(s) === norm(statusVal)) ?? "To Connect";
        const { error } = await insertCompanyConnection(activeEventId, {
          source_company_name: source,
          target_company_name: target,
          connection_type: connectionType,
          connection_status: connectionStatus,
          notes: notesVal || null,
          created_by: userId,
        });
        if (error) {
          errors.push(`${source} в†’ ${target}: ${error.message}`);
        } else {
          added++;
        }
      }
      const { data } = await getCompanyConnectionsByEvent(activeEventId);
      if (data) setCompanyConnections(data as typeof companyConnections);
      if (errors.length > 0) {
        toast({ title: `Imported ${added}`, description: `${skipped} skipped, ${errors.length} errors. ${errors.slice(0, 2).join("; ")}`, variant: "destructive" });
      } else {
        toast({ title: "Connections imported", description: `${added} connection(s) added. ${skipped} row(s) skipped (missing source/target).` });
      }
    } catch (err) {
      console.error("Import connections failed:", err);
      toast({ title: "Import failed", description: err instanceof Error ? err.message : "Could not parse file.", variant: "destructive" });
    }
  }, [activeEventId, profile?.id, user?.id, toast]);

  const evidence = initialKOs;
  const buildStamp =
    (import.meta.env.VITE_BUILD_STAMP as string | undefined) ||
    (import.meta.env.VITE_VERCEL_GIT_COMMIT_SHA as string | undefined) ||
    "local";
  const lastTokens = useMemo(() => {
    if (!lastEvidence?.question) return [];
    return lastEvidence.question
      .toLowerCase()
      .split(/\W+/)
      .map((t) => t.trim())
      .filter((t) => t.length > 3);
  }, [lastEvidence?.question]);

  /** Replace verbose [[Source: ...]] tags with [N] so inline badge renderer shows them (Verifiable RAG).
   *  Also strips raw `doc_id:` and `chunk:` fragments that leak from the backend prompt.
   *  Handles nested brackets like [[Source: [Kuration AI] - Meeting Notes | doc_id:... | chunk:1]]. */
  const cleanVerifiableCitationTags = useCallback((text: string) => {
    let n = 0;
    const byTag: Record<string, number> = {};
    return text
      // Replace full [[Source: ... ]] blocks (allowing nested [ ] inside via greedy match to last ]])
      .replace(/\[\[Source:[\s\S]*?\]\]/gi, (tag) => {
        if (byTag[tag] == null) {
          n += 1;
          byTag[tag] = n;
        }
        return `[${byTag[tag]}]`;
      })
      // Catch single-bracket leftover: [Source: ...]
      .replace(/\[Source:[^\]]*\]/gi, (tag) => {
        if (byTag[tag] == null) {
          n += 1;
          byTag[tag] = n;
        }
        return `[${byTag[tag]}]`;
      })
      // Catch naked doc_id / chunk refs that leak outside brackets
      .replace(/\s*\|?\s*doc_id:[^\]|)}\n]*/gi, "")
      .replace(/\s*\|?\s*chunk:\d+/gi, "")
      // Catch stray (doc_id:...) or (20230519)) leftovers from partially stripped tags
      .replace(/\s*\(\s*(?:doc_id:|20\d{6}\))\)?\s*/gi, "");
  }, []);

  const renderAssistantContent = useCallback((text: string) => {
    // в”Ђв”Ђ Inline markdown renderer в”Ђв”Ђ
    // Converts **bold**, *italic*, `code`, [n] references into React elements
    const renderInline = (raw: string, keyPrefix: string = ""): React.ReactNode[] => {
      const parts: React.ReactNode[] = [];
      // Process in two passes: first links, then other markdown
      // This ensures [text](url) links are always matched correctly
      
      // Pass 1: Extract and replace markdown links [text](url) with placeholders
      const linkPlaceholders: { [key: string]: { text: string; url: string } } = {};
      let linkCounter = 0;
      let processedText = raw.replace(/\[([^\]]+)\]\(([^)]+)\)/g, (match, text, url) => {
        const placeholder = `__LINK_${linkCounter}__`;
        linkPlaceholders[placeholder] = { text, url };
        linkCounter++;
        return placeholder;
      });
      
      // Pass 2: Process remaining markdown (bold, italic, code, source refs)
      const inlineRegex = /(\*\*\*(.+?)\*\*\*|\*\*(.+?)\*\*|\*(.+?)\*|`([^`]+)`|\[(\d+)\])/g;
      let lastIndex = 0;
      let match: RegExpExecArray | null;
      let i = 0;
      while ((match = inlineRegex.exec(processedText)) !== null) {
        // Text before the match (check for link placeholders)
        if (match.index > lastIndex) {
          const beforeText = processedText.slice(lastIndex, match.index);
          // Replace link placeholders with actual links
          const beforeParts = beforeText.split(/(__LINK_\d+__)/g);
          beforeParts.forEach((part, idx) => {
            if (part.startsWith('__LINK_') && part.endsWith('__')) {
              const linkData = linkPlaceholders[part];
              if (linkData) {
                parts.push(
                  <a
                    key={`${keyPrefix}link-${i}-${idx}`}
                    href={linkData.url}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="text-blue-600 hover:text-blue-600/80 underline decoration-[#3b82f6]/50 hover:decoration-[#3b82f6] transition-colors"
                  >
                    {linkData.text}
                  </a>
                );
              }
            } else if (part) {
              parts.push(<span key={`${keyPrefix}t-${i}-${idx}`}>{part}</span>);
            }
          });
        }
        if (match[2]) {
          // ***bold italic***
          parts.push(<strong key={`${keyPrefix}bi${i}`} className="font-bold italic text-blue-600">{match[2]}</strong>);
        } else if (match[3]) {
          // **bold**
          parts.push(<strong key={`${keyPrefix}b${i}`} className="font-bold text-blue-600">{match[3]}</strong>);
        } else if (match[4]) {
          // *italic*
          parts.push(<em key={`${keyPrefix}i${i}`} className="italic text-slate-700">{match[4]}</em>);
        } else if (match[5]) {
          // `code`
          parts.push(<code key={`${keyPrefix}c${i}`} className="bg-slate-100 px-1.5 py-0.5 rounded text-xs text-blue-600 font-mono">{match[5]}</code>);
        } else if (match[6]) {
          // [1] source reference
          parts.push(<span key={`${keyPrefix}r${i}`} className="inline-flex items-center justify-center bg-blue-600/20 text-blue-600 text-[10px] font-bold rounded-full w-4 h-4 mx-0.5 align-text-top">{match[6]}</span>);
        }
        lastIndex = match.index + match[0].length;
        i++;
      }
      // Handle remaining text after last match (check for link placeholders)
      if (lastIndex < processedText.length) {
        const remainingText = processedText.slice(lastIndex);
        const remainingParts = remainingText.split(/(__LINK_\d+__)/g);
        remainingParts.forEach((part, idx) => {
          if (part.startsWith('__LINK_') && part.endsWith('__')) {
            const linkData = linkPlaceholders[part];
            if (linkData) {
              parts.push(
                <a
                  key={`${keyPrefix}link-end-${idx}`}
                  href={linkData.url}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="text-blue-600 hover:text-blue-600/80 underline decoration-[#3b82f6]/50 hover:decoration-[#3b82f6] transition-colors"
                >
                  {linkData.text}
                </a>
              );
            }
          } else if (part) {
            parts.push(<span key={`${keyPrefix}end-${idx}`}>{part}</span>);
          }
        });
      }
      return parts.length > 0 ? parts : [<span key={`${keyPrefix}plain`}>{raw}</span>];
    };

    const lines = text.split("\n");
    type Block =
      | { type: "h1"; content: string }
      | { type: "h2"; content: string }
      | { type: "h3"; content: string }
      | { type: "p"; content: string }
      | { type: "ul"; items: string[] }
      | { type: "ol"; items: string[] }
      | { type: "table"; rows: string[][] }
      | { type: "hr" }
      | { type: "blank" };

    const blocks: Block[] = [];
    let ulItems: string[] = [];
    let olItems: string[] = [];
    let paragraph: string[] = [];
    let tableRows: string[][] = [];

    const isTableRow = (s: string) => /^\|.+\|$/.test(s.trim());
    const parseTableRow = (s: string) => s.split("|").map((c) => c.trim()).filter((_, i, arr) => i > 0 && i < arr.length - 1);
    const isTableSeparator = (cells: string[]) => cells.every((c) => /^[-:\s]+$/.test(c));

    const flushTable = () => {
      if (tableRows.length) {
        const rows = tableRows.filter((row) => row.some((c) => c.length > 0));
        if (rows.length) blocks.push({ type: "table", rows });
        tableRows = [];
      }
    };

    const flushParagraph = () => {
      if (paragraph.length) {
        blocks.push({ type: "p", content: paragraph.join(" ") });
        paragraph = [];
      }
    };
    const flushUl = () => {
      if (ulItems.length) {
        blocks.push({ type: "ul", items: [...ulItems] });
        ulItems = [];
      }
    };
    const flushOl = () => {
      if (olItems.length) {
        blocks.push({ type: "ol", items: [...olItems] });
        olItems = [];
      }
    };
    const flushAll = () => { flushParagraph(); flushUl(); flushOl(); flushTable(); };

    for (const raw of lines) {
      const line = raw.trimEnd();
      const trimmed = line.trim();

      // Blank line
      if (!trimmed) { flushAll(); continue; }

      // Table row: | cell | cell |
      if (isTableRow(trimmed)) {
        flushParagraph(); flushUl(); flushOl();
        const cells = parseTableRow(trimmed);
        if (cells.length && !(tableRows.length === 1 && isTableSeparator(cells))) {
          tableRows.push(cells);
        }
        continue;
      } else {
        flushTable();
      }

      // Horizontal rule
      if (/^(---+|\*\*\*+|___+)$/.test(trimmed)) { flushAll(); blocks.push({ type: "hr" }); continue; }

      // Headings: ## or ###
      const headingMatch = trimmed.match(/^(#{1,3})\s+(.+)$/);
      if (headingMatch) {
        flushAll();
        const level = headingMatch[1].length;
        const content = headingMatch[2].replace(/\s*#+$/, ""); // strip trailing #
        if (level === 1) blocks.push({ type: "h1", content });
        else if (level === 2) blocks.push({ type: "h2", content });
        else blocks.push({ type: "h3", content });
        continue;
      }

      // Unordered list: - item or * item
      if (/^[-*]\s+/.test(trimmed)) {
        flushParagraph(); flushOl();
        ulItems.push(trimmed.replace(/^[-*]\s+/, ""));
        continue;
      }

      // Ordered list: 1. item, 2. item
      if (/^\d+[.)]\s+/.test(trimmed)) {
        flushParagraph(); flushUl();
        olItems.push(trimmed.replace(/^\d+[.)]\s+/, ""));
        continue;
      }

      // Lines ending with ":" that are short в†’ treat as sub-heading
      if (trimmed.endsWith(":") && trimmed.length < 80 && !trimmed.startsWith("http")) {
        flushAll();
        blocks.push({ type: "h3", content: trimmed.replace(/:$/, "") });
        continue;
      }

      // Normal paragraph text
      flushUl(); flushOl();
      paragraph.push(trimmed);
    }
    flushAll();

    return (
      <div className="space-y-3">
        {blocks.map((block, idx) => {
          switch (block.type) {
            case "h1":
              return (
                <h2 key={idx} className="text-base font-bold text-blue-600 font-mono mt-3 mb-1 border-b border-slate-200 pb-1">
                  {renderInline(block.content, `h1-${idx}-`)}
                </h2>
              );
            case "h2":
              return (
                <h3 key={idx} className="text-sm font-bold text-blue-600 font-mono mt-3 mb-1">
                  {renderInline(block.content, `h2-${idx}-`)}
                </h3>
              );
            case "h3":
              return (
                <h4 key={idx} className="text-sm font-semibold text-slate-700 font-mono mt-2 mb-0.5">
                  {renderInline(block.content, `h3-${idx}-`)}
                </h4>
              );
            case "ul":
              return (
                <ul key={idx} className="list-disc pl-5 text-sm text-slate-900 space-y-1.5">
                  {block.items.map((item, i) => (
                    <li key={i} className="text-slate-900 leading-relaxed">{renderInline(item, `ul-${idx}-${i}-`)}</li>
                  ))}
                </ul>
              );
            case "ol":
              return (
                <ol key={idx} className="list-decimal pl-5 text-sm text-slate-900 space-y-1.5">
                  {block.items.map((item, i) => (
                    <li key={i} className="text-slate-900 leading-relaxed">{renderInline(item, `ol-${idx}-${i}-`)}</li>
                  ))}
                </ol>
              );
            case "hr":
              return <hr key={idx} className="border-slate-200 my-3" />;
            case "table": {
              const { rows } = block;
              const [head, ...body] = rows;
              return (
                <div key={idx} className="my-3 overflow-x-auto">
                  <table className="w-full border-collapse text-sm font-mono text-slate-900">
                    {head && head.length > 0 && (
                      <thead>
                        <tr>
                          {head.map((cell, cidx) => (
                            <th key={cidx} className="border border-slate-300 px-3 py-2 text-left font-bold text-blue-600 bg-slate-50">
                              {renderInline(cell, `th-${idx}-${cidx}-`)}
                            </th>
                          ))}
                        </tr>
                      </thead>
                    )}
                    <tbody>
                      {body.map((row, ridx) => (
                        <tr key={ridx}>
                          {row.map((cell, cidx) => (
                            <td key={cidx} className="border border-slate-200 px-3 py-2 text-slate-700">
                              {renderInline(cell, `td-${idx}-${ridx}-${cidx}-`)}
                            </td>
                          ))}
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              );
            }
            case "p":
              return (
                <p key={idx} className="text-sm text-slate-900 leading-relaxed">
                  {renderInline(block.content, `p-${idx}-`)}
                </p>
              );
            default:
              return null;
          }
        })}
      </div>
    );
  }, []);

  return (
    <div className="min-h-screen bg-white text-slate-900 relative overflow-hidden cis-app">
      {/* Black + yellow background: grid + subtle mesh */}
      <div className="fixed inset-0 cis-grid-bg cis-mesh-bg pointer-events-none" />

      <div className="relative z-10 max-w-[1600px] mx-auto px-4 py-3 space-y-3">
        {/* Top navigation bar */}
        <header className="flex items-center justify-between gap-4 border-b border-slate-200/20 pb-3 cis-fade-in">
          <div className="flex items-center gap-6 min-w-0">
            <div className="flex items-center gap-2 shrink-0">
              <span className="flex h-9 w-9 items-center justify-center rounded-lg bg-blue-600/15 text-blue-600">
                <Brain className="h-5 w-5" />
              </span>
              <span className="font-bold text-slate-900 text-lg tracking-tight hidden sm:inline">Platform</span>
            </div>
            <nav className="flex items-center gap-0.5 flex-wrap">
              {profile?.role !== "lp" && (
                <button
                  onClick={() => setActiveTab("chat")}
                  className={`px-3 py-2 rounded-lg text-sm font-medium transition-all ${
                    activeTab === "chat" ? "bg-blue-600/15 text-blue-600" : "text-slate-500 hover:text-slate-900 hover:bg-slate-50"
                  }`}
                >
                  Chat
                </button>
              )}
              {(profile?.role === "managing_partner" || profile?.role === "organizer" || profile?.role === "admin") && (
                <button
                  onClick={() => setActiveTab("onboarding")}
                  className={`px-3 py-2 rounded-lg text-sm font-medium transition-all ${
                    activeTab === "onboarding" ? "bg-blue-600/15 text-blue-600" : "text-slate-500 hover:text-slate-900 hover:bg-slate-50"
                  }`}
                >
                  Onboarding
                </button>
              )}
              {(profile?.role === "managing_partner" || profile?.role === "organizer" || profile?.role === "admin") && (
                <button
                  onClick={() => setActiveTab("admin")}
                  className={`px-3 py-2 rounded-lg text-sm font-medium transition-all ${
                    activeTab === "admin" ? "bg-blue-600/15 text-blue-600" : "text-slate-500 hover:text-slate-900 hover:bg-slate-50"
                  }`}
                >
                  Admin
                </button>
              )}
              <button
                onClick={() => setActiveTab("overview")}
                className={`px-3 py-2 rounded-lg text-sm font-medium transition-all ${
                  activeTab === "overview" ? "bg-blue-600/15 text-blue-600" : "text-slate-500 hover:text-slate-900 hover:bg-slate-50"
                }`}
              >
                Dashboard
              </button>
              {profile?.role !== "lp" && (
                <button
                  onClick={() => setActiveTab("sources")}
                  className={`px-3 py-2 rounded-lg text-sm font-medium transition-all ${
                    activeTab === "sources" ? "bg-blue-600/15 text-blue-600" : "text-slate-500 hover:text-slate-900 hover:bg-slate-50"
                  }`}
                >
                  Sources
                </button>
              )}
              {profile?.role !== "lp" && (
                <button
                  onClick={() => setActiveTab("decisions")}
                  className={`px-3 py-2 rounded-lg text-sm font-medium transition-all ${
                    activeTab === "decisions" ? "bg-blue-600/15 text-blue-600" : "text-slate-500 hover:text-slate-900 hover:bg-slate-50"
                  }`}
                >
                  Decisions
                </button>
              )}
            </nav>
          </div>
          <DropdownMenu>
            <DropdownMenuTrigger asChild>
              <button className="flex items-center gap-2 px-3 py-2 rounded-lg hover:bg-blue-500/10 hover:border-blue-500/30 text-slate-900 font-medium text-sm shrink-0 transition-all">
                <span className="flex h-7 w-7 items-center justify-center rounded-full bg-blue-600/20 text-blue-600 text-xs font-bold">
                  {(profile?.full_name || profile?.email || "U").charAt(0).toUpperCase()}
                </span>
                <span className="max-w-[140px] truncate hidden sm:inline">{profile?.full_name || profile?.email || "Account"}</span>
                <ChevronDown className="h-3.5 w-3.5 text-slate-400" />
              </button>
            </DropdownMenuTrigger>
            <DropdownMenuContent align="end" className="w-48 border border-slate-200 bg-white/95 backdrop-blur-xl text-slate-900">
              <DropdownMenuItem
                onClick={() => setActiveTab("account")}
                className="cursor-pointer text-slate-600 focus:bg-slate-100 focus:text-slate-900"
              >
                <User className="h-4 w-4 mr-2" />
                My Account
              </DropdownMenuItem>
              <DropdownMenuSeparator className="bg-slate-100" />
              <DropdownMenuItem onClick={signOut} className="cursor-pointer text-red-400 focus:bg-red-500/20 focus:text-red-200">
                <LogOut className="h-4 w-4 mr-2" />
                Log out
              </DropdownMenuItem>
            </DropdownMenuContent>
          </DropdownMenu>
        </header>

        {/* Main Layout вЂ” sidebar only when on Chat tab */}
        <div className="flex gap-4">
          {activeTab === "chat" && (
          <div className="w-64 flex-shrink-0 flex flex-col gap-4">
            {/* Chat Threads */}
            <div className="cis-surface p-4 sticky top-4 cis-fade-in-up cis-stagger-1 opacity-0 [animation-fill-mode:forwards]">
                <div className="text-xs text-slate-500 font-semibold uppercase tracking-wider mb-4 pb-2 border-b border-slate-200/20">
                  Chat Threads
                </div>
                <div className="space-y-3">
                  <Button
                    onClick={async () => {
                      const newThreadId = await createChatThread(`Chat ${threads.length + 1}`);
                      if (newThreadId) {
                        setActiveThread(newThreadId);
                        setMessages([]);
                        // Reload threads to show the new one
                        const eventId = activeEventId || (await ensureActiveEventId());
                        if (eventId) {
                          const { data: threadRows } = await supabase
                            .from("chat_threads")
                            .select("*")
                            .eq("event_id", eventId)
                            .order("created_at", { ascending: true });
                          if (threadRows?.length) {
                            const mappedThreads = threadRows.map((t: any) => ({
                              id: t.id,
                              title: t.title,
                              parentId: t.parent_id || undefined,
                            }));
                            setThreads(mappedThreads);
                          }
                        }
                      }
                    }}
                    className="w-full cis-btn-primary text-sm py-2.5"
                  >
                    <MessageSquarePlus className="h-4 w-4 mr-2" />
                    Create New Chat
                  </Button>
                  {threads.length > 0 ? (
                    <div className="max-h-64 overflow-y-auto">
                      <ThreadTree
                        threads={threads}
                        active={activeThread}
                        onSelect={(id) => {
                          setActiveThread(id);
                          setMessages([]);
                        }}
                      />
                    </div>
                  ) : (
                    <div className="text-xs text-slate-400 text-center py-4">
                      No threads yet
                    </div>
                  )}
                </div>
              </div>

            {/* Knowledge Scope вЂ” compact, scrollable, with grouped Drive folders */}
            <div className="cis-surface p-2 sticky top-4 cis-fade-in-up cis-stagger-3 opacity-0 [animation-fill-mode:forwards] max-h-[min(340px,45vh)] flex flex-col min-w-0">
                <div className="flex items-center justify-between mb-1.5 flex-shrink-0">
                  <div className="text-[10px] text-slate-500 font-semibold uppercase tracking-wider truncate">
                    Knowledge Scope
                  </div>
                  <button
                    type="button"
                    onClick={() => {
                      const allChecked = scopes.every((s) => s.checked);
                      setScopes((prev) => prev.map((s) => ({ ...s, checked: !allChecked })));
                    }}
                    className="text-[10px] text-slate-400 hover:text-blue-600 font-mono transition-colors shrink-0"
                  >
                    {scopes.every((s) => s.checked) ? "Deselect All" : "Select All"}
                  </button>
                </div>
                <div className="flex flex-col gap-0.5 overflow-y-auto min-h-0">
                  {/* Non-folder scopes */}
                  <div className="flex flex-wrap gap-1">
                    {scopes.filter((s) => s.type !== "folder").map((s) => (
                      <label
                        key={s.id}
                        className={`inline-flex items-center gap-1 text-[10px] border px-1.5 py-0.5 rounded cursor-pointer transition-all font-mono shrink-0 max-w-full min-w-0 ${
                          s.checked
                            ? "border-blue-500/50 bg-blue-600/10 text-blue-600"
                            : "border-slate-200 bg-white text-slate-500 hover:border-slate-300 hover:text-slate-600"
                        }`}
                      >
                        <Checkbox
                          checked={s.checked}
                          onCheckedChange={(val) => toggleScope(s.id, val === true)}
                          className="h-2.5 w-2.5 border-slate-300 data-[state=checked]:bg-blue-600 data-[state=checked]:border-blue-500 shrink-0"
                        />
                        <span className="truncate">{s.label}</span>
                      </label>
                    ))}
                  </div>
                </div>
              </div>

          </div>
          )}

          {/* Main Content Area */}
          <div className="flex-1 min-w-0 cis-fade-in-up cis-stagger-4 opacity-0 [animation-fill-mode:forwards]">
            <Tabs value={activeTab} onValueChange={setActiveTab} className="space-y-4">

          {/* Onboarding Tab */}
          <TabsContent value="onboarding">
            <OnboardingTab
              profile={profile}
              sources={sources}
              documents={documents}
              decisions={decisions}
              tasks={tasks}
              onNavigate={setActiveTab}
            />
          </TabsContent>

          {/* Chat Tab */}
          <TabsContent value="chat" className="overflow-hidden">
            {isDeveloper && (
              <Card className="border border-slate-200 bg-white mb-3">
                <CardHeader className="pb-2 border-b border-slate-200">
                  <CardTitle className="text-sm text-slate-900 font-mono font-black uppercase tracking-tight">Developer Cost Log</CardTitle>
                  <CardDescription className="text-xs text-slate-500 font-mono">
                    Estimated Claude spend (local only).
                  </CardDescription>
                </CardHeader>
                <CardContent className="text-xs space-y-2">
                  <div className="font-medium">
                    Total: $
                    {costLog.reduce((sum, entry) => sum + entry.estCostUsd, 0).toFixed(4)}
                  </div>
                  {costLog.length === 0 ? (
                    <div className="text-slate-500 font-mono">No Claude calls logged yet.</div>
                  ) : (
                    costLog.slice(0, 5).map((entry) => (
                      <div key={entry.ts} className="border rounded-md p-2">
                        <div className="font-medium">${entry.estCostUsd} вЂў {entry.ts}</div>
                        <div className="text-slate-500 font-mono">Q: {entry.question}</div>
                        <div className="text-slate-500 font-mono">
                          Tokens: {entry.estInputTokens} in / {entry.estOutputTokens} out
                        </div>
                      </div>
                    ))
                  )}
                </CardContent>
              </Card>
            )}
            {/* Chat Container вЂ” fills viewport, input at bottom of card */}
            <div className="flex flex-col overflow-hidden" style={{ height: "calc(100vh - 140px)", minHeight: "600px" }}>
              <Card className="flex-1 flex flex-col border border-slate-200/20 bg-white/30 backdrop-blur-sm min-h-0 h-full overflow-hidden rounded-xl">
                {/* Chat header */}
                <div className="flex items-center justify-between px-5 py-3 border-b border-slate-200/15 bg-white/40 flex-shrink-0">
                  <div className="flex items-center gap-3">
                    <span className="flex h-8 w-8 items-center justify-center rounded-lg bg-blue-600/15 text-blue-600">
                      <Brain className="h-4 w-4" />
                    </span>
                    <div className="min-w-0 flex-1">
                      <div className="text-sm font-bold text-slate-900 tracking-tight">Intelligence Chat</div>
                      <div className="text-[10px] text-slate-400 font-mono uppercase tracking-wider truncate" title={scopes.filter((s) => s.checked).map((s) => s.label).join(", ") || "None"}>
                        Scope: {(() => {
                          const labels = scopes.filter((s) => s.checked).map((s) => s.label);
                          if (labels.length === 0) return "None";
                          if (labels.length <= 2) return labels.join(", ");
                          return `${labels.slice(0, 2).join(", ")} +${labels.length - 2}`;
                        })()}
                      </div>
                    </div>
                  </div>
                  {chatIsLoading && (
                    <div className="flex items-center gap-2 px-3 py-1.5 rounded-full bg-blue-600/10 border border-blue-500/25">
                      <span className="relative flex h-2 w-2">
                        <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-blue-600 opacity-75" />
                        <span className="relative inline-flex rounded-full h-2 w-2 bg-blue-600" />
                      </span>
                      <span className="text-[11px] text-blue-600 font-mono font-semibold animate-pulse">Processing</span>
                    </div>
                  )}
                </div>

                {/* Messages area вЂ” scrollable, content aligned to bottom */}
                <div 
                  ref={chatContainerRef}
                  className="flex-1 overflow-y-auto px-5 py-5 bg-white scroll-smooth flex flex-col"
                >
                  {/* Spacer pushes messages to bottom when few messages */}
                  <div className="flex-1" />
                  {scopedMessages.length === 0 ? (
                    <div className="flex items-center justify-center min-h-[300px]">
                      <div className="text-center space-y-4 max-w-md">
                        <div className="flex justify-center">
                          <span className="flex h-16 w-16 items-center justify-center rounded-2xl bg-blue-600/10 border-2 border-blue-500/20">
                            <Brain className="h-8 w-8 text-blue-600" />
                          </span>
                        </div>
                        <div className="text-xl font-bold text-slate-900">Start a conversation</div>
                        <div className="text-sm text-slate-400 font-mono leading-relaxed">
                          Ask questions about your documents, companies, and portfolio. The AI will search your knowledge base to provide intelligent answers.
                        </div>
                        <div className="flex flex-wrap gap-2 justify-center pt-2">
                          {["Summarize recent deal flow", "Compare company financials", "What risks have been flagged?"].map((suggestion) => (
                            <button
                              key={suggestion}
                              onClick={() => { setInput(suggestion); }}
                              className="text-xs px-3 py-1.5 rounded-full border border-slate-200 text-slate-500 hover:text-blue-600 hover:border-blue-500/40 transition-all font-mono"
                            >
                              {suggestion}
                            </button>
                          ))}
                        </div>
                      </div>
                    </div>
                  ) : (
                    <div className="space-y-5">
                      {scopedMessages.map((m, index) => (
                        <div
                          key={m.id}
                          className={`flex gap-3 animate-in fade-in slide-in-from-bottom-2 duration-300 ${
                            m.author === "user" ? "justify-end" : "justify-start"
                          }`}
                          style={{ animationDelay: `${Math.min(index, 5) * 50}ms` }}
                        >
                          {m.author === "assistant" && (
                            <div className="flex-shrink-0 w-8 h-8 rounded-full bg-blue-600/15 border border-blue-500/30 flex items-center justify-center mt-1">
                              <Brain className="w-4 h-4 text-blue-600" />
                            </div>
                          )}
                          <div
                            className={`max-w-[80%] rounded-2xl px-4 py-3 ${
                              m.author === "user"
                                ? "bg-blue-600 text-slate-900 font-mono shadow-lg shadow-[#3b82f6]/10"
                                : "bg-white/[0.04] border border-slate-200/15 text-slate-900 font-mono backdrop-blur-sm"
                            }`}
                          >
                            {m.author === "assistant" ? (
                              <>
                                <div className="prose prose-sm dark:prose-invert max-w-none text-slate-900 [&_*]:text-slate-900 [&_p]:text-slate-900 [&_strong]:text-slate-900 [&_em]:text-slate-900 [&_ul]:text-slate-900 [&_ol]:text-slate-900 [&_li]:text-slate-900 [&_h1]:text-slate-900 [&_h2]:text-slate-900 [&_h3]:text-slate-900 [&_h4]:text-slate-900 [&_code]:text-slate-900 [&_pre]:text-slate-900">
                                  {m.isStreaming && m.text === "..." ? (
                                    <span className="inline-flex items-center gap-1 text-slate-900">
                                      <span className="animate-pulse">.</span>
                                      <span className="animate-pulse delay-75">.</span>
                                      <span className="animate-pulse delay-150">.</span>
                                    </span>
                                  ) : (
                                    <>
                                      {renderAssistantContent(cleanVerifiableCitationTags(m.text))}
                                      {m.isStreaming && (
                                        <span className="inline-block w-2 h-5 ml-1 bg-blue-600 animate-pulse" />
                                      )}
                                    </>
                                  )}
                                </div>
                                {/* Sources: one button that expands a clickable list */}
                                {!m.isStreaming && ((m.sourceDocs?.length ?? 0) > 0 || (m.contextLabels?.length ?? 0) > 0) && (
                                  <div className="mt-2 pt-2 border-t border-slate-200">
                                    <DropdownMenu>
                                      <DropdownMenuTrigger asChild>
                                        <Button
                                          size="sm"
                                          variant="outline"
                                          className="text-[11px] h-auto py-1 px-2.5 border border-slate-200 bg-white text-slate-500 hover:bg-blue-500/10 hover:border-blue-500/40 hover:text-blue-600 font-mono transition-all"
                                        >
                                          <FileText className="h-3 w-3 mr-1.5" />
                                          Sources
                                          <span className="ml-1 text-slate-400">({(m.sourceDocs?.length ?? 0) + (m.contextLabels?.length ?? 0)})</span>
                                          <ChevronDown className="h-3 w-3 ml-1.5" />
                                        </Button>
                                      </DropdownMenuTrigger>
                                      <DropdownMenuContent align="start" className="w-80 bg-white/95 border border-slate-200 text-slate-900">
                                        {m.sourceDocs && m.sourceDocs.length > 0 && (
                                          <>
                                            <div className="px-2 py-1.5 text-[10px] font-mono uppercase tracking-wider text-slate-400">
                                              Documents
                                            </div>
                                            {m.sourceDocs.map((doc, idx) => (
                                              <DropdownMenuItem
                                                key={doc.id}
                                                className="font-mono text-xs cursor-pointer"
                                                onSelect={(e) => {
                                                  e.preventDefault();
                                                  handleOpenDocument(doc.id);
                                                }}
                                              >
                                                {idx + 1}. {doc.title || "Document"}
                                              </DropdownMenuItem>
                                            ))}
                                          </>
                                        )}
                                        {m.contextLabels && m.contextLabels.length > 0 && (
                                          <>
                                            {m.sourceDocs && m.sourceDocs.length > 0 && <DropdownMenuSeparator />}
                                            <div className="px-2 py-1.5 text-[10px] font-mono uppercase tracking-wider text-slate-400">
                                              Also used
                                            </div>
                                            {m.contextLabels.map((label) => (
                                              <DropdownMenuItem key={label} disabled className="font-mono text-xs text-slate-500">
                                                {label}
                                              </DropdownMenuItem>
                                            ))}
                                          </>
                                        )}
                                      </DropdownMenuContent>
                                    </DropdownMenu>
                                  </div>
                                )}
                                {/* Log Connection button - appears after each AI response */}
                                {!m.isStreaming && m.text && m.text !== "..." && (
                                  <div className="mt-2 pt-2 border-t border-slate-200">
                                    <Button
                                      variant="ghost"
                                      size="sm"
                                      onClick={() => handleLogDecisionFromChat(m.text)}
                                      className="text-xs h-6 px-2 text-slate-400 hover:text-blue-600 hover:bg-slate-50 font-mono"
                                    >
                                      <Link2 className="h-3 w-3 mr-1" />
                                      Log Connection
                                    </Button>
                                  </div>
                                )}
                              </>
                            ) : (
                              <div className="group flex items-start gap-2">
                                <div className="text-sm leading-relaxed whitespace-pre-wrap text-black flex-1">{m.text}</div>
                                {!chatIsLoading && (
                                  <button
                                    type="button"
                                    onClick={() => {
                                      setEditingMessageId(m.id);
                                      setInput(m.text);
                                      setTimeout(() => document.querySelector<HTMLTextAreaElement>("textarea[placeholder*='Ask a question']")?.focus(), 50);
                                    }}
                                    className="flex-shrink-0 p-1.5 rounded-lg text-black/50 hover:text-black hover:bg-white/10 transition-all opacity-60 hover:opacity-100"
                                    title="Edit and resend"
                                  >
                                    <Pencil className="h-3.5 w-3.5" />
                                  </button>
                                )}
                              </div>
                            )}
                          </div>
                          {m.author === "user" && (
                            <div className="flex-shrink-0 w-8 h-8 rounded-full bg-blue-600/15 border border-blue-500/30 flex items-center justify-center mt-1">
                              <svg className="w-4 h-4 text-blue-600" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M16 7a4 4 0 11-8 0 4 4 0 018 0zM12 14a7 7 0 00-7 7h14a7 7 0 00-7-7z" />
                              </svg>
                            </div>
                          )}
                        </div>
                      ))}

                      {/* Animated thinking indicator вЂ” shows while loading */}
                      {chatIsLoading && (
                        <div className="flex gap-3 justify-start animate-in fade-in slide-in-from-bottom-2 duration-300">
                          <div className="flex-shrink-0 w-8 h-8 rounded-full bg-blue-600/15 border border-blue-500/30 flex items-center justify-center mt-1">
                            <Brain className="w-4 h-4 text-blue-600 animate-pulse" />
                          </div>
                          <div className="max-w-[80%] rounded-2xl px-4 py-3 bg-white/[0.04] border border-blue-500/20 text-slate-900 font-mono backdrop-blur-sm">
                            <div className="flex items-center gap-3">
                              <div className="flex gap-1">
                                <span className="w-2 h-2 rounded-full bg-blue-600 animate-bounce" style={{ animationDelay: "0ms" }} />
                                <span className="w-2 h-2 rounded-full bg-blue-600 animate-bounce" style={{ animationDelay: "150ms" }} />
                                <span className="w-2 h-2 rounded-full bg-blue-600 animate-bounce" style={{ animationDelay: "300ms" }} />
                              </div>
                              <span className="text-sm text-blue-600/80 font-semibold transition-all duration-500">
                                {chatLoadingStage}
                              </span>
                            </div>
                          </div>
                        </div>
                      )}

                      <div ref={messagesEndRef} />
                    </div>
                  )}
                </div>

                {/* Sources Used вЂ” collapsible strip */}
                {lastEvidence && lastEvidence.docs.length > 0 && (
                  <div className="border-t border-slate-200 bg-white/30 px-5 py-2.5 flex-shrink-0">
                    <div className="flex items-center gap-2 mb-1.5">
                      <FileText className="h-3 w-3 text-slate-400" />
                      <span className="text-[10px] font-mono font-bold text-slate-400 uppercase tracking-wider">Sources Used</span>
                    </div>
                    <div className="flex flex-wrap gap-1.5">
                      {lastEvidence.docs.slice(0, 3).map((doc, index) => (
                        <Button
                          key={doc.id}
                          size="sm"
                          variant="outline"
                          className="text-[11px] h-auto py-1 px-2.5 border border-slate-200 bg-white text-slate-500 hover:bg-blue-500/10 hover:border-blue-500/40 hover:text-blue-600 font-mono transition-all"
                          onClick={() => handleOpenDocument(doc.id)}
                        >
                          {index + 1}. {doc.title || doc.file_name || "Untitled"}
                        </Button>
                      ))}
                    </div>
                  </div>
                )}

                {/* Input bar вЂ” at bottom of card */}
                <div className="border-t-2 border-slate-200/15 bg-white/50 backdrop-blur-md p-4 flex-shrink-0">
                  {editingMessageId && (
                    <div className="flex items-center gap-2 mb-2 px-1">
                      <span className="text-xs font-mono text-blue-600/90">Editing message</span>
                      <button
                        type="button"
                        onClick={() => { setEditingMessageId(null); setInput(""); }}
                        className="text-xs font-mono text-slate-500 hover:text-slate-900 underline"
                      >
                        Cancel
                      </button>
                    </div>
                  )}
                  <div className="flex gap-3 items-end">
                    <div className="flex-1 relative">
                      <Textarea
                        value={input}
                        onChange={(e) => setInput(e.target.value)}
                        placeholder={editingMessageId ? "Edit your question and press Enter to resend..." : "Ask a question about your portfolio..."}
                        className="min-h-[52px] max-h-[180px] resize-none border border-slate-200/20 bg-white/[0.04] text-slate-900 placeholder:text-slate-400/35 font-mono rounded-xl pr-4 focus:border-blue-500/50 focus:ring-1 focus:ring-[#3b82f6]/20 transition-colors"
                        onKeyDown={(e) => {
                          if (e.key === "Enter" && !e.shiftKey && !(chatIsLoading || isClaudeLoading)) {
                            e.preventDefault();
                            addMessage();
                          }
                        }}
                      />
                    </div>
                    <Button 
                      onClick={(chatIsLoading || isClaudeLoading) ? stopGenerating : addMessage} 
                      disabled={!(chatIsLoading || isClaudeLoading) && !input.trim()}
                      size="lg"
                      title={(chatIsLoading || isClaudeLoading) ? "Cancel" : "Send"}
                      className={`h-[52px] w-[52px] p-0 font-bold border-2 rounded-xl transition-all ${
                        (chatIsLoading || isClaudeLoading)
                          ? "bg-white/[0.08] text-blue-600 border-slate-300 hover:bg-white/[0.14] hover:border-blue-500/50 hover:shadow-lg hover:shadow-blue-200/0.18)]"
                          : "bg-blue-600 text-slate-900 hover:bg-blue-600/90 border-blue-500 hover:shadow-[0_0_24px_rgba(59,130,246,0.4)]"
                      } disabled:opacity-40 disabled:hover:shadow-none`}
                    >
                      {(chatIsLoading || isClaudeLoading) ? (
                        <Square className="h-4 w-4 fill-current animate-pulse" />
                      ) : (
                        <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2.5} d="M12 19l9 2-9-18-9 18 9-2zm0 0v-8" />
                        </svg>
                      )}
                    </Button>
                  </div>
                  <div className="flex items-center justify-between mt-2.5">
                    <div className="flex items-center gap-2">
                      <button
                        type="button"
                        onClick={() => setWebSearchEnabled((prev) => !prev)}
                        className={`flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-mono font-bold transition-all border ${
                          webSearchEnabled
                            ? "border-blue-500/50 bg-blue-600/15 text-blue-600"
                            : "border-slate-200 bg-white text-slate-400 hover:border-slate-300 hover:text-slate-500"
                        }`}
                        title="Enable web search to find information about companies not in your documents"
                      >
                        <Globe className="h-3.5 w-3.5" />
                        Web Search {webSearchEnabled ? "ON" : "OFF"}
                      </button>
                      <button
                        type="button"
                        onClick={() => setMultiAgentEnabled((prev) => !prev)}
                        className={`flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-mono font-bold transition-all border ${
                          multiAgentEnabled
                            ? "border-blue-500/50 bg-blue-600/15 text-blue-600"
                            : "border-slate-200 bg-white text-slate-400 hover:border-slate-300 hover:text-slate-500"
                        }`}
                        title="Multi-Agent RAG: Orchestrator routes your query to Vector, Graph, KPI, and Web agents in parallel. When OFF, uses standard single-path RAG."
                      >
                        <Sparkles className="h-3.5 w-3.5" />
                        Multi-Agent {multiAgentEnabled ? "ON" : "OFF"}
                      </button>
                    </div>
                    <span className="text-[11px] text-slate-400 font-mono">
                      {chatIsLoading ? (
                        <span className="text-blue-600/70 flex items-center gap-1.5">
                          <Loader2 className="h-3 w-3 animate-spin" />
                          {chatLoadingStage}
                        </span>
                      ) : (
                        "Press Enter to send  вЂў  Shift+Enter for new line"
                      )}
                    </span>
                  </div>
                </div>
              </Card>
            </div>
          </TabsContent>

          {/* Sources Tab */}
          <TabsContent value="sources">
            <SourcesTab
              sources={sources}
              documents={documents}
              sourceFolders={sourceFolders}
              onCreateSource={handleCreateSource}
              onCreateFolder={handleCreateFolder}
              onDeleteFolderAndContents={handleDeleteFolderAndContents}
              onFolderCategoryUpdated={handleFolderCategoryUpdated}
              onSyncCategoriesFromDrive={handleSyncCategoriesFromDrive}
              onFoldersCategoriesSaved={handleFoldersCategoriesSaved}
              onDriveSyncConfigChanged={(folders) => {
                setInitialDriveSyncConfig((prev) => prev ? { ...prev, folders } : null);
              }}
              onDeleteSource={handleDeleteSource}
              getGoogleAccessToken={getGoogleAccessToken}
              onAutoLogDecision={handleAutoLogDecision}
              onDocumentSaved={(doc) =>
                setDocuments((prev) => [
                  { id: doc.id, title: doc.title, storage_path: doc.storage_path, folder_id: doc.folder_id },
                  ...prev,
                ])
              }
              activeEventId={activeEventId}
              ensureActiveEventId={ensureActiveEventId}
              currentUserId={profile?.id || user?.id || null}
              indexDocumentEmbeddings={indexDocumentEmbeddings}
              onRefreshCompanyCards={async () => {
                if (activeEventId) {
                  const res = await getAllEntityCards(activeEventId);
                  if (res.data) setCompanyCards(res.data as typeof companyCards);
                }
              }}
              initialDriveSyncConfig={initialDriveSyncConfig}
              onSourceFoldersRefetch={async () => {
                if (activeEventId) {
                  const { data } = await getSourceFoldersByEvent(activeEventId);
                  setSourceFolders((data || []) as SourceFolder[]);
                }
              }}
            />
          </TabsContent>

          {/* Admin Tab — invitation code + team members (MD/organizer only) */}
          {(profile?.role === "managing_partner" || profile?.role === "organizer" || profile?.role === "admin") && (
            <TabsContent value="admin">
              <div className="space-y-6 max-w-2xl">
                <div className="flex items-center gap-2 mb-2">
                  <Key className="h-5 w-5 text-blue-600" />
                  <h2 className="text-lg font-bold text-slate-900">Admin — invite your team</h2>
                </div>
                <p className="text-sm text-slate-500 mb-4">
                  Share the invitation code below with team members. They sign up, choose <strong>Team</strong> on the role screen, then enter the code to join your organization.
                </p>
                <TeamInvitationForm />
                {profile?.organization_id && (
                  <>
                    <TeamMembersList />
                    <SyncStatus />
                  </>
                )}
              </div>
            </TabsContent>
          )}

          {/* Dashboard Tab (task hub: MD assigns, team sees my tasks + Gantt) */}
          <TabsContent value="overview">
            <DashboardTab
              profile={profile}
              activeEventId={activeEventId}
              currentUserId={profile?.id || user?.id || null}
              tasks={tasks}
              onRefetchTasks={async () => {
                if (activeEventId) {
                  try {
                    const { data } = await getTasksByEvent(activeEventId);
                    setTasks((data || []) as Task[]);
                  } catch { /* tasks table may not exist yet */ }
                }
              }}
              decisions={decisions}
              documents={documents}
              sources={sources}
              companyCards={companyCards}
            />
          </TabsContent>

          {/* Decisions Tab */}
          <TabsContent value="decisions">
              <DecisionLoggerTab
              decisions={decisions}
              setDecisions={setDecisions}
              activeEventId={activeEventId}
              actorDefault={profile?.full_name || profile?.email || ""}
              draftDecision={draftDecision}
              onDraftConsumed={() => setDraftDecision(null)}
                draftDocumentId={draftDocumentId}
                onDraftDocumentConsumed={() => setDraftDocumentId(null)}
              documents={documents}
              onOpenDocument={handleOpenDocument}
                onOpenConverter={() => setActiveTab("sources")}
                currentUserId={profile?.id || user?.id || null}
            />
          </TabsContent>



              {/* Account Tab */}
              <TabsContent value="account">
                <Card className="border border-slate-200/20 bg-white/30 backdrop-blur-sm rounded-xl max-w-2xl">
                  <CardHeader className="border-b border-slate-200/15 bg-white/40">
                    <CardTitle className="text-lg font-bold text-slate-900 flex items-center gap-3">
                      <span className="flex h-10 w-10 items-center justify-center rounded-full bg-blue-600/20 text-blue-600 text-lg font-bold">
                        {(profile?.full_name || profile?.email || "U").charAt(0).toUpperCase()}
                      </span>
                      My Account
                    </CardTitle>
                  </CardHeader>
                  <CardContent className="p-6 space-y-5">
                    <div className="space-y-4">
                      <div className="flex items-center justify-between py-3 border-b border-slate-200">
                        <span className="text-sm text-slate-400 font-mono uppercase tracking-wider">Name</span>
                        <span className="text-sm text-slate-900 font-medium">{profile?.full_name || "вЂ”"}</span>
                      </div>
                      <div className="flex items-center justify-between py-3 border-b border-slate-200">
                        <span className="text-sm text-slate-400 font-mono uppercase tracking-wider">Email</span>
                        <span className="text-sm text-slate-900 font-medium">{profile?.email || "вЂ”"}</span>
                      </div>
                      <div className="flex items-center justify-between py-3 border-b border-slate-200">
                        <span className="text-sm text-slate-400 font-mono uppercase tracking-wider">Role</span>
                        <Badge className="bg-blue-600/20 text-blue-600 border-blue-500/40 text-xs font-semibold">
                          {profile?.role?.toUpperCase() || "MEMBER"}
                        </Badge>
                      </div>
                      {profile?.organization_id && (
                        <div className="flex items-center justify-between py-3 border-b border-slate-200">
                          <span className="text-sm text-slate-400 font-mono uppercase tracking-wider">Organization</span>
                          <span className="text-xs text-slate-500 font-mono">{profile.organization_id.slice(0, 8)}…</span>
                        </div>
                      )}
                      {profile?.organization_id && (
                        <>
                          <div className="space-y-2 py-3 border-b border-slate-200">
                            <Label className="text-sm text-slate-400 font-mono uppercase tracking-wider">Company name</Label>
                            {companyAccountLoading ? (
                              <div className="flex items-center gap-2 text-slate-500 text-sm"><Loader2 className="h-4 w-4 animate-spin" /> Loading…</div>
                            ) : (
                              <Input
                                value={companyAccountName}
                                onChange={(e) => setCompanyAccountName(e.target.value)}
                                placeholder="Your company name"
                                className="bg-white border-slate-200"
                              />
                            )}
                          </div>
                          <div className="space-y-2 py-3 border-b border-slate-200">
                            <Label className="text-sm text-slate-400 font-mono uppercase tracking-wider">Company description</Label>
                            {companyAccountLoading ? null : (
                              <Textarea
                                value={companyAccountDescription}
                                onChange={(e) => setCompanyAccountDescription(e.target.value)}
                                placeholder="Brief description of your company (used by AI context)"
                                rows={4}
                                className="bg-white border-slate-200 resize-none"
                              />
                            )}
                          </div>
                          {!companyAccountLoading && (
                            <div className="flex items-center gap-3 pt-2">
                              <Button
                                onClick={async () => {
                                  if (!profile?.organization_id) return;
                                  setCompanyAccountSaving(true);
                                  try {
                                    await setupCompany({
                                      organizationId: profile.organization_id,
                                      companyName: companyAccountName.trim() || undefined,
                                      companyDescription: companyAccountDescription.trim(),
                                    });
                                    toast({ title: "Saved", description: "Company details updated." });
                                  } catch (e) {
                                    toast({ title: "Failed to save", description: e instanceof Error ? e.message : "Unknown error", variant: "destructive" });
                                  } finally {
                                    setCompanyAccountSaving(false);
                                  }
                                }}
                                disabled={companyAccountSaving}
                                className="bg-blue-600 hover:bg-blue-700 text-white"
                              >
                                {companyAccountSaving ? <Loader2 className="h-4 w-4 mr-2 animate-spin" /> : <Save className="h-4 w-4 mr-2" />}
                                Save company details
                              </Button>
                            </div>
                          )}
                        </>
                      )}
                    </div>
                    <div className="flex items-center gap-3 pt-4">
                      <Button
                        onClick={signOut}
                        variant="outline"
                        className="border-red-500/40 text-red-400 hover:bg-red-500/20 hover:text-red-300 hover:border-red-500/60"
                      >
                        <LogOut className="h-4 w-4 mr-2" />
                        Log out
                      </Button>
                    </div>
                  </CardContent>
                </Card>
              </TabsContent>
            </Tabs>
          </div>
        </div>
      </div>

      {/* Document Viewer Modal */}
      <Dialog open={!!viewingDocument} onOpenChange={(open) => !open && setViewingDocument(null)}>
        <DialogContent className="max-w-4xl max-h-[90vh] overflow-hidden flex flex-col">
          <DialogHeader>
            <DialogTitle className="flex items-center justify-between gap-2">
              <span>{viewingDocument?.title || "Document Viewer"}</span>
              <Button variant="secondary" onClick={handleLogDecisionFromDocument}>
                Add decision
              </Button>
            </DialogTitle>
            <DialogDescription>
              {viewingDocument?.file_name && `File: ${viewingDocument.file_name}`}
            </DialogDescription>
          </DialogHeader>
          <div className="flex-1 overflow-auto space-y-4">
            <Tabs defaultValue="extracted" className="w-full">
              <TabsList className="grid w-full grid-cols-2">
                <TabsTrigger value="extracted">Extracted JSON</TabsTrigger>
                <TabsTrigger value="raw">Raw Content</TabsTrigger>
              </TabsList>
              <TabsContent value="extracted" className="mt-4">
                {viewingDocument?.extracted_json ? (
                  <div className="space-y-2">
                    <div className="text-sm text-slate-400 mb-2">
                      Structured data extracted by AI
                    </div>
                    <pre className="p-4 bg-white border border-slate-200 rounded-lg overflow-auto max-h-[500px] text-xs text-slate-900 font-mono">
                      {JSON.stringify(viewingDocument.extracted_json, null, 2)}
                    </pre>
                    <Button
                      size="sm"
                      variant="outline"
                      onClick={() => {
                        const blob = new Blob([JSON.stringify(viewingDocument?.extracted_json, null, 2)], {
                          type: "application/json",
                        });
                        const url = URL.createObjectURL(blob);
                        const a = document.createElement("a");
                        a.href = url;
                        a.download = `${viewingDocument?.title || "document"}-extracted.json`;
                        a.click();
                        URL.revokeObjectURL(url);
                      }}
                    >
                      <Download className="h-4 w-4 mr-2" />
                      Download JSON
                    </Button>
                  </div>
                ) : (
                  <div className="text-center py-8 text-slate-400">
                    <FileText className="h-12 w-12 mx-auto mb-4 opacity-50" />
                    <p>No extracted JSON available</p>
                  </div>
                )}
              </TabsContent>
              <TabsContent value="raw" className="mt-4">
                {viewingDocument?.raw_content ? (
                  <div className="space-y-2">
                    <div className="text-sm text-slate-400 mb-2">
                      Original text content ({viewingDocument.raw_content.length} characters)
                    </div>
                    <pre className="p-4 bg-slate-100 rounded-lg overflow-auto max-h-[500px] text-xs whitespace-pre-wrap">
                      {viewingDocument.raw_content}
                    </pre>
                    <Button
                      size="sm"
                      variant="outline"
                      onClick={() => {
                        const blob = new Blob([viewingDocument?.raw_content || ""], { type: "text/plain" });
                        const url = URL.createObjectURL(blob);
                        const a = document.createElement("a");
                        a.href = url;
                        a.download = `${viewingDocument?.title || "document"}-raw.txt`;
                        a.click();
                        URL.revokeObjectURL(url);
                      }}
                    >
                      <Download className="h-4 w-4 mr-2" />
                      Download Raw Text
                    </Button>
                  </div>
                ) : (
                  <div className="text-center py-8 text-slate-400">
                    <FileText className="h-12 w-12 mx-auto mb-4 opacity-50" />
                    <p>No raw content stored</p>
                    {viewingDocument?.storage_path && (
                      <Button
                        size="sm"
                        variant="outline"
                        className="mt-4"
                        onClick={async () => {
                          if (!viewingDocument?.storage_path) return;
                          const { data, error } = await supabase.storage
                            .from("cis-documents")
                            .createSignedUrl(viewingDocument.storage_path, 60);
                          if (error || !data?.signedUrl) {
                            toast({
                              title: "File not found",
                              description: "Could not access stored file.",
                              variant: "destructive",
                            });
                            return;
                          }
                          window.open(data.signedUrl, "_blank", "noopener,noreferrer");
                        }}
                      >
                        <Link2 className="h-4 w-4 mr-2" />
                        Open Stored File
                      </Button>
                    )}
                  </div>
                )}
              </TabsContent>
            </Tabs>
          </div>
        </DialogContent>
      </Dialog>

      {/* Log Decision Dialog - Create company connections from chat */}
      <Dialog open={logDecisionDialogOpen} onOpenChange={setLogDecisionDialogOpen}>
        <DialogContent className="sm:max-w-lg bg-white border border-slate-200">
          <DialogHeader>
            <DialogTitle className="flex items-center gap-2 text-slate-900 font-mono font-black uppercase">
              <Link2 className="h-5 w-5 text-blue-600" />
              Log Decision / Connection
            </DialogTitle>
            <DialogDescription className="text-slate-500 font-mono">
              Record a connection between two companies based on AI insight
            </DialogDescription>
          </DialogHeader>

          <LogDecisionForm
            documents={documents}
            pendingContext={pendingDecisionContext}
            onSubmit={handleCreateConnection}
            onCancel={() => {
              setLogDecisionDialogOpen(false);
              setPendingDecisionContext(null);
            }}
          />
        </DialogContent>
      </Dialog>
    </div>
  );
}

// Log Decision Form Component
function LogDecisionForm({
  documents,
  pendingContext,
  onSubmit,
  onCancel,
}: {
  documents: Array<{ id: string; title: string | null; storage_path: string | null }>;
  pendingContext: { aiReasoning: string; sourceDocIds?: string[] } | null;
  onSubmit: (data: {
    source_company_name: string;
    target_company_name: string;
    source_document_id?: string | null;
    target_document_id?: string | null;
    connection_type: ConnectionType;
    connection_status: ConnectionStatus;
    ai_reasoning?: string | null;
    notes?: string | null;
  }) => Promise<void>;
  onCancel: () => void;
}) {
  const [sourceCompany, setSourceCompany] = useState("");
  const [targetCompany, setTargetCompany] = useState("");
  const [sourceDocId, setSourceDocId] = useState<string>("none");
  const [targetDocId, setTargetDocId] = useState<string>("none");
  const [connectionType, setConnectionType] = useState<ConnectionType>("BD");
  const [connectionStatus, setConnectionStatus] = useState<ConnectionStatus>("To Connect");
  const [notes, setNotes] = useState("");
  const [editableRationale, setEditableRationale] = useState("");
  const [isSubmitting, setIsSubmitting] = useState(false);

  // Initialize editable rationale from AI context
  useEffect(() => {
    if (pendingContext?.aiReasoning) {
      setEditableRationale(pendingContext.aiReasoning.substring(0, 500));
    }
  }, [pendingContext?.aiReasoning]);

  // Extract company names from AI reasoning using known document titles + bold patterns
  useEffect(() => {
    if (!pendingContext?.aiReasoning) return;
    const text = pendingContext.aiReasoning;

    // Strategy 1: Match known company names from uploaded documents
    const knownNames = documents
      .map((d) => d.title?.trim())
      .filter((t): t is string => !!t && t.length > 1 && t.length < 60);

    // Find which known names appear in the AI text (case-insensitive)
    const foundNames = knownNames.filter((name) =>
      text.toLowerCase().includes(name.toLowerCase())
    );

    if (foundNames.length >= 2) {
      setSourceCompany(foundNames[0]);
      setTargetCompany(foundNames[1]);
      return;
    }

    // Strategy 2: Extract **bold** company names from markdown (Claude often bolds company names)
    const boldPattern = /\*\*([A-Z][a-zA-Z0-9 ]+?)\*\*/g;
    const boldMatches: string[] = [];
    let bm;
    while ((bm = boldPattern.exec(text)) !== null) {
      const name = bm[1].trim();
      // Skip common non-company bold words
      if (!/^(Note|Warning|Status|Connection|Type|Why|How|Key|Summary|Position|Duration|Schedule|Current|Potential|Recommended)$/i.test(name)) {
        boldMatches.push(name);
      }
    }
    if (boldMatches.length >= 2) {
      setSourceCompany(boldMatches[0]);
      setTargetCompany(boldMatches[1]);
      return;
    }

    // Strategy 3: Look for "X в†’ Y" or "X and Y" connection patterns
    const arrowMatch = text.match(/([A-Z][a-zA-Z0-9 ]+?)\s*[в†’в†’>]\s*([A-Z][a-zA-Z0-9 ]+?)(?:\s|$|\n|,|\()/);
    if (arrowMatch) {
      setSourceCompany(arrowMatch[1].trim());
      setTargetCompany(arrowMatch[2].trim());
      return;
    }

    // Strategy 4: Classic sentence patterns (broader than before)
    const sentencePatterns = [
      /connect(?:ing)?\s+([A-Z][a-zA-Z0-9 ]+?)\s+(?:to|with)\s+([A-Z][a-zA-Z0-9 ]+?)(?:\s|$|\n|,|\.)/i,
      /partner(?:ship)?\s+(?:between|with)\s+([A-Z][a-zA-Z0-9 ]+?)\s+(?:and|&)\s+([A-Z][a-zA-Z0-9 ]+?)(?:\s|$|\n|,|\.)/i,
      /([A-Z][a-zA-Z0-9 ]+?)\s+(?:could|should|would|can|might)\s+(?:partner|connect|collaborate|work)\s+with\s+([A-Z][a-zA-Z0-9 ]+?)(?:\s|$|\n|,|\.)/i,
      /introduce\s+([A-Z][a-zA-Z0-9 ]+?)\s+to\s+([A-Z][a-zA-Z0-9 ]+?)(?:\s|$|\n|,|\.)/i,
    ];
    for (const pattern of sentencePatterns) {
      const match = text.match(pattern);
      if (match) {
        setSourceCompany(match[1].trim());
        setTargetCompany(match[2].trim());
        return;
      }
    }

    // Strategy 5: If we found exactly 1 known name, use it as source
    if (foundNames.length === 1) {
      setSourceCompany(foundNames[0]);
      // Try to find one bold name that's different
      const other = boldMatches.find((b) => b.toLowerCase() !== foundNames[0].toLowerCase());
      if (other) setTargetCompany(other);
    }
  }, [pendingContext?.aiReasoning, documents]);

  const handleSubmit = async () => {
    if (!sourceCompany.trim() || !targetCompany.trim()) {
      return;
    }

    setIsSubmitting(true);
    try {
      await onSubmit({
        source_company_name: sourceCompany.trim(),
        target_company_name: targetCompany.trim(),
        source_document_id: sourceDocId === "none" ? null : sourceDocId,
        target_document_id: targetDocId === "none" ? null : targetDocId,
        connection_type: connectionType,
        connection_status: connectionStatus,
        ai_reasoning: editableRationale.trim() || pendingContext?.aiReasoning || null,
        notes: notes.trim() || null,
      });
    } finally {
      setIsSubmitting(false);
    }
  };

  return (
    <div className="space-y-4 py-4">
      <div className="grid grid-cols-2 gap-4">
        <div className="space-y-2">
          <Label className="text-slate-900 font-mono font-bold">Source Company</Label>
          <Input
            value={sourceCompany}
            onChange={(e) => setSourceCompany(e.target.value)}
            placeholder="e.g., Ridelink"
            className="border border-slate-200 bg-white text-slate-900 placeholder:text-slate-400 font-mono"
          />
        </div>
        <div className="space-y-2">
          <Label className="text-slate-900 font-mono font-bold">Target Company</Label>
          <Input
            value={targetCompany}
            onChange={(e) => setTargetCompany(e.target.value)}
            placeholder="e.g., Weego"
            className="border border-slate-200 bg-white text-slate-900 placeholder:text-slate-400 font-mono"
          />
        </div>
      </div>

      <div className="grid grid-cols-2 gap-4">
        <div className="space-y-2">
          <Label className="text-slate-900 font-mono font-bold">Source Document (optional)</Label>
          <Select value={sourceDocId} onValueChange={setSourceDocId}>
            <SelectTrigger className="border border-slate-200 bg-white text-slate-900 font-mono">
              <SelectValue placeholder="Link to document..." />
            </SelectTrigger>
            <SelectContent className="bg-white border border-slate-200 shadow-lg rounded-md">
              <SelectItem value="none" className="text-slate-900 font-mono">None</SelectItem>
              {documents.map((doc) => (
                <SelectItem key={doc.id} value={doc.id} className="text-slate-900 font-mono">
                  {doc.title || "Untitled"}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>
        <div className="space-y-2">
          <Label className="text-slate-900 font-mono font-bold">Target Document (optional)</Label>
          <Select value={targetDocId} onValueChange={setTargetDocId}>
            <SelectTrigger className="border border-slate-200 bg-white text-slate-900 font-mono">
              <SelectValue placeholder="Link to document..." />
            </SelectTrigger>
            <SelectContent className="bg-white border border-slate-200 shadow-lg rounded-md">
              <SelectItem value="none" className="text-slate-900 font-mono">None</SelectItem>
              {documents.filter((d) => d.id !== sourceDocId || sourceDocId === "none").map((doc) => (
                <SelectItem key={doc.id} value={doc.id} className="text-slate-900 font-mono">
                  {doc.title || "Untitled"}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>
      </div>

      <div className="grid grid-cols-2 gap-4">
        <div className="space-y-2">
          <Label className="text-slate-900 font-mono font-bold">Connection Type</Label>
          <Select value={connectionType} onValueChange={(v) => setConnectionType(v as ConnectionType)}>
            <SelectTrigger className="border border-slate-200 bg-white text-slate-900 font-mono">
              <SelectValue />
            </SelectTrigger>
            <SelectContent className="bg-white border border-slate-200 shadow-lg rounded-md">
              <SelectItem value="BD" className="text-slate-900 font-mono">
                <span className="flex items-center gap-2">
                  <span className="w-2 h-2 rounded-full bg-green-500" />
                  BD (Business Dev)
                </span>
              </SelectItem>
              <SelectItem value="INV" className="text-slate-900 font-mono">
                <span className="flex items-center gap-2">
                  <span className="w-2 h-2 rounded-full bg-blue-600" />
                  INV (Investment)
                </span>
              </SelectItem>
              <SelectItem value="Knowledge" className="text-slate-900 font-mono">
                <span className="flex items-center gap-2">
                  <span className="w-2 h-2 rounded-full bg-indigo-500" />
                  Knowledge
                </span>
              </SelectItem>
              <SelectItem value="Partnership" className="text-slate-900 font-mono">
                <span className="flex items-center gap-2">
                  <span className="w-2 h-2 rounded-full bg-purple-500" />
                  Partnership
                </span>
              </SelectItem>
              <SelectItem value="Project" className="text-slate-900 font-mono">
                <span className="flex items-center gap-2">
                  <span className="w-2 h-2 rounded-full bg-cyan-500" />
                  Project
                </span>
              </SelectItem>
            </SelectContent>
          </Select>
        </div>
        <div className="space-y-2">
          <Label className="text-slate-900 font-mono font-bold">Status</Label>
          <Select value={connectionStatus} onValueChange={(v) => setConnectionStatus(v as ConnectionStatus)}>
            <SelectTrigger className="border border-slate-200 bg-white text-slate-900 font-mono">
              <SelectValue />
            </SelectTrigger>
            <SelectContent className="bg-white border border-slate-200 shadow-lg rounded-md">
              <SelectItem value="To Connect" className="text-slate-900 font-mono">
                <span className="flex items-center gap-2">
                  <Clock className="h-3 w-3 text-yellow-500" />
                  To Connect
                </span>
              </SelectItem>
              <SelectItem value="In Progress" className="text-slate-900 font-mono">
                <span className="flex items-center gap-2">
                  <Loader2 className="h-3 w-3 text-blue-500" />
                  In Progress
                </span>
              </SelectItem>
              <SelectItem value="Connected" className="text-slate-900 font-mono">
                <span className="flex items-center gap-2">
                  <CheckCircle className="h-3 w-3 text-green-500" />
                  Connected
                </span>
              </SelectItem>
              <SelectItem value="Rejected" className="text-slate-900 font-mono">
                <span className="flex items-center gap-2">
                  <AlertTriangle className="h-3 w-3 text-red-500" />
                  Rejected
                </span>
              </SelectItem>
              <SelectItem value="Completed" className="text-slate-900 font-mono">
                <span className="flex items-center gap-2">
                  <CheckCircle className="h-3 w-3 text-emerald-500" />
                  Completed
                </span>
              </SelectItem>
            </SelectContent>
          </Select>
        </div>
      </div>

      <div className="space-y-2">
        <Label className="text-slate-900 font-mono font-bold">Additional Notes</Label>
        <Textarea
          value={notes}
          onChange={(e) => setNotes(e.target.value)}
          placeholder="Add any additional context..."
          className="border border-slate-200 bg-white text-slate-900 placeholder:text-slate-400 font-mono min-h-[80px]"
        />
      </div>

      {pendingContext?.aiReasoning && (
        <div className="space-y-2">
          <Label className="text-slate-900 font-mono font-bold text-xs">AI Rationale (editable вЂ” this is your decision record)</Label>
          <Textarea
            value={editableRationale}
            onChange={(e) => setEditableRationale(e.target.value)}
            className="border border-slate-200/30 bg-slate-50 text-slate-600 placeholder:text-slate-300 font-mono text-xs min-h-[80px] max-h-[150px]"
            placeholder="Edit the AI rationale to capture your reasoning..."
          />
          <p className="text-[10px] text-slate-400 font-mono">
            This rationale will be saved with the connection for your team's reference.
          </p>
        </div>
      )}

      <div className="flex justify-end gap-2 pt-2">
        <Button
          variant="outline"
          onClick={onCancel}
          className="border border-slate-200 bg-white text-slate-900 hover:bg-blue-500/10 font-mono"
        >
          Cancel
        </Button>
        <Button
          onClick={handleSubmit}
          disabled={isSubmitting || !sourceCompany.trim() || !targetCompany.trim()}
          className="bg-blue-600 text-slate-900 hover:bg-blue-600/80 font-bold border-2 border-blue-500"
        >
          {isSubmitting ? (
            <>
              <Loader2 className="h-4 w-4 mr-2 animate-spin" />
              Creating...
            </>
          ) : (
            <>
              <Link2 className="h-4 w-4 mr-2" />
              Log Connection
            </>
          )}
        </Button>
      </div>
    </div>
  );
}

