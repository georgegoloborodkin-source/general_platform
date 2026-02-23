import { useState, useEffect, useCallback } from "react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { useToast } from "@/hooks/use-toast";
import { supabase } from "@/integrations/supabase/client";
import { useAuth } from "@/hooks/useAuth";
import { RefreshCw, CheckCircle, AlertCircle, Clock, Cloud, Database, Folder, Mail } from "lucide-react";
import { Alert, AlertDescription } from "@/components/ui/alert";

interface SyncConfig {
  id: string;
  source_type: "clickup" | "google_drive" | "gmail";
  config: {
    clickup_list_id?: string;
    google_drive_folder_id?: string;
    google_drive_folder_name?: string;
    gmail_query?: string;
    gmail_label_ids?: string[];
    max_emails_per_sync?: number;
    include_attachments?: boolean;
    sync_frequency?: string;
  };
  sync_frequency: string | null;
  last_sync_at: string | null;
  last_sync_status: "success" | "error" | "pending" | null;
  last_sync_error: string | null;
  next_sync_at: string | null;
  event_id: string;
  is_active: boolean;
}

export function SyncStatus() {
  const { profile } = useAuth();
  const { toast } = useToast();
  const [syncConfigs, setSyncConfigs] = useState<SyncConfig[]>([]);
  const [loading, setLoading] = useState(true);
  const [syncing, setSyncing] = useState<string | null>(null);

  const isMD = profile?.role === "managing_partner" || profile?.role === "organizer";
  const orgId = profile?.organization_id;

  const loadSyncConfigs = useCallback(async () => {
    if (!orgId) {
      setLoading(false);
      return;
    }
    try {
      const { data, error } = await supabase
        .from("sync_configurations")
        .select("*")
        .eq("organization_id", orgId)
        .eq("is_active", true);

      if (error) {
        console.warn("[SyncStatus] Failed to load configs:", error);
        setSyncConfigs([]);
      } else {
        setSyncConfigs((data as SyncConfig[]) || []);
      }
    } catch (err) {
      console.warn("[SyncStatus] Error loading sync configs:", err);
      setSyncConfigs([]);
    } finally {
      setLoading(false);
    }
  }, [orgId]);

  useEffect(() => {
    if (isMD && orgId) {
      loadSyncConfigs();
    } else {
      setLoading(false);
    }
  }, [isMD, orgId, loadSyncConfigs]);

  const triggerSync = async (configId: string, sourceType: "clickup" | "google_drive" | "gmail") => {
    setSyncing(configId);
    try {
      if (sourceType === "google_drive") {
        // Mark as pending — the actual sync is handled by SourcesTab's syncGoogleDriveFolder
        await supabase
          .from("sync_configurations")
          .update({ last_sync_status: "pending" })
          .eq("id", configId);

        toast({
          title: "Sync triggered",
          description: "Google Drive sync has been queued. Go to the Sources tab to see progress.",
        });
      } else if (sourceType === "gmail") {
        await supabase
          .from("sync_configurations")
          .update({ last_sync_status: "pending" })
          .eq("id", configId);

        toast({
          title: "Gmail sync triggered",
          description: "Gmail sync has been queued. Go to the Sources tab to see progress.",
        });
      } else if (sourceType === "clickup") {
        toast({
          title: "ClickUp sync",
          description: "ClickUp sync should be triggered from the Sources tab.",
        });
      }
      await loadSyncConfigs();
    } catch (err: any) {
      console.error("Sync error:", err);
      toast({
        title: "Sync failed",
        description: err.message || "Failed to trigger sync.",
        variant: "destructive",
      });
      await loadSyncConfigs();
    } finally {
      setSyncing(null);
    }
  };

  const getStatusBadge = (status: string | null) => {
    switch (status) {
      case "success":
        return <Badge variant="default" className="bg-green-600"><CheckCircle className="h-3 w-3 mr-1" />Synced</Badge>;
      case "error":
        return <Badge variant="destructive"><AlertCircle className="h-3 w-3 mr-1" />Error</Badge>;
      case "pending":
        return <Badge variant="secondary"><Clock className="h-3 w-3 mr-1" />Pending</Badge>;
      default:
        return <Badge variant="outline">Never synced</Badge>;
    }
  };

  const formatLastSync = (dateString: string | null) => {
    if (!dateString) return "Never";
    const date = new Date(dateString);
    const now = new Date();
    const diffMs = now.getTime() - date.getTime();
    const diffMins = Math.floor(diffMs / 60000);
    const diffHours = Math.floor(diffMs / 3600000);
    const diffDays = Math.floor(diffMs / 86400000);

    if (diffMins < 1) return "Just now";
    if (diffMins < 60) return `${diffMins}m ago`;
    if (diffHours < 24) return `${diffHours}h ago`;
    return `${diffDays}d ago`;
  };

  if (!isMD || !orgId) {
    return null;
  }

  if (loading) {
    return (
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Cloud className="h-5 w-5" />
            Sync Status
          </CardTitle>
          <CardDescription>Loading...</CardDescription>
        </CardHeader>
      </Card>
    );
  }

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <Cloud className="h-5 w-5" />
          Document Sync Status
        </CardTitle>
        <CardDescription>
          Monitor and manage automatic document synchronization
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-4">
        {syncConfigs.length === 0 ? (
          <Alert>
            <AlertDescription>
              No sync configurations found. Configure Google Drive sync in the Sources tab.
            </AlertDescription>
          </Alert>
        ) : (
          syncConfigs.map((config) => (
            <div key={config.id} className="border rounded-md p-4 space-y-3">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  {config.source_type === "clickup" ? (
                    <Database className="h-4 w-4 text-muted-foreground" />
                  ) : config.source_type === "gmail" ? (
                    <Mail className="h-4 w-4 text-muted-foreground" />
                  ) : (
                    <Cloud className="h-4 w-4 text-muted-foreground" />
                  )}
                  <span className="font-medium capitalize">
                    {config.source_type === "clickup" ? "ClickUp" : config.source_type === "gmail" ? "Gmail" : "Google Drive"}
                  </span>
                  {getStatusBadge(config.last_sync_status)}
                </div>
                <Button
                  variant="outline"
                  size="sm"
                  onClick={() => triggerSync(config.id, config.source_type)}
                  disabled={syncing === config.id}
                >
                  {syncing === config.id ? (
                    <>
                      <RefreshCw className="h-4 w-4 mr-1 animate-spin" />
                      Syncing...
                    </>
                  ) : (
                    <>
                      <RefreshCw className="h-4 w-4 mr-1" />
                      Sync Now
                    </>
                  )}
                </Button>
              </div>

              <div className="space-y-1 text-sm text-muted-foreground">
                {config.source_type === "google_drive" && config.config.google_drive_folder_name && (
                  <div className="flex items-center justify-between">
                    <span className="flex items-center gap-1">
                      <Folder className="h-3 w-3" />
                      Folder:
                    </span>
                    <span className="font-medium">{config.config.google_drive_folder_name}</span>
                  </div>
                )}
                {config.source_type === "gmail" && config.config.gmail_query && (
                  <div className="flex items-center justify-between">
                    <span className="flex items-center gap-1">
                      <Mail className="h-3 w-3" />
                      Query:
                    </span>
                    <span className="font-medium truncate max-w-[200px]">{config.config.gmail_query}</span>
                  </div>
                )}
                {config.source_type === "gmail" && (
                  <div className="flex items-center justify-between">
                    <span>Attachments:</span>
                    <span>{config.config.include_attachments ? "Included" : "Skipped"}</span>
                  </div>
                )}
                <div className="flex items-center justify-between">
                  <span>Last sync:</span>
                  <span>{formatLastSync(config.last_sync_at)}</span>
                </div>
                {config.sync_frequency && (
                  <div className="flex items-center justify-between">
                    <span>Frequency:</span>
                    <span className="capitalize">{config.sync_frequency === "on_login" ? "On login" : config.sync_frequency}</span>
                  </div>
                )}
                {config.last_sync_error && (
                  <Alert variant="destructive" className="mt-2">
                    <AlertCircle className="h-4 w-4" />
                    <AlertDescription className="text-xs">
                      {config.last_sync_error}
                    </AlertDescription>
                  </Alert>
                )}
              </div>
            </div>
          ))
        )}

        <Alert>
          <AlertDescription className="text-xs">
            <strong>Auto-sync:</strong> Google Drive folders and Gmail are automatically synced on login and every 15 minutes.
            Use the Sources tab to connect folders or Gmail, and trigger manual syncs.
            Re-login is normal when the token expires; reconnect via Sources if needed.
          </AlertDescription>
        </Alert>
      </CardContent>
    </Card>
  );
}
