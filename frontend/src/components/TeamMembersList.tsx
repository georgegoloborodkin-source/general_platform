import { useState, useEffect, useRef } from "react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { useToast } from "@/hooks/use-toast";
import { supabase } from "@/integrations/supabase/client";
import { useAuth } from "@/hooks/useAuth";
import { Users, Mail, Calendar, UserX, Shield, RefreshCw } from "lucide-react";
import { Alert, AlertDescription } from "@/components/ui/alert";
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

interface TeamMember {
  id: string;
  email: string | null;
  full_name: string | null;
  role: string;
  created_at: string;
  organization_id: string;
}

export function TeamMembersList() {
  const { profile } = useAuth();
  const { toast } = useToast();
  const [teamMembers, setTeamMembers] = useState<TeamMember[]>([]);
  const [loading, setLoading] = useState(true);
  const [removingMemberId, setRemovingMemberId] = useState<string | null>(null);
  const [showRemoveDialog, setShowRemoveDialog] = useState(false);
  const [memberToRemove, setMemberToRemove] = useState<TeamMember | null>(null);
  const hasLoadedRef = useRef(false);

  const isMD = profile?.role === "managing_partner" || profile?.role === "organizer";
  const orgId = profile?.organization_id;

  useEffect(() => {
    // Only load once when component mounts and conditions are met
    if (isMD && orgId && !hasLoadedRef.current) {
      hasLoadedRef.current = true;
      loadTeamMembers();
    }
  }, [isMD, orgId]);

  const loadTeamMembers = async () => {
    if (!orgId) {
      setTeamMembers([]);
      setLoading(false);
      return;
    }

    setLoading(true);
    try {
      const { data, error } = await supabase
        .from("user_profiles")
        .select("id, email, full_name, role, created_at, organization_id")
        .eq("organization_id", orgId)
        .order("created_at", { ascending: false });

      if (error) {
        console.error("Supabase error loading team members:", error);
        setTeamMembers([]);
        // Defer toast to avoid render issues
        setTimeout(() => {
          toast({
            title: "Failed to load team members",
            description: error.message || "Please check your permissions or try refreshing.",
            variant: "destructive",
          });
        }, 0);
        return;
      }

      setTeamMembers((data as TeamMember[]) || []);
    } catch (err: any) {
      console.error("Unexpected error loading team members:", err);
      setTeamMembers([]);
      setTimeout(() => {
        toast({
          title: "Failed to load team members",
          description: err?.message || "An unexpected error occurred. Please try again.",
          variant: "destructive",
        });
      }, 0);
    } finally {
      setLoading(false);
    }
  };

  const handleRemoveMember = async () => {
    if (!memberToRemove || !orgId) return;

    try {
      // Use RPC function to remove team member (more secure and handles permissions)
      const { data, error } = await supabase.rpc("remove_team_member", {
        member_id: memberToRemove.id,
      });

      if (error) {
        throw error;
      }

      // Check if RPC returned success
      if (!data || !data.success) {
        throw new Error(data?.error || "Failed to remove team member");
      }

      toast({
        title: "Team member removed",
        description: `${memberToRemove.full_name || memberToRemove.email} has been removed from your fund. They will lose access immediately and cannot log back in to your organization.`,
      });

      // Reload team members - reset the ref to force reload
      hasLoadedRef.current = false;
      await loadTeamMembers();
      setShowRemoveDialog(false);
      setMemberToRemove(null);
    } catch (err: any) {
      console.error("Error removing team member:", err);
      toast({
        title: "Failed to remove team member",
        description: err.message || "Please try again. If the issue persists, check your permissions.",
        variant: "destructive",
      });
    }
  };

  const getRoleBadgeVariant = (role: string) => {
    switch (role) {
      case "managing_partner":
        return "default";
      case "organizer":
        return "secondary";
      case "team_member":
        return "outline";
      default:
        return "outline";
    }
  };

  const getRoleLabel = (role: string) => {
    switch (role) {
      case "managing_partner":
        return "Managing Partner";
      case "organizer":
        return "Organizer";
      case "team_member":
        return "Team Member";
      default:
        return role;
    }
  };

  if (!isMD || !orgId) {
    return null;
  }

  if (loading) {
    return (
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Users className="h-5 w-5" />
            Team Members
          </CardTitle>
          <CardDescription>Loading...</CardDescription>
        </CardHeader>
      </Card>
    );
  }

  // Calculate members directly - no need for useMemo here as filtering is cheap
  const mdMembers = teamMembers.filter((m) => m.role === "managing_partner" || m.role === "organizer");
  const regularMembers = teamMembers.filter((m) => m.role === "team_member");

  return (
    <>
      <Card>
        <CardHeader>
          <div className="flex items-center justify-between">
            <div>
              <CardTitle className="flex items-center gap-2">
                <Users className="h-5 w-5" />
                Team Members ({teamMembers.length})
              </CardTitle>
              <CardDescription>
                Manage your investment team members and their access
              </CardDescription>
            </div>
            <Button
              variant="outline"
              size="sm"
              onClick={() => {
                hasLoadedRef.current = false;
                loadTeamMembers();
              }}
              disabled={loading}
            >
              <RefreshCw className={`h-4 w-4 mr-1 ${loading ? "animate-spin" : ""}`} />
              Refresh
            </Button>
          </div>
        </CardHeader>
        <CardContent className="space-y-4">
          {teamMembers.length === 0 ? (
            <Alert>
              <AlertDescription>
                No team members yet. Share your invitation code to invite team members.
              </AlertDescription>
            </Alert>
          ) : (
            <>
              {mdMembers.length > 0 && (
                <div className="space-y-2">
                  <div className="flex items-center gap-2 text-sm font-medium text-muted-foreground">
                    <Shield className="h-4 w-4" />
                    Administrators ({mdMembers.length})
                  </div>
                  <div className="space-y-2">
                    {mdMembers.map((member) => (
                      <div
                        key={member.id}
                        className="flex items-center justify-between p-3 border rounded-md"
                      >
                        <div className="flex items-center gap-3">
                          <div className="flex flex-col">
                            <div className="flex items-center gap-2">
                              <span className="font-medium">
                                {member.full_name || "Unknown"}
                              </span>
                              <Badge variant={getRoleBadgeVariant(member.role)}>
                                {getRoleLabel(member.role)}
                              </Badge>
                            </div>
                            <div className="flex items-center gap-4 text-sm text-muted-foreground mt-1">
                              {member.email && (
                                <div className="flex items-center gap-1">
                                  <Mail className="h-3 w-3" />
                                  {member.email}
                                </div>
                              )}
                              <div className="flex items-center gap-1">
                                <Calendar className="h-3 w-3" />
                                Joined {new Date(member.created_at).toLocaleDateString()}
                              </div>
                            </div>
                          </div>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {regularMembers.length > 0 ? (
                <div className="space-y-2">
                  <div className="flex items-center gap-2 text-sm font-medium text-muted-foreground">
                    <Users className="h-4 w-4" />
                    Investment Team ({regularMembers.length})
                  </div>
                  <div className="space-y-2">
                    {regularMembers.map((member) => (
                      <div
                        key={member.id}
                        className="flex items-center justify-between p-3 border rounded-md"
                      >
                        <div className="flex items-center gap-3">
                          <div className="flex flex-col">
                            <div className="flex items-center gap-2">
                              <span className="font-medium">
                                {member.full_name || "Unknown"}
                              </span>
                              <Badge variant={getRoleBadgeVariant(member.role)}>
                                {getRoleLabel(member.role)}
                              </Badge>
                            </div>
                            <div className="flex items-center gap-4 text-sm text-muted-foreground mt-1">
                              {member.email && (
                                <div className="flex items-center gap-1">
                                  <Mail className="h-3 w-3" />
                                  {member.email}
                                </div>
                              )}
                              <div className="flex items-center gap-1">
                                <Calendar className="h-3 w-3" />
                                Joined {new Date(member.created_at).toLocaleDateString()}
                              </div>
                            </div>
                          </div>
                        </div>
                        <Button
                          variant="ghost"
                          size="sm"
                          onClick={() => {
                            setMemberToRemove(member);
                            setShowRemoveDialog(true);
                          }}
                          className="text-destructive hover:text-destructive"
                        >
                          <UserX className="h-4 w-4 mr-1" />
                          Remove
                        </Button>
                      </div>
                    ))}
                  </div>
                </div>
              ) : (
                <Alert>
                  <AlertDescription>
                    No investment team members yet. Share your invitation code with team members to invite them.
                  </AlertDescription>
                </Alert>
              )}
            </>
          )}
        </CardContent>
      </Card>

      <AlertDialog open={showRemoveDialog} onOpenChange={setShowRemoveDialog}>
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>Remove Team Member?</AlertDialogTitle>
            <AlertDialogDescription>
              Are you sure you want to remove{" "}
              <strong>{memberToRemove?.full_name || memberToRemove?.email}</strong> from your fund?
              They will lose access to all fund data and documents.
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel>Cancel</AlertDialogCancel>
            <AlertDialogAction
              onClick={handleRemoveMember}
              className="bg-destructive text-destructive-foreground hover:bg-destructive/90"
            >
              Remove
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
    </>
  );
}
