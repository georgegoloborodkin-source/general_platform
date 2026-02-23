import { useState, useEffect } from "react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import { Label } from "@/components/ui/label";
import { useToast } from "@/hooks/use-toast";
import { supabase } from "@/integrations/supabase/client";
import { useAuth } from "@/hooks/useAuth";
import { Key, Copy, CheckCircle, Users, Mail, UserPlus, Loader2 } from "lucide-react";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";

export function TeamInvitationForm() {
  const { profile } = useAuth();
  const { toast } = useToast();
  const [invitationCode, setInvitationCode] = useState<string | null>(null);
  const [copied, setCopied] = useState(false);
  const [loading, setLoading] = useState(true);
  
  // Email invitation state
  const [inviteEmail, setInviteEmail] = useState("");
  const [inviteRole, setInviteRole] = useState<"team_member" | "organizer" | "managing_partner">("team_member");
  const [isSendingInvite, setIsSendingInvite] = useState(false);

  const isMD = profile?.role === "managing_partner" || profile?.role === "organizer";
  const orgId = profile?.organization_id;

  useEffect(() => {
    if (isMD && orgId) {
      loadInvitationCode();
    }
  }, [isMD, orgId]);

  const loadInvitationCode = async () => {
    if (!orgId) return;

    setLoading(true);
    try {
      const { data, error } = await supabase
        .from("organizations")
        .select("invitation_code")
        .eq("id", orgId)
        .single();

      if (error) {
        throw error;
      }

      let code = data?.invitation_code || null;

      // If org has no invitation code (e.g. created via ensure_user_organization), generate one
      if (!code) {
        const { data: ensureData, error: ensureError } = await supabase.rpc("ensure_organization_invitation_code");
        if (!ensureError && ensureData?.invitation_code) {
          code = ensureData.invitation_code;
        }
      }

      setInvitationCode(code);
    } catch (err: any) {
      console.error("Error loading invitation code:", err);
      toast({
        title: "Failed to load invitation code",
        description: err.message || "Please try again.",
        variant: "destructive",
      });
    } finally {
      setLoading(false);
    }
  };

  const copyToClipboard = async () => {
    if (!invitationCode) return;
    await navigator.clipboard.writeText(invitationCode);
    setCopied(true);
    toast({
      title: "Copied!",
      description: "Invitation code copied to clipboard.",
    });
    setTimeout(() => setCopied(false), 2000);
  };

  if (!isMD || !orgId) {
    return null;
  }

  const handleSendEmailInvite = async () => {
    if (!inviteEmail.trim() || !orgId) {
      toast({
        title: "Missing information",
        description: "Please enter an email address.",
        variant: "destructive",
      });
      return;
    }

    // Basic email validation
    const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
    if (!emailRegex.test(inviteEmail.trim())) {
      toast({
        title: "Invalid email",
        description: "Please enter a valid email address.",
        variant: "destructive",
      });
      return;
    }

    setIsSendingInvite(true);
    try {
      const { data, error } = await supabase
        .from("invitations")
        .insert({
          organization_id: orgId,
          invited_by: profile?.id,
          email: inviteEmail.trim().toLowerCase(),
          role: inviteRole,
        })
        .select("token")
        .single();

      if (error) {
        throw error;
      }

      // Generate invitation URL
      const inviteUrl = `${window.location.origin}/invite/${data.token}`;

      toast({
        title: "Invitation sent!",
        description: `Send this link to ${inviteEmail}: ${inviteUrl}`,
      });

      // Copy to clipboard
      await navigator.clipboard.writeText(inviteUrl);
      
      setInviteEmail("");
      toast({
        title: "Link copied!",
        description: "Invitation link copied to clipboard. Share it with the invitee.",
      });
    } catch (err: any) {
      console.error("Error sending invitation:", err);
      toast({
        title: "Failed to send invitation",
        description: err.message || "Please try again.",
        variant: "destructive",
      });
    } finally {
      setIsSendingInvite(false);
    }
  };

  if (loading) {
    return (
      <Card>
        <CardHeader>
          <CardTitle>Invitation Code</CardTitle>
          <CardDescription>Loading...</CardDescription>
        </CardHeader>
      </Card>
    );
  }

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <Users className="h-5 w-5" />
          Invite Your Team
        </CardTitle>
        <CardDescription>
          Invite team members via code or email invitation
        </CardDescription>
      </CardHeader>
      <CardContent>
        <Tabs defaultValue="code" className="w-full">
          <TabsList className="grid w-full grid-cols-2">
            <TabsTrigger value="code">
              <Key className="h-4 w-4 mr-1" />
              Invitation Code
            </TabsTrigger>
            <TabsTrigger value="email">
              <Mail className="h-4 w-4 mr-1" />
              Email Invitation
            </TabsTrigger>
          </TabsList>

          <TabsContent value="code" className="space-y-4 mt-4">
            {invitationCode ? (
              <>
                <Alert>
                  <Key className="h-4 w-4" />
                  <AlertDescription>
                    Team members should select "Investment Team Member" during signup and enter this code.
                  </AlertDescription>
                </Alert>

                <div className="space-y-2">
                  <Label>Your Fund's Invitation Code</Label>
                  <div className="flex gap-2">
                    <Input
                      value={invitationCode}
                      readOnly
                      className="font-mono text-xl text-center font-bold tracking-wider"
                    />
                    <Button
                      variant="outline"
                      onClick={copyToClipboard}
                      className="min-w-[100px]"
                    >
                      {copied ? (
                        <>
                          <CheckCircle className="h-4 w-4 mr-1" />
                          Copied
                        </>
                      ) : (
                        <>
                          <Copy className="h-4 w-4 mr-1" />
                          Copy
                        </>
                      )}
                    </Button>
                  </div>
                  <p className="text-xs text-muted-foreground">
                    Share this code via email, Slack, or any communication channel. Team members will enter it when they sign up.
                  </p>
                </div>
              </>
            ) : (
              <Alert variant="destructive">
                <AlertDescription>
                  No invitation code found. Please contact support if this persists.
                </AlertDescription>
              </Alert>
            )}
          </TabsContent>

          <TabsContent value="email" className="space-y-4 mt-4">
            <Alert>
              <Mail className="h-4 w-4" />
              <AlertDescription>
                Send personalized email invitations. The invitee will receive a link to join your fund.
              </AlertDescription>
            </Alert>

            <div className="space-y-4">
              <div className="space-y-2">
                <Label>Email Address</Label>
                <Input
                  type="email"
                  placeholder="colleague@example.com"
                  value={inviteEmail}
                  onChange={(e) => setInviteEmail(e.target.value)}
                  disabled={isSendingInvite}
                />
              </div>

              <div className="space-y-2">
                <Label>Role</Label>
                <Select
                  value={inviteRole}
                  onValueChange={(v) => setInviteRole(v as typeof inviteRole)}
                  disabled={isSendingInvite}
                >
                  <SelectTrigger>
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="team_member">Investment Team Member</SelectItem>
                    <SelectItem value="organizer">Organizer</SelectItem>
                    {profile?.role === "managing_partner" && (
                      <SelectItem value="managing_partner">Managing Partner</SelectItem>
                    )}
                  </SelectContent>
                </Select>
                <p className="text-xs text-muted-foreground">
                  {inviteRole === "managing_partner" 
                    ? "Managing Partners have full access to manage the fund, invite members, and remove users."
                    : inviteRole === "organizer"
                    ? "Organizers can manage events and view team members."
                    : "Team members can upload documents, log decisions, and view fund data."}
                </p>
              </div>

              <Button
                onClick={handleSendEmailInvite}
                disabled={isSendingInvite || !inviteEmail.trim()}
                className="w-full"
              >
                {isSendingInvite ? (
                  <>
                    <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                    Sending...
                  </>
                ) : (
                  <>
                    <UserPlus className="h-4 w-4 mr-2" />
                    Send Invitation
                  </>
                )}
              </Button>
            </div>
          </TabsContent>
        </Tabs>
      </CardContent>
    </Card>
  );
}
