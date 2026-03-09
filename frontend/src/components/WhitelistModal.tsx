import { useState } from "react";
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogDescription } from "@/components/ui/dialog";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { toast } from "sonner";

interface WhitelistModalProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
}

const WhitelistModal = ({ open, onOpenChange }: WhitelistModalProps) => {
  const [name, setName] = useState("");
  const [email, setEmail] = useState("");
  const [firm, setFirm] = useState("");

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (!name.trim() || !email.trim() || !firm.trim()) {
      toast.error("Please fill out all fields.");
      return;
    }
    toast.success("You're on the list! We'll be in touch soon.");
    onOpenChange(false);
    setName("");
    setEmail("");
    setFirm("");
  };

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="bg-card border-border sm:max-w-md">
        <DialogHeader>
          <DialogTitle className="font-bold text-xl" style={{ fontFamily: "'Space Grotesk', sans-serif" }}>
            Join the VentureOS Whitelist
          </DialogTitle>
          <DialogDescription className="text-muted-foreground">
            Get early access to cross-portfolio intelligence.
          </DialogDescription>
        </DialogHeader>
        <form onSubmit={handleSubmit} className="space-y-4 pt-2">
          <div className="space-y-2">
            <Label htmlFor="name">Name</Label>
            <Input id="name" value={name} onChange={e => setName(e.target.value)} placeholder="Jane Doe" className="bg-secondary border-border" />
          </div>
          <div className="space-y-2">
            <Label htmlFor="email">Email</Label>
            <Input id="email" type="email" value={email} onChange={e => setEmail(e.target.value)} placeholder="jane@firm.vc" className="bg-secondary border-border" />
          </div>
          <div className="space-y-2">
            <Label htmlFor="firm">Firm</Label>
            <Input id="firm" value={firm} onChange={e => setFirm(e.target.value)} placeholder="Acme Ventures" className="bg-secondary border-border" />
          </div>
          <Button type="submit" className="w-full font-semibold">Join Whitelist</Button>
        </form>
      </DialogContent>
    </Dialog>
  );
};

export default WhitelistModal;
