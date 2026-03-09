import { useState } from "react";
import { Link } from "react-router-dom";
import { motion } from "framer-motion";
import { Mail, Brain, BarChart3, Link2, TrendingUp, AlertTriangle } from "lucide-react";
import { Button } from "@/components/ui/button";
import WhitelistModal from "@/components/WhitelistModal";

const fadeUp = {
  hidden: { opacity: 0, y: 30 },
  visible: { opacity: 1, y: 0, transition: { duration: 0.6, ease: "easeOut" as const } },
};

const stagger = {
  visible: { transition: { staggerChildren: 0.15 } },
};

const Landing = () => {
  const [modalOpen, setModalOpen] = useState(false);

  return (
    <div className="min-h-screen bg-background text-foreground">
      <WhitelistModal open={modalOpen} onOpenChange={setModalOpen} />

      {/* Nav */}
      <nav className="fixed top-0 z-50 w-full border-b border-border/50 bg-background/80 backdrop-blur-md">
        <div className="mx-auto flex max-w-6xl items-center justify-between px-6 py-4">
          <span className="text-lg font-bold tracking-tight" style={{ fontFamily: "'Space Grotesk', sans-serif" }}>
            VentureOS
          </span>
          <div className="flex items-center gap-3">
            <Button variant="ghost" size="sm" asChild>
              <Link to="/login">Log in</Link>
            </Button>
            <Button size="sm" onClick={() => setModalOpen(true)}>Join Whitelist</Button>
          </div>
        </div>
      </nav>

      {/* Hero */}
      <section className="relative flex min-h-screen items-center justify-center overflow-hidden px-6 pt-20">
        <div className="pointer-events-none absolute top-1/3 left-1/2 -translate-x-1/2 -translate-y-1/2 h-[500px] w-[700px] rounded-full bg-primary/10 blur-[120px]" />
        <motion.div initial="hidden" animate="visible" variants={stagger} className="relative z-10 mx-auto max-w-3xl text-center">
          <motion.div variants={fadeUp} className="mb-6 inline-block rounded-full border border-border bg-secondary px-4 py-1.5 text-xs font-medium text-muted-foreground">
            Gmail integration · 2 min setup
          </motion.div>
          <motion.h1 variants={fadeUp} className="text-4xl font-bold leading-tight tracking-tight sm:text-6xl" style={{ fontFamily: "'Space Grotesk', sans-serif" }}>
            Proactive Cross-Portfolio{" "}
            <span className="text-primary">Alerts</span>
          </motion.h1>
          <motion.p variants={fadeUp} className="mx-auto mt-6 max-w-xl text-lg text-muted-foreground">
            We read everything your portfolio sends you. We tell you what you missed. VentureOS synthesizes portfolio updates across all your companies — surfacing connections, patterns, and early warnings.
          </motion.p>
          <motion.div variants={fadeUp} className="mt-8">
            <Button size="lg" className="px-8 text-base font-semibold" onClick={() => setModalOpen(true)}>
              Join Whitelist
            </Button>
          </motion.div>
        </motion.div>
      </section>

      {/* What We're Not */}
      <Section>
        <motion.div variants={fadeUp} className="mx-auto max-w-3xl rounded-2xl border border-border bg-secondary/50 p-8 text-center">
          <h2 className="mb-3 text-2xl font-bold" style={{ fontFamily: "'Space Grotesk', sans-serif" }}>What we're not</h2>
          <p className="text-muted-foreground leading-relaxed">
            Not a CRM. Not a dashboard. Not an LP reporting tool. You already have those. VentureOS is the layer that reads everything they can't — and finds what no one on your team has time to find.
          </p>
        </motion.div>
      </Section>

      {/* How It Works */}
      <Section>
        <SectionHeading>How It Works</SectionHeading>
        <motion.div variants={stagger} className="mx-auto mt-12 grid max-w-4xl gap-6 sm:grid-cols-3">
          {[
            { icon: Mail, step: "Step 1", title: "Email Ingestion", desc: "Auto-reads portfolio updates from your inbox — no manual input needed." },
            { icon: Brain, step: "Step 2", title: "AI Analysis", desc: "Cross-references patterns, connections, and warnings across all companies." },
            { icon: BarChart3, step: "Step 3", title: "Intelligence Report", desc: "Weekly or monthly reports delivered with actionable insights." },
          ].map((item) => (
            <motion.div key={item.title} variants={fadeUp} className="rounded-2xl border border-border bg-card p-6 text-center">
              <div className="mx-auto mb-4 flex h-12 w-12 items-center justify-center rounded-xl bg-primary/10 text-primary">
                <item.icon className="h-6 w-6" />
              </div>
              <p className="text-xs font-medium uppercase tracking-wider text-muted-foreground">{item.step}</p>
              <h3 className="mt-2 text-lg font-semibold" style={{ fontFamily: "'Space Grotesk', sans-serif" }}>{item.title}</h3>
              <p className="mt-2 text-sm text-muted-foreground">{item.desc}</p>
            </motion.div>
          ))}
        </motion.div>
      </Section>

      {/* Features */}
      <Section>
        <SectionHeading>Intelligence that compounds</SectionHeading>
        <p className="mx-auto mt-4 max-w-2xl text-center text-muted-foreground">
          Connecting information across emails, companies, and time — in ways humans can't do manually at scale.
        </p>
        <motion.div variants={stagger} className="mx-auto mt-12 grid max-w-5xl gap-6 sm:grid-cols-3">
          {[
            { icon: Link2, title: "Cross-Portfolio Connections", desc: "Find which companies should be talking to each other — for BD, knowledge sharing, or solving shared challenges." },
            { icon: TrendingUp, title: "Pattern Recognition", desc: "Spot hiring freezes, financial trends, and recurring challenges across your portfolio before anyone else." },
            { icon: AlertTriangle, title: "Early Warnings", desc: "Flag problems from portfolio updates before they become emergencies — from data sitting in your inbox for weeks." },
          ].map((item) => (
            <motion.div key={item.title} variants={fadeUp} className="rounded-2xl border border-border bg-card p-6">
              <div className="mb-4 flex h-10 w-10 items-center justify-center rounded-lg bg-primary/10 text-primary">
                <item.icon className="h-5 w-5" />
              </div>
              <h3 className="text-lg font-semibold" style={{ fontFamily: "'Space Grotesk', sans-serif" }}>{item.title}</h3>
              <p className="mt-2 text-sm text-muted-foreground leading-relaxed">{item.desc}</p>
            </motion.div>
          ))}
        </motion.div>
      </Section>

      {/* Sample Alert */}
      <Section>
        <motion.div variants={fadeUp} className="mx-auto max-w-3xl rounded-2xl border border-primary/30 bg-primary/5 p-8">
          <p className="mb-2 text-xs font-medium uppercase tracking-wider text-primary">Sample Alert · Cross-Portfolio Intelligence</p>
          <p className="text-lg leading-relaxed italic text-foreground/90">
            "3 of your portfolio companies mentioned hiring freezes this month — that's a pattern worth watching. Also, Company X just said they're looking for logistics partners in Bangladesh, and Company Y in your portfolio does exactly that. Here's the connection."
          </p>
          <p className="mt-4 text-sm text-muted-foreground">From an actual VentureOS report across 19 companies</p>
        </motion.div>
      </Section>

      {/* Problems We Solve */}
      <Section>
        <SectionHeading>Problems we solve</SectionHeading>
        <motion.div variants={stagger} className="mx-auto mt-12 grid max-w-5xl gap-6 sm:grid-cols-3">
          {[
            { title: "Monitoring at scale is overwhelming", desc: "Reading updates from 20+ companies and retaining context across all of them is impossible manually." },
            { title: "CRM data is dirty and disconnected", desc: "Your CRM tracks deals, not insights. Real context lives in emails and updates — unstructured and unsearchable." },
            { title: "Knowledge walks out the door", desc: "When team members leave, years of portfolio context disappear. Onboarding new hires takes months." },
          ].map((item) => (
            <motion.div key={item.title} variants={fadeUp} className="rounded-2xl border border-border bg-card p-6">
              <h3 className="text-base font-semibold" style={{ fontFamily: "'Space Grotesk', sans-serif" }}>{item.title}</h3>
              <p className="mt-2 text-sm text-muted-foreground leading-relaxed">{item.desc}</p>
            </motion.div>
          ))}
        </motion.div>
        <motion.div variants={fadeUp} className="mx-auto mt-8 max-w-3xl rounded-2xl border border-border bg-secondary/50 p-6 text-center">
          <p className="text-sm text-muted-foreground leading-relaxed">
            <span className="font-semibold text-foreground">Beyond portfolio monitoring</span> — VentureOS also captures dealflow progress, deal stages, legal work and holdups, introductions to other VCs and corporates — giving you a single source of truth across your entire operation.
          </p>
        </motion.div>
      </Section>

      {/* Final CTA */}
      <section className="relative py-32 px-6">
        <div className="pointer-events-none absolute bottom-0 left-1/2 -translate-x-1/2 h-[400px] w-[600px] rounded-full bg-primary/8 blur-[100px]" />
        <motion.div initial="hidden" whileInView="visible" viewport={{ once: true, margin: "-100px" }} variants={stagger} className="relative z-10 mx-auto max-w-2xl text-center">
          <motion.h2 variants={fadeUp} className="text-3xl font-bold sm:text-4xl" style={{ fontFamily: "'Space Grotesk', sans-serif" }}>
            See what you're missing.
          </motion.h2>
          <motion.p variants={fadeUp} className="mt-4 text-muted-foreground">
            Join VC teams already using VentureOS to unlock cross-portfolio intelligence.
          </motion.p>
          <motion.div variants={fadeUp} className="mt-8">
            <Button size="lg" className="px-8 text-base font-semibold" onClick={() => setModalOpen(true)}>
              Join Whitelist
            </Button>
          </motion.div>
          <motion.p variants={fadeUp} className="mt-4 text-sm text-muted-foreground">
            Free for up to 50 portfolio companies.
          </motion.p>
        </motion.div>
      </section>

      {/* Footer */}
      <footer className="border-t border-border px-6 py-8">
        <div className="mx-auto flex max-w-6xl flex-col items-center justify-between gap-4 text-sm text-muted-foreground sm:flex-row">
          <p>Built by a VC team, for VC teams.</p>
          <div className="flex items-center gap-4">
            <a href="mailto:hello@ventureos.com" className="hover:text-foreground transition-colors">hello@ventureos.com</a>
            <span>·</span>
            <a href="#" className="hover:text-foreground transition-colors">Privacy</a>
            <span>·</span>
            <a href="#" className="hover:text-foreground transition-colors">Terms</a>
          </div>
        </div>
      </footer>
    </div>
  );
};

const Section = ({ children }: { children: React.ReactNode }) => (
  <motion.section
    initial="hidden"
    whileInView="visible"
    viewport={{ once: true, margin: "-100px" }}
    variants={stagger}
    className="px-6 py-24"
  >
    {children}
  </motion.section>
);

const SectionHeading = ({ children }: { children: React.ReactNode }) => (
  <motion.h2 variants={fadeUp} className="text-center text-3xl font-bold sm:text-4xl" style={{ fontFamily: "'Space Grotesk', sans-serif" }}>
    {children}
  </motion.h2>
);

export default Landing;
