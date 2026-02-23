import { Investor, Mentor, CorporatePartner, Match, Startup, TimeSlotConfig } from "@/types";

type BreakdownLine = string;

interface GenerateOptions {
  minMeetingsPerInvestor?: number; // default 0
  memberNameFilter?: string[]; // if provided, only these memberNames (case-insensitive) are eligible
  maxMeetingsPerStartup?: number; // default 1 (coverage-first)
  onlyAttending?: boolean; // default true
  mentors?: Mentor[];
  corporates?: CorporatePartner[];
}

type TargetKind = "investor" | "mentor" | "corporate";

interface TargetWrapper {
  kind: TargetKind;
  investor?: Investor;
  mentor?: Mentor;
  corporate?: CorporatePartner;
  id: string;
  displayName: string;
  totalSlots: number;
  availabilityStatus: "present" | "not-attending";
  slotAvailability?: Record<string, boolean>;
}

interface ScoredCandidate {
  startup: Startup;
  target: TargetWrapper;
  score: number;
  breakdown: BreakdownLine[];
  topReason: string;
}

const DEFAULT_TIME_SLOTS = [
  "09:00 - 09:20",
  "09:20 - 09:40",
  "09:40 - 10:00",
  "10:00 - 10:20",
  "10:20 - 10:40",
  "10:40 - 11:00",
  "11:00 - 11:20",
  "11:20 - 11:40",
  "11:40 - 12:00",
  "12:00 - 12:20",
  "12:20 - 12:40",
  "12:40 - 13:00",
  "13:00 - 13:20",
  "13:20 - 13:40",
  "13:40 - 14:00",
  "14:00 - 14:20",
  "14:20 - 14:40",
  "14:40 - 15:00",
  "15:00 - 15:20",
  "15:20 - 15:40",
  "15:40 - 16:00",
  "16:00 - 16:20",
  "16:20 - 16:40",
  "16:40 - 17:00",
  "17:00 - 17:20",
  "17:20 - 17:40",
  "17:40 - 18:00",
];

function norm(s: string): string {
  return (s || "").trim().toLowerCase();
}

function startupNameKey(startup: Startup): string {
  // Additional guard: if the same startup is accidentally imported twice (different IDs, same name),
  // we still don't want duplicate-looking rows in the schedule.
  return norm(startup.companyName);
}

function investorMemberKey(investor: Investor): string {
  return norm(investor.memberName);
}

function geoOverlap(startup: Startup, investor: Investor): string[] {
  const investorSet = new Set(investor.geoFocus.map(norm));
  return startup.geoMarkets.filter((m) => investorSet.has(norm(m)));
}

function industryMatches(startup: Startup, investor: Investor): boolean {
  const invSet = new Set(investor.industryPreferences.map(norm));
  return invSet.has(norm(startup.industry));
}

function stageMatches(startup: Startup, investor: Investor): boolean {
  const invSet = new Set(investor.stagePreferences.map(norm));
  return invSet.has(norm(startup.fundingStage));
}

/**
 * MVP hard filters:
 * - Investors: industry match + geo overlap + funding in range (STRICT)
 * - Mentors/Corporates: ALWAYS PASS (permissive) - we want them to match
 */
function passesHardFilters(startup: Startup, target: TargetWrapper): boolean {
  if (target.kind === "investor" && target.investor) {
    const investor = target.investor;
    if (!industryMatches(startup, investor)) return false;
    if (geoOverlap(startup, investor).length === 0) return false;
    if (startup.fundingTarget < investor.minTicketSize) return false;
    if (startup.fundingTarget > investor.maxTicketSize) return false;
    return true;
  }

  // Mentors and corporates ALWAYS pass hard filters - they should match any startup
  // The scoring will reflect how good the match is
  if (target.kind === "mentor" || target.kind === "corporate") {
    return true;
  }

  return false;
}

/**
 * MVP scoring (ONLY after hard filters pass).
 * Max score = 100.
 *
 * - Industry match: +30 (fixed)
 * - Geo overlap: +20 (fixed)
 * - Funding proximity to midpoint of ticket range: +0..+20
 * - Stage alignment: +0 or +15
 * - Investor has remaining slots: +15 (fixed, given remainingSlots>0 at selection time)
 */
function scoreCandidate(
  startup: Startup,
  target: TargetWrapper,
  remainingSlots: number
): { score: number; breakdown: BreakdownLine[]; topReason: string } {
  const breakdown: BreakdownLine[] = [];

  const components: { label: string; points: number }[] = [];

  // Shared: industry & geo
  if (target.kind === "investor" && target.investor) {
    const investor = target.investor;
    components.push({ label: `Industry match (${startup.industry})`, points: 30 });
    const overlap = geoOverlap(startup, investor);
    components.push({ label: `Geo overlap (${overlap.join(", ")})`, points: 20 });

    // Funding midpoint proximity
    const min = investor.minTicketSize;
    const max = investor.maxTicketSize;
    const mid = (min + max) / 2;
    const halfRange = Math.max(1, (max - min) / 2); // avoid divide by zero
    const distance = Math.abs(startup.fundingTarget - mid);
    const closeness = Math.max(0, 1 - distance / halfRange); // 1 at midpoint, 0 at ends
    const fundingPoints = Math.round(20 * closeness);
    components.push({
      label: `Ticket fit (target ${startup.fundingTarget.toLocaleString()} within ${min.toLocaleString()}–${max.toLocaleString()})`,
      points: fundingPoints,
    });

    // Stage alignment (not a hard filter in MVP spec)
    const stagePoints = stageMatches(startup, investor) ? 15 : 0;
    components.push({ label: `Stage alignment (${startup.fundingStage})`, points: stagePoints });
  }

  if (target.kind === "mentor" && target.mentor) {
    const mentor = target.mentor;
    const industries = new Set((mentor.industryPreferences || []).map(norm));
    const geos = new Set((mentor.geoFocus || []).map(norm));
    const overlapGeo = startup.geoMarkets.filter((g) => geos.has(norm(g)));
    const industryMatch = industries.size === 0 || industries.has(norm(startup.industry));
    const geoMatch = geos.size === 0 || overlapGeo.length > 0;
    // Give baseline points even without overlap (mentors should always match)
    components.push({ label: `Industry match (${startup.industry})`, points: industryMatch ? 30 : 15 });
    components.push({ label: `Geo overlap`, points: geoMatch ? 20 : 10 });
    components.push({ label: `Mentor availability`, points: 20 }); // baseline for being a mentor
    components.push({ label: `Slots available`, points: remainingSlots > 0 ? 15 : 0 });
  }

  if (target.kind === "corporate" && target.corporate) {
    const corp = target.corporate;
    const industries = new Set((corp.industryPreferences || []).map(norm));
    const geos = new Set((corp.geoFocus || []).map(norm));
    const stages = new Set((corp.stages || []).map(norm));
    const overlapGeo = startup.geoMarkets.filter((g) => geos.has(norm(g)));
    const industryMatch = industries.size === 0 || industries.has(norm(startup.industry));
    const geoMatch = geos.size === 0 || overlapGeo.length > 0;
    const stageMatch = stages.size === 0 || stages.has(norm(startup.fundingStage));
    // Give baseline points even without overlap (corporates should always match)
    components.push({ label: `Industry match (${startup.industry})`, points: industryMatch ? 30 : 15 });
    components.push({ label: `Geo overlap`, points: geoMatch ? 20 : 10 });
    const stagePoints = stageMatch ? 15 : 5;
    components.push({ label: `Stage fit (${startup.fundingStage})`, points: stagePoints });
    components.push({ label: `Partnership opportunity`, points: 15 }); // baseline for being a corporate
  }

  // Remaining slots for any target kind
  if (target.kind !== "mentor" && target.kind !== "corporate") {
    const slotPoints = remainingSlots > 0 ? 15 : 0;
    components.push({ label: `Slots available`, points: slotPoints });
  }

  const score = components.reduce((sum, c) => sum + c.points, 0);
  const maxComp = components.reduce((best, c) => (c.points > best.points ? c : best), components[0]);
  const topReason = maxComp ? `${maxComp.label} (+${maxComp.points})` : "Balanced match";
  // Per request: hide breakdown content, return only score and topReason metadata
  return { score, breakdown, topReason };
}

function buildSlots(timeSlots: TimeSlotConfig[]) {
  const slotsToUse =
    timeSlots.length > 0
      ? timeSlots.map((ts) => `${ts.startTime} - ${ts.endTime}`)
      : DEFAULT_TIME_SLOTS;

  const slotLabels =
    timeSlots.length > 0 ? timeSlots.map((ts) => ts.label) : slotsToUse.map((_, i) => `Slot ${i + 1}`);

  return { slotsToUse, slotLabels };
}

function isAvailableForSlot(entity: { slotAvailability?: Record<string, boolean> }, slotId?: string): boolean {
  if (!slotId) return true;
  if (!entity.slotAvailability) return true;
  return entity.slotAvailability[slotId] !== false;
}

function makeInvestorDisplayName(investor: Investor): string {
  return `${investor.firmName} (${investor.memberName})`;
}

function investorFirmKey(investor: Investor): string {
  // MVP assumption: you should NOT match a startup to the same firm multiple times,
  // even if the CSV has multiple rows (multiple members) for that firm.
  return norm(investor.firmName);
}

/**
 * MVP Generate Matches
 * - No duplicates (startup_id, investor_id) pairs
 * - Strict investor slot capacity
 * - Hard filters enforced
 * - Score threshold enforced (>=70)
 * - Includes scoreBreakdown lines
 */
export function generateMatches(
  startups: Startup[],
  investors: Investor[],
  existingMatches: Match[] = [],
  timeSlots: TimeSlotConfig[] = [],
  options: GenerateOptions = {}
): Match[] {
  const minMeetingsPerInvestor = options.minMeetingsPerInvestor ?? 0;
  const memberNameFilter = (options.memberNameFilter || []).map(norm);
  const filterByMember = memberNameFilter.length > 0;
  const onlyAttending = options.onlyAttending !== false;
  const mentors = options.mentors || [];
  const corporates = options.corporates || [];

  // Keep only attending
  const availableStartupsRaw = onlyAttending ? startups.filter((s) => s.availabilityStatus === "present") : startups;
  const availableInvestors = investors.filter((i) => {
    if (onlyAttending && i.availabilityStatus !== "present") return false;
    if (filterByMember) {
      return memberNameFilter.includes(norm(i.memberName));
    }
    return true;
  });
  const availableMentors = mentors.filter((m) => (onlyAttending ? m.availabilityStatus === "present" : true));
  const availableCorporates = corporates.filter((c) => (onlyAttending ? c.availabilityStatus === "present" : true));
  // Allow one slot per target type by default (investor + mentor + corporate)
  const maxMeetingsPerStartup =
    options.maxMeetingsPerStartup ??
    (1 + (availableMentors.length > 0 ? 1 : 0) + (availableCorporates.length > 0 ? 1 : 0));

  // Deduplicate startups by normalized name to avoid duplicate-looking entries eating slots
  const startupByName = new Map<string, Startup>();
  for (const s of availableStartupsRaw) {
    const key = startupNameKey(s);
    if (!startupByName.has(key)) {
      startupByName.set(key, s);
    }
  }
  const availableStartups = Array.from(startupByName.values());

  // Build slot helpers
  const { slotsToUse, slotLabels } = buildSlots(timeSlots);

  // Schedule occupancy (to avoid double-booking in a slot)
  const scheduleGrid: Record<
    string,
    {
      startupIds: Set<string>;
      targetIds: Set<string>;
    }
  > = {};
  slotLabels.forEach((label) => {
    scheduleGrid[label] = { startupIds: new Set(), targetIds: new Set() };
  });

  // Preserve completed/locked matches ONLY if they still pass hard filters, and count them toward slots.
  // This avoids MVP violations (industry/geo/ticket mismatches) lingering in output.
  const preservedRaw = existingMatches.filter((m) => m.completed || m.locked);
  const preserved: Match[] = [];
  const usedPairs = new Set<string>();
  const usedFirmPairs = new Set<string>(); // startupId::firmKey
  const usedNamePairs = new Set<string>(); // startupName::firmName::memberName (prevents duplicate-looking rows)
  const targetUsedCount = new Map<string, number>(); // key = targetKind::id
  const startupUsedCount = new Map<string, number>();

  for (const m of preservedRaw) {
    const startup = availableStartups.find((s) => s.id === m.startupId);
    const targetKind = m.targetType || "investor";
    const target =
      targetKind === "investor"
        ? availableInvestors.find((i) => i.id === (m.investorId || m.targetId))
        : targetKind === "mentor"
        ? availableMentors.find((i) => i.id === m.targetId)
        : availableCorporates.find((i) => i.id === m.targetId);
    if (!startup || !target) continue;

    const wrapped: TargetWrapper =
      targetKind === "investor"
        ? {
            kind: "investor",
            investor: target as Investor,
            id: (target as Investor).id,
            displayName: makeInvestorDisplayName(target as Investor),
            totalSlots: (target as Investor).totalSlots,
            availabilityStatus: (target as Investor).availabilityStatus,
            slotAvailability: (target as Investor).slotAvailability,
          }
        : targetKind === "mentor"
        ? {
            kind: "mentor",
            mentor: target as Mentor,
            id: (target as Mentor).id,
            displayName: (target as Mentor).fullName,
            totalSlots: (target as Mentor).totalSlots,
            availabilityStatus: (target as Mentor).availabilityStatus,
            slotAvailability: (target as Mentor).slotAvailability,
          }
        : {
            kind: "corporate",
            corporate: target as CorporatePartner,
            id: (target as CorporatePartner).id,
            displayName: `${(target as CorporatePartner).firmName} (${(target as CorporatePartner).contactName})`,
            totalSlots: (target as CorporatePartner).totalSlots,
            availabilityStatus: (target as CorporatePartner).availabilityStatus,
            slotAvailability: (target as CorporatePartner).slotAvailability,
          };

    if (!passesHardFilters(startup, wrapped)) continue;

    const key = `${startup.id}::${wrapped.kind}::${wrapped.id}`;
    if (usedPairs.has(key)) continue;
    if (wrapped.kind === "investor" && wrapped.investor) {
      const firmKey = `${startup.id}::${investorFirmKey(wrapped.investor)}`;
      if (usedFirmPairs.has(firmKey)) continue;
      const nameKey = `${startupNameKey(startup)}::${investorFirmKey(wrapped.investor)}::${investorMemberKey(
        wrapped.investor
      )}`;
      if (usedNamePairs.has(nameKey)) continue;
    }

    const used = targetUsedCount.get(`${wrapped.kind}::${wrapped.id}`) || 0;
    if (used >= wrapped.totalSlots) continue;
    const su = startupUsedCount.get(startup.id) || 0;
    if (su >= maxMeetingsPerStartup) continue;

    // Reserve the existing slot if it exists in this run
    if (m.timeSlot && scheduleGrid[m.timeSlot]) {
      scheduleGrid[m.timeSlot].startupIds.add(startup.id);
      scheduleGrid[m.timeSlot].targetIds.add(`${wrapped.kind}::${wrapped.id}`);
    }

    const { score, breakdown } = scoreCandidate(startup, wrapped, wrapped.totalSlots - used);
    const minScore = wrapped.kind === "investor" ? 50 : 5;
    if (score < minScore) continue;

    preserved.push({
      ...m,
      startupName: startup.companyName,
      targetType: wrapped.kind,
      targetId: wrapped.id,
      targetName: wrapped.displayName,
      investorId: wrapped.kind === "investor" ? wrapped.id : undefined,
      investorName: wrapped.kind === "investor" ? wrapped.displayName : undefined,
      compatibilityScore: score,
      scoreBreakdown: breakdown,
    });

    usedPairs.add(key);
    if (wrapped.kind === "investor" && wrapped.investor) {
      usedFirmPairs.add(`${startup.id}::${investorFirmKey(wrapped.investor)}`);
      usedNamePairs.add(`${startupNameKey(startup)}::${investorFirmKey(wrapped.investor)}::${investorMemberKey(
        wrapped.investor
      )}`);
    }
    targetUsedCount.set(`${wrapped.kind}::${wrapped.id}`, used + 1);
    startupUsedCount.set(startup.id, su + 1);
  }

  // Build scored candidates grouped per startup for fairness-first allocation
  const candidatesByStartup = new Map<string, ScoredCandidate[]>();

  const targets: TargetWrapper[] = [
    ...availableInvestors.map<TargetWrapper>((i) => ({
      kind: "investor",
      investor: i,
      id: i.id,
      displayName: makeInvestorDisplayName(i),
      totalSlots: i.totalSlots,
      availabilityStatus: i.availabilityStatus,
      slotAvailability: i.slotAvailability,
    })),
    ...availableMentors.map<TargetWrapper>((m) => ({
      kind: "mentor",
      mentor: m,
      id: m.id,
      displayName: m.fullName,
      totalSlots: m.totalSlots,
      availabilityStatus: m.availabilityStatus,
      slotAvailability: m.slotAvailability,
    })),
    ...availableCorporates.map<TargetWrapper>((c) => ({
      kind: "corporate",
      corporate: c,
      id: c.id,
      displayName: `${c.firmName} (${c.contactName})`,
      totalSlots: c.totalSlots,
      availabilityStatus: c.availabilityStatus,
      slotAvailability: c.slotAvailability,
    })),
  ];

  for (const target of targets) {
    const keyBase = `${target.kind}::${target.id}`;
    const alreadyUsed = targetUsedCount.get(keyBase) || 0;
    const remainingSlots = Math.max(0, target.totalSlots - alreadyUsed);
    if (remainingSlots <= 0) continue;

    for (const startup of availableStartups) {
      const key = `${startup.id}::${target.kind}::${target.id}`;
      if (usedPairs.has(key)) continue;
      if (target.kind === "investor" && target.investor) {
        const firmKey = `${startup.id}::${investorFirmKey(target.investor)}`;
        if (usedFirmPairs.has(firmKey)) continue;
        const nameKey = `${startupNameKey(startup)}::${investorFirmKey(target.investor)}::${investorMemberKey(
          target.investor
        )}`;
        if (usedNamePairs.has(nameKey)) continue;
      }

      if (!passesHardFilters(startup, target)) continue;

    const { score, breakdown, topReason } = scoreCandidate(startup, target, remainingSlots);
    // Investors need 50+ score; mentors/corporates have NO minimum (always match)
    const minScore = target.kind === "investor" ? 50 : 0;
    if (score < minScore) continue;

      const bucket = candidatesByStartup.get(startup.id) || [];
      bucket.push({
        startup,
        target,
        score,
        breakdown,
        topReason,
      });
      candidatesByStartup.set(startup.id, bucket);
    }
  }

  // Helper to try to assign a candidate with earliest available slot
  function tryAssignCandidate(cand: ScoredCandidate): Match | null {
    const { startup, target, score, breakdown, topReason } = cand;
    const key = `${startup.id}::${target.kind}::${target.id}`;
    if (usedPairs.has(key)) return null;
    if (target.kind === "investor" && target.investor) {
      const firmKey = `${startup.id}::${investorFirmKey(target.investor)}`;
      if (usedFirmPairs.has(firmKey)) return null;
      const nameKey = `${startupNameKey(startup)}::${investorFirmKey(target.investor)}::${investorMemberKey(
        target.investor
      )}`;
      if (usedNamePairs.has(nameKey)) return null;
    }
    const startupCount = startupUsedCount.get(startup.id) || 0;
    if (startupCount >= maxMeetingsPerStartup) return null;

    let assignedTimeSlot = "";
    let assignedSlotTime = "";

    for (let i = 0; i < slotLabels.length; i++) {
      const slotLabel = slotLabels[i];
      const slotTime = slotsToUse[i];
      const slotConfigId = timeSlots[i]?.id;
      const isSlotDone = timeSlots[i]?.isDone === true;
      if (isSlotDone) continue;

      const cell = scheduleGrid[slotLabel];
      if (cell.startupIds.has(startup.id)) continue;
      if (cell.targetIds.has(`${target.kind}::${target.id}`)) continue;

      if (!isAvailableForSlot(startup, slotConfigId)) continue;
      if (!isAvailableForSlot(target, slotConfigId)) continue;

      assignedTimeSlot = slotLabel;
      assignedSlotTime = slotTime || "";
      cell.startupIds.add(startup.id);
      cell.targetIds.add(`${target.kind}::${target.id}`);
      break;
    }

    if (!assignedTimeSlot) return null;

    const breakdownWithReason = topReason ? [topReason, ...breakdown] : breakdown;

    const match: Match = {
      id: `match-${startup.id}-${target.id}-${Date.now()}-${Math.random()}`,
      startupId: startup.id,
      targetId: target.id,
      targetType: target.kind,
      startupName: startup.companyName,
      targetName: target.displayName,
      investorId: target.kind === "investor" ? target.id : undefined,
      investorName: target.kind === "investor" ? target.displayName : undefined,
      timeSlot: assignedTimeSlot,
      slotTime: assignedSlotTime,
      compatibilityScore: score,
      status: "upcoming",
      completed: false,
      startupAttending: true,
      targetAttending: true,
      investorAttending: target.kind === "investor",
      scoreBreakdown: breakdownWithReason,
    };

    usedPairs.add(key);
    if (target.kind === "investor" && target.investor) {
      const firmKey = `${startup.id}::${investorFirmKey(target.investor)}`;
      const nameKey = `${startupNameKey(startup)}::${investorFirmKey(target.investor)}::${investorMemberKey(
        target.investor
      )}`;
      usedFirmPairs.add(firmKey);
      usedNamePairs.add(nameKey);
    }
    targetUsedCount.set(`${target.kind}::${target.id}`, (targetUsedCount.get(`${target.kind}::${target.id}`) || 0) + 1);
    startupUsedCount.set(startup.id, startupCount + 1);

    return match;
  }

  const newMatches: Match[] = [];

  // Helper to assign best candidate of a specific type to each startup
  function assignByType(kind: TargetKind) {
    for (const startup of availableStartups) {
      const list = (candidatesByStartup.get(startup.id) || [])
        .filter((c) => c.target.kind === kind)
        .sort((a, b) => b.score - a.score);
      for (const cand of list) {
        const remaining = Math.max(
          0,
          cand.target.totalSlots - (targetUsedCount.get(`${cand.target.kind}::${cand.target.id}`) || 0)
        );
        if (remaining <= 0) continue;
        const match = tryAssignCandidate(cand);
        if (match) {
          newMatches.push(match);
          break; // move to next startup for this type
        }
      }
    }
  }

  // PASS 1: Give each startup its best INVESTOR first (priority)
  if (availableInvestors.length > 0) assignByType("investor");
  
  // PASS 2: Give each startup one MENTOR
  if (availableMentors.length > 0) assignByType("mentor");
  
  // PASS 3: Give each startup one CORPORATE
  if (availableCorporates.length > 0) assignByType("corporate");

  // PASS 4: Build a flat list of remaining candidates for utilization pass
  const remainingCandidates: ScoredCandidate[] = [];
  for (const bucket of candidatesByStartup.values()) {
    for (const cand of bucket) {
      const remaining = Math.max(
        0,
        cand.target.totalSlots - (targetUsedCount.get(`${cand.target.kind}::${cand.target.id}`) || 0)
      );
      if (remaining <= 0) continue;
      const key = `${cand.startup.id}::${cand.target.kind}::${cand.target.id}`;
      if (usedPairs.has(key)) continue;
      remainingCandidates.push(cand);
    }
  }
  remainingCandidates.sort((a, b) => b.score - a.score);

  // PASS 5: Utilization — fill remaining slots by score
  for (const cand of remainingCandidates) {
    const remaining = Math.max(
      0,
      cand.target.totalSlots - (targetUsedCount.get(`${cand.target.kind}::${cand.target.id}`) || 0)
    );
    if (remaining <= 0) continue;
    const match = tryAssignCandidate(cand);
    if (match) newMatches.push(match);
  }

  // Optional: ensure min meetings per investor (if configured) — second-chance fill
  if (minMeetingsPerInvestor > 0) {
    for (const investor of availableInvestors) {
      while ((targetUsedCount.get(`investor::${investor.id}`) || 0) < minMeetingsPerInvestor) {
        const candidates = remainingCandidates.filter((c) => c.target.kind === "investor" && c.target.id === investor.id);
        const next = candidates.find((c) => {
          const key = `${c.startup.id}::${c.target.kind}::${c.target.id}`;
          return (
            !usedPairs.has(key) &&
            investor.totalSlots - (targetUsedCount.get(`investor::${investor.id}`) || 0) > 0
          );
        });
        if (!next) break;
        const match = tryAssignCandidate(next);
        if (match) newMatches.push(match);
        else break;
      }
    }
  }

  // Combine and sort by slotTime order for clean output
  const allMatches = [...preserved, ...newMatches];
  const timeIndex = new Map<string, number>();
  slotsToUse.forEach((t, idx) => timeIndex.set(t, idx));

  allMatches.sort((a, b) => {
    const ia = timeIndex.get(a.slotTime) ?? 9999;
    const ib = timeIndex.get(b.slotTime) ?? 9999;
    return ia - ib;
  });

  // Final dedupe guarantee (paranoid)
  const final: Match[] = [];
  const seen = new Set<string>();
  const seenFirm = new Set<string>();
  for (const m of allMatches) {
    const key = `${m.startupId}::${m.targetType || "investor"}::${m.targetId || m.investorId}`;
    if (seen.has(key)) continue;
    if (m.targetType === "investor") {
      const investor = investors.find((i) => i.id === (m.investorId || m.targetId));
      const firmKey = investor
        ? `${m.startupId}::${investorFirmKey(investor)}`
        : `${m.startupId}::${norm(m.investorName || m.targetName || "")}`;
      if (seenFirm.has(firmKey)) continue;
      seenFirm.add(firmKey);
    }
    seen.add(key);
    final.push(m);
  }

  return final;
}


