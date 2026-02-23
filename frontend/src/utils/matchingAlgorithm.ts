import { Startup, Investor, Mentor, CorporatePartner, Match, TimeSlotConfig } from "@/types";

interface CompatibilityScore {
  geoMatch: number;
  industryMatch: number;
  fundingMatch: number;
  stageMatch: number;
  totalScore: number;
}

type MatchTarget = Investor | Mentor | CorporatePartner;

interface TargetInfo {
  id: string;
  type: 'investor' | 'mentor' | 'corporate';
  name: string;
  geoFocus: string[];
  industryPreferences: string[];
  totalSlots: number;
  availabilityStatus: 'present' | 'not-attending';
  slotAvailability?: Record<string, boolean>;
}

// Time slots configuration - Extended from 9:00 to 18:00 (20-minute slots)
const TIME_SLOTS = [
  '09:00 - 09:20',
  '09:20 - 09:40', 
  '09:40 - 10:00',
  '10:00 - 10:20',
  '10:20 - 10:40',
  '10:40 - 11:00',
  '11:00 - 11:20',
  '11:20 - 11:40',
  '11:40 - 12:00',
  '12:00 - 12:20',
  '12:20 - 12:40',
  '12:40 - 13:00',
  '13:00 - 13:20',
  '13:20 - 13:40',
  '13:40 - 14:00',
  '14:00 - 14:20',
  '14:20 - 14:40',
  '14:40 - 15:00',
  '15:00 - 15:20',
  '15:20 - 15:40',
  '15:40 - 16:00',
  '16:00 - 16:20',
  '16:20 - 16:40',
  '16:40 - 17:00',
  '17:00 - 17:20',
  '17:20 - 17:40',
  '17:40 - 18:00'
];

function investorDisplayName(investor: Investor): string {
  return `${investor.firmName} (${investor.memberName})`;
}

function mentorDisplayName(mentor: Mentor): string {
  return mentor.fullName;
}

function corporateDisplayName(corporate: CorporatePartner): string {
  return `${corporate.firmName} (${corporate.contactName})`;
}

function toTargetInfo(target: MatchTarget, type: 'investor' | 'mentor' | 'corporate'): TargetInfo {
  if (type === 'investor') {
    const inv = target as Investor;
    return {
      id: inv.id,
      type: 'investor',
      name: investorDisplayName(inv),
      geoFocus: inv.geoFocus,
      industryPreferences: inv.industryPreferences,
      totalSlots: inv.totalSlots,
      availabilityStatus: inv.availabilityStatus,
      slotAvailability: inv.slotAvailability
    };
  } else if (type === 'mentor') {
    const men = target as Mentor;
    return {
      id: men.id,
      type: 'mentor',
      name: mentorDisplayName(men),
      geoFocus: men.geoFocus,
      industryPreferences: men.industryPreferences,
      totalSlots: men.totalSlots,
      availabilityStatus: men.availabilityStatus,
      slotAvailability: men.slotAvailability
    };
  } else {
    const corp = target as CorporatePartner;
    return {
      id: corp.id,
      type: 'corporate',
      name: corporateDisplayName(corp),
      geoFocus: corp.geoFocus,
      industryPreferences: corp.industryPreferences,
      totalSlots: corp.totalSlots,
      availabilityStatus: corp.availabilityStatus,
      slotAvailability: corp.slotAvailability
    };
  }
}

export function calculateCompatibilityScore(startup: Startup, investor: Investor): CompatibilityScore {
  // Geographic Match (40% weight) - Calculate overlap percentage
  const geoOverlap = startup.geoMarkets.filter(market => 
    investor.geoFocus.some(focus => focus.toLowerCase() === market.toLowerCase())
  );
  const geoMatch = geoOverlap.length > 0 
    ? (geoOverlap.length / Math.max(startup.geoMarkets.length, investor.geoFocus.length)) * 100 
    : 0;

  // Industry Match (25% weight) - Check if startup industry is in investor preferences
  const industryMatch = investor.industryPreferences.some(pref => 
    pref.toLowerCase() === startup.industry.toLowerCase()
  ) ? 100 : 0;

  // Stage Match (20% weight) - Check if startup stage is in investor preferences
  const stageMatch = investor.stagePreferences.some(pref => 
    pref.toLowerCase() === startup.fundingStage.toLowerCase()
  ) ? 100 : 0;

  // Funding Match (15% weight) - More sophisticated funding alignment
  let fundingMatch = 0;
  const target = startup.fundingTarget;
  const minTicket = investor.minTicketSize;
  const maxTicket = investor.maxTicketSize;

  if (target >= minTicket && target <= maxTicket) {
    // Perfect match - funding target is within investor's range
    fundingMatch = 100;
  } else if (target < minTicket) {
    // Below minimum - calculate proximity score
    const gap = minTicket - target;
    const tolerance = minTicket * 0.3; // 30% tolerance below minimum
    if (gap <= tolerance) {
      fundingMatch = Math.max(20, (1 - gap / tolerance) * 70); // 20-70% score
    }
  } else if (target > maxTicket) {
    // Above maximum - calculate proximity score  
    const gap = target - maxTicket;
    const tolerance = maxTicket * 0.5; // 50% tolerance above maximum
    if (gap <= tolerance) {
      fundingMatch = Math.max(20, (1 - gap / tolerance) * 70); // 20-70% score
    }
  }

  // Calculate weighted total score
  const totalScore = (geoMatch * 0.4) + (industryMatch * 0.25) + (stageMatch * 0.2) + (fundingMatch * 0.15);

  return {
    geoMatch: Math.round(geoMatch),
    industryMatch: Math.round(industryMatch),
    stageMatch: Math.round(stageMatch),
    fundingMatch: Math.round(fundingMatch),
    totalScore: Math.round(Math.max(0, Math.min(100, totalScore)))
  };
}

export function generateMatches(
  startups: Startup[], 
  investors: Investor[], 
  existingMatches: Match[] = [],
  timeSlots: TimeSlotConfig[] = [],
  options?: {
    mentors?: Mentor[];
    corporates?: CorporatePartner[];
  }
): Match[] {
  // Filter out unavailable participants
  const availableStartups = startups.filter(s => s.availabilityStatus === 'present');
  const availableInvestors = investors.filter(i => i.availabilityStatus === 'present');
  const availableMentors = (options?.mentors || []).filter(m => m.availabilityStatus === 'present');
  const availableCorporates = (options?.corporates || []).filter(c => c.availabilityStatus === 'present');
  
  // Convert all targets to a unified format
  const allTargets: TargetInfo[] = [
    ...availableInvestors.map(i => toTargetInfo(i, 'investor')),
    ...availableMentors.map(m => toTargetInfo(m, 'mentor')),
    ...availableCorporates.map(c => toTargetInfo(c, 'corporate'))
  ];

  // Get completed matches to preserve
  const completedMatches = existingMatches.filter(match => match.completed);
  // Get locked matches - only preserve the pairing, not the time slot
  const lockedMatches = existingMatches.filter(match => match.locked && !match.completed);
  // Only completed matches preserve their time slots unchanged
  const preservedMatches = [...completedMatches];
  
  // Track pairs that have already met (completed or newly scheduled in this run)
  const hasMet = new Set<string>();
  preservedMatches.forEach(m => {
    const targetId = m.targetId || m.investorId || '';
    hasMet.add(`${m.startupId}::${targetId}`);
  });

  // Calculate all possible compatibility scores
  const allPossibleMatches: (Match & { score: CompatibilityScore })[] = [];

  availableStartups.forEach(startup => {
    allTargets.forEach(target => {
      // Skip if already have a completed or locked match
      const hasPreservedMatch = preservedMatches.some(
        match => {
          const matchTargetId = match.targetId || match.investorId || '';
          return match.startupId === startup.id && matchTargetId === target.id;
        }
      );
      
      if (!hasPreservedMatch) {
        // Calculate compatibility based on target type
        let score: CompatibilityScore;
        if (target.type === 'investor') {
          const investor = availableInvestors.find(i => i.id === target.id)!;
          score = calculateCompatibilityScore(startup, investor);
        } else {
          // For mentors and corporates, use simplified scoring (geo + industry)
          const geoOverlap = startup.geoMarkets.filter(market => 
            target.geoFocus.some(focus => focus.toLowerCase() === market.toLowerCase())
          );
          const geoMatch = geoOverlap.length > 0 
            ? (geoOverlap.length / Math.max(startup.geoMarkets.length, target.geoFocus.length)) * 100 
            : 0;
          const industryMatch = target.industryPreferences.some(pref => 
            pref.toLowerCase() === startup.industry.toLowerCase()
          ) ? 100 : 0;
          score = {
            geoMatch: Math.round(geoMatch),
            industryMatch: Math.round(industryMatch),
            fundingMatch: 0,
            stageMatch: 0,
            totalScore: Math.round((geoMatch * 0.5) + (industryMatch * 0.5))
          };
        }
        
        allPossibleMatches.push({
          id: `${startup.id}-${target.id}-${Date.now()}`,
          startupId: startup.id,
          targetId: target.id,
          targetType: target.type,
          startupName: startup.companyName,
          targetName: target.name,
          timeSlot: '',
          slotTime: '',
          compatibilityScore: score.totalScore,
          status: 'upcoming',
          completed: false,
          score,
          // Legacy compatibility
          investorId: target.type === 'investor' ? target.id : undefined,
          investorName: target.type === 'investor' ? target.name : undefined
        });
      }
    });
  });

  // Sort by compatibility score (highest first)
  allPossibleMatches.sort((a, b) => b.compatibilityScore - a.compatibilityScore);

  // Track slot usage for each target and startup
  const targetSlotUsage = new Map<string, number>();
  const startupMeetingCount = new Map<string, number>();
  
  // Initialize counts
  allTargets.forEach(target => {
    targetSlotUsage.set(target.id, 0);
  });
  
  availableStartups.forEach(startup => {
    startupMeetingCount.set(startup.id, 0);
  });

  // Account for preserved matches (completed and locked)
  preservedMatches.forEach(match => {
    const matchTargetId = match.targetId || match.investorId || '';
    const currentTargetUsage = targetSlotUsage.get(matchTargetId) || 0;
    const currentStartupCount = startupMeetingCount.get(match.startupId) || 0;
    
    targetSlotUsage.set(matchTargetId, currentTargetUsage + 1);
    startupMeetingCount.set(match.startupId, currentStartupCount + 1);
  });

  // Calculate target meetings per startup (evenly distributed)
  const totalAvailableSlots = allTargets.reduce((sum, target) => {
    const usedSlots = targetSlotUsage.get(target.id) || 0;
    return sum + Math.max(0, target.totalSlots - usedSlots);
  }, 0);
  
  const targetMeetingsPerStartup = Math.floor(totalAvailableSlots / availableStartups.length);
  const extraMeetings = totalAvailableSlots % availableStartups.length;

  // Use custom time slots if provided, otherwise fall back to default
  const slotsToUse = timeSlots.length > 0 ? timeSlots.map(ts => ts.startTime + ' - ' + ts.endTime) : TIME_SLOTS;
  const slotLabels = timeSlots.length > 0 ? timeSlots.map(ts => ts.label) : TIME_SLOTS.map((_, i) => `Slot ${i + 1}`);

  // Compute per-slot capacity (number of available targets for that slot)
  const slotCapacity = slotLabels.map((_, i) => {
    const isSlotDone = timeSlots[i]?.isDone === true;
    if (isSlotDone) return 0;
    return allTargets.filter(target => {
      const targetAvailable = !target.slotAvailability || target.slotAvailability[timeSlots[i]?.id] !== false;
      return targetAvailable;
    }).length;
  });

  // Create schedule grid for time slot assignment
  const scheduleGrid: { [timeSlot: string]: { startupIds: Set<string>, targetIds: Set<string> } } = {};
  slotsToUse.forEach((slot, index) => {
    scheduleGrid[slotLabels[index]] = { startupIds: new Set(), targetIds: new Set() };
  });

  // Add preserved matches to schedule grid (use label key)
  preservedMatches.forEach(match => {
    if (match.timeSlot && scheduleGrid[match.timeSlot]) {
      const matchTargetId = match.targetId || match.investorId || '';
      scheduleGrid[match.timeSlot].startupIds.add(match.startupId);
      scheduleGrid[match.timeSlot].targetIds.add(matchTargetId);
    }
  });

  const newMatches: Match[] = [];

  // Pre-assign locked pairs (lock the pairing only, assign any valid slot)
  for (const lm of lockedMatches) {
    const startup = availableStartups.find(s => s.id === lm.startupId);
    const lmTargetId = lm.targetId || lm.investorId || '';
    const target = allTargets.find(t => t.id === lmTargetId);
    if (!startup || !target) continue;

    const currentTargetUsage = targetSlotUsage.get(target.id) || 0;
    if (currentTargetUsage >= target.totalSlots) continue;

    let assignedTimeSlot = '';
    let assignedSlotTime = '';
    for (let i = 0; i < slotLabels.length; i++) {
      const slotLabel = slotLabels[i];
      const gridCell = scheduleGrid[slotLabel];

      const isSlotDone = timeSlots[i]?.isDone === true;
      if (isSlotDone) continue;

      // Respect per-slot capacity
      if (gridCell.targetIds.size >= slotCapacity[i]) continue;

      const startupAvailable = !startup.slotAvailability || startup.slotAvailability[timeSlots[i]?.id] !== false;
      const targetAvailable = !target.slotAvailability || target.slotAvailability[timeSlots[i]?.id] !== false;

      if (
        !gridCell.startupIds.has(startup.id) &&
        !gridCell.targetIds.has(target.id) &&
        startupAvailable &&
        targetAvailable
      ) {
        assignedTimeSlot = slotLabel;
        assignedSlotTime = slotsToUse[i];
        gridCell.startupIds.add(startup.id);
        gridCell.targetIds.add(target.id);
        break;
      }
    }

    if (!assignedTimeSlot) continue;

    const lockedMatch: Match = {
      id: lm.id,
      startupId: startup.id,
      targetId: target.id,
      targetType: target.type,
      startupName: startup.companyName,
      targetName: target.name,
      timeSlot: assignedTimeSlot,
      slotTime: assignedSlotTime,
      compatibilityScore: lm.compatibilityScore || 0,
      status: 'upcoming',
      completed: false,
      locked: true,
      startupAttending: true,
      targetAttending: true,
      // Legacy
      investorId: target.type === 'investor' ? target.id : undefined,
      investorName: target.type === 'investor' ? target.name : undefined,
      investorAttending: target.type === 'investor' ? true : undefined
    };

    newMatches.push(lockedMatch);
    hasMet.add(`${startup.id}::${target.id}`);

    targetSlotUsage.set(target.id, (targetSlotUsage.get(target.id) || 0) + 1);
    const currentStartupCount = startupMeetingCount.get(startup.id) || 0;
    startupMeetingCount.set(startup.id, currentStartupCount + 1);
  }

  // Fairness-first pass: ensure every startup reaches at least the average (floor)
  const minTargetPerStartup = targetMeetingsPerStartup; // floor average

  // Build candidate lists per startup (sorted by compatibility desc)
  const candidatesByStartup = new Map<string, (Match & { score: CompatibilityScore })[]>();
  for (const pm of allPossibleMatches) {
    const list = candidatesByStartup.get(pm.startupId) || [];
    list.push(pm);
    candidatesByStartup.set(pm.startupId, list);
  }
  for (const [sid, list] of candidatesByStartup) {
    list.sort((a, b) => b.compatibilityScore - a.compatibilityScore);
  }

  // Helper to try assigning the best candidate of a specific target type per startup
  const tryAssignPreferredType = (targetType: 'mentor' | 'corporate') => {
    for (const startup of availableStartups) {
      const candidates = (candidatesByStartup.get(startup.id) || []).filter(
        (c) => c.targetType === targetType
      );
      if (candidates.length === 0) continue;

      const currentStartupCount = startupMeetingCount.get(startup.id) || 0;
      // Try the best-scoring candidate of this type
      for (const cand of candidates) {
        const candTargetId = cand.targetId || cand.investorId || "";
        if (hasMet.has(`${cand.startupId}::${candTargetId}`)) continue;
        const target = allTargets.find((t) => t.id === candTargetId);
        if (!target) continue;

        const currentTargetUsage = targetSlotUsage.get(target.id) || 0;
        if (currentTargetUsage >= target.totalSlots) continue; // target full

        // Find an available time slot for both
        let assignedTimeSlot = "";
        let assignedSlotTime = "";
        for (let i = 0; i < slotLabels.length; i++) {
          const slotLabel = slotLabels[i];
          const gridCell = scheduleGrid[slotLabel];

          const isSlotDone = timeSlots[i]?.isDone === true;
          if (isSlotDone) continue;

          // Respect per-slot capacity
          if (gridCell.targetIds.size >= slotCapacity[i]) continue;

          const startupAvailable =
            !startup.slotAvailability || startup.slotAvailability[timeSlots[i]?.id] !== false;
          const targetAvailable =
            !target.slotAvailability || target.slotAvailability[timeSlots[i]?.id] !== false;

          if (
            !gridCell.startupIds.has(startup.id) &&
            !gridCell.targetIds.has(target.id) &&
            startupAvailable &&
            targetAvailable
          ) {
            assignedTimeSlot = slotLabel;
            assignedSlotTime = slotsToUse[i];
            gridCell.startupIds.add(startup.id);
            gridCell.targetIds.add(target.id);
            break;
          }
        }

        if (!assignedTimeSlot) continue;

        const newMatch: Match = {
          id: `${startup.id}-${target.id}-${Date.now()}-${targetType}`,
          startupId: startup.id,
          targetId: target.id,
          targetType: target.type,
          startupName: startup.companyName,
          targetName: target.name,
          timeSlot: assignedTimeSlot,
          slotTime: assignedSlotTime,
          compatibilityScore: cand.compatibilityScore,
          status: "upcoming",
          completed: false,
          startupAttending: true,
          targetAttending: true,
          investorId: target.type === "investor" ? target.id : undefined,
          investorName: target.type === "investor" ? target.name : undefined,
          investorAttending: target.type === "investor" ? true : undefined,
        };

        newMatches.push(newMatch);
        hasMet.add(`${startup.id}::${target.id}`);
        targetSlotUsage.set(target.id, currentTargetUsage + 1);
        startupMeetingCount.set(startup.id, currentStartupCount + 1);
        break; // only one preferred match of this type per startup
      }
    }
  };

  // Prefer to secure one mentor and one corporate (if available) per startup before general rounds
  tryAssignPreferredType('mentor');
  tryAssignPreferredType('corporate');

  // Round-robin assign one meeting per round to the startups with the fewest meetings
  for (let round = 1; round <= minTargetPerStartup; round++) {
    // Sort startups by current meeting count (ascending) to always prioritize those with fewer meetings
    const startupsByNeed = [...availableStartups].sort(
      (a, b) => (startupMeetingCount.get(a.id) || 0) - (startupMeetingCount.get(b.id) || 0)
    );

    for (const startup of startupsByNeed) {
      const currentCount = startupMeetingCount.get(startup.id) || 0;
      if (currentCount >= round) continue; // this startup already satisfied this round

      const candidates = candidatesByStartup.get(startup.id) || [];

      // Try candidates in score order
      for (const cand of candidates) {
        const candTargetId = cand.targetId || cand.investorId || '';
        if (hasMet.has(`${cand.startupId}::${candTargetId}`)) continue;
        const target = allTargets.find(t => t.id === candTargetId);
        if (!target) continue;

        const currentTargetUsage = targetSlotUsage.get(target.id) || 0;
        if (currentTargetUsage >= target.totalSlots) continue; // target full

        // Find an available time slot for both
        let assignedTimeSlot = '';
        let assignedSlotTime = '';
        for (let i = 0; i < slotLabels.length; i++) {
          const slotLabel = slotLabels[i];
          const gridCell = scheduleGrid[slotLabel];

          // Skip slots marked as done
          const isSlotDone = timeSlots[i]?.isDone === true;
          if (isSlotDone) continue;

          // Respect per-slot capacity
          if (gridCell.targetIds.size >= slotCapacity[i]) continue;

          const startupAvailable = !startup.slotAvailability || startup.slotAvailability[timeSlots[i]?.id] !== false;
          const targetAvailable = !target.slotAvailability || target.slotAvailability[timeSlots[i]?.id] !== false;

          if (
            !gridCell.startupIds.has(startup.id) &&
            !gridCell.targetIds.has(target.id) &&
            startupAvailable &&
            targetAvailable
          ) {
            assignedTimeSlot = slotLabel;
            assignedSlotTime = slotsToUse[i];

            // Reserve this slot
            gridCell.startupIds.add(startup.id);
            gridCell.targetIds.add(target.id);
            break;
          }
        }

        if (!assignedTimeSlot) {
          // Couldn't find a slot with this target; try next candidate
          continue;
        }

        // Create the match
        const newMatch: Match = {
          id: `${startup.id}-${target.id}-${Date.now()}-rr${round}`,
          startupId: startup.id,
          targetId: target.id,
          targetType: target.type,
          startupName: startup.companyName,
          targetName: target.name,
          timeSlot: assignedTimeSlot,
          slotTime: assignedSlotTime,
          compatibilityScore: cand.compatibilityScore,
          status: 'upcoming',
          completed: false,
          startupAttending: true,
          targetAttending: true,
          // Legacy
          investorId: target.type === 'investor' ? target.id : undefined,
          investorName: target.type === 'investor' ? target.name : undefined,
          investorAttending: target.type === 'investor' ? true : undefined
        };

        newMatches.push(newMatch);
        hasMet.add(`${startup.id}::${target.id}`);

        targetSlotUsage.set(target.id, currentTargetUsage + 1);
        startupMeetingCount.set(startup.id, currentCount + 1);
        break; // assign only one meeting for this round for this startup
      }
    }
  }

  // Fill up to per-startup maximums using remaining high-score opportunities
  for (const potentialMatch of allPossibleMatches) {
    const potTargetId = potentialMatch.targetId || potentialMatch.investorId || '';
    const target = allTargets.find(t => t.id === potTargetId);
    const startup = availableStartups.find(s => s.id === potentialMatch.startupId);
    
    if (!target || !startup) continue;
    if (hasMet.has(`${startup.id}::${target.id}`)) continue;

    const currentTargetUsage = targetSlotUsage.get(target.id) || 0;
    const currentStartupCount = startupMeetingCount.get(startup.id) || 0;

    // Respect target capacity
    if (currentTargetUsage >= target.totalSlots) continue;

    // Respect per-startup cap (balanced distribution: max = base + extras)
    const startupIndex = availableStartups.findIndex(s => s.id === startup.id);
    const maxMeetingsForThisStartup = targetMeetingsPerStartup + (startupIndex < extraMeetings ? 1 : 0);
    if (currentStartupCount >= maxMeetingsForThisStartup) continue;

    // Find available time slot where both are free and attending
    let assignedTimeSlot = '';
    let assignedSlotTime = '';
    for (let i = 0; i < slotLabels.length; i++) {
      const slotLabel = slotLabels[i];
      const slot = scheduleGrid[slotLabel];

      const isSlotDone = timeSlots[i]?.isDone === true;
      if (isSlotDone) continue;

      // Respect per-slot capacity
      if (slot.targetIds.size >= slotCapacity[i]) continue;

      const startupAvailable = !startup.slotAvailability || startup.slotAvailability[timeSlots[i]?.id] !== false;
      const targetAvailable = !target.slotAvailability || target.slotAvailability[timeSlots[i]?.id] !== false;

      if (
        !slot.startupIds.has(startup.id) &&
        !slot.targetIds.has(target.id) &&
        startupAvailable &&
        targetAvailable
      ) {
        assignedTimeSlot = slotLabel;
        assignedSlotTime = slotsToUse[i];
        slot.startupIds.add(startup.id);
        slot.targetIds.add(target.id);
        break;
      }
    }

    if (!assignedTimeSlot) continue;

    const newMatch: Match = {
      id: potentialMatch.id,
      startupId: startup.id,
      targetId: target.id,
      targetType: target.type,
      startupName: startup.companyName,
      targetName: target.name,
      timeSlot: assignedTimeSlot,
      slotTime: assignedSlotTime,
      compatibilityScore: potentialMatch.compatibilityScore,
      status: 'upcoming',
      completed: false,
      startupAttending: true,
      targetAttending: true,
      // Legacy
      investorId: target.type === 'investor' ? target.id : undefined,
      investorName: target.type === 'investor' ? target.name : undefined,
      investorAttending: target.type === 'investor' ? true : undefined
    };

    newMatches.push(newMatch);
    hasMet.add(`${startup.id}::${target.id}`);
    targetSlotUsage.set(target.id, currentTargetUsage + 1);
    startupMeetingCount.set(startup.id, currentStartupCount + 1);
  }

  // Second pass: fill remaining open target slots per time slot to maximize meetings
  for (let i = 0; i < slotLabels.length; i++) {
    const slotLabel = slotLabels[i];
    const slot = scheduleGrid[slotLabel];
    const isSlotDone = timeSlots[i]?.isDone === true;
    if (isSlotDone) continue;
    if (slot.targetIds.size >= slotCapacity[i]) continue;

    for (const target of allTargets) {
      const currentTargetUsage = targetSlotUsage.get(target.id) || 0;
      if (currentTargetUsage >= target.totalSlots) continue;
      if (slot.targetIds.has(target.id)) continue;

      const targetAvailable = !target.slotAvailability || target.slotAvailability[timeSlots[i]?.id] !== false;
      if (!targetAvailable) continue;

      const candidates = allPossibleMatches
        .filter(pm => {
          const pmTargetId = pm.targetId || pm.investorId || '';
          return pmTargetId === target.id;
        })
        .sort((a, b) => b.compatibilityScore - a.compatibilityScore);

      for (const cand of candidates) {
        const startup = availableStartups.find(s => s.id === cand.startupId);
        if (!startup) continue;
        const startupAvailable = !startup.slotAvailability || startup.slotAvailability[timeSlots[i]?.id] !== false;

        if (!slot.startupIds.has(startup.id) && startupAvailable && !hasMet.has(`${startup.id}::${target.id}`)) {
          const newMatch: Match = {
            id: `${startup.id}-${target.id}-${Date.now()}-${i}`,
            startupId: startup.id,
            targetId: target.id,
            targetType: target.type,
            startupName: startup.companyName,
            targetName: target.name,
            timeSlot: slotLabel,
            slotTime: slotsToUse[i],
            compatibilityScore: cand.compatibilityScore,
            status: 'upcoming',
            completed: false,
            startupAttending: true,
            targetAttending: true,
            // Legacy
            investorId: target.type === 'investor' ? target.id : undefined,
            investorName: target.type === 'investor' ? target.name : undefined,
            investorAttending: target.type === 'investor' ? true : undefined
          };
          newMatches.push(newMatch);
          hasMet.add(`${startup.id}::${target.id}`);

          targetSlotUsage.set(target.id, (targetSlotUsage.get(target.id) || 0) + 1);
          const currentStartupCount = startupMeetingCount.get(startup.id) || 0;
          startupMeetingCount.set(startup.id, currentStartupCount + 1);

          slot.targetIds.add(target.id);
          slot.startupIds.add(startup.id);
          break;
        }
      }
    }
  }

  // Post-pass: ensure each startup gets at least one mentor and one corporate if available
  const ensureTypeForStartups = (desiredType: 'mentor' | 'corporate') => {
    const targetsOfType = allTargets.filter(t => t.type === desiredType);
    if (targetsOfType.length === 0) return;

    for (const startup of availableStartups) {
      const alreadyHas = newMatches.some(m => m.startupId === startup.id && m.targetType === desiredType);
      if (alreadyHas) continue;

      const candidates = (candidatesByStartup.get(startup.id) || [])
        .filter(c => c.targetType === desiredType)
        .sort((a, b) => b.compatibilityScore - a.compatibilityScore);
      if (candidates.length === 0) continue;

      const cand = candidates[0];
      const candTargetId = cand.targetId || cand.investorId || '';
      const target = allTargets.find(t => t.id === candTargetId);
      if (!target) continue;

      const currentTargetUsage = targetSlotUsage.get(target.id) || 0;
      if (currentTargetUsage >= target.totalSlots) continue;

      let assignedTimeSlot = '';
      let assignedSlotTime = '';
      for (let j = 0; j < slotLabels.length; j++) {
        const slotLabel = slotLabels[j];
        const gridCell = scheduleGrid[slotLabel];
        const isSlotDone = timeSlots[j]?.isDone === true;
        if (isSlotDone) continue;
        if (gridCell.targetIds.size >= slotCapacity[j]) continue;

        const startupAvailable = !startup.slotAvailability || startup.slotAvailability[timeSlots[j]?.id] !== false;
        const targetAvailable = !target.slotAvailability || target.slotAvailability[timeSlots[j]?.id] !== false;

        if (
          !gridCell.startupIds.has(startup.id) &&
          !gridCell.targetIds.has(target.id) &&
          startupAvailable &&
          targetAvailable
        ) {
          assignedTimeSlot = slotLabel;
          assignedSlotTime = slotsToUse[j];
          gridCell.startupIds.add(startup.id);
          gridCell.targetIds.add(target.id);
          break;
        }
      }

      if (!assignedTimeSlot) continue;

      const newMatch: Match = {
        id: `${startup.id}-${target.id}-${Date.now()}-ensure-${desiredType}`,
        startupId: startup.id,
        targetId: target.id,
        targetType: target.type,
        startupName: startup.companyName,
        targetName: target.name,
        timeSlot: assignedTimeSlot,
        slotTime: assignedSlotTime,
        compatibilityScore: cand.compatibilityScore,
        status: 'upcoming',
        completed: false,
        startupAttending: true,
        targetAttending: true,
        investorId: target.type === 'investor' ? target.id : undefined,
        investorName: target.type === 'investor' ? target.name : undefined,
        investorAttending: target.type === 'investor' ? true : undefined
      };

      newMatches.push(newMatch);
      hasMet.add(`${startup.id}::${target.id}`);
      targetSlotUsage.set(target.id, currentTargetUsage + 1);
      const currentStartupCount = startupMeetingCount.get(startup.id) || 0;
      startupMeetingCount.set(startup.id, currentStartupCount + 1);
    }
  };

  ensureTypeForStartups('mentor');
  ensureTypeForStartups('corporate');

  // Combine preserved matches with new matches and sort by time slot
  const allMatches = [...preservedMatches, ...newMatches];
  
  // Sort by time slot for better organization
  allMatches.sort((a, b) => {
    const timeA = slotsToUse.indexOf(a.slotTime);
    const timeB = slotsToUse.indexOf(b.slotTime);
    return timeA - timeB;
  });

  return allMatches;
}