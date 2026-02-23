import { Startup, Investor, Match, TimeSlotConfig } from "@/types";

interface CompatibilityScore {
  geoMatch: number;
  industryMatch: number;
  fundingMatch: number;
  stageMatch: number;
  diversityBonus: number; // New: portfolio diversity
  totalScore: number;
  breakdown: {
    geoWeight: number;
    industryWeight: number;
    stageWeight: number;
    fundingWeight: number;
    diversityWeight: number;
  };
}

interface MatchingConfig {
  weights: {
    geo: number;
    industry: number;
    stage: number;
    funding: number;
    diversity: number;
  };
  enableDiversityBonus: boolean;
  minCompatibilityThreshold: number; // Minimum score to create a match
  maxMeetingsPerStartup?: number;
  prioritizeFairness: boolean;
}

function investorDisplayName(investor: Investor): string {
  return `${investor.firmName} (${investor.memberName})`;
}

const DEFAULT_CONFIG: MatchingConfig = {
  weights: {
    geo: 0.35,
    industry: 0.25,
    stage: 0.20,
    funding: 0.15,
    diversity: 0.05,
  },
  enableDiversityBonus: true,
  minCompatibilityThreshold: 30,
  prioritizeFairness: true,
};

// Industry similarity matrix (for partial matching)
const INDUSTRY_SIMILARITY: Record<string, Record<string, number>> = {
  'AI/ML': { 'SaaS': 0.7, 'Fintech': 0.5, 'EdTech': 0.6 },
  'SaaS': { 'AI/ML': 0.7, 'Fintech': 0.6, 'E-commerce': 0.4 },
  'Fintech': { 'SaaS': 0.6, 'AI/ML': 0.5, 'E-commerce': 0.5 },
  'Healthtech': { 'EdTech': 0.4, 'Consumer Goods': 0.3 },
  'EdTech': { 'Healthtech': 0.4, 'SaaS': 0.5, 'AI/ML': 0.6 },
  'E-commerce': { 'SaaS': 0.4, 'Fintech': 0.5, 'Logistics': 0.6 },
};

// Stage proximity scoring
const STAGE_PROXIMITY: Record<string, Record<string, number>> = {
  'Pre-seed': { 'Seed': 0.8, 'Series A': 0.5 },
  'Seed': { 'Pre-seed': 0.8, 'Series A': 0.9, 'Series B+': 0.6 },
  'Series A': { 'Seed': 0.9, 'Series B+': 0.8 },
  'Series B+': { 'Series A': 0.8, 'Seed': 0.6 },
};

/**
 * Enhanced compatibility scoring with partial matches and diversity bonus
 */
export function calculateCompatibilityScoreV2(
  startup: Startup,
  investor: Investor,
  investorMatches: Match[] = [], // Existing matches for diversity calculation
  config: MatchingConfig = DEFAULT_CONFIG
): CompatibilityScore {
  // 1. Geographic Match (35% weight) - Enhanced with market importance
  const geoOverlap = startup.geoMarkets.filter(market =>
    investor.geoFocus.some(focus => focus.toLowerCase() === market.toLowerCase())
  );
  
  // Calculate overlap percentage (improved formula)
  const geoMatch = geoOverlap.length > 0
    ? (geoOverlap.length / Math.max(startup.geoMarkets.length, investor.geoFocus.length)) * 100
    : 0;

  // 2. Industry Match (25% weight) - Partial matching support
  let industryMatch = 0;
  const startupIndustry = startup.industry.toLowerCase();
  
  // Check for exact match
  const exactMatch = investor.industryPreferences.some(pref =>
    pref.toLowerCase() === startupIndustry
  );
  
  if (exactMatch) {
    industryMatch = 100;
  } else {
    // Check for partial/similar industry matches
    const similarities = investor.industryPreferences.map(pref => {
      const prefLower = pref.toLowerCase();
      const similarity = INDUSTRY_SIMILARITY[startupIndustry]?.[prefLower] ||
                         INDUSTRY_SIMILARITY[prefLower]?.[startupIndustry] ||
                         0;
      return similarity * 100;
    });
    
    industryMatch = similarities.length > 0 ? Math.max(...similarities) : 0;
  }

  // 3. Stage Match (20% weight) - Proximity scoring
  let stageMatch = 0;
  const startupStage = startup.fundingStage.toLowerCase();
  
  const exactStageMatch = investor.stagePreferences.some(pref =>
    pref.toLowerCase() === startupStage
  );
  
  if (exactStageMatch) {
    stageMatch = 100;
  } else {
    // Calculate proximity scores
    const proximities = investor.stagePreferences.map(pref => {
      const prefLower = pref.toLowerCase();
      const proximity = STAGE_PROXIMITY[startupStage]?.[prefLower] ||
                       STAGE_PROXIMITY[prefLower]?.[startupStage] ||
                       0;
      return proximity * 100;
    });
    
    stageMatch = proximities.length > 0 ? Math.max(...proximities) : 0;
  }

  // 4. Funding Match (15% weight) - Enhanced with better tolerance
  let fundingMatch = 0;
  const target = startup.fundingTarget;
  const minTicket = investor.minTicketSize;
  const maxTicket = investor.maxTicketSize;
  const range = maxTicket - minTicket;

  if (target >= minTicket && target <= maxTicket) {
    // Perfect match - calculate how centered it is
    const center = (minTicket + maxTicket) / 2;
    const distanceFromCenter = Math.abs(target - center);
    const maxDistance = range / 2;
    const centeringScore = 1 - (distanceFromCenter / maxDistance);
    fundingMatch = 70 + (centeringScore * 30); // 70-100% range
  } else if (target < minTicket) {
    const gap = minTicket - target;
    const tolerance = minTicket * 0.3;
    if (gap <= tolerance) {
      fundingMatch = Math.max(30, (1 - gap / tolerance) * 70);
    }
  } else if (target > maxTicket) {
    const gap = target - maxTicket;
    const tolerance = maxTicket * 0.5;
    if (gap <= tolerance) {
      fundingMatch = Math.max(30, (1 - gap / tolerance) * 70);
    }
  }

  // 5. Diversity Bonus (5% weight) - New feature
  let diversityBonus = 0;
  if (config.enableDiversityBonus && investorMatches.length > 0) {
    // Check if investor already has many matches in same industry
    const sameIndustryCount = investorMatches.filter(m => {
      const matchedStartup = investorMatches.find(im => im.startupId === m.startupId);
      // This would need startup data - simplified here
      return false; // Placeholder
    }).length;
    
    const totalMatches = investorMatches.length;
    const industryConcentration = sameIndustryCount / totalMatches;
    
    // Bonus for diversifying portfolio
    if (industryConcentration < 0.3) {
      diversityBonus = 100; // High diversity
    } else if (industryConcentration < 0.5) {
      diversityBonus = 50; // Medium diversity
    }
    // Low diversity = 0 bonus
  }

  // Calculate weighted total score
  const weights = config.weights;
  const totalScore =
    (geoMatch * weights.geo) +
    (industryMatch * weights.industry) +
    (stageMatch * weights.stage) +
    (fundingMatch * weights.funding) +
    (diversityBonus * weights.diversity);

  return {
    geoMatch: Math.round(geoMatch),
    industryMatch: Math.round(industryMatch),
    stageMatch: Math.round(stageMatch),
    fundingMatch: Math.round(fundingMatch),
    diversityBonus: Math.round(diversityBonus),
    totalScore: Math.round(Math.max(0, Math.min(100, totalScore))),
    breakdown: {
      geoWeight: weights.geo,
      industryWeight: weights.industry,
      stageWeight: weights.stage,
      fundingWeight: weights.funding,
      diversityWeight: weights.diversity,
    },
  };
}

/**
 * Enhanced matching algorithm with improved fairness and performance
 */
export function generateMatchesV2(
  startups: Startup[],
  investors: Investor[],
  existingMatches: Match[] = [],
  timeSlots: TimeSlotConfig[] = [],
  config: MatchingConfig = DEFAULT_CONFIG
): Match[] {
  // Filter available participants
  const availableStartups = startups.filter(s => s.availabilityStatus === 'present');
  const availableInvestors = investors.filter(i => i.availabilityStatus === 'present');

  if (availableStartups.length === 0 || availableInvestors.length === 0) {
    return [];
  }

  // Preserve completed matches
  const completedMatches = existingMatches.filter(match => match.completed);
  const lockedMatches = existingMatches.filter(match => match.locked && !match.completed);
  const preservedMatches = [...completedMatches];

  // Track pairs that have met
  const hasMet = new Set<string>();
  preservedMatches.forEach(m => hasMet.add(`${m.startupId}::${m.investorId}`));

  // Calculate compatibility scores with diversity awareness
  const allPossibleMatches: (Match & { score: CompatibilityScore })[] = [];

  availableStartups.forEach(startup => {
    availableInvestors.forEach(investor => {
      // Skip if already have preserved match
      const hasPreservedMatch = preservedMatches.some(
        match => match.startupId === startup.id && match.investorId === investor.id
      );

      if (!hasPreservedMatch) {
        // Get existing matches for this investor (for diversity calculation)
        const investorExistingMatches = existingMatches.filter(m => m.investorId === investor.id);
        
        const score = calculateCompatibilityScoreV2(
          startup,
          investor,
          investorExistingMatches,
          config
        );

        // Apply minimum threshold
        if (score.totalScore >= config.minCompatibilityThreshold) {
          allPossibleMatches.push({
            id: `${startup.id}-${investor.id}-${Date.now()}`,
            startupId: startup.id,
            investorId: investor.id,
            startupName: startup.companyName,
            investorName: investorDisplayName(investor),
            timeSlot: '',
            slotTime: '',
            compatibilityScore: score.totalScore,
            status: 'upcoming',
            completed: false,
            score,
          });
        }
      }
    });
  });

  // Sort by compatibility score
  allPossibleMatches.sort((a, b) => b.compatibilityScore - a.compatibilityScore);

  // Initialize tracking maps
  const investorSlotUsage = new Map<string, number>();
  const startupMeetingCount = new Map<string, number>();
  const investorMatchesByInvestor = new Map<string, Match[]>();

  availableInvestors.forEach(investor => {
    investorSlotUsage.set(investor.id, 0);
    investorMatchesByInvestor.set(investor.id, []);
  });

  availableStartups.forEach(startup => {
    startupMeetingCount.set(startup.id, 0);
  });

  // Account for preserved matches
  preservedMatches.forEach(match => {
    investorSlotUsage.set(match.investorId, (investorSlotUsage.get(match.investorId) || 0) + 1);
    startupMeetingCount.set(match.startupId, (startupMeetingCount.get(match.startupId) || 0) + 1);
    const existing = investorMatchesByInvestor.get(match.investorId) || [];
    investorMatchesByInvestor.set(match.investorId, [...existing, match]);
  });

  // Calculate target meetings
  const totalAvailableSlots = availableInvestors.reduce((sum, inv) => {
    const usedSlots = investorSlotUsage.get(inv.id) || 0;
    return sum + Math.max(0, inv.totalSlots - usedSlots);
  }, 0);

  const targetMeetingsPerStartup = Math.floor(totalAvailableSlots / availableStartups.length);
  const extraMeetings = totalAvailableSlots % availableStartups.length;

  // Setup time slots - Extended from 9:00 to 18:00 (20-minute slots)
  const defaultSlots = [
    '09:00 - 09:20', '09:20 - 09:40', '09:40 - 10:00', '10:00 - 10:20', '10:20 - 10:40', '10:40 - 11:00',
    '11:00 - 11:20', '11:20 - 11:40', '11:40 - 12:00', '12:00 - 12:20', '12:20 - 12:40', '12:40 - 13:00',
    '13:00 - 13:20', '13:20 - 13:40', '13:40 - 14:00', '14:00 - 14:20', '14:20 - 14:40', '14:40 - 15:00',
    '15:00 - 15:20', '15:20 - 15:40', '15:40 - 16:00', '16:00 - 16:20', '16:20 - 16:40', '16:40 - 17:00',
    '17:00 - 17:20', '17:20 - 17:40', '17:40 - 18:00'
  ];
  
  const slotsToUse = timeSlots.length > 0
    ? timeSlots.map(ts => `${ts.startTime} - ${ts.endTime}`)
    : defaultSlots;
  
  const slotLabels = timeSlots.length > 0
    ? timeSlots.map(ts => ts.label)
    : slotsToUse.map((_, i) => `Slot ${i + 1}`);

  // Compute slot capacity
  const slotCapacity = slotLabels.map((_, i) => {
    const isSlotDone = timeSlots[i]?.isDone === true;
    if (isSlotDone) return 0;
    return availableInvestors.filter(inv => {
      const investorAvailable = !inv.slotAvailability || inv.slotAvailability[timeSlots[i]?.id] !== false;
      return investorAvailable;
    }).length;
  });

  // Create schedule grid
  const scheduleGrid: { [timeSlot: string]: { startupIds: Set<string>, investorIds: Set<string> } } = {};
  slotLabels.forEach(label => {
    scheduleGrid[label] = { startupIds: new Set(), investorIds: new Set() };
  });

  // Add preserved matches to grid
  preservedMatches.forEach(match => {
    if (match.timeSlot && scheduleGrid[match.timeSlot]) {
      scheduleGrid[match.timeSlot].startupIds.add(match.startupId);
      scheduleGrid[match.timeSlot].investorIds.add(match.investorId);
    }
  });

  const newMatches: Match[] = [];

  // Pre-assign locked matches
  for (const lm of lockedMatches) {
    const startup = availableStartups.find(s => s.id === lm.startupId);
    const investor = availableInvestors.find(i => i.id === lm.investorId);
    if (!startup || !investor) continue;

    const currentInvestorUsage = investorSlotUsage.get(investor.id) || 0;
    if (currentInvestorUsage >= investor.totalSlots) continue;

    let assignedTimeSlot = '';
    let assignedSlotTime = '';

    for (let i = 0; i < slotLabels.length; i++) {
      const slotLabel = slotLabels[i];
      const gridCell = scheduleGrid[slotLabel];
      const isSlotDone = timeSlots[i]?.isDone === true;
      if (isSlotDone) continue;
      if (gridCell.investorIds.size >= slotCapacity[i]) continue;

      const startupAvailable = !startup.slotAvailability || startup.slotAvailability[timeSlots[i]?.id] !== false;
      const investorAvailable = !investor.slotAvailability || investor.slotAvailability[timeSlots[i]?.id] !== false;

      if (
        !gridCell.startupIds.has(startup.id) &&
        !gridCell.investorIds.has(investor.id) &&
        startupAvailable &&
        investorAvailable
      ) {
        assignedTimeSlot = slotLabel;
        assignedSlotTime = slotsToUse[i];
        gridCell.startupIds.add(startup.id);
        gridCell.investorIds.add(investor.id);
        break;
      }
    }

    if (!assignedTimeSlot) continue;

    const score = calculateCompatibilityScoreV2(startup, investor, [], config);
    const lockedMatch: Match = {
      id: lm.id,
      startupId: startup.id,
      investorId: investor.id,
      startupName: startup.companyName,
      investorName: investorDisplayName(investor),
      timeSlot: assignedTimeSlot,
      slotTime: assignedSlotTime,
      compatibilityScore: score.totalScore,
      status: 'upcoming',
      completed: false,
      locked: true,
      startupAttending: true,
      investorAttending: true,
    };

    newMatches.push(lockedMatch);
    hasMet.add(`${startup.id}::${investor.id}`);
    investorSlotUsage.set(investor.id, (investorSlotUsage.get(investor.id) || 0) + 1);
    startupMeetingCount.set(startup.id, (startupMeetingCount.get(startup.id) || 0) + 1);
    
    const existing = investorMatchesByInvestor.get(investor.id) || [];
    investorMatchesByInvestor.set(investor.id, [...existing, lockedMatch]);
  }

  // Fairness-first round-robin pass
  if (config.prioritizeFairness) {
    const minTargetPerStartup = targetMeetingsPerStartup;

    // Build candidate lists per startup
    const candidatesByStartup = new Map<string, (Match & { score: CompatibilityScore })[]>();
    for (const pm of allPossibleMatches) {
      const list = candidatesByStartup.get(pm.startupId) || [];
      list.push(pm);
      candidatesByStartup.set(pm.startupId, list);
    }

    for (const [sid, list] of candidatesByStartup) {
      list.sort((a, b) => b.compatibilityScore - a.compatibilityScore);
    }

    // Round-robin assignment
    for (let round = 1; round <= minTargetPerStartup; round++) {
      const startupsByNeed = [...availableStartups].sort(
        (a, b) => (startupMeetingCount.get(a.id) || 0) - (startupMeetingCount.get(b.id) || 0)
      );

      for (const startup of startupsByNeed) {
        const currentCount = startupMeetingCount.get(startup.id) || 0;
        if (currentCount >= round) continue;

        const candidates = candidatesByStartup.get(startup.id) || [];

        for (const cand of candidates) {
          if (hasMet.has(`${cand.startupId}::${cand.investorId}`)) continue;

          const investor = availableInvestors.find(i => i.id === cand.investorId);
          if (!investor) continue;

          const currentInvestorUsage = investorSlotUsage.get(investor.id) || 0;
          if (currentInvestorUsage >= investor.totalSlots) continue;

          // Find available time slot
          let assignedTimeSlot = '';
          let assignedSlotTime = '';

          for (let i = 0; i < slotLabels.length; i++) {
            const slotLabel = slotLabels[i];
            const gridCell = scheduleGrid[slotLabel];
            const isSlotDone = timeSlots[i]?.isDone === true;
            if (isSlotDone) continue;
            if (gridCell.investorIds.size >= slotCapacity[i]) continue;

            const startupAvailable = !startup.slotAvailability || startup.slotAvailability[timeSlots[i]?.id] !== false;
            const investorAvailable = !investor.slotAvailability || investor.slotAvailability[timeSlots[i]?.id] !== false;

            if (
              !gridCell.startupIds.has(startup.id) &&
              !gridCell.investorIds.has(investor.id) &&
              startupAvailable &&
              investorAvailable
            ) {
              assignedTimeSlot = slotLabel;
              assignedSlotTime = slotsToUse[i];
              gridCell.startupIds.add(startup.id);
              gridCell.investorIds.add(investor.id);
              break;
            }
          }

          if (!assignedTimeSlot) continue;

          const newMatch: Match = {
            id: `${startup.id}-${investor.id}-${Date.now()}-rr${round}`,
            startupId: startup.id,
            investorId: investor.id,
            startupName: startup.companyName,
              investorName: investorDisplayName(investor),
            timeSlot: assignedTimeSlot,
            slotTime: assignedSlotTime,
            compatibilityScore: cand.compatibilityScore,
            status: 'upcoming',
            completed: false,
            startupAttending: true,
            investorAttending: true,
          };

          newMatches.push(newMatch);
          hasMet.add(`${startup.id}::${investor.id}`);
          investorSlotUsage.set(investor.id, currentInvestorUsage + 1);
          startupMeetingCount.set(startup.id, currentCount + 1);

          const existing = investorMatchesByInvestor.get(investor.id) || [];
          investorMatchesByInvestor.set(investor.id, [...existing, newMatch]);

          break;
        }
      }
    }
  }

  // Fill remaining slots with high-score matches
  for (const potentialMatch of allPossibleMatches) {
    if (hasMet.has(`${potentialMatch.startupId}::${potentialMatch.investorId}`)) continue;

    const investor = availableInvestors.find(i => i.id === potentialMatch.investorId);
    const startup = availableStartups.find(s => s.id === potentialMatch.startupId);
    if (!investor || !startup) continue;

    const currentInvestorUsage = investorSlotUsage.get(investor.id) || 0;
    const currentStartupCount = startupMeetingCount.get(startup.id) || 0;

    if (currentInvestorUsage >= investor.totalSlots) continue;

    const startupIndex = availableStartups.findIndex(s => s.id === startup.id);
    const maxMeetingsForThisStartup = targetMeetingsPerStartup + (startupIndex < extraMeetings ? 1 : 0);
    if (currentStartupCount >= maxMeetingsForThisStartup) continue;

    // Find available time slot
    let assignedTimeSlot = '';
    let assignedSlotTime = '';

    for (let i = 0; i < slotLabels.length; i++) {
      const slotLabel = slotLabels[i];
      const slot = scheduleGrid[slotLabel];
      const isSlotDone = timeSlots[i]?.isDone === true;
      if (isSlotDone) continue;
      if (slot.investorIds.size >= slotCapacity[i]) continue;

      const startupAvailable = !startup.slotAvailability || startup.slotAvailability[timeSlots[i]?.id] !== false;
      const investorAvailable = !investor.slotAvailability || investor.slotAvailability[timeSlots[i]?.id] !== false;

      if (
        !slot.startupIds.has(startup.id) &&
        !slot.investorIds.has(investor.id) &&
        startupAvailable &&
        investorAvailable
      ) {
        assignedTimeSlot = slotLabel;
        assignedSlotTime = slotsToUse[i];
        slot.startupIds.add(startup.id);
        slot.investorIds.add(investor.id);
        break;
      }
    }

    if (!assignedTimeSlot) continue;

    const newMatch: Match = {
      id: potentialMatch.id,
      startupId: startup.id,
      investorId: investor.id,
      startupName: startup.companyName,
      investorName: investorDisplayName(investor),
      timeSlot: assignedTimeSlot,
      slotTime: assignedSlotTime,
      compatibilityScore: potentialMatch.compatibilityScore,
      status: 'upcoming',
      completed: false,
      startupAttending: true,
      investorAttending: true,
    };

    newMatches.push(newMatch);
    hasMet.add(`${startup.id}::${investor.id}`);
    investorSlotUsage.set(investor.id, currentInvestorUsage + 1);
    startupMeetingCount.set(startup.id, currentStartupCount + 1);

    const existing = investorMatchesByInvestor.get(investor.id) || [];
    investorMatchesByInvestor.set(investor.id, [...existing, newMatch]);
  }

  // Combine and sort
  const allMatches = [...preservedMatches, ...newMatches];
  allMatches.sort((a, b) => {
    const timeA = slotsToUse.indexOf(a.slotTime);
    const timeB = slotsToUse.indexOf(b.slotTime);
    return timeA - timeB;
  });

  return allMatches;
}

/**
 * Export configuration for customization
 */
export { DEFAULT_CONFIG, type MatchingConfig, type CompatibilityScore };

