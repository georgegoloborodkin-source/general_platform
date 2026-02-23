import { Investor, Match, Startup, TimeSlotConfig } from "@/types";

function pad2(n: number): string {
  return String(n).padStart(2, "0");
}

function yyyymmdd(date: Date): string {
  return `${date.getFullYear()}${pad2(date.getMonth() + 1)}${pad2(date.getDate())}`;
}

function hhmm(time: string): { hh: number; mm: number } | null {
  const m = /^(\d{2}):(\d{2})$/.exec(time.trim());
  if (!m) return null;
  return { hh: Number(m[1]), mm: Number(m[2]) };
}

function dtLocal(date: Date, time: string): string | null {
  const t = hhmm(time);
  if (!t) return null;
  return `${yyyymmdd(date)}T${pad2(t.hh)}${pad2(t.mm)}00`;
}

export function buildMemberCallSheetIcs(
  memberName: string,
  matches: Match[],
  startups: Startup[],
  investors: Investor[],
  timeSlots: TimeSlotConfig[],
  day: Date = new Date()
): string {
  const byLabel = new Map(timeSlots.map((ts) => [ts.label, ts]));

  const lines: string[] = [];
  lines.push("BEGIN:VCALENDAR");
  lines.push("VERSION:2.0");
  lines.push("PRODID:-//Investormatching-platform//CallSheet//EN");
  lines.push("CALSCALE:GREGORIAN");

  const memberMatches = matches.filter((m) => {
    const inv = investors.find((i) => i.id === m.investorId);
    return inv?.memberName === memberName;
  });

  for (const m of memberMatches) {
    const slot = byLabel.get(m.timeSlot);
    if (!slot) continue;
    const dtStart = dtLocal(day, slot.startTime);
    const dtEnd = dtLocal(day, slot.endTime);
    if (!dtStart || !dtEnd) continue;

    const inv = investors.find((i) => i.id === m.investorId);
    const st = startups.find((s) => s.id === m.startupId);
    const summary = `Call: ${st?.companyName || m.startupName}`;
    const location = inv?.tableNumber ? `Table ${inv.tableNumber}` : "";
    const description = `Investor: ${inv?.firmName || ""} (${inv?.memberName || ""})\\nStartup: ${st?.companyName || m.startupName}`;

    lines.push("BEGIN:VEVENT");
    lines.push(`UID:${m.id}@investormatching`);
    lines.push(`DTSTART:${dtStart}`);
    lines.push(`DTEND:${dtEnd}`);
    lines.push(`SUMMARY:${summary}`);
    if (location) lines.push(`LOCATION:${location}`);
    lines.push(`DESCRIPTION:${description}`);
    lines.push("END:VEVENT");
  }

  lines.push("END:VCALENDAR");
  return lines.join("\r\n");
}


