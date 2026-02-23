import { Startup, Investor, Match, FUNDING_STAGES } from "@/types";

export function parseStartupCSV(csvContent: string): Startup[] {
  const lines = csvContent.split('\n').filter(line => line.trim());
  const headers = lines[0].split(',').map(h => h.trim().toLowerCase());
  
  const startups: Startup[] = [];
  
  for (let i = 1; i < lines.length; i++) {
    const values = lines[i].split(',').map(v => v.trim());
    
    if (values.length < headers.length) continue;
    
    try {
      const startup: Startup = {
        id: `startup-${Date.now()}-${i}`,
        companyName: values[headers.indexOf('company_name')] || values[headers.indexOf('companyname')] || '',
        geoMarkets: (values[headers.indexOf('geo_markets')] || values[headers.indexOf('geomarkets')] || '')
          .split(';').map(m => m.trim()).filter(m => m),
        industry: values[headers.indexOf('industry')] || '',
        fundingTarget: parseInt(values[headers.indexOf('funding_target')] || values[headers.indexOf('fundingtarget')] || '0'),
        fundingStage: values[headers.indexOf('funding_stage')] || values[headers.indexOf('fundingstage')] || '',
        availabilityStatus: 'present'
      };
      
      if (startup.companyName) {
        startups.push(startup);
      }
    } catch (error) {
      console.error(`Error parsing startup row ${i}:`, error);
    }
  }
  
  return startups;
}

export function parseInvestorCSV(csvContent: string): Investor[] {
  const lines = csvContent.split('\n').filter(line => line.trim());
  const headers = lines[0].split(',').map(h => h.trim().toLowerCase());
  
  const investors: Investor[] = [];
  
  for (let i = 1; i < lines.length; i++) {
    const values = lines[i].split(',').map(v => v.trim());
    
    if (values.length < headers.length) continue;
    
    try {
      const investor: Investor = {
        id: `investor-${Date.now()}-${i}`,
        firmName: values[headers.indexOf('firm_name')] || values[headers.indexOf('firmname')] || '',
        memberName:
          values[headers.indexOf('investor_member_name')] ||
          values[headers.indexOf('investment_member')] ||
          values[headers.indexOf('investmentmember')] ||
          values[headers.indexOf('investormembername')] ||
          values[headers.indexOf('member_name')] ||
          values[headers.indexOf('membername')] ||
          values[headers.indexOf('contact_name')] ||
          values[headers.indexOf('contactname')] ||
          values[headers.indexOf('partner_name')] ||
          values[headers.indexOf('partnername')] ||
          '',
        geoFocus: (values[headers.indexOf('geo_focus')] || values[headers.indexOf('geofocus')] || '')
          .split(';').map(g => g.trim()).filter(g => g),
        industryPreferences: (values[headers.indexOf('industry_preferences')] || values[headers.indexOf('industrypreferences')] || '')
          .split(';').map(i => i.trim()).filter(i => i),
        stagePreferences: (values[headers.indexOf('stage_preferences')] || values[headers.indexOf('stagepreferences')] || '')
          .split(';').map(s => s.trim()).filter(s => s),
        minTicketSize: parseInt(values[headers.indexOf('min_ticket_size')] || values[headers.indexOf('minticketsize')] || '0'),
        maxTicketSize: parseInt(values[headers.indexOf('max_ticket_size')] || values[headers.indexOf('maxticketsize')] || '0'),
        totalSlots: parseInt(values[headers.indexOf('total_slots')] || values[headers.indexOf('totalslots')] || '3'),
        tableNumber: values[headers.indexOf('table_number')] || values[headers.indexOf('tablenumber')] || '',
        availabilityStatus: 'present'
      };

      // If stage preferences not provided, default to all stages
      if (!investor.stagePreferences || investor.stagePreferences.length === 0) {
        investor.stagePreferences = [...FUNDING_STAGES];
      }
      
      if (investor.firmName && investor.memberName) {
        investors.push(investor);
      }
    } catch (error) {
      console.error(`Error parsing investor row ${i}:`, error);
    }
  }
  
  return investors;
}

export function exportMatchesToCSV(matches: Match[], startups: Startup[], investors: Investor[]): string {
  const headers = [
    'Startup Name',
    'Investor Name',
    'Investor Table Number',
    'Time Slot',
    'Slot Time',
    'Compatibility Score',
    'Score Breakdown',
    'Status',
    'Completed',
    'Startup Industry',
    'Startup Funding Target',
    'Startup Funding Stage',
    'Investor Ticket Range',
    'Startup Geo Markets',
    'Investor Geo Focus'
  ];
  
  const rows = matches.map(match => {
    const startup = startups.find(s => s.id === match.startupId);
    const investor = investors.find(i => i.id === match.investorId);
    
    return [
      match.startupName,
      match.investorName,
      investor?.tableNumber || '',
      match.timeSlot,
      match.slotTime || '',
      match.compatibilityScore + '%',
      (match.scoreBreakdown || []).join(' | '),
      match.status,
      match.completed ? 'Yes' : 'No',
      startup?.industry || '',
      startup ? `$${startup.fundingTarget.toLocaleString()}` : '',
      startup?.fundingStage || '',
      investor ? `$${investor.minTicketSize.toLocaleString()} - $${investor.maxTicketSize.toLocaleString()}` : '',
      startup?.geoMarkets.join('; ') || '',
      investor?.geoFocus.join('; ') || ''
    ];
  });
  
  return [headers, ...rows].map(row => row.join(',')).join('\n');
}

export function downloadCSV(content: string, filename: string): void {
  const blob = new Blob([content], { type: 'text/csv;charset=utf-8;' });
  const link = document.createElement('a');
  
  if (link.download !== undefined) {
    const url = URL.createObjectURL(blob);
    link.setAttribute('href', url);
    link.setAttribute('download', filename);
    link.style.visibility = 'hidden';
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  }
}

export function generateStartupCSVTemplate(): string {
  const headers = [
    'company_name',
    'geo_markets', // semicolon separated: "North America;Europe"
    'industry',
    'funding_target', // number
    'funding_stage'
  ];
  
  const sampleData = [
    'TechFlow AI',
    'North America;Europe',
    'AI/ML',
    '2000000',
    'Series A'
  ];
  
  return [headers, sampleData].map(row => row.join(',')).join('\n');
}

export function generateInvestorCSVTemplate(): string {
  const headers = [
    'firm_name',
    'investment_member',
    'geo_focus', // semicolon separated: "North America;Europe"
    'industry_preferences', // semicolon separated: "AI/ML;SaaS"
    'min_ticket_size', // number
    'max_ticket_size', // number
    'total_slots', // number
    'table_number'
  ];
  
  const sampleData = [
    'Venture Capital Partners',
    'Jane Doe',
    'North America;Europe',
    'AI/ML;SaaS',
    '1000000',
    '5000000',
    '4',
    'A1'
  ];
  
  return [headers, sampleData].map(row => row.join(',')).join('\n');
}