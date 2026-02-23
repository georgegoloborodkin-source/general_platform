import { Startup, Investor, FUNDING_STAGES } from "@/types";

// Column name variations mapping
const STARTUP_COLUMN_MAPPINGS: Record<string, string[]> = {
  companyName: [
    'company_name', 'companyname', 'company', 'name', 'startup_name', 
    'startupname', 'company name', 'startup name', 'business name', 'firm'
  ],
  geoMarkets: [
    'geo_markets', 'geomarkets', 'geographic_markets', 'geographicmarkets',
    'markets', 'geography', 'regions', 'locations', 'geo', 'geographic regions',
    'target markets', 'market', 'region'
  ],
  industry: [
    'industry', 'sector', 'vertical', 'category', 'industry_type',
    'business_sector', 'businesssector', 'industry category'
  ],
  fundingTarget: [
    'funding_target', 'fundingtarget', 'target', 'funding_amount',
    'fundingamount', 'amount', 'raise_amount', 'raiseamount',
    'funding goal', 'funding_goal', 'fundinggoal', 'investment_target',
    'investmenttarget', 'seeking', 'amount_raising', 'amountraising'
  ],
  fundingStage: [
    'funding_stage', 'fundingstage', 'stage', 'round', 'funding_round',
    'fundinground', 'investment_stage', 'investmentstage', 'round_stage',
    'roundstage', 'stage of funding', 'current_stage', 'currentstage'
  ]
};

const INVESTOR_COLUMN_MAPPINGS: Record<string, string[]> = {
  firmName: [
    'firm_name', 'firmname', 'firm', 'name', 'investor_name', 'investorname',
    'company_name', 'companyname', 'company', 'vc_name', 'vcname',
    'venture_capital', 'venturecapital', 'fund_name', 'fundname'
  ],
  memberName: [
    'investor_member_name', 'investormembername', 'member_name', 'membername',
    'investment_member', 'investmentmember',
    'contact_name', 'contactname', 'partner_name', 'partnername',
    'person_name', 'personname', 'representative', 'rep_name', 'repname',
    'primary_contact', 'primarycontact', 'investment_manager', 'investmentmanager'
  ],
  geoFocus: [
    'geo_focus', 'geofocus', 'geographic_focus', 'geographicfocus',
    'focus_regions', 'focusregions', 'target_regions', 'targetregions',
    'geography', 'regions', 'markets', 'locations', 'geo', 'geographic regions'
  ],
  industryPreferences: [
    'industry_preferences', 'industrypreferences', 'preferred_industries',
    'preferredindustries', 'industries', 'sectors', 'vertical_preferences',
    'verticalpreferences', 'industry_focus', 'industryfocus', 'sector_focus',
    'sectorfocus', 'preferences', 'interested_industries', 'interestedindustries'
  ],
  stagePreferences: [
    'stage_preferences', 'stagepreferences', 'preferred_stages', 'preferredstages',
    'stages', 'funding_stages', 'fundingstages', 'round_preferences',
    'roundpreferences', 'investment_stages', 'investmentstages', 'stage_focus',
    'stagefocus', 'preferred_rounds', 'preferredrounds'
  ],
  minTicketSize: [
    'min_ticket_size', 'minticketsize', 'minimum_ticket', 'minimumticket',
    'min_investment', 'mininvestment', 'min_amount', 'minamount',
    'minimum_investment', 'minimuminvestment', 'min_ticket', 'minticket',
    'ticket_min', 'ticketmin', 'min check', 'min_check', 'mincheck'
  ],
  maxTicketSize: [
    'max_ticket_size', 'maxticketsize', 'maximum_ticket', 'maximumticket',
    'max_investment', 'maxinvestment', 'max_amount', 'maxamount',
    'maximum_investment', 'maximuminvestment', 'max_ticket', 'maxticket',
    'ticket_max', 'ticketmax', 'max check', 'max_check', 'maxcheck'
  ],
  totalSlots: [
    'total_slots', 'totalslots', 'slots', 'meeting_slots', 'meetingslots',
    'number_of_slots', 'numberofslots', 'available_slots', 'availableslots',
    'capacity', 'meetings', 'max_meetings', 'maxmeetings', 'slot_count',
    'slotcount', 'meeting_capacity', 'meetingcapacity'
  ],
  tableNumber: [
    'table_number', 'tablenumber', 'table', 'table_num', 'tablenum',
    'location', 'booth', 'room', 'station', 'table_id', 'tableid'
  ]
};

interface ColumnMapping {
  originalName: string;
  mappedField: string;
  confidence: number; // 0-1
}

interface ConversionResult<T> {
  data: T[];
  mappings: ColumnMapping[];
  warnings: string[];
  errors: string[];
  detectedSeparator: string;
  totalRows: number;
  validRows: number;
  skippedRows: number;
}

/**
 * Detect CSV separator (comma, semicolon, tab)
 */
function detectSeparator(firstLine: string): string {
  const commaCount = (firstLine.match(/,/g) || []).length;
  const semicolonCount = (firstLine.match(/;/g) || []).length;
  const tabCount = (firstLine.match(/\t/g) || []).length;

  if (tabCount > commaCount && tabCount > semicolonCount) return '\t';
  if (semicolonCount > commaCount) return ';';
  return ',';
}

/**
 * Normalize column name for matching
 */
function normalizeColumnName(name: string): string {
  return name
    .trim()
    .toLowerCase()
    .replace(/[_\s-]/g, '') // Remove underscores, spaces, hyphens
    .replace(/[^\w]/g, ''); // Remove special characters
}

/**
 * Find best column mapping
 */
function findColumnMapping(
  columnName: string,
  mappings: Record<string, string[]>
): { field: string; confidence: number } | null {
  const normalized = normalizeColumnName(columnName);
  
  // Exact match
  for (const [field, variations] of Object.entries(mappings)) {
    for (const variation of variations) {
      if (normalizeColumnName(variation) === normalized) {
        return { field, confidence: 1.0 };
      }
    }
  }

  // Partial match (contains)
  for (const [field, variations] of Object.entries(mappings)) {
    for (const variation of variations) {
      const normalizedVariation = normalizeColumnName(variation);
      if (normalized.includes(normalizedVariation) || normalizedVariation.includes(normalized)) {
        return { field, confidence: 0.7 };
      }
    }
  }

  // Fuzzy match (similar words)
  const columnWords = normalized.split(/\s+/);
  for (const [field, variations] of Object.entries(mappings)) {
    for (const variation of variations) {
      const variationWords = normalizeColumnName(variation).split(/\s+/);
      const commonWords = columnWords.filter(w => variationWords.includes(w));
      if (commonWords.length > 0 && commonWords.length / Math.max(columnWords.length, variationWords.length) > 0.5) {
        return { field, confidence: 0.5 };
      }
    }
  }

  return null;
}

/**
 * Parse CSV with smart column detection
 */
function parseCSV(content: string): { headers: string[]; rows: string[][]; separator: string } {
  const lines = content.split(/\r?\n/).filter(line => line.trim());
  if (lines.length === 0) {
    throw new Error('CSV file is empty');
  }

  const separator = detectSeparator(lines[0]);
  const headers = lines[0].split(separator).map(h => h.trim().replace(/^["']|["']$/g, ''));
  const rows: string[][] = [];

  for (let i = 1; i < lines.length; i++) {
    // Handle quoted values with commas
    const row: string[] = [];
    let current = '';
    let inQuotes = false;
    
    for (let j = 0; j < lines[i].length; j++) {
      const char = lines[i][j];
      const nextChar = lines[i][j + 1];

      if (char === '"' || char === "'") {
        if (inQuotes && nextChar === char) {
          // Escaped quote
          current += char;
          j++; // Skip next quote
        } else {
          inQuotes = !inQuotes;
        }
      } else if (char === separator && !inQuotes) {
        row.push(current.trim());
        current = '';
      } else {
        current += char;
      }
    }
    row.push(current.trim()); // Last value

    // Pad row if needed
    while (row.length < headers.length) {
      row.push('');
    }

    rows.push(row.slice(0, headers.length));
  }

  return { headers, rows, separator };
}

/**
 * Parse number with various formats
 */
function parseNumber(value: string): number {
  if (!value || value.trim() === '') return 0;
  
  // Remove currency symbols, commas, spaces
  const cleaned = value
    .replace(/[$€£¥,\s]/g, '')
    .replace(/[^\d.-]/g, '');
  
  const num = parseFloat(cleaned);
  return isNaN(num) ? 0 : num;
}

/**
 * Parse list (semicolon, comma, or pipe separated)
 */
function parseList(value: string): string[] {
  if (!value || value.trim() === '') return [];
  
  // Try different separators
  const separators = [';', ',', '|', '\n'];
  for (const sep of separators) {
    if (value.includes(sep)) {
      return value.split(sep).map(item => item.trim()).filter(item => item);
    }
  }
  
  // Single value
  return [value.trim()].filter(item => item);
}

/**
 * Smart startup CSV converter
 */
export function smartConvertStartupCSV(csvContent: string): ConversionResult<Startup> {
  const result: ConversionResult<Startup> = {
    data: [],
    mappings: [],
    warnings: [],
    errors: [],
    detectedSeparator: ',',
    totalRows: 0,
    validRows: 0,
    skippedRows: 0
  };

  try {
    const { headers, rows, separator } = parseCSV(csvContent);
    result.detectedSeparator = separator;
    result.totalRows = rows.length;

    // Map columns
    const columnMap: Record<number, string> = {};
    const requiredFields = ['companyName'];
    const foundFields = new Set<string>();

    headers.forEach((header, index) => {
      const mapping = findColumnMapping(header, STARTUP_COLUMN_MAPPINGS);
      if (mapping) {
        columnMap[index] = mapping.field;
        result.mappings.push({
          originalName: header,
          mappedField: mapping.field,
          confidence: mapping.confidence
        });
        foundFields.add(mapping.field);
      } else {
        result.warnings.push(`Column "${header}" not recognized - will be ignored`);
      }
    });

    // Check for required fields
    if (!foundFields.has('companyName')) {
      result.errors.push('Required column "company_name" or "company name" not found');
      return result;
    }

    // Parse rows
    rows.forEach((row, rowIndex) => {
      try {
        const startup: Partial<Startup> = {
          id: `startup-${Date.now()}-${rowIndex}`,
          availabilityStatus: 'present' as const
        };

        // Map each column
        Object.entries(columnMap).forEach(([colIndex, field]) => {
          const value = row[parseInt(colIndex)] || '';
          
          switch (field) {
            case 'companyName':
              startup.companyName = value || '';
              break;
            case 'geoMarkets':
              startup.geoMarkets = parseList(value);
              break;
            case 'industry':
              startup.industry = value || '';
              break;
            case 'fundingTarget':
              startup.fundingTarget = parseNumber(value);
              break;
            case 'fundingStage':
              startup.fundingStage = value || '';
              break;
          }
        });

        // Validate required fields (row-level, MVP strict)
        const missing: string[] = [];
        if (!startup.companyName || !startup.companyName.trim()) missing.push('company_name');
        if (!startup.geoMarkets || startup.geoMarkets.length === 0) missing.push('geo_markets');
        if (!startup.industry || !startup.industry.trim()) missing.push('industry');
        if (!startup.fundingTarget || startup.fundingTarget <= 0) missing.push('funding_target');
        if (!startup.fundingStage || !startup.fundingStage.trim()) missing.push('funding_stage');

        if (missing.length > 0) {
          result.errors.push(`Row ${rowIndex + 2}: Missing/invalid ${missing.join(', ')}`);
          result.skippedRows++;
          return;
        }

        result.data.push(startup as Startup);
        result.validRows++;
      } catch (error) {
        result.warnings.push(`Row ${rowIndex + 2}: ${error instanceof Error ? error.message : 'Parse error'}`);
        result.skippedRows++;
      }
    });

    if (result.data.length === 0) {
      result.errors.push('No valid startup data found after conversion');
    }

  } catch (error) {
    result.errors.push(error instanceof Error ? error.message : 'Failed to parse CSV');
  }

  return result;
}

/**
 * Smart investor CSV converter
 */
export function smartConvertInvestorCSV(csvContent: string): ConversionResult<Investor> {
  const result: ConversionResult<Investor> = {
    data: [],
    mappings: [],
    warnings: [],
    errors: [],
    detectedSeparator: ',',
    totalRows: 0,
    validRows: 0,
    skippedRows: 0
  };

  try {
    const { headers, rows, separator } = parseCSV(csvContent);
    result.detectedSeparator = separator;
    result.totalRows = rows.length;

    // Map columns
    const columnMap: Record<number, string> = {};
    const requiredFields = ['firmName', 'memberName'];
    const foundFields = new Set<string>();

    headers.forEach((header, index) => {
      const mapping = findColumnMapping(header, INVESTOR_COLUMN_MAPPINGS);
      if (mapping) {
        columnMap[index] = mapping.field;
        result.mappings.push({
          originalName: header,
          mappedField: mapping.field,
          confidence: mapping.confidence
        });
        foundFields.add(mapping.field);
      } else {
        result.warnings.push(`Column "${header}" not recognized - will be ignored`);
      }
    });

    // Check for required fields
    if (!foundFields.has('firmName')) {
      result.errors.push('Required column "firm_name" or "firm name" not found');
      return result;
    }
    if (!foundFields.has('memberName')) {
      result.errors.push('Required column "investment_member" (or similar: investor_member_name/member_name/contact_name/partner_name) not found');
      return result;
    }

    // Parse rows
    rows.forEach((row, rowIndex) => {
      try {
        const investor: Partial<Investor> = {
          id: `investor-${Date.now()}-${rowIndex}`,
          availabilityStatus: 'present' as const,
          totalSlots: 3 // Default
        };

        // Map each column
        Object.entries(columnMap).forEach(([colIndex, field]) => {
          const value = row[parseInt(colIndex)] || '';
          
          switch (field) {
            case 'firmName':
              investor.firmName = value || '';
              break;
            case 'memberName':
              investor.memberName = value || '';
              break;
            case 'geoFocus':
              investor.geoFocus = parseList(value);
              break;
            case 'industryPreferences':
              investor.industryPreferences = parseList(value);
              break;
            case 'stagePreferences':
              investor.stagePreferences = parseList(value);
              break;
            case 'minTicketSize':
              investor.minTicketSize = parseNumber(value);
              break;
            case 'maxTicketSize':
              investor.maxTicketSize = parseNumber(value);
              break;
            case 'totalSlots':
              investor.totalSlots = parseInt(value) || 3;
              break;
            case 'tableNumber':
              investor.tableNumber = value || '';
              break;
          }
        });

        // Default stage preferences if missing (template does not include stage_preferences)
        if (!investor.stagePreferences || investor.stagePreferences.length === 0) {
          investor.stagePreferences = [...FUNDING_STAGES];
        }

        // Validate required fields (row-level, MVP strict)
        const missing: string[] = [];
        if (!investor.firmName || !investor.firmName.trim()) missing.push('firm_name');
        if (!investor.memberName || !investor.memberName.trim()) missing.push('investment_member');
        if (!investor.geoFocus || investor.geoFocus.length === 0) missing.push('geo_focus');
        if (!investor.industryPreferences || investor.industryPreferences.length === 0) missing.push('industry_preferences');
        if (!investor.minTicketSize || investor.minTicketSize <= 0) missing.push('min_ticket_size');
        if (!investor.maxTicketSize || investor.maxTicketSize <= 0) missing.push('max_ticket_size');
        if (!investor.totalSlots || investor.totalSlots <= 0) missing.push('total_slots');

        if (missing.length > 0) {
          result.errors.push(`Row ${rowIndex + 2}: Missing/invalid ${missing.join(', ')}`);
          result.skippedRows++;
          return;
        }

        result.data.push(investor as Investor);
        result.validRows++;
      } catch (error) {
        result.warnings.push(`Row ${rowIndex + 2}: ${error instanceof Error ? error.message : 'Parse error'}`);
        result.skippedRows++;
      }
    });

    if (result.data.length === 0) {
      result.errors.push('No valid investor data found after conversion');
    }

  } catch (error) {
    result.errors.push(error instanceof Error ? error.message : 'Failed to parse CSV');
  }

  return result;
}

/**
 * Auto-detect if CSV is for startups or investors
 */
export function detectCSVType(csvContent: string): 'startups' | 'investors' | 'unknown' {
  try {
    const { headers } = parseCSV(csvContent);
    const normalizedHeaders = headers.map(h => normalizeColumnName(h));

    // Check for startup indicators (more comprehensive)
    const startupIndicators = [
      'company', 'startup', 'fundingtarget', 'fundingstage', 'fundinggoal',
      'raise', 'amountraising', 'business', 'venture', 'fundingamount',
      'investmenttarget', 'seeking', 'round', 'stage'
    ];
    const startupScore = normalizedHeaders.filter(h => 
      startupIndicators.some(indicator => h.includes(indicator))
    ).length;

    // Check for investor indicators (more comprehensive)
    const investorIndicators = [
      'firm', 'vc', 'fund', 'investor', 'ticketsize', 'industrypreferences',
      'stagepreferences', 'slots', 'meeting', 'capacity', 'preferred',
      'mininvestment', 'maxinvestment', 'check', 'ticket'
    ];
    const investorScore = normalizedHeaders.filter(h => 
      investorIndicators.some(indicator => h.includes(indicator))
    ).length;

    if (startupScore > investorScore && startupScore > 0) return 'startups';
    if (investorScore > startupScore && investorScore > 0) return 'investors';
    return 'unknown';
  } catch {
    return 'unknown';
  }
}

/**
 * Detect if CSV contains both startups and investors (mixed format)
 */
export function detectMixedCSV(csvContent: string): boolean {
  try {
    const { headers } = parseCSV(csvContent);
    const normalizedHeaders = headers.map(h => normalizeColumnName(h));

    // Check for both startup and investor indicators
    const hasStartupIndicators = normalizedHeaders.some(h => 
      ['company', 'startup', 'fundingtarget', 'fundingstage'].some(ind => h.includes(ind))
    );
    
    const hasInvestorIndicators = normalizedHeaders.some(h => 
      ['firm', 'ticketsize', 'industrypreferences', 'slots'].some(ind => h.includes(ind))
    );

    return hasStartupIndicators && hasInvestorIndicators;
  } catch {
    return false;
  }
}

/**
 * Parse mixed CSV that contains both startups and investors
 * Uses a "type" column or auto-detects based on data patterns
 */
export function parseMixedCSV(csvContent: string): {
  startups: Startup[];
  investors: Investor[];
  mappings: {
    startups: ColumnMapping[];
    investors: ColumnMapping[];
  };
  warnings: string[];
  errors: string[];
} {
  const result = {
    startups: [] as Startup[],
    investors: [] as Investor[],
    mappings: {
      startups: [] as ColumnMapping[],
      investors: [] as ColumnMapping[]
    },
    warnings: [] as string[],
    errors: [] as string[]
  };

  try {
    const { headers, rows } = parseCSV(csvContent);
    
    // Check if there's a "type" column
    const typeColumnIndex = headers.findIndex(h => 
      normalizeColumnName(h).includes('type') || 
      normalizeColumnName(h).includes('category') ||
      normalizeColumnName(h).includes('kind')
    );

    // If no type column, try to detect per row
    rows.forEach((row, rowIndex) => {
      let rowType: 'startup' | 'investor' | 'unknown' = 'unknown';

      if (typeColumnIndex >= 0) {
        const typeValue = normalizeColumnName(row[typeColumnIndex] || '');
        if (typeValue.includes('startup') || typeValue.includes('company')) {
          rowType = 'startup';
        } else if (typeValue.includes('investor') || typeValue.includes('firm') || typeValue.includes('vc')) {
          rowType = 'investor';
        }
      }

      // Auto-detect based on data patterns
      if (rowType === 'unknown') {
        const rowData = row.join(' ').toLowerCase();
        const hasStartupData = 
          (row[headers.findIndex(h => normalizeColumnName(h).includes('company'))] || '').length > 0 ||
          (row[headers.findIndex(h => normalizeColumnName(h).includes('fundingtarget'))] || '').length > 0;
        
        const hasInvestorData = 
          (row[headers.findIndex(h => normalizeColumnName(h).includes('firm'))] || '').length > 0 ||
          (row[headers.findIndex(h => normalizeColumnName(h).includes('ticketsize'))] || '').length > 0;

        if (hasStartupData && !hasInvestorData) rowType = 'startup';
        else if (hasInvestorData && !hasStartupData) rowType = 'investor';
        else if (hasStartupData && hasInvestorData) {
          // Ambiguous - check more indicators
          const startupScore = ['company', 'startup', 'funding'].filter(term => 
            rowData.includes(term)
          ).length;
          const investorScore = ['firm', 'vc', 'investor', 'ticket'].filter(term => 
            rowData.includes(term)
          ).length;
          rowType = startupScore > investorScore ? 'startup' : 'investor';
        }
      }

      // Parse based on detected type
      if (rowType === 'startup') {
        // Build a row object from headers and values
        const rowData: Record<string, string> = {};
        headers.forEach((header, idx) => {
          rowData[header] = row[idx] || '';
        });
        
        // Create a temporary CSV with just this row (properly quoted)
        const quotedRow = row.map(val => {
          const str = String(val || '');
          if (str.includes(',') || str.includes('"') || str.includes('\n')) {
            return `"${str.replace(/"/g, '""')}"`;
          }
          return str;
        });
        const tempCSV = [headers.join(','), quotedRow.join(',')].join('\n');
        
        const startupResult = smartConvertStartupCSV(tempCSV);
        if (startupResult.data.length > 0) {
          result.startups.push(startupResult.data[0]);
          // Only add unique mappings
          startupResult.mappings.forEach(mapping => {
            if (!result.mappings.startups.find(m => m.originalName === mapping.originalName)) {
              result.mappings.startups.push(mapping);
            }
          });
        }
        result.warnings.push(...startupResult.warnings.map(w => `Row ${rowIndex + 2}: ${w}`));
      } else if (rowType === 'investor') {
        // Build a row object from headers and values
        const rowData: Record<string, string> = {};
        headers.forEach((header, idx) => {
          rowData[header] = row[idx] || '';
        });
        
        // Create a temporary CSV with just this row (properly quoted)
        const quotedRow = row.map(val => {
          const str = String(val || '');
          if (str.includes(',') || str.includes('"') || str.includes('\n')) {
            return `"${str.replace(/"/g, '""')}"`;
          }
          return str;
        });
        const tempCSV = [headers.join(','), quotedRow.join(',')].join('\n');
        
        const investorResult = smartConvertInvestorCSV(tempCSV);
        if (investorResult.data.length > 0) {
          result.investors.push(investorResult.data[0]);
          // Only add unique mappings
          investorResult.mappings.forEach(mapping => {
            if (!result.mappings.investors.find(m => m.originalName === mapping.originalName)) {
              result.mappings.investors.push(mapping);
            }
          });
        }
        result.warnings.push(...investorResult.warnings.map(w => `Row ${rowIndex + 2}: ${w}`));
      } else {
        result.warnings.push(`Row ${rowIndex + 2}: Could not determine if startup or investor - skipped`);
      }
    });

  } catch (error) {
    result.errors.push(error instanceof Error ? error.message : 'Failed to parse mixed CSV');
  }

  return result;
}

