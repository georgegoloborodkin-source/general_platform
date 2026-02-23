export interface Organization {
  id: string;
  name: string;
  slug: string;
  company_description: string;
  system_prompt: string;
  industry_hint: string;
  subscription_tier: 'free' | 'starter' | 'professional' | 'enterprise';
  invitation_code: string | null;
  created_at: string;
  updated_at: string;
}

export type UserRole = 'admin' | 'team_member';

export interface UserProfile {
  id: string;
  email: string | null;
  full_name: string | null;
  role: UserRole;
  organization_id: string | null;
  created_at: string;
  updated_at: string;
}

export interface Event {
  id: string;
  organization_id: string;
  name: string;
  status: 'draft' | 'active' | 'completed';
  created_at: string;
  updated_at: string;
}

export interface DocumentRecord {
  id: string;
  event_id: string;
  title: string | null;
  source_type: string;
  file_name: string | null;
  raw_content: string | null;
  detected_type: string | null;
  extracted_json: Record<string, any> | null;
  created_by: string | null;
  created_at: string;
  updated_at: string;
}

export interface ChatThread {
  id: string;
  event_id: string;
  title: string;
  created_by: string | null;
  created_at: string;
}

export interface ChatMessage {
  id: string;
  event_id: string;
  thread_id: string;
  role: 'user' | 'assistant' | 'system';
  content: string;
  model: string | null;
  created_by: string | null;
  created_at: string;
}
