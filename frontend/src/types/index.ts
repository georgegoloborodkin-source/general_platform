// User & Organization Types
// Roles: organizer/managing_partner (MD), team_member, lp (limited partner read-only)
export type UserRole = 'organizer' | 'managing_partner' | 'team_member' | 'lp';

export interface UserProfile {
  id: string;
  email: string | null;
  full_name: string | null;
  role: UserRole;
  organization_id: string | null;
  created_at: string;
  updated_at: string;
}

export interface Organization {
  id: string;
  name: string;
  slug: string;
  subscription_tier: 'free' | 'starter' | 'professional' | 'enterprise';
  created_at: string;
  updated_at: string;
}

export interface Event {
  id: string;
  organization_id: string;
  name: string;
  date: string | null;
  status: 'draft' | 'active' | 'completed';
  created_at: string;
  updated_at: string;
}

export interface DecisionLog {
  id: string;
  event_id: string;
  actor_id: string | null;
  actor_name: string;
  action_type: string;
  startup_name: string;
  context: Record<string, any> | null;
  confidence_score: number;
  outcome: string | null;
  notes: string | null;
  document_id?: string | null;
  created_at: string;
  updated_at: string;
}

export interface DocumentRecord {
  id: string;
  event_id: string;
  title: string | null;
  source_type: string;
  file_name: string | null;
  storage_path: string | null;
  detected_type: string | null;
  extracted_json: Record<string, any> | null;
  created_by: string | null;
  folder_id?: string | null;
  gmail_message_id?: string | null;
  gmail_thread_id?: string | null;
  gmail_labels?: string[] | null;
  email_from?: string | null;
  email_to?: string[] | null;
  email_cc?: string[] | null;
  email_subject?: string | null;
  email_sent_at?: string | null;
  email_has_attachments?: boolean;
  created_at: string;
  updated_at: string;
}

export interface ChatThread {
  id: string;
  event_id: string;
  parent_id: string | null;
  title: string;
  created_by: string | null;
  created_at: string;
}

export interface ChatMessage {
  id: string;
  event_id: string;
  thread_id: string;
  role: "user" | "assistant" | "system";
  content: string;
  model: string | null;
  source_doc_ids: string[] | null;
  created_by: string | null;
  created_at: string;
}

export interface SourceRecord {
  id: string;
  event_id: string;
  title: string | null;
  source_type: 'syndicate' | 'company' | 'deck' | 'notes' | 'other';
  external_url: string | null;
  storage_path: string | null;
  tags: string[] | null;
  notes: string | null;
  status: 'active' | 'archived';
  created_by: string | null;
  created_at: string;
  updated_at: string;
}

export type TaskStatus = 'not_started' | 'in_progress' | 'done' | 'cancelled';

export interface Task {
  id: string;
  event_id: string;
  assignee_user_id: string | null;
  title: string;
  description: string | null;
  status: TaskStatus;
  start_date: string | null;
  deadline: string | null;
  status_note: string | null;
  created_by: string | null;
  created_at: string;
  updated_at: string;
}