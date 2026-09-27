export type SentimentLabel = 'positive' | 'negative' | 'neutral' | string;

export interface ChatMessage {
  id?: string;
  timestamp: number;
  channel: string;
  user: string;
  message: string;
  label: SentimentLabel;
  score: number;
  latency: number;
}

export interface RealtimeMetrics {
  channel: string;
  window_seconds: number;
  total_messages: number;
  pos_ratio: number;
  neg_ratio: number;
  neu_ratio: number;
  avg_latency_ms: number;
  messages_per_second: number;
}

export interface TimelineMetricItem {
  channel: string;
  minute_timestamp: number;
  pos_count: number;
  neu_count: number;
  neg_count: number;
  avg_pos_score: number;
  avg_neg_score: number;
  total_messages: number;
}

export interface ChannelStatus {
  channel: string;
  connected: boolean;
  vod_id: string | null;
  stream_start: number | null;
  active_subscribers: number;
}

export interface HealthResponse {
  status: string;
  db_type: string;
  model_loaded: boolean;
  active_channels: string[];
}

export type SocketStatus = 'disconnected' | 'connecting' | 'connected' | 'error';
