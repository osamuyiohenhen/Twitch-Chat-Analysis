import type {
  ChannelStatus,
  ChatMessage,
  HealthResponse,
  RealtimeMetrics,
  TimelineMetricItem,
} from '../types/sentiment';

const API_BASE = import.meta.env.VITE_API_URL || 'http://localhost:8000';
export const WS_BASE = import.meta.env.VITE_WS_URL || 'ws://localhost:8000';

async function handleResponse<T>(res: Response): Promise<T> {
  if (!res.ok) {
    const errorBody = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(errorBody.detail || errorBody.message || `Request failed with status ${res.status}`);
  }
  return res.json() as Promise<T>;
}

export const api = {
  async getHealth(): Promise<HealthResponse> {
    const res = await fetch(`${API_BASE}/health`);
    return handleResponse<HealthResponse>(res);
  },

  async connectChannel(channel: string): Promise<{ channel: string; status: string; message: string; vod_id?: string }> {
    const res = await fetch(`${API_BASE}/api/v1/channels/${encodeURIComponent(channel)}/connect`, {
      method: 'POST',
    });
    return handleResponse(res);
  },

  async disconnectChannel(channel: string): Promise<{ channel: string; status: string; message: string }> {
    const res = await fetch(`${API_BASE}/api/v1/channels/${encodeURIComponent(channel)}/disconnect`, {
      method: 'POST',
    });
    return handleResponse(res);
  },

  async getChannelStatus(channel: string): Promise<ChannelStatus> {
    const res = await fetch(`${API_BASE}/api/v1/channels/${encodeURIComponent(channel)}/status`);
    return handleResponse<ChannelStatus>(res);
  },

  async getRealtimeMetrics(channel: string, windowSeconds = 30): Promise<RealtimeMetrics> {
    const res = await fetch(
      `${API_BASE}/api/v1/channels/${encodeURIComponent(channel)}/sentiment/realtime?window_seconds=${windowSeconds}`
    );
    return handleResponse<RealtimeMetrics>(res);
  },

  async getTimeline(channel: string, limit = 60): Promise<TimelineMetricItem[]> {
    const res = await fetch(
      `${API_BASE}/api/v1/channels/${encodeURIComponent(channel)}/sentiment/timeline?limit=${limit}`
    );
    return handleResponse<TimelineMetricItem[]>(res);
  },

  async getHistoricalMessages(channel: string, limit = 50, sentiment = 'all'): Promise<ChatMessage[]> {
    const res = await fetch(
      `${API_BASE}/api/v1/channels/${encodeURIComponent(channel)}/messages?limit=${limit}&sentiment=${sentiment}`
    );
    return handleResponse<ChatMessage[]>(res);
  },
};
