import { defineStore } from 'pinia';
import { api, WS_BASE } from '../services/api';
import type {
  ChatMessage,
  HealthResponse,
  RealtimeMetrics,
  SocketStatus,
  TimelineMetricItem,
} from '../types/sentiment';

const MAX_MESSAGES = 150;

export const useSentimentStore = defineStore('sentiment', {
  state: () => ({
    currentChannel: '',
    isConnected: false,
    isConnecting: false,
    socketStatus: 'disconnected' as SocketStatus,
    error: null as string | null,
    backendHealth: null as HealthResponse | null,
    isBackendOnline: false,

    // Stream & VOD metadata
    vodId: null as string | null,
    streamStart: null as number | null,

    // Live Metrics & Smoothed Bars (0.0 to 1.0)
    metrics: null as RealtimeMetrics | null,
    smoothedPos: 0.5,
    smoothedNeg: 0.5,
    smoothedNeu: 0.0,
    sessionMessageCount: 0, // Starts at 0 fresh for current connection session

    // Neutral Bar Toggle
    showNeutralBar: localStorage.getItem('show_neutral_bar') === 'true',

    // Latest single message (throttled ~1 sec for calm reading)
    latestMessage: null as ChatMessage | null,
    _lastMsgUpdateTime: 0,

    // Full message stream buffer
    messages: [] as ChatMessage[],
    autoScroll: true,

    // Timeline data & VOD selection
    timeline: [] as TimelineMetricItem[],
    selectedMinuteLabel: null as string | null,
    pendingVodUrl: null as string | null,

    // Theme (dark / light)
    theme: (localStorage.getItem('twitch_theme') as 'dark' | 'light') || 'dark',

    // Instructions expander state
    showInstructions: localStorage.getItem('seen_instructions') !== 'true',

    // Internal socket & timers
    _ws: null as WebSocket | null,
    _healthTimer: null as number | null,
    _reconnectTimer: null as number | null,
    _timelineTimer: null as number | null,
  }),

  getters: {
    // Smoothed bar percentages
    posPercent: (state) => {
      if (state.showNeutralBar) {
        const total = (state.smoothedPos + state.smoothedNeg + state.smoothedNeu) || 1;
        return Math.round((state.smoothedPos / total) * 100);
      }
      const total = (state.smoothedPos + state.smoothedNeg) || 1;
      return Math.round((state.smoothedPos / total) * 100);
    },
    negPercent: (state) => {
      if (state.showNeutralBar) {
        const total = (state.smoothedPos + state.smoothedNeg + state.smoothedNeu) || 1;
        return Math.round((state.smoothedNeg / total) * 100);
      }
      const total = (state.smoothedPos + state.smoothedNeg) || 1;
      return Math.round((state.smoothedNeg / total) * 100);
    },
    neuPercent: (state) => {
      if (!state.showNeutralBar) return 0;
      const total = (state.smoothedPos + state.smoothedNeg + state.smoothedNeu) || 1;
      return Math.round((state.smoothedNeu / total) * 100);
    },
    avgSentimentLabel: (state) => {
      if (state.sessionMessageCount === 0) return 'Waiting for chat...';
      if (state.smoothedPos > state.smoothedNeg && state.smoothedPos > state.smoothedNeu) return 'Positive';
      if (state.smoothedNeg > state.smoothedPos && state.smoothedNeg > state.smoothedNeu) return 'Negative';
      if (state.showNeutralBar && state.smoothedNeu > state.smoothedPos && state.smoothedNeu > state.smoothedNeg) return 'Neutral';
      return state.smoothedPos >= state.smoothedNeg ? 'Positive' : 'Negative';
    },

    // Raw rolling metrics
    positiveRatio: (state) => Math.round((state.metrics?.pos_ratio ?? 0) * 100),
    negativeRatio: (state) => Math.round((state.metrics?.neg_ratio ?? 0) * 100),
    neutralRatio: (state) => Math.round((state.metrics?.neu_ratio ?? 0) * 100),
    velocity: (state) => Number((state.metrics?.messages_per_second ?? 0).toFixed(1)),
    avgLatency: (state) => Math.round(state.metrics?.avg_latency_ms ?? 0),
    totalCount: (state) => state.sessionMessageCount,

    isDark: (state) => state.theme === 'dark',
  },

  actions: {
    toggleTheme() {
      this.theme = this.theme === 'dark' ? 'light' : 'dark';
      localStorage.setItem('twitch_theme', this.theme);
      if (this.theme === 'dark') {
        document.documentElement.classList.add('dark');
      } else {
        document.documentElement.classList.remove('dark');
      }
    },

    initTheme() {
      if (this.theme === 'dark') {
        document.documentElement.classList.add('dark');
      } else {
        document.documentElement.classList.remove('dark');
      }
    },

    toggleNeutralBar() {
      this.showNeutralBar = !this.showNeutralBar;
      localStorage.setItem('show_neutral_bar', String(this.showNeutralBar));
    },

    dismissInstructions() {
      this.showInstructions = false;
      localStorage.setItem('seen_instructions', 'true');
    },

    toggleInstructions() {
      this.showInstructions = !this.showInstructions;
    },

    toggleAutoScroll() {
      this.autoScroll = !this.autoScroll;
    },

    clearFeed() {
      this.messages = [];
    },

    async checkBackendHealth() {
      try {
        const health = await api.getHealth();
        this.backendHealth = health;
        this.isBackendOnline = health.status === 'healthy';
        this.error = null;
      } catch (err: any) {
        this.isBackendOnline = false;
        this.backendHealth = null;
        this.error = `Backend unreachable: ${err.message}`;
      }
    },

    startHealthPolling() {
      this.checkBackendHealth();
      if (this._healthTimer) clearInterval(this._healthTimer);
      this._healthTimer = window.setInterval(() => {
        this.checkBackendHealth();
      }, 10000);
    },

    stopHealthPolling() {
      if (this._healthTimer) {
        clearInterval(this._healthTimer);
        this._healthTimer = null;
      }
    },

    async connect(channelName: string) {
      const channel = channelName.trim().toLowerCase().replace(/^#/, '');
      if (!channel) return;

      this.isConnecting = true;
      this.error = null;

      try {
        // 1. Connect via REST
        const connectRes = await api.connectChannel(channel);
        this.currentChannel = channel;
        this.isConnected = true;
        this.isConnecting = false;
        this.vodId = connectRes.vod_id || null;

        // Fresh Start: Reset metrics & session count for this connection
        this.sessionMessageCount = 0;
        this.smoothedPos = 0.5;
        this.smoothedNeg = 0.5;
        this.smoothedNeu = 0.0;
        this.latestMessage = null;
        this.selectedMinuteLabel = null;
        this.pendingVodUrl = null;
        this.timeline = [];

        // 2. Fetch channel status (stream_start, vod_id)
        this.fetchChannelStatus(channel);

        // 3. Load timeline history immediately
        this.refreshTimeline(channel);

        // Poll timeline every 10 seconds for fast responsive updates
        if (this._timelineTimer) clearInterval(this._timelineTimer);
        this._timelineTimer = window.setInterval(() => {
          if (this.isConnected && this.currentChannel) {
            this.refreshTimeline(this.currentChannel);
          }
        }, 10000);

        // 4. Connect WebSocket for 4-5 fps live streaming
        this.openWebSocket(channel);
      } catch (err: any) {
        this.isConnecting = false;
        this.isConnected = false;
        this.error = `Failed to connect to #${channel}: ${err.message}`;
      }
    },

    async fetchChannelStatus(channel: string) {
      try {
        const status = await api.getChannelStatus(channel);
        if (status.vod_id) this.vodId = status.vod_id;
        if (status.stream_start) this.streamStart = status.stream_start;
      } catch {
        // Ignore
      }
    },

    async refreshTimeline(channel: string) {
      try {
        const items = await api.getTimeline(channel, 60);
        if (items && items.length > 0) {
          this.timeline = items;
        }
      } catch {
        // Ignore
      }
    },

    openWebSocket(channel: string) {
      if (this._ws) {
        this._ws.close();
        this._ws = null;
      }

      this.socketStatus = 'connecting';
      const wsUrl = `${WS_BASE}/ws/channels/${encodeURIComponent(channel)}`;
      const ws = new WebSocket(wsUrl);

      ws.onopen = () => {
        this.socketStatus = 'connected';
      };

      ws.onmessage = (event) => {
        try {
          const payload = JSON.parse(event.data);
          this.handleSocketEvent(payload);
        } catch (e) {
          console.error('Failed to parse WebSocket message:', e);
        }
      };

      ws.onerror = (e) => {
        console.warn('WebSocket error:', e);
        this.socketStatus = 'error';
      };

      ws.onclose = () => {
        this.socketStatus = 'disconnected';
        if (this.isConnected && this.currentChannel === channel) {
          if (this._reconnectTimer) clearTimeout(this._reconnectTimer);
          this._reconnectTimer = window.setTimeout(() => {
            if (this.isConnected && this.currentChannel === channel) {
              this.openWebSocket(channel);
            }
          }, 3000);
        }
      };

      this._ws = ws;
    },

    handleSocketEvent(event: { type: string; data?: any; message?: string }) {
      if (event.type === 'chat' && event.data) {
        const chatData: ChatMessage = {
          ...event.data,
          id: `${Date.now()}-${Math.random().toString(36).substring(2, 9)}`,
        };

        // Increment session message count (fresh per session)
        this.sessionMessageCount++;

        // Buffer full list for optional chat stream panel
        this.messages.push(chatData);
        if (this.messages.length > MAX_MESSAGES) {
          this.messages.shift();
        }

        // Throttle latest message to ~1 sec for calm reading
        const now = Date.now();
        if (now - this._lastMsgUpdateTime >= 1000 || !this.latestMessage) {
          this.latestMessage = chatData;
          this._lastMsgUpdateTime = now;
        }

        // Micro-adjustment on incoming chat for ultra-fluid real-time bar movement
        const score = chatData.score ?? 0.8;
        const label = chatData.label?.toLowerCase();
        const microRate = 0.08;
        if (label === 'positive') {
          this.smoothedPos = this.smoothedPos * (1 - microRate) + score * microRate;
          this.smoothedNeg = this.smoothedNeg * (1 - microRate);
        } else if (label === 'negative') {
          this.smoothedNeg = this.smoothedNeg * (1 - microRate) + score * microRate;
          this.smoothedPos = this.smoothedPos * (1 - microRate);
        } else if (label === 'neutral') {
          this.smoothedNeu = this.smoothedNeu * (1 - microRate) + 0.5 * microRate;
        }

      } else if (event.type === 'metrics' && event.data) {
        this.metrics = event.data;

        // Smooth bars at 4Hz from backend metrics snapshot
        const targetPos = event.data.pos_ratio ?? 0.5;
        const targetNeg = event.data.neg_ratio ?? 0.5;
        const targetNeu = event.data.neu_ratio ?? 0.0;
        const smoothing = 0.35;

        this.smoothedPos = smoothing * targetPos + (1 - smoothing) * this.smoothedPos;
        this.smoothedNeg = smoothing * targetNeg + (1 - smoothing) * this.smoothedNeg;
        this.smoothedNeu = smoothing * targetNeu + (1 - smoothing) * this.smoothedNeu;

      } else if (event.type === 'warning') {
        this.error = event.message || 'Stream warning received';
      }
    },

    selectTimelinePoint(timeLabel: string, totalMinutes: number) {
      this.selectedMinuteLabel = timeLabel;
      if (this.vodId) {
        this.pendingVodUrl = `https://twitch.tv/videos/${this.vodId}?t=${totalMinutes}m0s`;
      } else {
        this.pendingVodUrl = `https://twitch.tv/${this.currentChannel}`;
      }
    },

    async disconnect() {
      const channel = this.currentChannel;
      this.isConnected = false;
      this.isConnecting = false;
      this.socketStatus = 'disconnected';

      if (this._reconnectTimer) {
        clearTimeout(this._reconnectTimer);
        this._reconnectTimer = null;
      }
      if (this._timelineTimer) {
        clearInterval(this._timelineTimer);
        this._timelineTimer = null;
      }
      if (this._ws) {
        this._ws.close();
        this._ws = null;
      }

      if (channel) {
        try {
          await api.disconnectChannel(channel);
        } catch {
          // ignore
        }
      }
    },
  },
});