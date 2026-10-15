<script setup lang="ts">
import { computed, ref } from 'vue';
import { useSentimentStore } from '../stores/sentimentStore';
import type { StreamSessionItem, TimelineMetricItem } from '../types/sentiment';
import {
  Chart as ChartJS,
  Title,
  Tooltip,
  Legend,
  BarElement,
  CategoryScale,
  LinearScale,
} from 'chart.js';
import zoomPlugin from 'chartjs-plugin-zoom';
import { Bar } from 'vue-chartjs';
import { ExternalLink, Video, RotateCw, ZoomOut } from '@lucide/vue';

ChartJS.register(
  Title,
  Tooltip,
  Legend,
  BarElement,
  CategoryScale,
  LinearScale,
  zoomPlugin
);

const store = useSentimentStore();
const chartRef = ref<any>(null);
const isRefreshing = ref(false);

function resetZoom() {
  if (chartRef.value?.chart) {
    chartRef.value.chart.resetZoom();
  }
}

async function manualRefresh() {
  if (!store.currentChannel) return;
  isRefreshing.value = true;
  await store.refreshTimeline(store.currentChannel);
  await store.fetchSessions(store.currentChannel);
  setTimeout(() => {
    isRefreshing.value = false;
  }, 400);
}

function onSessionChange(val: string) {
  store.selectSession(val ? val : null);
  resetZoom();
}

function formatSessionLabel(sess: StreamSessionItem): string {
  const isCurrent = sess.vod_id === store.vodId;
  const dateSource = sess.stream_start_time && sess.stream_start_time > 0
    ? sess.stream_start_time * 1000
    : sess.first_minute * 1000;
  const dateStr = new Date(dateSource).toLocaleDateString(undefined, {
    month: 'short',
    day: 'numeric',
    hour: 'numeric',
    minute: '2-digit',
  });
  const msgs = sess.total_messages >= 1000
    ? `${(sess.total_messages / 1000).toFixed(1)}k msgs`
    : `${sess.total_messages} msgs`;

  if (sess.stream_title) {
    const cleanTitle = sess.stream_title.length > 40
      ? sess.stream_title.slice(0, 37) + '...'
      : sess.stream_title;
    return `"${cleanTitle}" • ${dateStr} (${msgs})${isCurrent ? ' [Current]' : ''}`;
  }
  return `${dateStr} (VOD ${sess.vod_id}, ${msgs})${isCurrent ? ' [Current]' : ''}`;
}

// Calculate time labels and minutes
const timelineDataFormatted = computed(() => {
  if (!store.timeline || store.timeline.length === 0) {
    return { labels: [], pos: [], neu: [], neg: [], minuteValues: [], items: [] };
  }

  const labels: string[] = [];
  const pos: number[] = [];
  const neu: number[] = [];
  const neg: number[] = [];
  const minuteValues: number[] = [];
  const items: TimelineMetricItem[] = [];

  // Determine stream start baseline
  let baselineStart: number | null = null;
  if (store.selectedSessionVodId) {
    const sess = store.currentSessionInfo;
    if (sess?.stream_start_time && sess.stream_start_time > 0) {
      baselineStart = sess.stream_start_time;
    } else if (sess?.first_minute) {
      baselineStart = sess.first_minute;
    }
  } else {
    baselineStart = store.streamStart;
  }

  store.timeline.forEach((item) => {
    let label = '';
    let totalMinutes = 0;

    if (baselineStart && baselineStart > 0 && item.minute_timestamp >= baselineStart) {
      // True VOD relative time
      const gapSec = item.minute_timestamp - baselineStart;
      totalMinutes = Math.floor(gapSec / 60);
      const h = Math.floor(totalMinutes / 60);
      const m = totalMinutes % 60;
      label = `${h}:${m.toString().padStart(2, '0')}`;
    } else {
      // Fallback to local clock time HH:MM
      const d = new Date(item.minute_timestamp * 1000);
      totalMinutes = d.getHours() * 60 + d.getMinutes();
      label = `${d.getHours()}:${d.getMinutes().toString().padStart(2, '0')}`;
    }

    labels.push(label);
    pos.push(item.pos_count);
    neu.push(item.neu_count);
    neg.push(item.neg_count);
    minuteValues.push(totalMinutes);
    items.push(item);
  });

  return { labels, pos, neu, neg, minuteValues, items };
});

const chartData = computed(() => {
  const { labels, pos, neu, neg } = timelineDataFormatted.value;
  return {
    labels: labels.length > 0 ? labels : ['0:00'],
    datasets: [
      {
        label: 'Positive',
        data: pos.length > 0 ? pos : [0],
        backgroundColor: '#00CC96',
        borderRadius: 2,
        stack: 'sentimentStack',
      },
      {
        label: 'Neutral',
        data: neu.length > 0 ? neu : [0],
        backgroundColor: '#636EFA',
        borderRadius: 2,
        stack: 'sentimentStack',
      },
      {
        label: 'Negative',
        data: neg.length > 0 ? neg : [0],
        backgroundColor: '#EF553B',
        borderRadius: 2,
        stack: 'sentimentStack',
      },
    ],
  };
});

const chartOptions = computed<any>(() => ({
  responsive: true,
  maintainAspectRatio: false,
  plugins: {
    legend: {
      position: 'right',
      labels: {
        color: store.isDark ? '#e5e7eb' : '#374151',
        boxWidth: 10,
        boxHeight: 10,
        usePointStyle: false,
        font: { size: 11 },
      },
    },
    tooltip: {
      backgroundColor: store.isDark ? '#1e2026' : '#ffffff',
      titleColor: store.isDark ? '#ffffff' : '#111827',
      bodyColor: store.isDark ? '#e5e7eb' : '#374151',
      borderColor: store.isDark ? '#374151' : '#d1d5db',
      borderWidth: 1,
      padding: 10,
    },
    zoom: {
      pan: {
        enabled: true,
        mode: 'x',
      },
      zoom: {
        wheel: {
          enabled: true,
        },
        pinch: {
          enabled: true,
        },
        mode: 'x',
      },
    },
  },
  scales: {
    x: {
      stacked: true,
      grid: {
        display: false,
      },
      ticks: {
        color: store.isDark ? '#9ca3af' : '#6b7280',
        font: { size: 10 },
        maxRotation: 0,
      },
    },
    y: {
      stacked: true,
      title: {
        display: true,
        text: 'Messages',
        color: store.isDark ? '#9ca3af' : '#6b7280',
        font: { size: 11 },
      },
      grid: {
        color: store.isDark ? 'rgba(55, 65, 81, 0.4)' : 'rgba(229, 231, 235, 0.8)',
      },
      ticks: {
        color: store.isDark ? '#9ca3af' : '#6b7280',
        font: { size: 10 },
      },
    },
  },
  onClick: (_event: any, elements: any[]) => {
    if (elements.length > 0) {
      const idx = elements[0].index;
      const { labels, minuteValues, items } = timelineDataFormatted.value;
      if (labels[idx] && minuteValues[idx] !== undefined) {
        const itemVodId = items[idx]?.vod_id || null;
        store.selectTimelinePoint(labels[idx], minuteValues[idx], itemVodId);
      }
    }
  },
}));

function openVodLink() {
  if (store.pendingVodUrl) {
    window.open(store.pendingVodUrl, '_blank');
  }
}
</script>

<template>
  <div v-if="store.isConnected" class="space-y-3 pt-2">
    <!-- Timeline Title & Action Bar -->
    <div class="flex flex-wrap items-center justify-between gap-2">
      <div>
        <h4 class="text-base font-bold tracking-tight capitalize">
          {{ store.currentChannel }}'s Chat Activity Timeline
        </h4>
        <p class="text-xs opacity-60">
          <span v-if="store.selectedSessionVodId">
            Viewing: {{ store.currentSessionInfo?.stream_title ? `"${store.currentSessionInfo.stream_title}" (VOD ${store.selectedSessionVodId})` : `Historical VOD ${store.selectedSessionVodId}` }}
          </span>
          <span v-else>
            Stream Time (VOD timestamp) • Auto-updates every 10s
          </span>
          <span class="ml-2 text-[11px] opacity-75">
            (Scroll wheel / pinch to zoom, drag to pan)
          </span>
        </p>
      </div>

      <div class="flex items-center gap-2">
        <!-- Session Selector Dropdown -->
        <div v-if="store.availableSessions.length > 0" class="flex items-center gap-1.5">
          <label class="text-xs opacity-60 font-medium">Session:</label>
          <select
            :value="store.selectedSessionVodId || ''"
            @change="onSessionChange(($event.target as HTMLSelectElement).value)"
            class="text-xs rounded px-2.5 py-1 border transition-colors outline-none font-medium cursor-pointer"
            :class="store.isDark
              ? 'bg-[#14171f] border-[#262730] text-zinc-300 hover:border-zinc-500'
              : 'bg-white border-zinc-300 text-zinc-800 hover:border-zinc-400'"
            title="Switch between live stream and recorded past sessions"
          >
            <option value="">🔴 Current Stream (Live)</option>
            <option
              v-for="sess in store.availableSessions"
              :key="sess.vod_id"
              :value="sess.vod_id"
            >
              {{ formatSessionLabel(sess) }}
            </option>
          </select>
        </div>

        <!-- Reset Zoom Button -->
        <button
          @click="resetZoom"
          class="inline-flex items-center gap-1 px-2.5 py-1 rounded text-xs border transition-colors"
          :class="store.isDark 
            ? 'bg-[#14171f] border-[#262730] text-zinc-400 hover:text-white' 
            : 'bg-white border-zinc-300 text-zinc-600 hover:text-black'"
          title="Reset zoom and pan"
        >
          <ZoomOut class="w-3.5 h-3.5" />
          <span>Reset Zoom</span>
        </button>

        <!-- Refresh Button -->
        <button
          @click="manualRefresh"
          class="inline-flex items-center gap-1.5 px-2.5 py-1 rounded text-xs border transition-colors"
          :class="store.isDark 
            ? 'bg-[#14171f] border-[#262730] text-zinc-400 hover:text-white' 
            : 'bg-white border-zinc-300 text-zinc-600 hover:text-black'"
          title="Refresh timeline from database"
        >
          <RotateCw class="w-3.5 h-3.5" :class="{ 'animate-spin': isRefreshing }" />
          <span>Refresh</span>
        </button>
      </div>
    </div>

    <!-- Stacked Bar Chart Box -->
    <div
      class="p-4 rounded-lg border transition-colors relative"
      :class="store.isDark ? 'bg-[#14171f] border-[#262730]' : 'bg-[#f9fafb] border-[#e5e7eb]'"
    >
      <div v-if="store.timeline.length > 0" style="height: 280px; position: relative; width: 100%;">
        <Bar ref="chartRef" :data="chartData" :options="chartOptions" />
      </div>
      <div
        v-else
        class="h-[280px] flex flex-col items-center justify-center text-center opacity-60 text-xs space-y-1"
      >
        <span>Accumulating minute-level chat volume...</span>
        <span class="text-[11px] opacity-75">Entries will appear as soon as chats are logged (checking every 10s)</span>
      </div>

      <p v-if="store.timeline.length > 0" class="text-[11px] text-center opacity-40 mt-1">
        Click any bar to jump to that moment in the VOD
      </p>
    </div>

    <!-- VOD Button Action -->
    <div v-if="store.selectedMinuteLabel" class="flex items-center gap-3 pt-1">
      <button
        @click="openVodLink"
        class="inline-flex items-center gap-2 px-4 py-2 rounded-md text-xs font-semibold text-white bg-[#636EFA] hover:bg-[#525cee] transition-all shadow-sm active:scale-95"
      >
        <Video class="w-3.5 h-3.5" />
        <span>Open VOD at {{ store.selectedMinuteLabel }}</span>
        <ExternalLink class="w-3 h-3 opacity-75" />
      </button>
      <span class="text-xs opacity-60">
        {{ store.selectedPointVodId ? `Twitch VOD ID: ${store.selectedPointVodId}` : (store.vodId ? `Twitch VOD ID: ${store.vodId}` : 'Opens live Twitch channel') }}
      </span>
    </div>
  </div>
</template>