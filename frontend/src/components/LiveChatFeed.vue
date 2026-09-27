<script setup lang="ts">
import { ref, computed, watch, nextTick } from 'vue';
import { useSentimentStore } from '../stores/sentimentStore';
import {
  MessageSquare,
  Pause,
  Play,
  Trash2,
  Zap,
} from '@lucide/vue';

const store = useSentimentStore();
const feedContainer = ref<HTMLElement | null>(null);
const activeFilter = ref<'all' | 'positive' | 'negative' | 'neutral'>('all');
const searchQuery = ref('');

const filteredMessages = computed(() => {
  let list = store.messages;

  if (activeFilter.value !== 'all') {
    list = list.filter((m) => m.label?.toLowerCase() === activeFilter.value);
  }

  if (searchQuery.value.trim()) {
    const q = searchQuery.value.toLowerCase();
    list = list.filter(
      (m) =>
        m.user?.toLowerCase().includes(q) ||
        m.message?.toLowerCase().includes(q)
    );
  }

  return list;
});

// Auto-scroll handler
watch(
  () => store.messages.length,
  async () => {
    if (store.autoScroll && feedContainer.value) {
      await nextTick();
      feedContainer.value.scrollTop = feedContainer.value.scrollHeight;
    }
  }
);

function formatTime(timestamp: number) {
  if (!timestamp) return '';
  const d = new Date(timestamp * 1000);
  return d.toLocaleTimeString([], { hour12: false, hour: '2-digit', minute: '2-digit', second: '2-digit' });
}

function getBadgeClass(label: string) {
  const norm = label?.toLowerCase();
  if (norm === 'positive') {
    return 'bg-emerald-500/15 text-emerald-400 border-emerald-500/30';
  } else if (norm === 'negative') {
    return 'bg-rose-500/15 text-rose-400 border-rose-500/30';
  }
  return 'bg-amber-500/15 text-amber-400 border-amber-500/30';
}
</script>

<template>
  <div class="bg-[#18181b] border border-[#2e2e35] rounded-xl flex flex-col h-[520px] shadow-sm">
    <!-- Feed Header & Controls -->
    <div class="p-4 border-b border-[#2e2e35] flex flex-col sm:flex-row items-stretch sm:items-center justify-between gap-3 bg-[#18181b]/80 backdrop-blur rounded-t-xl">
      
      <!-- Title & Live Counter -->
      <div class="flex items-center gap-2">
        <MessageSquare class="w-4 h-4 text-purple-400" />
        <h2 class="text-sm font-semibold text-white">Live Stream Chat & Sentiment Stream</h2>
        <span class="text-xs px-2 py-0.5 rounded-full bg-zinc-800 text-zinc-300 border border-zinc-700 font-mono">
          {{ filteredMessages.length }} / {{ store.messages.length }}
        </span>
      </div>

      <!-- Controls: Filters, Pause, Clear -->
      <div class="flex items-center gap-2 flex-wrap">
        <!-- Sentiment Filter Pills -->
        <div class="flex rounded-lg bg-[#0e0e10] p-0.5 border border-[#2e2e35] text-xs">
          <button
            v-for="filter in (['all', 'positive', 'negative', 'neutral'] as const)"
            :key="filter"
            @click="activeFilter = filter"
            class="px-2.5 py-1 rounded-md capitalize font-medium transition-colors"
            :class="activeFilter === filter ? 'bg-purple-600 text-white shadow-sm' : 'text-zinc-400 hover:text-zinc-200'"
          >
            {{ filter }}
          </button>
        </div>

        <!-- Search Input -->
        <input
          v-model="searchQuery"
          type="text"
          placeholder="Filter messages..."
          class="px-2.5 py-1 bg-[#0e0e10] border border-[#2e2e35] rounded-lg text-xs text-zinc-200 placeholder-zinc-500 focus:outline-none focus:border-purple-500 w-32"
        />

        <!-- Auto-scroll Pause / Resume -->
        <button
          @click="store.toggleAutoScroll()"
          class="flex items-center gap-1.5 px-2.5 py-1 rounded-lg border text-xs font-medium transition-colors"
          :class="store.autoScroll
            ? 'bg-[#0e0e10] border-[#2e2e35] text-zinc-300 hover:bg-zinc-800'
            : 'bg-amber-500/20 border-amber-500/40 text-amber-300'"
          :title="store.autoScroll ? 'Pause autoscroll' : 'Resume autoscroll'"
        >
          <Pause v-if="store.autoScroll" class="w-3.5 h-3.5" />
          <Play v-else class="w-3.5 h-3.5 text-amber-400" />
          <span>{{ store.autoScroll ? 'Auto-scroll' : 'Paused' }}</span>
        </button>

        <!-- Clear Feed Button -->
        <button
          @click="store.clearFeed()"
          class="p-1.5 rounded-lg bg-[#0e0e10] hover:bg-zinc-800 border border-[#2e2e35] text-zinc-400 hover:text-rose-400 transition-colors"
          title="Clear message feed"
        >
          <Trash2 class="w-3.5 h-3.5" />
        </button>
      </div>
    </div>

    <!-- Live Chat List -->
    <div
      ref="feedContainer"
      class="flex-1 overflow-y-auto p-4 space-y-2.5 bg-[#0e0e10]/50"
    >
      <!-- Empty State -->
      <div
        v-if="filteredMessages.length === 0"
        class="h-full flex flex-col items-center justify-center text-center p-6 text-zinc-500"
      >
        <MessageSquare class="w-10 h-10 mb-2 stroke-[1.5] text-zinc-600 animate-bounce" />
        <p class="text-sm font-medium text-zinc-400">
          {{ store.isConnected ? 'Waiting for messages...' : 'Connect to a channel to begin streaming chat' }}
        </p>
        <p class="text-xs text-zinc-600 mt-1 max-w-sm">
          Incoming messages will be scored by the RoBERTa model in real-time with latency profiling.
        </p>
      </div>

      <!-- Message Items -->
      <div
        v-for="msg in filteredMessages"
        :key="msg.id || `${msg.timestamp}-${msg.user}`"
        class="flex flex-col sm:flex-row sm:items-center justify-between gap-2 p-2.5 rounded-lg bg-[#18181b] border border-[#2e2e35] hover:border-zinc-700 transition-all text-xs"
      >
        <!-- User & Message Body -->
        <div class="flex items-start gap-2.5 flex-1 min-w-0">
          <span class="text-[11px] font-mono text-zinc-500 shrink-0 pt-0.5">
            {{ formatTime(msg.timestamp) }}
          </span>
          <span class="font-bold text-purple-400 shrink-0">
            {{ msg.user || 'anonymous' }}:
          </span>
          <span class="text-zinc-200 break-words flex-1">
            {{ msg.message }}
          </span>
        </div>

        <!-- Sentiment Badge & Latency Tag -->
        <div class="flex items-center gap-2 self-end sm:self-center shrink-0">
          <!-- Sentiment Badge -->
          <span
            class="px-2 py-0.5 rounded-md font-semibold text-[11px] border capitalize flex items-center gap-1 shadow-sm"
            :class="getBadgeClass(msg.label)"
          >
            <span>{{ msg.label }}</span>
            <span class="opacity-75 font-mono text-[10px]">
              {{ (msg.score * 100).toFixed(0) }}%
            </span>
          </span>

          <!-- Inference Latency Tag -->
          <span
            v-if="msg.latency"
            class="flex items-center gap-0.5 text-[10px] font-mono text-zinc-500 bg-[#0e0e10] px-1.5 py-0.5 rounded border border-[#2e2e35]"
            title="Inference execution latency"
          >
            <Zap class="w-3 h-3 text-purple-400" />
            {{ msg.latency.toFixed(0) }}ms
          </span>
        </div>
      </div>
    </div>
  </div>
</template>
