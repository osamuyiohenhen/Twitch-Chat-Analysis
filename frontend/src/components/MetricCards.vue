<script setup lang="ts">
import { useSentimentStore } from '../stores/sentimentStore';
import {
  Smile,
  Frown,
  Activity,
  Zap,
  MessageSquare,
} from '@lucide/vue';

const store = useSentimentStore();
</script>

<template>
  <div class="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
    <!-- Positive Sentiment Card -->
    <div class="bg-[#18181b] border border-[#2e2e35] rounded-xl p-4 shadow-sm hover:border-emerald-500/40 transition-colors">
      <div class="flex items-center justify-between">
        <span class="text-xs font-semibold uppercase tracking-wider text-zinc-400">Positive Sentiment</span>
        <div class="p-2 rounded-lg bg-emerald-500/10 text-emerald-400">
          <Smile class="w-5 h-5" />
        </div>
      </div>
      <div class="mt-3 flex items-baseline gap-2">
        <span class="text-3xl font-extrabold text-white tracking-tight">
          {{ store.positiveRatio }}%
        </span>
        <span class="text-xs text-zinc-500">rolling 30s</span>
      </div>
      <!-- Progress Bar -->
      <div class="mt-3 w-full bg-[#0e0e10] rounded-full h-1.5 overflow-hidden">
        <div
          class="bg-emerald-500 h-full rounded-full transition-all duration-500"
          :style="{ width: `${store.positiveRatio}%` }"
        ></div>
      </div>
    </div>

    <!-- Negative Sentiment Card -->
    <div class="bg-[#18181b] border border-[#2e2e35] rounded-xl p-4 shadow-sm hover:border-rose-500/40 transition-colors">
      <div class="flex items-center justify-between">
        <span class="text-xs font-semibold uppercase tracking-wider text-zinc-400">Negative Sentiment</span>
        <div class="p-2 rounded-lg bg-rose-500/10 text-rose-400">
          <Frown class="w-5 h-5" />
        </div>
      </div>
      <div class="mt-3 flex items-baseline gap-2">
        <span class="text-3xl font-extrabold text-white tracking-tight">
          {{ store.negativeRatio }}%
        </span>
        <span class="text-xs text-zinc-500">rolling 30s</span>
      </div>
      <!-- Progress Bar -->
      <div class="mt-3 w-full bg-[#0e0e10] rounded-full h-1.5 overflow-hidden">
        <div
          class="bg-rose-500 h-full rounded-full transition-all duration-500"
          :style="{ width: `${store.negativeRatio}%` }"
        ></div>
      </div>
    </div>

    <!-- Chat Velocity Card -->
    <div class="bg-[#18181b] border border-[#2e2e35] rounded-xl p-4 shadow-sm hover:border-amber-500/40 transition-colors">
      <div class="flex items-center justify-between">
        <span class="text-xs font-semibold uppercase tracking-wider text-zinc-400">Chat Velocity</span>
        <div class="p-2 rounded-lg bg-amber-500/10 text-amber-400">
          <Activity class="w-5 h-5" />
        </div>
      </div>
      <div class="mt-3 flex items-baseline gap-2">
        <span class="text-3xl font-extrabold text-white tracking-tight">
          {{ store.velocity }}
        </span>
        <span class="text-xs text-zinc-500">msgs / sec</span>
      </div>
      <div class="mt-3 text-xs text-zinc-400 flex items-center gap-1.5">
        <MessageSquare class="w-3.5 h-3.5 text-zinc-500" />
        <span>{{ store.totalCount.toLocaleString() }} total messages</span>
      </div>
    </div>

    <!-- AI Latency Card -->
    <div class="bg-[#18181b] border border-[#2e2e35] rounded-xl p-4 shadow-sm hover:border-purple-500/40 transition-colors">
      <div class="flex items-center justify-between">
        <span class="text-xs font-semibold uppercase tracking-wider text-zinc-400">Avg Model Latency</span>
        <div class="p-2 rounded-lg bg-purple-500/10 text-purple-400">
          <Zap class="w-5 h-5" />
        </div>
      </div>
      <div class="mt-3 flex items-baseline gap-2">
        <span class="text-3xl font-extrabold text-white tracking-tight">
          {{ store.avgLatency }}
        </span>
        <span class="text-xs text-zinc-500">ms / inference</span>
      </div>
      <div class="mt-3 text-xs text-zinc-400 flex items-center gap-1.5">
        <span
          class="inline-block w-2 h-2 rounded-full"
          :class="store.avgLatency < 80 ? 'bg-emerald-500' : 'bg-amber-500'"
        ></span>
        <span>{{ store.avgLatency < 80 ? 'Real-time performance' : 'Processing peak' }}</span>
      </div>
    </div>
  </div>
</template>
