<script setup lang="ts">
import { useSentimentStore } from '../stores/sentimentStore';
import { SlidersHorizontal } from '@lucide/vue';

const store = useSentimentStore();
</script>

<template>
  <div class="space-y-4">
    <!-- Header with Monitoring title and Neutral Toggle -->
    <div class="flex flex-col sm:flex-row sm:items-center justify-between gap-3">
      <h3
        v-if="store.isConnected"
        class="text-2xl font-bold tracking-tight capitalize"
      >
        Monitoring: {{ store.currentChannel }}
      </h3>
      <div
        v-else
        class="p-4 rounded-md text-sm border flex items-center gap-2 w-full"
        :class="store.isDark 
          ? 'bg-[#1e2026] text-zinc-300 border-[#31333f]' 
          : 'bg-[#f0f2f6] text-zinc-700 border-[#d6d9e0]'"
      >
        <span>👈 Enter a channel name in the sidebar and click <strong>Connect</strong> to start monitoring.</span>
      </div>

      <!-- Neutral Bar Toggle -->
      <div v-if="store.isConnected" class="flex items-center gap-2 self-end sm:self-auto">
        <button
          @click="store.toggleNeutralBar()"
          class="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-md text-xs font-medium border transition-all"
          :class="store.showNeutralBar
            ? 'bg-[#636EFA]/15 border-[#636EFA] text-[#636EFA]'
            : store.isDark
            ? 'bg-[#14171f] border-[#262730] text-zinc-400 hover:text-zinc-200'
            : 'bg-white border-zinc-300 text-zinc-600 hover:text-zinc-900'"
          title="Toggle 3-bar view with Neutral sentiment"
        >
          <SlidersHorizontal class="w-3.5 h-3.5" />
          <span>{{ store.showNeutralBar ? 'Neutral Bar: ON (100% sum)' : 'Show Neutral Bar' }}</span>
        </button>
      </div>
    </div>

    <!-- Active Monitoring Area (Two Columns) -->
    <div
      v-if="store.isConnected"
      class="grid grid-cols-1 md:grid-cols-4 gap-6 items-end pt-2"
    >
      <!-- Big Live Comparison Bars (3 cols) -->
      <div
        class="md:col-span-3 flex justify-center items-end h-[340px] px-6 pb-6 pt-4 rounded-lg border transition-colors"
        :class="store.isDark ? 'bg-[#14171f] border-[#262730]' : 'bg-[#f9fafb] border-[#e5e7eb]'"
      >
        <div class="flex items-end justify-center gap-8 sm:gap-14 h-full w-full max-w-lg">
          <!-- Positive Bar -->
          <div class="flex flex-col items-center justify-end h-full w-24 sm:w-32">
            <span class="text-xl sm:text-2xl font-extrabold mb-2 tracking-tight">
              {{ store.posPercent }}%
            </span>
            <div
              class="w-full rounded-t-md transition-all duration-200 ease-out shadow-lg"
              :style="{
                height: `${Math.max(store.posPercent, 3)}%`,
                backgroundColor: '#00CC96',
                boxShadow: '0 4px 20px rgba(0, 204, 150, 0.25)',
              }"
            ></div>
            <span class="text-sm font-semibold mt-3 opacity-90">Positive</span>
          </div>

          <!-- Neutral Bar (Toggleable) -->
          <div
            v-if="store.showNeutralBar"
            class="flex flex-col items-center justify-end h-full w-24 sm:w-32"
          >
            <span class="text-xl sm:text-2xl font-extrabold mb-2 tracking-tight">
              {{ store.neuPercent }}%
            </span>
            <div
              class="w-full rounded-t-md transition-all duration-200 ease-out shadow-lg"
              :style="{
                height: `${Math.max(store.neuPercent, 3)}%`,
                backgroundColor: '#636EFA',
                boxShadow: '0 4px 20px rgba(99, 110, 250, 0.25)',
              }"
            ></div>
            <span class="text-sm font-semibold mt-3 opacity-90">Neutral</span>
          </div>

          <!-- Negative Bar -->
          <div class="flex flex-col items-center justify-end h-full w-24 sm:w-32">
            <span class="text-xl sm:text-2xl font-extrabold mb-2 tracking-tight">
              {{ store.negPercent }}%
            </span>
            <div
              class="w-full rounded-t-md transition-all duration-200 ease-out shadow-lg"
              :style="{
                height: `${Math.max(store.negPercent, 3)}%`,
                backgroundColor: '#EF553B',
                boxShadow: '0 4px 20px rgba(239, 85, 59, 0.25)',
              }"
            ></div>
            <span class="text-sm font-semibold mt-3 opacity-90">Negative</span>
          </div>
        </div>
      </div>

      <!-- Avg Sentiment Callout (1 col) -->
      <div
        class="p-6 rounded-lg border flex flex-col justify-center h-[340px] transition-colors"
        :class="store.isDark ? 'bg-[#14171f] border-[#262730]' : 'bg-[#f9fafb] border-[#e5e7eb]'"
      >
        <span class="text-xs font-medium uppercase tracking-wider opacity-70">
          Avg Sentiment ({{ store.sessionMessageCount }} msgs)
        </span>
        <span
          class="text-3xl font-extrabold mt-2 tracking-tight"
          :class="store.avgSentimentLabel === 'Positive'
            ? 'text-[#00CC96]'
            : store.avgSentimentLabel === 'Negative'
            ? 'text-[#EF553B]'
            : store.avgSentimentLabel === 'Neutral'
            ? 'text-[#636EFA]'
            : 'text-amber-400'"
        >
          {{ store.avgSentimentLabel }}
        </span>

        <div class="mt-6 pt-6 border-t text-xs space-y-2 opacity-75"
             :class="store.isDark ? 'border-[#262730]' : 'border-[#e5e7eb]'">
          <div class="flex justify-between">
            <span>Velocity:</span>
            <span class="font-mono font-bold">{{ store.metrics?.messages_per_second?.toFixed(1) || '0.0' }} msg/s</span>
          </div>
          <div class="flex justify-between">
            <span>Latency:</span>
            <span class="font-mono font-bold">{{ store.metrics?.avg_latency_ms?.toFixed(0) || '0' }} ms</span>
          </div>
          <div class="flex justify-between">
            <span>Update Rate:</span>
            <span class="font-mono text-[#00CC96] font-semibold">~4 fps</span>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>