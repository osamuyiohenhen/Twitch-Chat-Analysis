<script setup lang="ts">
import { onMounted, onUnmounted } from 'vue';
import { useSentimentStore } from './stores/sentimentStore';
import Sidebar from './components/Sidebar.vue';
import InstructionsExpander from './components/InstructionsExpander.vue';
import LiveSentimentBars from './components/LiveSentimentBars.vue';
import LatestMessage from './components/LatestMessage.vue';
import ActivityTimeline from './components/ActivityTimeline.vue';
import { AlertCircle, X } from '@lucide/vue';

const store = useSentimentStore();

onMounted(() => {
  store.initTheme();
  store.startHealthPolling();
});

onUnmounted(() => {
  store.stopHealthPolling();
  store.disconnect();
});
</script>

<template>
  <div
    class="min-h-screen flex flex-col md:flex-row font-sans transition-colors duration-200 antialiased"
    :class="store.isDark ? 'bg-[#0e1117] text-[#fafafa]' : 'bg-[#ffffff] text-[#31333f]'"
  >
    <!-- Left Sidebar: Connection & Settings -->
    <Sidebar />

    <!-- Main Content Area -->
    <main class="flex-1 flex flex-col min-w-0 overflow-y-auto">
      <!-- Error Banner (if any) -->
      <div
        v-if="store.error"
        class="bg-rose-500/10 border-b border-rose-500/30 px-8 py-2.5 text-xs text-rose-400 flex items-center justify-between"
      >
        <div class="flex items-center gap-2">
          <AlertCircle class="w-4 h-4 shrink-0" />
          <span class="font-medium">{{ store.error }}</span>
        </div>
        <button @click="store.error = null" class="text-rose-400 hover:text-white p-1">
          <X class="w-3.5 h-3.5" />
        </button>
      </div>

      <div class="p-6 md:p-10 max-w-5xl w-full mx-auto space-y-6">
        <!-- Title -->
        <h1 class="text-3xl font-extrabold tracking-tight">
          Twitch Sentiment Engine
        </h1>

        <!-- Instructions Expander -->
        <InstructionsExpander />

        <!-- Live Sentiment Comparison & Subtitle -->
        <LiveSentimentBars />

        <!-- Throttled 1-second Latest Message -->
        <LatestMessage />

        <!-- Stacked Bar Chat Activity Timeline with VOD clickable timestamps -->
        <ActivityTimeline />
      </div>
    </main>
  </div>
</template>