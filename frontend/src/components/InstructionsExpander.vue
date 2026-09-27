<script setup lang="ts">
import { useSentimentStore } from '../stores/sentimentStore';
import { Info, X } from '@lucide/vue';

const store = useSentimentStore();
</script>

<template>
  <div
    v-if="store.showInstructions"
    class="border rounded-lg p-5 transition-all shadow-sm relative"
    :class="store.isDark 
      ? 'bg-[#181a20] border-[#31333f] text-zinc-300' 
      : 'bg-[#f8f9fb] border-[#d6d9e0] text-zinc-700'"
  >
    <!-- Dismiss (X) Button Top Right -->
    <button
      @click="store.dismissInstructions()"
      class="absolute top-3.5 right-3.5 p-1 rounded-md text-zinc-400 hover:text-white hover:bg-black/10 dark:hover:bg-white/10 transition-colors"
      title="Dismiss instructions (can be re-opened from sidebar)"
    >
      <X class="w-4 h-4" />
    </button>

    <!-- Header -->
    <div class="flex items-center gap-2 mb-3">
      <Info class="w-4 h-4 text-[#ff4b4b]" />
      <h3 class="text-sm font-bold tracking-tight">Getting Started</h3>
    </div>

    <!-- Content -->
    <div class="text-xs leading-relaxed space-y-2 opacity-90 pr-6">
      <p>1. Enter a Twitch channel name in the sidebar and click <strong>Connect</strong>.</p>
      <p>2. Watch the <strong>live sentiment bars</strong> update in real-time as chat messages stream in.</p>
      <p>3. The <strong>Chat Activity Timeline</strong> below plots volume per minute broken down by positive, neutral, and negative messages.</p>
      <p>4. Click any bar on the timeline to generate a direct link to that exact moment in the Twitch VOD.</p>
    </div>

    <!-- Bottom Actions -->
    <div class="flex items-center justify-between mt-4 pt-3 border-t text-xs opacity-75"
         :class="store.isDark ? 'border-[#31333f]' : 'border-[#d6d9e0]'">
      <span>Tip: You can re-open these instructions anytime from the sidebar.</span>
      <button
        @click="store.dismissInstructions()"
        class="px-3.5 py-1 rounded text-xs font-semibold bg-[#ff4b4b] hover:bg-[#ff3333] text-white transition-colors"
      >
        Dismiss
      </button>
    </div>
  </div>
</template>