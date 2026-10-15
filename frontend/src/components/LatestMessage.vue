<script setup lang="ts">
import { computed } from 'vue';
import { useSentimentStore } from '../stores/sentimentStore';

const store = useSentimentStore();

const sentimentDot = computed(() => {
  const label = store.latestMessage?.label?.toLowerCase();
  if (label === 'positive') return '🟢';
  if (label === 'negative') return '🔴';
  return '⚪';
});
</script>

<template>
  <div v-if="store.isConnected" class="space-y-1">
    <div class="flex items-center gap-1.5 text-sm font-semibold">
      <span>Latest Message</span>
      <span>{{ sentimentDot }}</span>
    </div>

    <div
      v-if="store.latestMessage"
      class="text-sm font-medium transition-all"
    >
      <span class="font-bold capitalize">{{ store.latestMessage.user || store.currentChannel }}: </span>
      <span class="italic opacity-90">"{{ store.latestMessage.message }}"</span>
    </div>
    <div v-else class="text-xs opacity-50 italic">
      Waiting for incoming chat message...
    </div>
  </div>
</template>