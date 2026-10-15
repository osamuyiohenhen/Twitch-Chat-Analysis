<script setup lang="ts">
import { ref } from 'vue';
import { useSentimentStore } from '../stores/sentimentStore';
import {
  Activity,
  Radio,
  Server,
  CheckCircle2,
  XCircle,
  ExternalLink,
} from '@lucide/vue';

const store = useSentimentStore();
const channelInput = ref('tarik');

const presetChannels = ['tarik', 'caedrel', 'shroud', 'caseoh_', 'hasanabi'];

function handleConnect() {
  if (!channelInput.value) return;
  if (store.isConnected && store.currentChannel.toLowerCase() === channelInput.value.toLowerCase()) {
    return;
  }
  store.connect(channelInput.value);
}

function handleDisconnect() {
  store.disconnect();
}

function selectPreset(name: string) {
  channelInput.value = name;
  store.connect(name);
}
</script>

<template>
  <header class="bg-[#18181b] border-b border-[#2e2e35] px-6 py-4 shadow-lg">
    <div class="max-w-7xl mx-auto flex flex-col md:flex-row md:items-center justify-between gap-4">
      
      <!-- Brand & Title -->
      <div class="flex items-center gap-3">
        <div class="h-10 w-10 rounded-xl bg-purple-600 flex items-center justify-center shadow-lg shadow-purple-600/30 text-white font-bold">
          <Activity class="w-6 h-6 animate-pulse" />
        </div>
        <div>
          <div class="flex items-center gap-2">
            <h1 class="text-xl font-bold text-white tracking-tight">Twitch Sentiment Pulse</h1>
            <span class="text-xs px-2 py-0.5 rounded-full bg-purple-500/20 text-purple-400 font-medium border border-purple-500/30">
              v2.0
            </span>
          </div>
          <p class="text-xs text-zinc-400">Real-time gaming dialect sentiment & velocity analytics</p>
        </div>
      </div>

      <!-- Channel Search & Controls -->
      <div class="flex flex-col sm:flex-row items-stretch sm:items-center gap-2">
        <div class="relative flex-1 sm:w-64">
          <span class="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none text-zinc-500">
            #
          </span>
          <input
            v-model="channelInput"
            type="text"
            placeholder="Channel name..."
            :disabled="store.isConnecting"
            @keyup.enter="handleConnect"
            class="w-full pl-8 pr-3 py-2 bg-[#0e0e10] border border-[#2e2e35] rounded-lg text-sm text-zinc-100 placeholder-zinc-500 focus:outline-none focus:border-purple-500 focus:ring-1 focus:ring-purple-500 transition-colors"
          />
        </div>

        <button
          v-if="!store.isConnected"
          @click="handleConnect"
          :disabled="store.isConnecting || !channelInput"
          class="flex items-center justify-center gap-2 px-4 py-2 bg-purple-600 hover:bg-purple-700 disabled:opacity-50 disabled:cursor-not-allowed text-white font-medium text-sm rounded-lg transition-colors shadow-md shadow-purple-600/20"
        >
          <Radio v-if="!store.isConnecting" class="w-4 h-4" />
          <span v-if="store.isConnecting" class="inline-block w-4 h-4 border-2 border-white/30 border-t-white rounded-full animate-spin"></span>
          {{ store.isConnecting ? 'Connecting...' : 'Connect' }}
        </button>

        <button
          v-else
          @click="handleDisconnect"
          class="flex items-center justify-center gap-2 px-4 py-2 bg-rose-600 hover:bg-rose-700 text-white font-medium text-sm rounded-lg transition-colors shadow-md shadow-rose-600/20"
        >
          <XCircle class="w-4 h-4" />
          Disconnect
        </button>
      </div>

      <!-- Backend & Channel Status -->
      <div class="flex items-center gap-3">
        <!-- Live Channel Badge -->
        <div v-if="store.isConnected" class="flex items-center gap-2 px-3 py-1.5 rounded-lg bg-emerald-500/10 border border-emerald-500/30 text-emerald-400 text-xs font-semibold">
          <span class="w-2 h-2 rounded-full bg-emerald-500 animate-ping"></span>
          <span>LIVE: #{{ store.currentChannel }}</span>
          <a
            :href="`https://twitch.tv/${store.currentChannel}`"
            target="_blank"
            rel="noopener noreferrer"
            class="text-zinc-400 hover:text-white transition-colors"
            title="Open stream in Twitch"
          >
            <ExternalLink class="w-3.5 h-3.5" />
          </a>
        </div>

        <!-- Backend Health Badge -->
        <div
          class="flex items-center gap-2 px-3 py-1.5 rounded-lg text-xs font-medium border"
          :class="store.isBackendOnline
            ? 'bg-zinc-900 border-emerald-500/20 text-emerald-400'
            : 'bg-zinc-900 border-rose-500/20 text-rose-400'"
        >
          <Server class="w-3.5 h-3.5" />
          <span>{{ store.isBackendOnline ? 'API Online' : 'API Offline' }}</span>
          <CheckCircle2 v-if="store.isBackendOnline" class="w-3.5 h-3.5 text-emerald-500" />
          <XCircle v-else class="w-3.5 h-3.5 text-rose-500" />
        </div>
      </div>
    </div>

    <!-- Quick Presets -->
    <div class="max-w-7xl mx-auto mt-3 flex items-center gap-2 text-xs text-zinc-400 overflow-x-auto pb-1">
      <span class="font-medium text-zinc-500">Quick Connect:</span>
      <button
        v-for="name in presetChannels"
        :key="name"
        @click="selectPreset(name)"
        class="px-2.5 py-1 rounded-md bg-[#0e0e10] hover:bg-[#2e2e35] text-zinc-300 border border-[#2e2e35] transition-colors"
        :class="{ 'border-purple-500 text-purple-400': store.currentChannel === name }"
      >
        #{{ name }}
      </button>
    </div>
  </header>
</template>
