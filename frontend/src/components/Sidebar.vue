<script setup lang="ts">
import { ref } from 'vue';
import { useSentimentStore } from '../stores/sentimentStore';
import { Sun, Moon, HelpCircle } from '@lucide/vue';

const store = useSentimentStore();
const channelInput = ref('jasontheween');

function handleConnect() {
  if (!channelInput.value.trim()) return;
  store.connect(channelInput.value.trim());
}

function handleDisconnect() {
  store.disconnect();
}
</script>

<template>
  <aside
    class="w-full md:w-72 shrink-0 p-6 flex flex-col justify-between border-b md:border-b-0 md:border-r transition-colors duration-200"
    :class="store.isDark ? 'bg-[#262730] text-[#fafafa] border-[#31333f]' : 'bg-[#f0f2f6] text-[#31333f] border-[#d6d9e0]'"
  >
    <div class="space-y-6">
      <!-- Title -->
      <h2 class="text-xl font-bold tracking-tight">Connection</h2>

      <!-- Connection Form -->
      <div class="space-y-3">
        <div>
          <label class="block text-xs font-medium mb-1.5 opacity-80">Channel Name</label>
          <input
            v-model="channelInput"
            type="text"
            placeholder="xQc"
            :disabled="store.isConnecting"
            @keyup.enter="handleConnect"
            class="w-full px-3 py-2 rounded-md text-sm transition-colors border focus:outline-none focus:ring-1"
            :class="store.isDark 
              ? 'bg-[#0e1117] border-[#41444c] text-white focus:border-[#ff4b4b] focus:ring-[#ff4b4b]' 
              : 'bg-white border-[#d6d9e0] text-[#31333f] focus:border-[#ff4b4b] focus:ring-[#ff4b4b]'"
          />
        </div>

        <!-- Connect Button -->
        <button
          @click="handleConnect"
          :disabled="store.isConnecting || !channelInput"
          class="w-full py-2 px-4 rounded-md text-sm font-semibold text-white transition-all disabled:opacity-50 disabled:cursor-not-allowed shadow-sm"
          :class="store.isConnected ? 'bg-zinc-600 hover:bg-zinc-700' : 'bg-[#ff4b4b] hover:bg-[#ff3333] active:scale-[0.99]'"
        >
          {{ store.isConnecting ? 'Connecting...' : 'Connect' }}
        </button>

        <!-- Connected Status Badge -->
        <div
          v-if="store.isConnected"
          class="p-3 rounded-md text-sm font-medium border"
          :class="store.isDark 
            ? 'bg-[#12281e] text-[#00cc96] border-[#1d4633]' 
            : 'bg-[#e6f9f3] text-[#00966e] border-[#b3edd9]'"
        >
          Connected to {{ store.currentChannel }}!
        </div>

        <!-- Stop / Disconnect Button -->
        <button
          v-if="store.isConnected"
          @click="handleDisconnect"
          class="w-full py-2 px-4 rounded-md text-sm font-medium border transition-colors"
          :class="store.isDark 
            ? 'bg-[#1e2026] hover:bg-[#2e313b] text-zinc-300 border-[#3f424e]' 
            : 'bg-white hover:bg-zinc-100 text-zinc-700 border-zinc-300'"
        >
          Stop / Disconnect
        </button>
      </div>

      <!-- Instructions Toggle Button in Sidebar -->
      <div class="pt-2">
        <button
          @click="store.toggleInstructions()"
          class="w-full py-2 px-3 rounded-md text-xs font-medium border flex items-center justify-center gap-2 transition-colors hover:bg-black/5 dark:hover:bg-white/5"
          :class="store.isDark ? 'border-[#3f424e] text-zinc-300' : 'border-[#d6d9e0] text-zinc-700'"
        >
          <HelpCircle class="w-3.5 h-3.5 text-[#ff4b4b]" />
          <span>{{ store.showInstructions ? 'Hide Instructions' : 'View Instructions' }}</span>
        </button>
      </div>
    </div>

    <!-- Bottom Controls: Theme Toggle & Backend Status -->
    <div class="pt-6 border-t mt-6 flex items-center justify-between text-xs opacity-75"
         :class="store.isDark ? 'border-[#3f424e]' : 'border-[#d6d9e0]'">
      <div class="flex items-center gap-1.5">
        <span class="w-2 h-2 rounded-full" :class="store.isBackendOnline ? 'bg-[#00cc96]' : 'bg-[#ef553b]'"></span>
        <span>{{ store.isBackendOnline ? 'Backend Online' : 'Backend Offline' }}</span>
      </div>

      <button
        @click="store.toggleTheme()"
        class="p-1.5 rounded-md hover:bg-black/10 dark:hover:bg-white/10 transition-colors"
        :title="store.isDark ? 'Switch to Light Mode' : 'Switch to Dark Mode'"
      >
        <Sun v-if="store.isDark" class="w-4 h-4 text-amber-300" />
        <Moon v-else class="w-4 h-4 text-zinc-600" />
      </button>
    </div>
  </aside>
</template>