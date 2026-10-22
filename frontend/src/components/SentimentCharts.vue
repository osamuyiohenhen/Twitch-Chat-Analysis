<script setup lang="ts">
import { computed } from 'vue';
import { useSentimentStore } from '../stores/sentimentStore';
import {
  Chart as ChartJS,
  Title,
  Tooltip,
  Legend,
  LineElement,
  LinearScale,
  PointElement,
  CategoryScale,
  ArcElement,
  Filler,
  type ChartOptions,
} from 'chart.js';
import { Line, Doughnut } from 'vue-chartjs';
import { TrendingUp, PieChart } from '@lucide/vue';

ChartJS.register(
  Title,
  Tooltip,
  Legend,
  LineElement,
  LinearScale,
  PointElement,
  CategoryScale,
  ArcElement,
  Filler
);

const store = useSentimentStore();

// --- Timeline Line Chart Data ---
const timelineChartData = computed(() => {
  const points = store.timeline.slice(-20);
  const labels = points.map((p) => {
    const d = new Date(p.minute_timestamp * 1000);
    return `${d.getHours().toString().padStart(2, '0')}:${d.getMinutes().toString().padStart(2, '0')}`;
  });

  return {
    labels: labels.length > 0 ? labels : ['--:--'],
    datasets: [
      {
        label: 'Positive Messages',
        data: points.length > 0 ? points.map((p) => p.pos_count) : [0],
        borderColor: '#10b981',
        backgroundColor: 'rgba(16, 185, 129, 0.1)',
        tension: 0.35,
        fill: true,
        pointRadius: 3,
        pointHoverRadius: 6,
      },
      {
        label: 'Negative Messages',
        data: points.length > 0 ? points.map((p) => p.neg_count) : [0],
        borderColor: '#f43f5e',
        backgroundColor: 'rgba(244, 63, 94, 0.1)',
        tension: 0.35,
        fill: true,
        pointRadius: 3,
        pointHoverRadius: 6,
      },
    ],
  };
});

const lineOptions: ChartOptions<'line'> = {
  responsive: true,
  maintainAspectRatio: false,
  plugins: {
    legend: {
      position: 'top',
      labels: {
        color: '#a1a1aa',
        boxWidth: 12,
        font: { size: 11, family: "'Inter', sans-serif" },
      },
    },
    tooltip: {
      backgroundColor: '#18181b',
      titleColor: '#fff',
      bodyColor: '#e4e4e7',
      borderColor: '#3f3f46',
      borderWidth: 1,
      padding: 10,
    },
  },
  scales: {
    x: {
      grid: {
        color: 'rgba(46, 46, 53, 0.5)',
      },
      ticks: {
        color: '#71717a',
        font: { size: 10 },
      },
    },
    y: {
      beginAtZero: true,
      grid: {
        color: 'rgba(46, 46, 53, 0.5)',
      },
      ticks: {
        color: '#71717a',
        font: { size: 10 },
      },
    },
  },
};

// --- Sentiment Doughnut Data ---
const doughnutData = computed(() => {
  const pos = store.positiveRatio;
  const neg = store.negativeRatio;
  const neu = store.neutralRatio;

  const total = pos + neg + neu;
  const data = total > 0 ? [pos, neu, neg] : [33, 34, 33];

  return {
    labels: ['Positive', 'Neutral', 'Negative'],
    datasets: [
      {
        data,
        backgroundColor: ['#10b981', '#f59e0b', '#f43f5e'],
        borderWidth: 0,
        hoverOffset: 4,
      },
    ],
  };
});

const doughnutOptions: ChartOptions<'doughnut'> = {
  responsive: true,
  maintainAspectRatio: false,
  cutout: '72%',
  plugins: {
    legend: {
      position: 'bottom',
      labels: {
        color: '#a1a1aa',
        boxWidth: 10,
        padding: 12,
        font: { size: 11 },
      },
    },
    tooltip: {
      backgroundColor: '#18181b',
      titleColor: '#fff',
      bodyColor: '#e4e4e7',
      borderColor: '#3f3f46',
      borderWidth: 1,
      callbacks: {
        label: (item) => ` ${item.label}: ${item.raw}%`,
      },
    },
  },
};
</script>

<template>
  <div class="grid grid-cols-1 lg:grid-cols-3 gap-6">
    <!-- Timeline Line Chart (2 Cols) -->
    <div class="lg:col-span-2 bg-[#18181b] border border-[#2e2e35] rounded-xl p-5 shadow-sm flex flex-col">
      <div class="flex items-center justify-between mb-4">
        <div class="flex items-center gap-2">
          <TrendingUp class="w-4 h-4 text-purple-400" />
          <h2 class="text-sm font-semibold text-white">Sentiment Trajectory Over Time</h2>
        </div>
        <span class="text-xs text-zinc-500">1-minute aggregations</span>
      </div>
      <div class="h-64 relative">
        <Line :data="timelineChartData" :options="lineOptions" />
      </div>
    </div>

    <!-- 30-Second Distribution Doughnut (1 Col) -->
    <div class="bg-[#18181b] border border-[#2e2e35] rounded-xl p-5 shadow-sm flex flex-col">
      <div class="flex items-center justify-between mb-4">
        <div class="flex items-center gap-2">
          <PieChart class="w-4 h-4 text-purple-400" />
          <h2 class="text-sm font-semibold text-white">Live Sentiment Breakdown</h2>
        </div>
        <span class="text-xs text-zinc-500">Last 30s</span>
      </div>
      <div class="h-64 relative flex items-center justify-center">
        <Doughnut :data="doughnutData" :options="doughnutOptions" />
        <!-- Center Text -->
        <div class="absolute inset-0 flex flex-col items-center justify-center pointer-events-none pb-6">
          <span class="text-2xl font-extrabold text-white">{{ store.positiveRatio }}%</span>
          <span class="text-[10px] uppercase tracking-wider text-emerald-400 font-semibold">Positive</span>
        </div>
      </div>
    </div>
  </div>
</template>
