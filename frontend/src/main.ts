import { createApp } from 'vue';
import { createPinia } from 'pinia';
import '@fontsource/inter/300.css';
import '@fontsource/inter/400.css';
import '@fontsource/inter/500.css';
import '@fontsource/inter/600.css';
import '@fontsource/inter/700.css';
import '@fontsource/inter/800.css';
import './style.css';
import App from './App.vue';
import { Chart as ChartJS } from 'chart.js';

ChartJS.defaults.font.family = "'Inter', system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif";

const app = createApp(App);
const pinia = createPinia();

app.use(pinia);
app.mount('#app');
