/* global Chart */

/**
 * Simple frontend for visualization layer.
 * - Connects to WebSocket /ws/realtime
 * - Renders last messages list and a basic line chart
 * - Allows toggling detection methods via /api/detection/config
 * - Provides placeholder root cause query
 */

const state = {
  ws: null,
  backendBase: "http://localhost:5000",
  maxMessages: 200,
  messages: [],
  chart: null,
  chartData: [],
};

function $(id) {
  return document.getElementById(id);
}

function setWsStatus(connected) {
  const el = $("ws-status");
  if (!el) return;
  if (connected) {
    el.textContent = "WS 已连接";
    el.classList.remove("status-disconnected");
    el.classList.add("status-connected");
  } else {
    el.textContent = "WS 未连接";
    el.classList.remove("status-connected");
    el.classList.add("status-disconnected");
  }
}

async function checkBackendHealth() {
  const el = $("health-status");
  if (!el) return;
  try {
    const res = await fetch(`${state.backendBase}/health`);
    if (res.ok) {
      el.textContent = "后端正常";
      el.classList.remove("status-error", "status-unknown");
      el.classList.add("status-ok");
    } else {
      el.textContent = "后端异常";
      el.classList.remove("status-ok", "status-unknown");
      el.classList.add("status-error");
    }
  } catch {
    el.textContent = "后端不可达";
    el.classList.remove("status-ok", "status-unknown");
    el.classList.add("status-error");
  }
}

function appendMessage(raw) {
  try {
    const obj = JSON.parse(raw);
    state.messages.push(obj);
  } catch {
    state.messages.push({ raw });
  }
  if (state.messages.length > state.maxMessages) {
    state.messages.shift();
  }
  renderMessageList();
  updateChartData();
}

function renderMessageList() {
  const container = $("realtime-list");
  if (!container) return;
  container.innerHTML = "";
  const metricFilter = $("metric-filter")?.value?.trim();

  const filtered = metricFilter
    ? state.messages.filter((m) => {
        const name = m.metric_name || (m.metadata && m.metadata.metric_name);
        const type = m.metric_type;
        return (
          (name && String(name).includes(metricFilter)) ||
          (type && String(type).includes(metricFilter))
        );
      })
    : state.messages;

  const latest = filtered.slice(-50).reverse();
  for (const msg of latest) {
    const div = document.createElement("div");
    div.className = "realtime-item";
    const ts = msg.timestamp_us
      ? new Date(msg.timestamp_us / 1000).toLocaleTimeString()
      : "";
    const metric = `${msg.metric_type || ""}/${msg.metric_name || ""}`;
    div.textContent = `[${ts}] ${msg.node_id || ""} ${
      msg.rank_id || ""
    } ${metric} = ${msg.value}`;
    container.appendChild(div);
  }
}

function initChart() {
  const ctx = $("realtime-chart");
  if (!ctx || !Chart) return;
  state.chart = new Chart(ctx, {
    type: "line",
    data: {
      datasets: [
        {
          label: "实时指标值",
          data: [],
          borderColor: "#4e79ff",
          borderWidth: 1.5,
          pointRadius: 0,
        },
      ],
    },
    options: {
      animation: false,
      responsive: true,
      scales: {
        x: {
          type: "time",
          time: { unit: "second" },
          ticks: { maxTicksLimit: 10 },
        },
        y: {
          beginAtZero: false,
        },
      },
      plugins: {
        legend: { display: true },
      },
    },
  });
}

function updateChartData() {
  if (!state.chart) return;
  const metricFilter = $("metric-filter")?.value?.trim();
  const maxPoints = parseInt($("max-points")?.value || "200", 10) || 200;

  const filtered = metricFilter
    ? state.messages.filter((m) => {
        const name = m.metric_name || (m.metadata && m.metadata.metric_name);
        const type = m.metric_type;
        return (
          (name && String(name).includes(metricFilter)) ||
          (type && String(type).includes(metricFilter))
        );
      })
    : state.messages;

  const points = filtered.slice(-maxPoints).map((m) => ({
    x: m.timestamp_us ? new Date(m.timestamp_us / 1000) : new Date(),
    y: m.value ?? 0,
  }));

  state.chart.data.datasets[0].data = points;
  state.chart.update("none");
}

function connectWebSocket() {
  const backendInput = $("backend-url");
  if (backendInput && backendInput.value) {
    state.backendBase = backendInput.value.replace(/\/+$/, "");
  }
  const wsUrl = state.backendBase.replace(/^http/, "ws") + "/ws/realtime";

  if (state.ws) {
    state.ws.close();
    state.ws = null;
  }

  try {
    const ws = new WebSocket(wsUrl);
    state.ws = ws;
    ws.onopen = () => {
      setWsStatus(true);
    };
    ws.onclose = () => {
      setWsStatus(false);
    };
    ws.onerror = () => {
      setWsStatus(false);
    };
    ws.onmessage = (event) => {
      appendMessage(event.data);
    };
  } catch (e) {
    console.error("Failed to open WebSocket:", e);
    setWsStatus(false);
  }
}

async function loadDetectorConfig() {
  const msgEl = $("config-message");
  try {
    const res = await fetch(`${state.backendBase}/api/detection/config`);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const cfg = await res.json();
    $("cfg-statistical").checked = !!cfg.statistical;
    $("cfg-comparison").checked = !!cfg.comparison;
    $("cfg-ml").checked = !!cfg.ml;
    msgEl.textContent = "已加载当前配置：" + JSON.stringify(cfg);
  } catch (e) {
    msgEl.textContent = "加载配置失败: " + e.message;
  }
}

async function saveDetectorConfig() {
  const msgEl = $("config-message");
  const payload = {
    statistical: $("cfg-statistical").checked,
    comparison: $("cfg-comparison").checked,
    ml: $("cfg-ml").checked,
  };
  try {
    const res = await fetch(`${state.backendBase}/api/detection/config`, {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const cfg = await res.json();
    msgEl.textContent = "已保存配置：" + JSON.stringify(cfg);
  } catch (e) {
    msgEl.textContent = "保存配置失败: " + e.message;
  }
}

async function queryRootCause() {
  const id = $("rootcause-anomaly-id").value.trim();
  const out = $("rootcause-result");
  if (!id) {
    out.textContent = "请先输入 anomaly_id。";
    return;
  }
  try {
    const res = await fetch(`${state.backendBase}/api/root_cause/${encodeURIComponent(id)}`);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();
    out.textContent = JSON.stringify(data, null, 2);
  } catch (e) {
    out.textContent = "查询失败: " + e.message;
  }
}

function initTabs() {
  const tabs = document.querySelectorAll(".tab");
  const panels = document.querySelectorAll(".tab-panel");
  tabs.forEach((btn) => {
    btn.addEventListener("click", () => {
      const target = btn.getAttribute("data-target");
      tabs.forEach((b) => b.classList.remove("active"));
      panels.forEach((p) => p.classList.remove("active"));
      btn.classList.add("active");
      const panel = document.getElementById(target);
      if (panel) panel.classList.add("active");
    });
  });
}

function initEvents() {
  $("reconnect-ws").addEventListener("click", () => {
    connectWebSocket();
  });
  $("metric-filter").addEventListener("input", () => {
    renderMessageList();
    updateChartData();
  });
  $("max-points").addEventListener("change", () => {
    updateChartData();
  });
  $("btn-load-config").addEventListener("click", () => {
    loadDetectorConfig();
  });
  $("btn-save-config").addEventListener("click", () => {
    saveDetectorConfig();
  });
  $("btn-rootcause-query").addEventListener("click", () => {
    queryRootCause();
  });
}

function main() {
  const backendInput = $("backend-url");
  if (backendInput && backendInput.value) {
    state.backendBase = backendInput.value.replace(/\/+$/, "");
  }
  initTabs();
  initChart();
  initEvents();
  connectWebSocket();
  checkBackendHealth();
}

document.addEventListener("DOMContentLoaded", main);


