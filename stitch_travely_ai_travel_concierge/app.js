const API_BASE = window.TRAVELY_API_BASE || "http://127.0.0.1:5000";

const elements = {
  apiBase: document.getElementById("api-base"),
  backendStatusChip: document.getElementById("backend-status-chip"),
  searchProvider: document.getElementById("search-provider"),
  requestStatus: document.getElementById("request-status"),
  requestForm: document.getElementById("request-form"),
  queryInput: document.getElementById("query-input"),
  submitButton: document.getElementById("submit-button"),
  refreshHealthButton: document.getElementById("refresh-health-button"),
  greetingTitle: document.getElementById("greeting-title"),
  resultsPanel: document.getElementById("results-panel"),
  resultsMeta: document.getElementById("results-meta"),
  resultsMessage: document.getElementById("results-message"),
  resultsGrid: document.getElementById("results-grid"),
};

elements.apiBase.textContent = API_BASE;

function applyGreeting(now = new Date()) {
  const hour = now.getHours();
  let title = "Chúng ta nên bắt đầu từ đâu nhỉ?";

  if (hour >= 5 && hour < 11) {
    title = "Chào buổi sáng, hôm nay anh muốn đi đâu?";
  } else if (hour >= 11 && hour < 14) {
    title = "Chào buổi trưa, mình lên lịch một chuyến đi nhé?";
  } else if (hour >= 14 && hour < 18) {
    title = "Chào buổi chiều, mình khám phá điểm đến mới nhé?";
  } else if (hour >= 18 && hour < 23) {
    title = "Chào buổi tối, mình lên một chuyến đi thật xịn nhé?";
  } else {
    title = "Khuya rồi, mình vẫn có thể lên plan du lịch thật mượt.";
  }

  elements.greetingTitle.textContent = title;
}

const moneyFormatter = new Intl.NumberFormat("vi-VN");

function formatMoney(value) {
  if (value === null || value === undefined) {
    return "Chưa có";
  }
  return `${moneyFormatter.format(Math.round(value))} VND`;
}

function joinOrFallback(items, fallback = "Chưa có") {
  return Array.isArray(items) && items.length ? items.join(", ") : fallback;
}

function setRequestStatus(message) {
  elements.requestStatus.textContent = message;
}

function setBusyState(isBusy) {
  elements.requestForm.dataset.busy = isBusy ? "true" : "false";
}

function setBackendChip(label, className) {
  elements.backendStatusChip.textContent = label;
  elements.backendStatusChip.className = className;
}

function showResultsPanel() {
  elements.resultsPanel.classList.remove("hidden");
}

async function readJson(response) {
  const text = await response.text();
  if (!text) {
    return {};
  }
  try {
    return JSON.parse(text);
  } catch (_error) {
    return { error: text };
  }
}

async function refreshHealth() {
  try {
    const response = await fetch(`${API_BASE}/health`);
    const payload = await readJson(response);
    const ready = !!payload.ready;

    setBackendChip(
      ready ? "Backend ready" : "Backend degraded",
      ready
        ? "rounded-full border border-emerald-400/25 bg-emerald-400/10 px-3 py-2 font-mono text-[11px] uppercase tracking-[0.18em] text-emerald-200"
        : "rounded-full border border-amber-300/25 bg-amber-300/10 px-3 py-2 font-mono text-[11px] uppercase tracking-[0.18em] text-amber-100",
    );
    elements.searchProvider.textContent = payload.search_provider || "Unknown";
  } catch (_error) {
    setBackendChip(
      "Backend offline",
      "rounded-full border border-rose-400/25 bg-rose-400/10 px-3 py-2 font-mono text-[11px] uppercase tracking-[0.18em] text-rose-200",
    );
    elements.searchProvider.textContent = "Unavailable";
  }
}

function renderRequestError(message) {
  showResultsPanel();
  elements.resultsMeta.textContent = "error";
  elements.resultsGrid.innerHTML = "";
  elements.resultsMessage.className = "mt-4 text-sm leading-7 text-rose-200";
  elements.resultsMessage.textContent = message;
  setTimeout(() => {
    elements.resultsPanel.scrollIntoView({ behavior: "smooth", block: "start" });
  }, 100);
}

function renderRecommendations(payload) {
  const recommendations = payload.recommendations || [];
  const parsedIntent = payload.parsed_intent || {};
  const priceBand = payload.price_band || {};

  showResultsPanel();
  elements.resultsMeta.textContent = `${recommendations.length} items`;
  elements.resultsMessage.className = "mt-4 text-sm leading-7 text-soft-text";
  elements.resultsMessage.textContent = `Điểm đến: ${parsedIntent.destination || "Chưa có"} · Khách: ${parsedIntent.guest_count || "Chưa có"} · Mức giá: ${priceBand.price_class_label || "Chưa có"} · Confidence: ${Math.round((priceBand.confidence || 0) * 100)}%`;

  if (!recommendations.length) {
    elements.resultsGrid.innerHTML = `
      <article class="rounded-[22px] border border-white/10 bg-white/[0.03] px-4 py-4 text-sm leading-7 text-soft-text">
        Backend không trả recommendation nào cho request này.
      </article>
    `;
    setTimeout(() => {
      elements.resultsPanel.scrollIntoView({ behavior: "smooth", block: "start" });
    }, 100);
    return;
  }

  elements.resultsGrid.innerHTML = recommendations
    .map((item, index) => {
      const candidate = item.candidate || {};
      return `
        <article class="rounded-[22px] border border-white/10 bg-white/[0.03] p-4 sm:p-5">
          <div class="flex items-start justify-between gap-4">
            <div>
              <p class="font-mono text-[11px] uppercase tracking-[0.18em] text-white/40">Candidate ${index + 1}</p>
              <h3 class="mt-2 text-xl font-semibold text-white">${candidate.name || "Unknown stay"}</h3>
              <p class="mt-2 text-sm leading-7 text-soft-text">${candidate.destination || "Unknown destination"} · ${candidate.property_type || "Property type missing"}</p>
            </div>
            <div class="font-mono text-[11px] uppercase tracking-[0.18em] text-white/45">${Number(item.score || 0).toFixed(4)}</div>
          </div>

          <div class="mt-4 space-y-2 text-sm leading-7 text-soft-text">
            <p><span class="text-white">Giá:</span> ${formatMoney(candidate.price_vnd)}</p>
            <p><span class="text-white">Amenities:</span> ${joinOrFallback(candidate.amenities)}</p>
            <p><span class="text-white">Lý do:</span> ${joinOrFallback(item.reasons, "Chưa có")}</p>
            <p><span class="text-white">Tradeoff:</span> ${joinOrFallback(item.tradeoffs, "Chưa có")}</p>
          </div>

          ${candidate.source_url ? `<a class="mt-4 inline-flex text-sm text-cyan-200 transition hover:text-white" href="${candidate.source_url}" target="_blank" rel="noreferrer">Mở nguồn</a>` : ""}
        </article>
      `;
    })
    .join("");

  setTimeout(() => {
    elements.resultsPanel.scrollIntoView({ behavior: "smooth", block: "start" });
  }, 100);
}

let thinkingTimeouts = [];

function clearThinking() {
  thinkingTimeouts.forEach(clearTimeout);
  thinkingTimeouts = [];
}

function startThinking() {
  clearThinking();
  showResultsPanel();
  elements.resultsMeta.textContent = "processing";
  elements.resultsMessage.className = "mt-4 text-sm leading-7 text-cyan-200/80 font-medium";
  elements.resultsMessage.textContent = "AI đang suy luận các bước...";
  
  elements.resultsGrid.innerHTML = `
    <div class="thinking-shell p-5">
      <div class="flex items-center gap-3">
        <div class="relative flex h-5 w-5 items-center justify-center">
          <span class="animate-ping absolute inline-flex h-full w-full rounded-full bg-cyan-400 opacity-75"></span>
          <span class="relative inline-flex rounded-full h-3 w-3 bg-cyan-500"></span>
        </div>
        <span class="font-mono text-xs uppercase tracking-[0.2em] text-cyan-200 font-semibold">Tiến trình suy luận (Thinking Process)</span>
      </div>
      
      <div class="mt-5 space-y-4 pl-8 text-sm">
        <div id="step-intent" class="flex items-center gap-3 text-white/30 transition-all duration-300">
          <span id="icon-intent" class="text-base">⚪</span>
          <span class="step-text">Phân tích ý định người dùng (LLM Intent Parser)</span>
        </div>
        <div id="step-price" class="flex items-center gap-3 text-white/30 transition-all duration-300">
          <span id="icon-price" class="text-base">⚪</span>
          <span class="step-text">Dự đoán phân khúc giá tối ưu (LightGBM Classifier)</span>
        </div>
        <div id="step-search" class="flex items-center gap-3 text-white/30 transition-all duration-300">
          <span id="icon-search" class="text-base">⚪</span>
          <span class="step-text">Tìm kiếm phòng thực tế thời gian thực (Perplexity Search API)</span>
        </div>
        <div id="step-rank" class="flex items-center gap-3 text-white/30 transition-all duration-300">
          <span id="icon-rank" class="text-base">⚪</span>
          <span class="step-text">Chấm điểm và xếp hạng danh sách đề xuất (Recommendation Ranker)</span>
        </div>
      </div>
    </div>
  `;

  setTimeout(() => {
    elements.resultsPanel.scrollIntoView({ behavior: "smooth", block: "start" });
  }, 100);

  const setStepState = (id, state) => {
    const el = document.getElementById(`step-${id}`);
    const icon = document.getElementById(`icon-${id}`);
    if (!el || !icon) return;
    
    if (state === "active") {
      el.className = "flex items-center gap-3 text-cyan-200 font-medium transition-all duration-300";
      icon.innerHTML = `<span class="inline-block animate-spin">🌀</span>`;
    } else if (state === "completed") {
      el.className = "flex items-center gap-3 text-emerald-400 transition-all duration-300";
      icon.innerHTML = "✅";
    } else {
      el.className = "flex items-center gap-3 text-white/30 transition-all duration-300";
      icon.innerHTML = "⚪";
    }
  };

  // Step 1: Active immediately
  setStepState("intent", "active");

  // Step 2: Active after 700ms
  thinkingTimeouts.push(setTimeout(() => {
    setStepState("intent", "completed");
    setStepState("price", "active");
  }, 700));

  // Step 3: Active after 1400ms
  thinkingTimeouts.push(setTimeout(() => {
    setStepState("price", "completed");
    setStepState("search", "active");
  }, 1400));
}

function completeThinkingAndShow(payload) {
  clearThinking();
  
  const setStepState = (id, state) => {
    const el = document.getElementById(`step-${id}`);
    const icon = document.getElementById(`icon-${id}`);
    if (!el || !icon) return;
    
    if (state === "active") {
      el.className = "flex items-center gap-3 text-cyan-200 font-medium transition-all duration-300";
      icon.innerHTML = `<span class="inline-block animate-spin">🌀</span>`;
    } else if (state === "completed") {
      el.className = "flex items-center gap-3 text-emerald-400 transition-all duration-300";
      icon.innerHTML = "✅";
    }
  };

  // Ensure prior steps are marked completed
  setStepState("intent", "completed");
  setStepState("price", "completed");
  setStepState("search", "completed");
  setStepState("rank", "active");

  setTimeout(() => {
    setStepState("rank", "completed");
    setTimeout(() => {
      renderRecommendations(payload);
    }, 300);
  }, 400);
}

async function submitQuery(event) {
  event.preventDefault();
  const query = elements.queryInput.value.trim();

  if (!query) {
    setRequestStatus("Nhập prompt trước đã");
    renderRequestError("Anh nhập travel request rồi hãy gửi backend.");
    setBusyState(false);
    return;
  }

  setRequestStatus("Đang gửi backend...");
  setBusyState(true);
  elements.submitButton.disabled = true;
  elements.submitButton.classList.add("opacity-70", "cursor-not-allowed");
  
  startThinking();

  try {
    const response = await fetch(`${API_BASE}/api/recommend`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query }),
    });
    const payload = await readJson(response);
    if (!response.ok) {
      throw new Error(payload.error || `Request failed with status ${response.status}`);
    }

    setRequestStatus("Đã nhận phản hồi");
    completeThinkingAndShow(payload);
  } catch (error) {
    clearThinking();
    setRequestStatus("Yêu cầu bị chặn");
    renderRequestError(error.message || "Backend request failed.");
  } finally {
    setBusyState(false);
    elements.submitButton.disabled = false;
    elements.submitButton.classList.remove("opacity-70", "cursor-not-allowed");
    await refreshHealth();
  }
}

elements.requestForm.addEventListener("submit", submitQuery);
elements.refreshHealthButton.addEventListener("click", refreshHealth);

applyGreeting();
setBusyState(false);
refreshHealth();
