const searchForm = document.getElementById("search-form");
const queryInput = document.getElementById("query-input");
const submitButton = document.getElementById("submit-button");
const statusText = document.getElementById("status-text");
const intentOutput = document.getElementById("intent-output");
const priceOutput = document.getElementById("price-output");
const providerBadge = document.getElementById("provider-badge");
const resultCount = document.getElementById("result-count");
const resultsContainer = document.getElementById("results");

const money = new Intl.NumberFormat("vi-VN");

function setStatus(message, tone = "") {
  statusText.textContent = message;
  statusText.dataset.tone = tone;
}

function setJsonBlock(node, value) {
  if (!value) {
    node.textContent = "Chua co du lieu.";
    node.classList.add("empty");
    return;
  }
  node.textContent = JSON.stringify(value, null, 2);
  node.classList.remove("empty");
}

function formatPrice(priceVnd) {
  if (priceVnd === null || priceVnd === undefined) {
    return "Khong ro gia";
  }
  return `${money.format(priceVnd)} VND`;
}

function renderRecommendations(items) {
  resultCount.textContent = `${items.length} items`;
  if (!items.length) {
    resultsContainer.className = "results empty-state";
    resultsContainer.textContent = "Khong co candidate nao duoc tra ve.";
    return;
  }

  resultsContainer.className = "results";
  resultsContainer.innerHTML = items
    .map((item) => {
      const candidate = item.candidate || {};
      const reasons = (item.reasons || []).join(", ") || "Khong co";
      const tradeoffs = (item.tradeoffs || []).join(", ") || "Khong co";
      const amenities = (candidate.amenities || []).join(", ") || "Khong ro";
      const locations = (candidate.location_tags || []).join(", ") || "Khong ro";
      const sourceUrl = candidate.source_url
        ? `<a href="${candidate.source_url}" target="_blank" rel="noreferrer">Mo nguon</a>`
        : `<span>Khong co URL</span>`;

      return `
        <article class="result-card">
          <div class="result-topline">
            <h3>${candidate.name || "Unknown stay"}</h3>
            <span class="score">score ${Number(item.score || 0).toFixed(4)}</span>
          </div>
          <p class="meta">
            ${formatPrice(candidate.price_vnd)} | ${candidate.property_type || "Khong ro loai"} | ${candidate.destination || "Khong ro diem den"}
          </p>
          <p class="meta">Amenities: ${amenities}</p>
          <p class="meta">Location tags: ${locations}</p>
          <p class="meta">Reasons: ${reasons}</p>
          <p class="meta">Tradeoffs: ${tradeoffs}</p>
          <div class="result-footer">${sourceUrl}</div>
        </article>
      `;
    })
    .join("");
}

async function refreshHealth() {
  try {
    const response = await fetch("/health");
    if (!response.ok) {
      return;
    }
    const payload = await response.json();
    providerBadge.textContent = `provider: ${payload.search_provider || "unknown"}`;
  } catch (error) {
    providerBadge.textContent = "provider: unavailable";
  }
}

searchForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  const query = queryInput.value.trim();
  if (!query) {
    setStatus("Can nhap query.", "error");
    return;
  }

  submitButton.disabled = true;
  setStatus("Dang goi /api/recommend ...", "loading");

  try {
    const response = await fetch("/api/recommend", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ query }),
    });
    const payload = await response.json();
    if (!response.ok) {
      throw new Error(payload.error || `Request failed with status ${response.status}`);
    }

    setJsonBlock(intentOutput, payload.parsed_intent);
    setJsonBlock(priceOutput, payload.price_band);
    renderRecommendations(payload.recommendations || []);
    setStatus("Response OK.", "success");
  } catch (error) {
    renderRecommendations([]);
    setStatus(error.message || "Khong goi duoc API.", "error");
  } finally {
    submitButton.disabled = false;
  }
});

refreshHealth();
