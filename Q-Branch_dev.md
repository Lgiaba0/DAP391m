# KẾ HOẠCH TIẾP THEO - GIỮ UI ĐẸP, NỐI FULL STACK THẬT

> Mục tiêu: giữ shell UI hiện tại, bỏ phần giả lập, nối backend thật, search thật, ML thật, test đủ.
>
> Nguyên tắc: không đổi UI thành layout khác. Chỉ sửa đúng phần cần để chạy thật.

## Kết luận sau khi rà soát thiết kế

- Giữ:
  - `stitch_travely_ai_travel_concierge/code.html` với shell SpaceX hiện tại
  - background `spaceX_AI.png`
  - hướng chat/input hiện tại
- Thiếu cần bổ sung:
  - gửi query thật
  - trạng thái `loading` / `error` / `degraded` / `success`
  - render `parsed_intent` + `price_band` + `recommendations` + link nguồn thật
  - trạng thái `health/readiness` từ backend
- Thừa nên cắt hoặc ẩn:
  - biểu đồ xử lý giả
  - thẻ khách sạn hard-code
  - các mục phụ không phục vụ luồng concierge thật: `Itineraries`, `Insights`, `History`, `Upgrade to Pro`, `Support`, `Archive`

## Definition of Done

- UI ở `stitch_travely_ai_travel_concierge/code.html` gọi API thật, không render dữ liệu giả.
- `api_server.py` trả `GET /health` và `POST /api/recommend` ổn định.
- `agents/web_search_agent.py` gọi provider thật, không silent mock.
- `ML_core/core/price_band_service.py` load `.joblib` thật, lấy `predict_proba()` thật.
- Test backend pass. UI build pass.
- Khi thiếu key/model: UI và API báo lỗi rõ, không giả vờ thành công.

## Các việc cần làm tiếp theo

### 1. Khóa lại phạm vi UI thật

Files:
- Modify: `stitch_travely_ai_travel_concierge/code.html`
- Modify: `stitch_travely_ai_travel_concierge/app.js`

Làm:
- Cắt/ẩn toàn bộ block demo giả.
- Giữ khung đẹp, thêm `id`/`data-slot` rõ cho:
  - input
  - submit button
  - health badge
  - loading state
  - error state
  - result list
  - source links

Verify:
- UI vẫn dùng shell cũ.
- Không còn hotel hard-code và fake processing chart.

### 2. Nối frontend vào backend thật

Files:
- Modify: `stitch_travely_ai_travel_concierge/code.html`
- Modify: `stitch_travely_ai_travel_concierge/app.js`

Làm:
- `app.js` gọi `GET /health` khi load trang.
- `app.js` gọi `POST /api/recommend` khi submit.
- Render 4 state:
  - backend ready
  - backend degraded
  - request loading
  - request fail
- Render dữ liệu thật:
  - `parsed_intent`
  - `price_band`
  - `recommendations`
  - `debug.search_queries` nếu có

Verify:
- `npm.cmd run dev`
- browser submit được
- DevTools thấy request thật tới API

### 3. Chốt contract API và readiness

Files:
- Modify: `api_server.py`
- Test: `tests/test_api_server.py`

Làm:
- Giữ contract JSON ổn định, không đổi vô lý.
- Đảm bảo:
  - query rỗng -> `400`
  - thiếu search key -> `503`
  - thiếu model artifact -> `503`
  - provider fail -> `502`
- `GET /health` phải báo:
  - `ready`
  - `issues`
  - `search_provider`
  - `price_classifier_path`

Verify:
- `python -m unittest tests.test_api_server`

### 4. Chốt Web Search thật

Files:
- Modify: `agents/web_search_agent.py`
- Maybe modify: `agents/source_normalizer.py`
- Test: `tests/test_web_search_agent.py`

Làm:
- Ưu tiên provider:
  - `SERPAPI_API_KEY`
  - hoặc `GOOGLE_API_KEY` + `GOOGLE_CSE_ID`
- Mọi result đi qua `normalize_search_result(...)`.
- Trả nhiều candidates thật, có dedupe, có metadata provider/query.
- Tuyệt đối không fallback mock trong production path.

Verify:
- test pass
- query thật không trả `metadata.mock = true`

### 5. Chốt ML V3 thật

Files:
- Modify: `ML_core/core/price_band_service.py`
- Maybe modify: `ML_core/core/config.py`
- Test: `tests/test_price_band_service.py`

Làm:
- Load artifact từ `PRICE_CLASSIFIER_PATH` hoặc default:
  - `ML_core/models/classify/v3/price_classification_v3_model.joblib`
- Lấy class bằng `predict_proba()`.
- `confidence = max(proba)`.
- Nếu thiếu artifact -> `ModelArtifactError` rõ ràng.
- Nếu model cần feature adapter nhỏ, đặt trong service, không đẩy complexity ra ngoài pipeline.

Verify:
- `python -m unittest tests.test_price_band_service`

### 6. Chạy full verify cuối

Files:
- Verify: `tests/test_api_server.py`
- Verify: `tests/test_web_search_agent.py`
- Verify: `tests/test_price_band_service.py`
- Verify: `stitch_travely_ai_travel_concierge/package.json`

Run:
```powershell
python -m unittest discover tests
cd stitch_travely_ai_travel_concierge
npm.cmd run build
```

Done khi:
- test pass
- build pass
- UI gọi API thật
- API trả data thật

## Anh cần chuẩn bị song song

Bắt buộc:
- 1 search provider thật:
  - `SERPAPI_API_KEY`
  - hoặc `GOOGLE_API_KEY` + `GOOGLE_CSE_ID`
- 1 file model thật:
  - `ML_core/models/classify/v3/price_classification_v3_model.joblib`
  - hoặc đưa em path chính xác để set `PRICE_CLASSIFIER_PATH`

Nên chuẩn bị thêm:
- file `.env` từ `.env.example`
- 2-3 query test thật anh muốn demo
- nếu anh muốn giữ item nav nào trong UI, nói em sớm để em không cắt nhầm

## Thứ tự thực thi nhanh nhất

- [ ] Dọn phạm vi UI trong `code.html`
- [ ] Nối fetch + render thật trong `app.js`
- [ ] Chốt readiness + error contract trong `api_server.py`
- [ ] Chốt web search provider thật trong `agents/web_search_agent.py`
- [ ] Chốt ML artifact + `predict_proba()` thật trong `price_band_service.py`
- [ ] Chạy test + build + demo end-to-end

# 45 PHĂT Tá»I â€” FULL STACK PHáº¢I XONG

Má»¥c tiĂªu má»›i:
- KhĂ´ng demo ná»­a vá»i.
- KhĂ´ng bá» qua API.
- KhĂ´ng bá» qua Web Search tháº­t.
- KhĂ´ng bá» qua ML V3 tháº­t.
- UI anh tá»± thiáº¿t káº¿, nhÆ°ng ká»¹ thuáº­t ná»‘i UI pháº£i cháº¡y.

## Äiá»u kiá»‡n báº¯t buá»™c

Cáº§n cĂ³ Ä‘á»§ 2 thá»© nĂ y Ä‘á»ƒ gá»i lĂ  "xong tháº­t":
- Web Search API key: Google Custom Search hoáº·c SerpAPI.
- ML model artifact V3: file `.joblib` Ä‘Ăºng path service load Ä‘Æ°á»£c.

Náº¿u thiáº¿u 1 trong 2:
- Váº«n code full integration.
- Váº«n cĂ³ error message rĂµ.
- KhĂ´ng Ä‘Æ°á»£c ghi "hoĂ n thĂ nh tháº­t".
- Tráº¡ng thĂ¡i chá»‰ lĂ  "code sáºµn, chá» key/artifact".

## Definition of Done

Xong tháº­t khi:
- `api_server.py` cháº¡y Flask API.
- `GET /` má»Ÿ UI.
- `POST /api/recommend` nháº­n query vĂ  tráº£ JSON tháº­t.
- UI gá»i API báº±ng `fetch()` vĂ  render káº¿t quáº£.
- `WebSearchAgent` gá»i provider tháº­t, khĂ´ng cĂ²n hard-coded mock candidate.
- Káº¿t quáº£ search Ä‘i qua `source_normalizer.normalize_search_result()`.
- `PriceBandService` load V3 `.joblib`.
- `PriceBandService` dĂ¹ng `predict_proba()` Ä‘á»ƒ láº¥y confidence tháº­t.
- KhĂ´ng Ä‘á»•i data contract.
- Test cÅ© pass.
- CĂ³ thĂªm test hoáº·c smoke test cho API + Web Search adapter + model loading.

KhĂ´ng xong náº¿u:
- `metadata.mock = true` váº«n lĂ  Ä‘Æ°á»ng chĂ­nh.
- PriceBandService váº«n chá»‰ dĂ¹ng `class_id_from_budget()` lĂ m káº¿t quáº£ cuá»‘i.
- UI chá»‰ lĂ  HTML tÄ©nh.
- API tráº£ 200 nhÆ°ng dĂ¹ng data giáº£.

## Thá»© tá»± lĂ m trong 45 phĂºt

### 0-5 phĂºt â€” Kiá»ƒm tra ná»n

LĂ m:
- Cháº¡y test hiá»‡n táº¡i:
  ```powershell
  python -m unittest discover tests
  ```
- Kiá»ƒm tra model artifact:
  ```powershell
  Get-ChildItem -Recurse ML_core -Filter *.joblib
  ```
- Kiá»ƒm tra env key:
  ```powershell
  Get-ChildItem Env:GOOGLE_API_KEY, Env:GOOGLE_CSE_ID, Env:SERPAPI_API_KEY
  ```

Verify:
- Biáº¿t test Ä‘ang pass/fail.
- Biáº¿t cĂ³ model chÆ°a.
- Biáº¿t dĂ¹ng provider nĂ o.

### 5-12 phĂºt â€” API Flask tháº­t

Táº¡o/sá»­a:
- `api_server.py`

Route:
- `GET /`
- `GET /health`
- `POST /api/recommend`

YĂªu cáº§u:
- query rá»—ng -> HTTP 400.
- lá»—i provider/model -> HTTP 502 hoáº·c 500 cĂ³ JSON rĂµ.
- success -> `jsonify(response.to_dict())`.

Verify:
```powershell
python api_server.py
curl http://localhost:5000/health
```

### 12-20 phĂºt â€” UI ná»‘i API

Táº¡o/sá»­a:
- `static/index.html`
- `static/app.js`
- `static/style.css`

YĂªu cáº§u ká»¹ thuáº­t:
- form nháº­p query.
- submit gá»i `/api/recommend`.
- loading state.
- error state.
- render results Ä‘Ăºng JSON tháº­t.
- khĂ´ng hard-code result.

UI visual:
- Anh tá»± chá»‰nh sau.
- KhĂ´ng khĂ³a style theo Ă½ agent.

Verify:
- Browser submit Ä‘Æ°á»£c.
- Network tháº¥y `/api/recommend`.
- Console khĂ´ng lá»—i.

### 20-30 phĂºt â€” Web Search tháº­t

Sá»­a:
- `agents/web_search_agent.py`

Giá»¯:
- Input váº«n lĂ  `SearchFeatureVector`.
- Output váº«n lĂ  `list[RecommendationCandidate]`.
- Query váº«n Ä‘i qua `SearchQueryBuilder`.
- Raw result pháº£i normalize qua `source_normalizer.normalize_search_result()`.

Provider chá»n theo env:
- CĂ³ `SERPAPI_API_KEY` -> dĂ¹ng SerpAPI.
- CĂ³ `GOOGLE_API_KEY` + `GOOGLE_CSE_ID` -> dĂ¹ng Google Custom Search.
- KhĂ´ng cĂ³ key -> raise lá»—i rĂµ, khĂ´ng silent mock.

YĂªu cáº§u:
- timeout.
- parse title/link/snippet/source.
- tráº£ nhiá»u candidate, khĂ´ng chá»‰ 1.
- mock chá»‰ Ä‘Æ°á»£c giá»¯ trong test hoáº·c explicit dev fallback, khĂ´ng pháº£i production path.

Verify:
- Query tháº­t tráº£ candidate tháº­t.
- `metadata.mock` khĂ´ng xuáº¥t hiá»‡n trong production response.

### 30-38 phĂºt â€” ML V3 tháº­t

Sá»­a:
- `ML_core/core/price_band_service.py`

YĂªu cáº§u:
- load `.joblib` tá»« config path.
- gá»i `model.predict_proba(features)`.
- class id = `argmax(proba)`.
- confidence = `max(proba)`.
- price bounds láº¥y tá»« `PRICE_CLASS_BOUNDS_VND`.
- náº¿u artifact thiáº¿u -> raise `ModelArtifactError`, khĂ´ng tá»± giáº£ confidence.

Cáº§n kiá»ƒm tra feature input model:
- Náº¿u V3 model cáº§n feature khĂ¡c hiá»‡n táº¡i, thĂªm adapter nhá» trong service.
- KhĂ´ng Ä‘á»ƒ pipeline biáº¿t chi tiáº¿t model.

Verify:
- CĂ³ unit test model load náº¿u artifact cĂ³.
- Náº¿u artifact thiáº¿u, lá»—i rĂµ.
- KhĂ´ng cĂ²n confidence hard-code.

### 38-43 phĂºt â€” End-to-end tháº­t

Cháº¡y:
```powershell
python -m unittest discover tests
python api_server.py
```

Test query:
```text
Khach san Da Nang gan bien 2 nguoi 1-2 trieu co ho boi
Resort Nha Trang for 4 people under 2 million near beach
```

Verify:
- API 200.
- UI render Ä‘Æ°á»£c.
- Search result cĂ³ URL tháº­t.
- Price confidence Ä‘áº¿n tá»« model.
- KhĂ´ng mock.

### 43-45 phĂºt â€” Ghi tráº¡ng thĂ¡i tháº­t

Ghi vĂ o log:
- ÄĂ£ xong pháº§n nĂ o.
- Key/artifact cĂ³ hay thiáº¿u.
- Command test Ä‘Ă£ cháº¡y.
- Lá»—i cĂ²n láº¡i náº¿u cĂ³.

## File cáº§n Ä‘á»¥ng

Báº¯t buá»™c:
- `api_server.py`
- `static/index.html`
- `static/app.js`
- `static/style.css`
- `agents/web_search_agent.py`
- `ML_core/core/price_band_service.py`

CĂ³ thá»ƒ Ä‘á»¥ng náº¿u cáº§n:
- `ML_core/core/config.py`
- `tests/*`
- `.env.example` hoáº·c docs hÆ°á»›ng dáº«n env.

KhĂ´ng Ä‘á»¥ng náº¿u khĂ´ng báº¯t buá»™c:
- `intent/intent_parser.py`
- `ML_core/core/feature_builder.py`
- `recommendation/ranker.py`
- `agents/search_query_builder.py`
- data contract schemas.

## Blocker pháº£i bĂ¡o ngay

BĂ¡o ngay náº¿u:
- KhĂ´ng cĂ³ Web Search API key.
- KhĂ´ng cĂ³ V3 `.joblib`.
- Model artifact khĂ´ng tÆ°Æ¡ng thĂ­ch feature input hiá»‡n táº¡i.
- Flask/flask-cors chÆ°a cĂ i.
- Test ná»n fail trÆ°á»›c khi sá»­a.

Quy táº¯c:
- Thiáº¿u key/artifact khĂ´ng Ä‘Æ°á»£c thay báº±ng fake vĂ  gá»i lĂ  done.
- CĂ³ thá»ƒ lĂ m code path Ä‘áº§y Ä‘á»§, nhÆ°ng tráº¡ng thĂ¡i pháº£i nĂ³i tháº­t: "chá» key/artifact".

---

# Q-Branch Implementation Plan â€” HoĂ n thiá»‡n DAP391m

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Biáº¿n DAP391m tá»« backend pipeline mock â†’ sáº£n pháº©m hoĂ n chá»‰nh cĂ³ Frontend + API tháº­t + Web Search tháº­t + ML Model tháº­t.

**Architecture:** Flask API bá»c `recommend()` pipeline, frontend thuáº§n (HTML+JS+CSS) gá»i API qua fetch, hiá»ƒn thá»‹ káº¿t quáº£ Ä‘Æ°á»£c xáº¿p háº¡ng. Web Search Agent Ä‘Æ°á»£c thay tháº¿ báº±ng Google Custom Search / SerpAPI. Price Band Service Ä‘Æ°á»£c nĂ¢ng cáº¥p tá»« rule-based â†’ dĂ¹ng V3 LGBM model.

**Tech Stack:** Python 3.10+, Flask, HTML/CSS/JS (vanilla, khĂ´ng framework), Google Custom Search API hoáº·c SerpAPI, joblib/scikit-learn.

**Current state (Ä‘Ă£ verified qua code exploration):**
- âœ… Pipeline end-to-end: `recommend()` cháº¡y hoĂ n chá»‰nh IntentParser â†’ PriceBandService â†’ FeatureBuilder â†’ WebSearchAgent â†’ Ranker
- âœ… Data contracts: IntentRequest, PriceBandPrediction, SearchFeatureVector, RecommendationCandidate, RankedRecommendation, RecommendationResponse
- âœ… Tests: test_intent_parser, test_feature_builder, test_recommend_pipeline (3 tests, all passing)
- â ï¸ WebSearchAgent: MOCK â€” luĂ´n tráº£ vá» 1 candidate giáº£ vá»›i `metadata={"mock": True}`
- â ï¸ PriceBandService: rule-based `class_id_from_budget()`, KHĂ”NG dĂ¹ng ML model V3
- âŒ ML model artifact: `ML_core/models/` directory khĂ´ng tá»“n táº¡i, chÆ°a cĂ³ file `.joblib`
- âŒ API endpoint: KhĂ´ng cĂ³ â€” pipeline chá»‰ gá»i Ä‘Æ°á»£c tá»« Python code
- âŒ Frontend: KhĂ´ng cĂ³ â€” khĂ´ng `static/`, khĂ´ng `index.html`, khĂ´ng `api_server.py`

## Global Constraints

- Python 3.10+ vá»›i type hints Ä‘áº§y Ä‘á»§
- Flask cho API (Ä‘Æ¡n giáº£n nháº¥t, khĂ´ng cáº§n FastAPI)
- Frontend vanilla HTML/CSS/JS â€” khĂ´ng React/Vue/Angular
- CORS enabled trĂªn Flask Ä‘á»ƒ frontend gá»i Ä‘Æ°á»£c tá»« browser
- Má»i dataclass Ä‘á»u cĂ³ `.to_dict()` â€” API tráº£ JSON qua `jsonify(response.to_dict())`
- Web search tháº­t pháº£i parse káº¿t quáº£ qua `source_normalizer.normalize_search_result()` â†’ `RecommendationCandidate`
- ML model pháº£i dĂ¹ng `predict_proba()` Ä‘á»ƒ cĂ³ confidence tháº­t, khĂ´ng pháº£i hard-coded 0.0/1.0/0.6
- Giá»¯ nguyĂªn data contract giá»¯a cĂ¡c module â€” khĂ´ng thay Ä‘á»•i schema
- KhĂ´ng thay Ä‘á»•i cĂ¡c module Ä‘Ă£ hoáº¡t Ä‘á»™ng á»•n Ä‘á»‹nh (intent_parser, feature_builder, ranker, search_query_builder)
- Code tiáº¿ng Viá»‡t cho UI text, tiáº¿ng Anh cho code

---

## đŸ” Hiá»‡n tráº¡ng dá»± Ă¡n â€” ÄĂƒ CĂ“ gĂ¬?

```text
                          â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”
                          â”‚       DAP391m (hiá»‡n táº¡i)         â”‚
                          â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜

  NgÆ°á»i dĂ¹ng gĂµ: "KhĂ¡ch sáº¡n ÄĂ  Náºµng 2 ngÆ°á»i, gáº§n biá»ƒn, 1-2 triá»‡u, cĂ³ há»“ bÆ¡i"
       â”‚
       â–¼
  â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”    â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”    â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”    â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”    â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”
  â”‚ â‘  Intent â”‚â”€â”€â”€â–¶â”‚ â‘¡ Price     â”‚â”€â”€â”€â–¶â”‚ â‘¢ Feature    â”‚â”€â”€â”€â–¶â”‚ â‘£ Web Search  â”‚â”€â”€â”€â–¶â”‚ â‘¤ Ranker    â”‚
  â”‚  Parser  â”‚    â”‚  Band Svc   â”‚    â”‚  Builder     â”‚    â”‚  Agent        â”‚    â”‚              â”‚
  â”‚          â”‚    â”‚             â”‚    â”‚              â”‚    â”‚ â ï¸ MOCK!      â”‚    â”‚              â”‚
  â”‚ âœ… xong  â”‚    â”‚ âœ… xong     â”‚    â”‚  âœ… xong     â”‚    â”‚ Tráº£ data giáº£  â”‚    â”‚ âœ… xong      â”‚
  â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜    â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜    â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜    â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜    â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜
       â”‚               â”‚                   â”‚                    â”‚                    â”‚
       â–¼               â–¼                   â–¼                    â–¼                    â–¼
  IntentRequest   PriceBandPrediction  SearchFeatureVector  1 candidate giáº£    Ranked output
  (Ä‘iá»ƒm Ä‘áº¿n,      (phĂ¢n khĂºc giĂ¡,     (vector nhá»‹ phĂ¢n     (metadata:         (cĂ³ score,
   sá»‘ khĂ¡ch,      confidence)          cho tĂ¬m kiáº¿m)        mock=True)         reasons,
   ngĂ¢n sĂ¡ch...)                                                              tradeoffs)
                                                                                  â”‚
       â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜
       â–¼
  â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”
  â”‚ â‘¥ RecommendPipeline â”‚  â†  Káº¿t ná»‘i â‘ â†’â‘¤ end-to-end. Gá»i recommend("query") lĂ  cháº¡y háº¿t.
  â”‚    âœ… xong          â”‚     Äáº§u ra: RecommendationResponse (JSON)
  â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜
```

### TĂ³m gá»n: Backend pipeline cháº¡y Ä‘Æ°á»£c end-to-end, nhÆ°ng Web Search Ä‘ang DĂ™NG Dá»® LIá»†U GIáº¢. ChÆ°a cĂ³ Frontend gĂ¬ cáº£.

---

## đŸ¯ Cáº§n LĂ€M gĂ¬ Ä‘á»ƒ hoĂ n thiá»‡n?

```text
                          â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”
                          â”‚         CĂ”NG VIá»†C Cá»¦A TEAM Q-BRANCH           â”‚
                          â”‚   (anh: UI Frontend + TĂ­ch há»£p API/Model)     â”‚
                          â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜

   PHáº¦N A â€” FRONTEND                           PHáº¦N B â€” BACKEND TĂCH Há»¢P
   â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”                 â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”
   â”‚ XĂ¢y UI tá»« con sá»‘ 0       â”‚                 â”‚ Biáº¿n mock â†’ tháº­t             â”‚
   â”‚                           â”‚                 â”‚                              â”‚
   â”‚ â–  Trang chá»§:              â”‚                 â”‚ â–  B.1 Web Search tháº­t        â”‚
   â”‚   - Ă” input cho user gĂµ   â”‚                 â”‚   - Gá»i API Google/Bing/etc  â”‚
   â”‚     truy váº¥n              â”‚                 â”‚   - Hoáº·c scrape web search   â”‚
   â”‚   - NĂºt "TĂ¬m"             â”‚                 â”‚   - Parse káº¿t quáº£ tháº­t       â”‚
   â”‚                           â”‚                 â”‚   - Äiá»n vĂ o Candidate       â”‚
   â”‚ â–  Trang káº¿t quáº£:          â”‚                 â”‚                              â”‚
   â”‚   - Hiá»ƒn thá»‹ danh sĂ¡ch    â”‚                 â”‚ â–  B.2 DĂ¹ng ML Model tháº­t     â”‚
   â”‚     khĂ¡ch sáº¡n Ä‘Æ°á»£c gá»£i Ă½  â”‚                 â”‚   - Load artifact V3         â”‚
   â”‚   - Má»—i item: tĂªn, giĂ¡,   â”‚                 â”‚   - DĂ¹ng predict_proba()    â”‚
   â”‚     Ä‘iá»ƒm, lĂ½ do phĂ¹ há»£p,  â”‚                 â”‚     thay vĂ¬ rule-based       â”‚
   â”‚     Ä‘Ă¡nh Ä‘á»•i              â”‚                 â”‚   - Tráº£ confidence tháº­t      â”‚
   â”‚   - Responsive, sáº¡ch Ä‘áº¹p  â”‚                 â”‚                              â”‚
   â”‚                           â”‚                 â”‚ â–  B.3 UI gá»i Backend         â”‚
   â”‚ â–  TĂ­ch há»£p:               â”‚                 â”‚   - Táº¡o API endpoint         â”‚
   â”‚   - UI â†’ gá»i API backend  â”‚                 â”‚   - Frontend fetch JSON      â”‚
   â”‚   - Nháº­n JSON response    â”‚                 â”‚   - Render káº¿t quáº£           â”‚
   â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜                 â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜
```

---

## đŸ§­ Lá»™ trĂ¬nh tá»«ng bÆ°á»›c

```text
  BÆ°á»›c 1           BÆ°á»›c 2            BÆ°á»›c 3             BÆ°á»›c 4            BÆ°á»›c 5
  â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”      â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”        â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”       â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”       â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”
  â”‚ Táº¡o APIâ”‚â”€â”€â”€â”€â”€â–¶â”‚ Build  â”‚â”€â”€â”€â”€â”€â”€â”€â–¶â”‚ TĂ­ch há»£p â”‚â”€â”€â”€â”€â”€â”€â–¶â”‚ Web     â”‚â”€â”€â”€â”€â”€â”€â–¶â”‚ HoĂ n     â”‚
  â”‚ endpointâ”‚      â”‚ UI cÆ¡  â”‚        â”‚ API â†” UI â”‚       â”‚ Search  â”‚       â”‚ thiá»‡n    â”‚
  â”‚ Flask / â”‚      â”‚ báº£n    â”‚        â”‚          â”‚       â”‚ THáº¬T    â”‚       â”‚ & polish â”‚
  â”‚ FastAPI â”‚      â”‚        â”‚        â”‚          â”‚       â”‚         â”‚       â”‚          â”‚
  â””â”€â”€â”€â”€â”€â”€â”€â”€â”˜      â””â”€â”€â”€â”€â”€â”€â”€â”€â”˜        â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜       â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜       â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜
   1-2 ngĂ y        2-4 ngĂ y          1-2 ngĂ y           2-4 ngĂ y          1-2 ngĂ y
```

### Chi tiáº¿t tá»«ng bÆ°á»›c:

| # | LĂ m gĂ¬ | File liĂªn quan | Output |
|---|--------|---------------|--------|
| **B1. API endpoint** | Bá»c `recommend()` trong 1 API (Flask/FastAPI). Nháº­n `POST /api/recommend` body `{"query": "..."}` â†’ tráº£ JSON `RecommendationResponse` | Táº¡o má»›i: `api_server.py` | API cháº¡y localhost:5000 |
| **B2. UI Frontend** | XĂ¢y giao diá»‡n web: input box + nĂºt tĂ¬m + danh sĂ¡ch káº¿t quáº£. Gá»i fetch Ä‘áº¿n API B1. | Táº¡o má»›i: `static/index.html`, `static/app.js`, `static/style.css` | Giao diá»‡n hoĂ n chá»‰nh |
| **B3. TĂ­ch há»£p** | Ná»‘i UI â†” API. Kiá»ƒm tra flow: gĂµ query â†’ fetch â†’ render káº¿t quáº£ Ä‘áº¹p. | Cáº£ 2 file trĂªn | Demo cháº¡y Ä‘Æ°á»£c |
| **B4. Web Search tháº­t** | Thay `WebSearchAgent.search()` Ä‘ang mock báº±ng code gá»i API tĂ¬m kiáº¿m tháº­t (Google Custom Search, SerpAPI, hoáº·c scrape). Chuáº©n hĂ³a káº¿t quáº£ qua `source_normalizer.py` | Sá»­a: `agents/web_search_agent.py` | Káº¿t quáº£ tháº­t, khĂ´ng mock |
| **B5. ML Model tháº­t** | Sá»­a `PriceBandService.predict_price_band()` Ä‘á»ƒ load artifact V3 vĂ  dĂ¹ng `model.predict_proba()` thay vĂ¬ Ă¡nh xáº¡ rule-based | Sá»­a: `ML_core/core/price_band_service.py` | Confidence tháº­t tá»« model |
| **B6. Polish** | Responsive CSS, loading spinner, error handling, empty state | `static/*` | Sáº£n pháº©m hoĂ n thiá»‡n |

---

## đŸ“‚ Cáº¥u trĂºc dá»± Ă¡n SAU KHI hoĂ n thiá»‡n

```text
DAP391m/
  ML_core/              â† CĂ³ sáºµn (backend ML)
  intent/               â† CĂ³ sáºµn (parser)
  agents/               â† Sá»¬A: web_search_agent.py (mock â†’ tháº­t)
  recommendation/       â† CĂ³ sáºµn (ranker)
  pipelines/            â† CĂ³ sáºµn (pipeline)
  tests/                â† CĂ³ sáºµn
  docs/                 â† CĂ³ sáºµn

  api_server.py         â† Táº O Má»I: Flask/FastAPI endpoint
  static/               â† Táº O Má»I: toĂ n bá»™ frontend
    index.html          â†   Giao diá»‡n chĂ­nh
    app.js              â†   Logic gá»i API + render
    style.css           â†   CSS responsive

  index.html            â† Placeholder hiá»‡n táº¡i, thay báº±ng UI tháº­t
  Q-Branch_dev.md       â† File nĂ y
  CLAUDE.md             â† TĂ i liá»‡u dá»± Ă¡n (tiáº¿ng Viá»‡t)
```

---

## â¡ Báº¯t Ä‘áº§u tá»« Ä‘Ă¢u?

```text
  Æ¯U TIĂN Sá» 1: API endpoint (BÆ°á»›c 1)
  â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”
  â”‚  Táº¡o file api_server.py                          â”‚
  â”‚  DĂ¹ng Flask (Ä‘Æ¡n giáº£n nháº¥t)                      â”‚
  â”‚  1 route: POST /api/recommend                    â”‚
  â”‚  Import recommend() tá»« pipelines                 â”‚
  â”‚  Tráº£ JSON ra lĂ  cĂ³ API Ä‘á»ƒ UI gá»i ngay            â”‚
  â”‚                                                  â”‚
  â”‚  Code máº«u:                                        â”‚
  â”‚  â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â” â”‚
  â”‚  â”‚ from flask import Flask, request, jsonify   â”‚ â”‚
  â”‚  â”‚ from pipelines.recommend_pipeline import    â”‚ â”‚
  â”‚  â”‚      recommend                              â”‚ â”‚
  â”‚  â”‚ app = Flask(__name__)                       â”‚ â”‚
  â”‚  â”‚ @app.route('/api/recommend', methods=['POST']â”‚ â”‚
  â”‚  â”‚ def api_recommend():                        â”‚ â”‚
  â”‚  â”‚   q = request.json.get('query', '')         â”‚ â”‚
  â”‚  â”‚   resp = recommend(q)                       â”‚ â”‚
  â”‚  â”‚   return jsonify(resp.to_dict())            â”‚ â”‚
  â”‚  â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜ â”‚
  â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜
```
