# CLAUDE.md

File này cung cấp hướng dẫn cho Claude Code (claude.ai/code) khi làm việc với mã nguồn trong repository này.

## Nhận diện Dự án

**DAP391m** — Công cụ gợi ý chỗ ở tại Việt Nam. Nhận truy vấn văn bản tự do từ người dùng
(tiếng Việt hoặc tiếng Anh), phân tích ý định, phân loại khoảng giá, tìm kiếm ứng viên,
và trả về các gợi ý đã được xếp hạng kèm giải thích.

Remote: `https://github.com/Lgiaba0/DAP391m.git`

## Tổng quan Kiến trúc

```text
truy vấn thô từ người dùng
  → intent/          (IntentParser: từ khóa + regex → IntentRequest)
  → ML_core/         (PriceBandService: ngân sách → phân khúc giá; FeatureBuilder: vector)
  → agents/          (WebSearchAgent: vector → truy vấn tìm kiếm + ứng viên)
  → recommendation/  (RecommendationRanker: chấm điểm có trọng số → kết quả đã xếp hạng)
  → pipelines/       (RecommendPipeline: kết nối mọi thứ end-to-end)
```

Dự án thực thi nguyên tắc **tách biệt ML-core / application-layer**: `ML_core/` sở hữu các model,
schema, feature engineering, và dịch vụ suy luận. Mọi thứ bên ngoài nó là
điều phối ứng dụng. Agent và pipeline chỉ tiêu thụ data contract từ ML_core,
không bao giờ đụng đến các artifact huấn luyện nội bộ.

### Data Contract (tất cả trong `ML_core/core/schemas.py`)

Ba dataclass tạo thành xương sống, mỗi cái đều có phương thức `to_dict()`:

| Contract | Mục đích | Được tạo bởi |
|---|---|---|
| `IntentRequest` | Ý định người dùng có cấu trúc (điểm đến, ngân sách, số khách, tiện nghi, vị trí ưa thích, loại hình chỗ ở) | `IntentParser.parse()` |
| `PriceBandPrediction` | Phân khúc giá 5 lớp + độ tin cậy + khoảng VND | `PriceBandService.predict_price_band()` |
| `SearchFeatureVector` | Tín hiệu tìm kiếm mã hóa nhị phân cho web agent | `FeatureBuilder.build()` |

Contract tầng recommendation nằm trong `recommendation/schemas.py`:
`RecommendationCandidate`, `RankedRecommendation`, `RecommendationResponse`.

### Sơ đồ Module

```
ML_core/
  core/
    schemas.py            — IntentRequest, PriceBandPrediction, SearchFeatureVector
    config.py             — ML_CORE_ROOT, PRICE_CLASS_LABELS, PRICE_CLASS_BOUNDS_VND,
                            DEFAULT_PRICE_CLASSIFIER_PATH, LOW_CONFIDENCE_THRESHOLD
    exceptions.py         — MLCoreError, ModelArtifactError, InvalidIntentError
    price_band_service.py — PriceBandService + class_id_from_budget()
    feature_builder.py    — FeatureBuilder: hợp nhất intent + phân khúc giá → vector
  scripts/
    classify/v1..v3/      — huấn luyện phân loại giá (V3 = LGBM phân cấp 2 tầng)
    reg/v1..v5/           — huấn luyện hồi quy giá VND
    validate_ml_core.py   — kiểm tra cú pháp AST toàn bộ core/, agents/, pipelines/, scripts/

intent/
  intent_parser.py        — IntentParser: lowercase → khớp từ khóa + trích xuất regex
                            (điểm đến, số khách, khoảng ngân sách, tiện nghi, vị trí, loại hình chỗ ở)

agents/
  search_query_builder.py — SearchQueryBuilder: vector → các cụm từ khóa tìm kiếm
  web_search_agent.py     — WebSearchAgent: vector → ứng viên mock (stub, CHƯA phải tìm kiếm thật)
  source_normalizer.py    — normalize_search_result(): dict thô → RecommendationCandidate

recommendation/
  schemas.py              — RecommendationCandidate, RankedRecommendation, RecommendationResponse
  ranker.py               — RecommendationRanker: chấm điểm có trọng số (giá 0.35, tiện nghi 0.25,
                            vị trí 0.20, sức chứa 0.10, nguồn 0.10)

pipelines/
  recommend_pipeline.py   — RecommendPipeline + hàm tiện ích recommend()

scripts/
  run_recommend_demo.py   — Demo CLI: python scripts/run_recommend_demo.py ["truy vấn"]

samples/
  intent_examples.json    — truy vấn mẫu với kết quả mong đợi
  mock_search_results.json — ứng viên mock mẫu

tests/
  test_intent_parser.py   — kiểm thử phân tích ý định tiếng Việt + tiếng Anh
  test_feature_builder.py — kiểm thử xây dựng vector từ intent + phân khúc giá
  test_recommend_pipeline.py — kiểm thử khói tích hợp pipeline đầy đủ

docs/
  phase_next_intent_to_recommendation_pipeline.md — kế hoạch triển khai đầy đủ, data contract, tiêu chí chấp nhận
```

## Chi tiết Thiết kế Quan trọng

### Phân loại Giá (5 phân khúc)

```
0: budget           < 500.000 VND
1: economy          500.000 – 1.000.000 VND
2: mid_range        1.000.000 – 2.000.000 VND
3: upscale          2.000.000 – 5.000.000 VND
4: premium_luxury   ≥ 5.000.000 VND
```

### Ánh xạ Ngân sách → Phân khúc (`class_id_from_budget`)
- Trung điểm của khoảng min/max → tra cứu ranh giới phân khúc.
- Không có ngân sách cụ thể → mặc định về phân khúc 2 (mid_range).
- `PriceBandService.predict_price_band()` hiện tại dùng ánh xạ dựa trên ngân sách, KHÔNG dùng
  artifact phân loại V3. Artifact tồn tại tại
  `models/classify/v3/price_classification_v3_model.joblib` và service có thể
  tải nó, nhưng suy luận hiện tại suy ra phân khúc trực tiếp từ con số ngân sách.

### Heuristic của Feature Builder
- `expand_budget = True` khi độ tin cậy về giá < 0.55 (`LOW_CONFIDENCE_THRESHOLD`).
- Ngân sách tường minh từ người dùng được giữ lại làm hướng dẫn tìm kiếm cứng.
- Tín hiệu boolean về tiện nghi/vị trí/loại hình chỗ ở được mã hóa thành số nguyên 0/1.

### Trạng thái Web Search Agent
**HIỆN TẠI LÀ MOCK.** `WebSearchAgent.search()` trả về chính xác một
`RecommendationCandidate` giả với `metadata={"mock": True}` bất kể đầu vào.
Đây là chủ ý theo kế hoạch triển khai — thay thế bằng tích hợp
web-search/provider thật ở bước cuối cùng.

### Thuật toán Xếp hạng
Tổ hợp tuyến tính có trọng số trên 5 điểm thành phần, mỗi cái [0.0, 1.0]:
- `_price_fit`: 0.3 nếu không có giá; 0.2 nếu ngoài ngân sách cứng (hoặc 0.6 nếu expand_budget);
  1.0 nếu trong khoảng.
- `_amenity_fit`: tỉ lệ tiện nghi mong muốn có mặt.
- `_location_fit`: tỉ lệ tag vị trí mong muốn có mặt.
- `_capacity_fit`: 1.0 nếu sức chứa ứng viên ≥ số khách yêu cầu, ngược lại 0.0.
- `source_quality`: min(max(source_quality, 0), 1) từ ứng viên.
Ghi chú penalty vào `tradeoffs` khi điểm thành phần < 0.5; ghi công vào `reasons` khi ≥ 0.95.

### Hạn chế của Intent Parser (Có chủ ý)
- Chỉ dựa trên luật: khớp chuỗi con từ khóa + regex.
- Không có tokenization, an toàn ranh giới từ, xử lý phủ định, hay chấm điểm độ tin cậy.
- Khớp điểm đến đầu tiên sẽ thắng; không phân giải nhập nhằng.
- Hỗ trợ từ khóa tiếng Việt + tiếng Anh; regex ngân sách xử lý được đơn vị VND
  (triệu/million, k/nghìn) và các mẫu khoảng/dưới/khoảng.
- Được thiết kế như một bộ chuẩn hóa tất định nhẹ trước ML core; phân tích LLM
  có thể được thêm sau phía sau cùng interface.

## Các Lệnh Thường Dùng

```powershell
# Chạy pipeline demo (dùng truy vấn mặc định nếu không có tham số)
python scripts/run_recommend_demo.py
python scripts/run_recommend_demo.py "Khach san Nha Trang duoi 1 trieu cho 2 nguoi"

# Chạy tất cả kiểm thử
python -m unittest discover tests

# Chạy một file kiểm thử đơn lẻ
python -m unittest tests.test_intent_parser
python -m unittest tests.test_feature_builder
python -m unittest tests.test_recommend_pipeline

# Kiểm tra cú pháp ML core (AST parse toàn bộ file Python trong core/, agents/, pipelines/, scripts/)
python ML_core/scripts/validate_ml_core.py

# Huấn luyện model phân loại giá (V3 phân cấp)
python ML_core/scripts/classify/v3/train_price_classification_v3.py
```

## Trạng thái Triển khai

Theo `docs/phase_next_intent_to_recommendation_pipeline.md`:

- [x] Giai đoạn 1: ML core schemas + interface price band service
- [x] Giai đoạn 2: Intent parser tất định
- [x] Giai đoạn 3: Feature vector builder
- [x] Giai đoạn 4: Web-search agent (stub với mock)
- [x] Giai đoạn 5: Xếp hạng gợi ý
- [x] Giai đoạn 6: Pipeline end-to-end (`RecommendPipeline.recommend()`)
- [ ] Thay thế mock web search bằng provider/tooling thật (bước cuối)

## Quy tắc Phát triển từ Tài liệu Dự án

Từ kế hoạch kiến trúc:
- **Cô lập ML core**: Agent và pipeline chỉ import từ `ML_core.core.*`,
  không bao giờ từ `ML_core.scripts.*`.
- **Kỷ luật data contract**: Mọi giao tiếp liên module dùng dataclass có kiểu,
  không dùng dict lỏng lẻo.
- **Feature vector là bàn giao**: Web-search agent nhận `SearchFeatureVector`,
  không nhận văn bản thô hay nội bộ model.
- **Chấm điểm minh bạch**: Xếp hạng phải tạo ra giải thích gắn với ý định + bằng chứng.
- **Feature vector tất định**: Đầu ra `FeatureBuilder.build()` phải
  tuần tự hóa được sang JSON và tái tạo được với cùng đầu vào.

## Thói quen Git

- Nhánh: `main`
- Phong cách commit: tin nhắn pha trộn tiếng Việt / tiếng Anh
- Dữ liệu sinh ra bị loại trừ qua `.gitignore`:
  - `ML_core/data/`, `ML_core/reports/`, `ML_core/models/**/*.joblib`
  - `reports/recommendation/*` (ngoại trừ `.gitkeep`)