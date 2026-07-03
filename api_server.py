import os
from pathlib import Path
from urllib.parse import urlparse

try:
    from flask import Flask, jsonify, make_response, request, send_from_directory

    HAS_FLASK = True
except ModuleNotFoundError:
    Flask = None
    jsonify = None
    make_response = None
    request = None
    send_from_directory = None
    HAS_FLASK = False


PROJECT_ROOT = Path(__file__).resolve().parent
STATIC_DIR = PROJECT_ROOT / "static"
ENV_FILE = PROJECT_ROOT / ".env"
DEFAULT_ALLOWED_ORIGINS = {
    "http://127.0.0.1:5173",
    "http://localhost:5173",
    "http://127.0.0.1:4173",
    "http://localhost:4173",
    "http://127.0.0.1:5000",
    "http://localhost:5000",
}


def load_env_file(env_file: Path = ENV_FILE) -> None:
    if not env_file.exists():
        return

    for raw_line in env_file.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key:
            os.environ[key] = value


load_env_file()

from agents.web_search_agent import WebSearchConfigurationError, WebSearchProviderError
from ML_core.core.exceptions import MLCoreError, ModelArtifactError
from pipelines.recommend_pipeline import RecommendPipeline


def _configured_allowed_origins() -> set[str]:
    raw = os.environ.get("ALLOWED_ORIGINS", "")
    configured = {item.strip() for item in raw.split(",") if item.strip()}
    return DEFAULT_ALLOWED_ORIGINS | configured


def _cors_origin_for_request() -> str | None:
    origin = request.headers.get("Origin")
    if not origin:
        return None
    if origin in _configured_allowed_origins():
        return origin
    parsed = urlparse(origin)
    if parsed.hostname in {"127.0.0.1", "localhost"}:
        return origin
    return None


def create_app(pipeline: RecommendPipeline | None = None):
    if not HAS_FLASK:
        raise RuntimeError("Flask is required to run the API server. Install dependencies from requirements.txt first.")

    app = Flask(__name__, static_folder=str(STATIC_DIR), static_url_path="/static")
    app.config["PIPELINE"] = pipeline or RecommendPipeline()

    @app.after_request
    def add_cors_headers(response):
        allowed_origin = _cors_origin_for_request()
        if allowed_origin:
            response.headers["Access-Control-Allow-Origin"] = allowed_origin
            response.headers["Vary"] = "Origin"
            response.headers["Access-Control-Allow-Headers"] = "Content-Type"
            response.headers["Access-Control-Allow-Methods"] = "GET,POST,OPTIONS"
        return response

    @app.get("/")
    def index():
        return send_from_directory(app.static_folder, "index.html")

    @app.get("/health")
    def health():
        active_pipeline: RecommendPipeline = app.config["PIPELINE"]
        issues = active_pipeline.price_band_service.get_readiness_issues()
        issues.extend(active_pipeline.web_search_agent.get_readiness_issues())
        return jsonify(
            {
                "status": "ok" if not issues else "degraded",
                "ready": not issues,
                "issues": issues,
                "search_provider": active_pipeline.web_search_agent.describe_provider(),
                "price_classifier_path": str(active_pipeline.price_band_service.model_path),
            }
        )

    @app.route("/api/recommend", methods=["POST", "OPTIONS"])
    def recommend_route():
        if request.method == "OPTIONS":
            return make_response("", 204)

        payload = request.get_json(silent=True) or {}
        query = str(payload.get("query") or "").strip()
        if not query:
            return jsonify({"error": "`query` is required."}), 400

        active_pipeline: RecommendPipeline = app.config["PIPELINE"]
        try:
            response = active_pipeline.recommend(query)
        except WebSearchConfigurationError as exc:
            return jsonify({"error": str(exc)}), 503
        except WebSearchProviderError as exc:
            return jsonify({"error": str(exc)}), 502
        except ModelArtifactError as exc:
            return jsonify({"error": str(exc)}), 503
        except MLCoreError as exc:
            return jsonify({"error": str(exc)}), 500
        except Exception as exc:
            return jsonify({"error": f"Unexpected server error: {exc}"}), 500

        return jsonify(response.to_dict())

    return app


def main():
    app = create_app()
    port = int(os.environ.get("PORT", "5000"))
    app.run(host="0.0.0.0", port=port, debug=True)


if __name__ == "__main__":
    main()
