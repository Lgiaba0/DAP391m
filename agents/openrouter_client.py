import json
import os
import urllib.request
import urllib.error
from typing import Any

def call_openrouter(model: str, messages: list[dict[str, str]], json_mode: bool = False) -> str:
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY environment variable is not set.")
    
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "http://localhost:5000",
        "X-Title": "DAP391m Travel Concierge"
    }
    
    data: dict[str, Any] = {
        "model": model,
        "messages": messages,
    }
    if json_mode:
        data["response_format"] = {"type": "json_object"}
        
    req = urllib.request.Request(url, data=json.dumps(data).encode("utf-8"), headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=30) as response:
            res = json.loads(response.read().decode("utf-8"))
            choices = res.get("choices") or []
            if not choices:
                raise ValueError(f"OpenRouter response had no choices: {res}")
            return str(choices[0]["message"]["content"])
    except urllib.error.HTTPError as e:
        error_body = e.read().decode("utf-8")
        raise RuntimeError(f"OpenRouter HTTP Error {e.code}: {error_body}") from e
    except Exception as e:
        raise RuntimeError(f"OpenRouter call failed: {e}") from e

def extract_json_from_text(text: str) -> str:
    text_stripped = text.strip()
    
    first_brace = text_stripped.find('{')
    last_brace = text_stripped.rfind('}')
    
    first_bracket = text_stripped.find('[')
    last_bracket = text_stripped.rfind(']')
    
    has_brace = (first_brace != -1 and last_brace != -1 and last_brace > first_brace)
    has_bracket = (first_bracket != -1 and last_bracket != -1 and last_bracket > first_bracket)
    
    if has_brace and has_bracket:
        if first_brace < first_bracket:
            return text_stripped[first_brace:last_brace+1]
        else:
            return text_stripped[first_bracket:last_bracket+1]
    elif has_brace:
        return text_stripped[first_brace:last_brace+1]
    elif has_bracket:
        return text_stripped[first_bracket:last_bracket+1]
        
    return text_stripped
