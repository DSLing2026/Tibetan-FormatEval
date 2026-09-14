#!/usr/bin/env python3
"""Run MCQA or open-answer evaluation through an OpenAI-compatible chat endpoint.

This uses only the Python standard library. Example:

python3 scripts/run_openai_compatible_eval.py \
  --gold eval_sets/tibetan_formateval_main/mcqa_4way.jsonl \
  --task mcqa \
  --model qwen \
  --base-url http://localhost:8000/v1 \
  --api-key EMPTY \
  --output predictions/qwen/mcqa_4way.jsonl
"""

from __future__ import annotations

import argparse
import concurrent.futures as futures
import json
import os
import re
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def format_mcqa_prompt(row: dict[str, Any]) -> str:
    labels = [chr(ord("A") + idx) for idx in range(len(row["choices"]))]
    options = "\n".join(f"{label}. {choice}" for label, choice in zip(labels, row["choices"]))
    return (
        "Answer the following Tibetan multiple-choice question. "
        "Return only the option letter, with no explanation.\n\n"
        f"Question:\n{row['question']}\n\n"
        f"Options:\n{options}\n\n"
        "Answer:"
    )


def format_open_prompt(row: dict[str, Any]) -> str:
    return (
        "Answer the following Tibetan question with a short answer only. "
        "If you do not know the answer, return \"I don't know\".\n\n"
        f"Question:\n{row['question']}\n\n"
        "Short answer:"
    )


def chat_completion(
    base_url: str,
    api_key: str,
    api_model: str,
    display_model: str,
    prompt: str,
    temperature: float,
    max_tokens: int,
    extra_body: dict[str, Any],
    timeout: int,
    max_retries: int,
) -> tuple[dict[str, Any], str]:
    url = base_url.rstrip("/") + "/chat/completions"
    payload = {
        "model": api_model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    payload.update(extra_body)
    data = json.dumps(payload).encode("utf-8")
    last_err = ""
    for attempt in range(1, max_retries + 1):
        req = urllib.request.Request(
            url,
            data=data,
            headers={
                "Accept": "application/json",
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
                "User-Agent": "curl/8.5.0",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout) as response:
                body = json.loads(response.read().decode("utf-8"))
            choice = (body.get("choices") or [{}])[0]
            message = choice.get("message") or {}
            text = (message.get("content") or choice.get("text") or "").strip()
            usage = body.get("usage") or {}
            completion_details = usage.get("completion_tokens_details") or {}
            raw = {
                "id": body.get("id", ""),
                "usage": usage,
                "api_model": api_model,
                "display_model": display_model,
                "finish_reason": choice.get("finish_reason", ""),
                "request": {"max_tokens": max_tokens, "temperature": temperature, **extra_body},
            }
            if not text:
                last_err = (
                    "empty prediction "
                    f"(finish_reason={choice.get('finish_reason', '')}, "
                    f"completion_tokens={usage.get('completion_tokens')}, "
                    f"reasoning_tokens={completion_details.get('reasoning_tokens')})"
                )
                time.sleep(min(3 * attempt, 30))
                continue
            if "上游模型未返回任何内容" in text or "upstream model" in text.lower():
                last_err = f"provider returned no-content placeholder: {text[:120]}"
                time.sleep(min(3 * attempt, 30))
                continue
            return raw, text
        except urllib.error.HTTPError as exc:
            err_body = exc.read().decode("utf-8", errors="replace")
            last_err = f"HTTP {exc.code}: {err_body}"
            if exc.code in {408, 409, 429, 500, 502, 503, 504, 520, 522, 524}:
                time.sleep(min(5 * attempt, 60))
            else:
                raise RuntimeError(last_err) from exc
        except Exception as exc:
            last_err = str(exc)
            time.sleep(min(3 * attempt, 30))
    raise RuntimeError(f"OpenAI-compatible chat call failed after retries: {last_err}")


def load_shell_config(path: Path | None) -> dict[str, str]:
    if path is None:
        return {}
    text = path.read_text(encoding="utf-8")
    key_match = re.search(r"(?:API_KEY|api_key|apikey)\s*[:=]\s*['\"]?([^'\"\n]+)", text)
    base_match = re.search(r"base_url\s*[:=]\s*['\"]?([^'\"\n]+)", text)
    openai_path_match = re.search(r"openai\s*:\s*([^\s#]+)", text)
    result = {}
    if key_match:
        result["api_key"] = key_match.group(1).strip()
    if base_match:
        base_url = base_match.group(1).strip().rstrip("/")
        if openai_path_match:
            openai_path = openai_path_match.group(1).strip()
            if openai_path.endswith("/chat/completions"):
                api_prefix = openai_path[: -len("/chat/completions")]
                base_url = base_url + api_prefix
        result["base_url"] = base_url
    return result


def load_done(path: Path) -> set[str]:
    if not path.exists():
        return set()
    done = set()
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                if not row.get("error") and str(row.get("prediction", "")).strip():
                    done.add(row["id"])
    return done


def run_one(
    row: dict[str, Any],
    task: str,
    base_url: str,
    api_key: str,
    api_model: str,
    display_model: str,
    temperature: float,
    max_tokens: int,
    extra_body: dict[str, Any],
    timeout: int,
    max_retries: int,
) -> dict[str, Any]:
    prompt = format_mcqa_prompt(row) if task == "mcqa" else format_open_prompt(row)
    try:
        raw, prediction = chat_completion(
            base_url,
            api_key,
            api_model,
            display_model,
            prompt,
            temperature,
            max_tokens,
            extra_body,
            timeout,
            max_retries,
        )
        return {
            "id": row["id"],
            "prediction": prediction,
            "model": display_model,
            "raw": raw,
            "error": "",
        }
    except Exception as exc:
        return {
            "id": row["id"],
            "prediction": "",
            "model": display_model,
            "raw": {"api_model": api_model, "display_model": display_model},
            "error": str(exc),
        }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gold", type=Path, required=True)
    parser.add_argument("--task", choices=["mcqa", "open"], required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--display-model", default="")
    parser.add_argument("--shell-file", type=Path)
    parser.add_argument("--base-url", default="")
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=16)
    parser.add_argument(
        "--extra-body-json",
        default="{}",
        help="Extra JSON object merged into the chat/completions request body.",
    )
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--max-retries", type=int, default=6)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--sleep", type=float, default=0.0)
    args = parser.parse_args()

    shell_config = load_shell_config(args.shell_file)
    base_url = args.base_url or shell_config.get("base_url", "")
    api_key = args.api_key if args.api_key != "EMPTY" else shell_config.get("api_key", os.environ.get("API_KEY", "EMPTY"))
    display_model = args.display_model or args.model
    if not base_url:
        raise ValueError("Missing --base-url or base_url in --shell-file")
    extra_body = json.loads(args.extra_body_json)
    if not isinstance(extra_body, dict):
        raise ValueError("--extra-body-json must decode to a JSON object")

    rows = read_jsonl(args.gold)[args.offset :]
    if args.limit:
        rows = rows[: args.limit]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    done = load_done(args.output)
    pending = [row for row in rows if row["id"] not in done]
    lock = threading.Lock()
    completed = 0
    with args.output.open("a", encoding="utf-8") as f:
        with futures.ThreadPoolExecutor(max_workers=args.concurrency) as executor:
            future_to_row = {
                executor.submit(
                    run_one,
                    row,
                    args.task,
                    base_url,
                    api_key,
                    args.model,
                    display_model,
                    args.temperature,
                    args.max_tokens,
                    extra_body,
                    args.timeout,
                    args.max_retries,
                ): row
                for row in pending
            }
            for future in futures.as_completed(future_to_row):
                out = future.result()
                with lock:
                    f.write(json.dumps(out, ensure_ascii=False) + "\n")
                    f.flush()
                    completed += 1
                    print(
                        f"[{completed}/{len(pending)}] {out['id']} "
                        f"error={bool(out.get('error'))}",
                        flush=True,
                    )
                if args.sleep:
                    time.sleep(args.sleep)


if __name__ == "__main__":
    main()
