from __future__ import annotations

import json
from typing import Any, Dict, Optional


TRAINER_EVENT_PREFIX = "TRAINER_EVENT "


def encode_trainer_event(event: str, **payload: Any) -> str:
    body: Dict[str, Any] = {"event": str(event)}
    body.update(payload)
    return TRAINER_EVENT_PREFIX + json.dumps(body, ensure_ascii=True)


def maybe_parse_trainer_event_line(line: str) -> Optional[Dict[str, Any]]:
    text = str(line).strip()
    if not text.startswith(TRAINER_EVENT_PREFIX):
        return None
    payload = text[len(TRAINER_EVENT_PREFIX) :].strip()
    if not payload:
        return None
    parsed = json.loads(payload)
    if not isinstance(parsed, dict):
        raise ValueError("Trainer event payload must be a JSON object.")
    return parsed


def encode_trainer_command(command: str, **payload: Any) -> str:
    body: Dict[str, Any] = {"command": str(command)}
    body.update(payload)
    return json.dumps(body, ensure_ascii=True)


def parse_trainer_command_line(line: str) -> Dict[str, Any]:
    text = str(line).strip()
    if not text:
        raise ValueError("Trainer command line is empty.")
    parsed = json.loads(text)
    if not isinstance(parsed, dict):
        raise ValueError("Trainer command must be a JSON object.")
    command = str(parsed.get("command", "")).strip()
    if not command:
        raise ValueError("Trainer command is missing a non-empty 'command' field.")
    parsed["command"] = command
    return parsed
