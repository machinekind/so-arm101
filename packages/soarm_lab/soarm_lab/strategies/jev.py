"""Jev, TypeSafe's decision model, as a judge: one POST to its System One endpoint per call.

    POST https://api.typesafe.ai/v1/systemone
    Authorization: Bearer $TYPESAFE_API_KEY
    {"model": "jev-1.14.0", "state": {...}, "questions": {"touched": {"type": "choice", ...}, ...}}

The reply holds the model that answered, an answer per question (for a choice: `choice`, `confidence` and
`probabilities` per option) and the token usage. The client is the standard library's `urllib`, on the
documented JSON contract (OpenAPI 0.2.0), not the `typesafe-sdk` package: the SDK would add httpx2,
pydantic, tenacity, truststore and anyio, retry for up to 30 s inside our own timeout, and switch to zstd
and multipart on its own. The strategy retries itself, once, after an overload or a dropped connection.

The key comes from `TYPESAFE_API_KEY` in the environment, read by `check`, never in `__init__`, so the
strategy builds and its settings load without one. It goes only into the Authorization header: no error,
note, record or log line holds it. The URL is a constant, so no config file can send the key elsewhere.

Images (Jev 1.14, which TypeSafe does not yet guarantee for production) go out only when the strategy is
given frames: each one as a PNG asset, base64 in the JSON body, named by a reference the state contains,
with the `TypeSafe-Danger-Zone-Images: true` header. Jev takes a frame only when both of its sides lie
within 280 to 1568 px (`frame_problem`); it never resizes one.
"""

from __future__ import annotations

import base64
import http.client
import json
import os
import struct
import threading
import urllib.error
import urllib.request
import zlib
from collections.abc import Callable, Mapping
from typing import Any

import numpy as np

from .judges import JudgeError, Questions, Reply, checked_reply

SYSTEM_ONE_URL = "https://api.typesafe.ai/v1/systemone"
#: The environment variable that holds the API key.
KEY_ENV = "TYPESAFE_API_KEY"
#: The model the thresholds were chosen for; an alias such as "jev-latest" moves with new releases.
DEFAULT_MODEL = "jev-1.14.0"
#: The header a request with images needs, with the value "true".
IMAGES_HEADER = "TypeSafe-Danger-Zone-Images"
#: The shortest and the longest side of an image Jev takes [px].
IMAGE_SIDE_PX = (280, 1568)
#: How much of an error body a reason quotes.
QUOTE_LIMIT = 200

#: Sends one request: (url, headers, body, timeout_s) -> (HTTP status, response body). Raises OSError (or
#: http.client.HTTPException) when no response came.
Transport = Callable[[str, Mapping[str, str], bytes, float], tuple[int, bytes]]


def urllib_transport(url: str, headers: Mapping[str, str], body: bytes,
                     timeout_s: float) -> tuple[int, bytes]:
    """POSTs `body` with `urllib`; an HTTP error status is returned with its body, not raised."""
    request = urllib.request.Request(url, data=body, headers=dict(headers), method="POST")
    try:
        with urllib.request.urlopen(request, timeout=timeout_s) as response:
            return int(response.status), response.read()
    except urllib.error.HTTPError as error:
        with error:
            return int(error.code), error.read()


def request_body(model: str, state: Mapping[str, Any], questions: Questions,
                 assets: Mapping[str, Mapping[str, str]]) -> dict[str, Any]:
    """The System One request: the model, the state and the questions, and the image assets when any."""
    body: dict[str, Any] = {"model": model, "state": state, "questions": questions}
    if assets:
        body["assets"] = assets
    return body


def parse_reply(status: int, payload: bytes, questions: Questions) -> Reply:
    """The reply to a System One request, or `JudgeError` naming why there is none."""
    if status == 200:
        try:
            body = json.loads(payload)
        except ValueError:
            raise JudgeError("jev: odpowiedz nie do uzycia (to nie JSON)", kind="unusable", status=status,
                             detail=payload[:QUOTE_LIMIT].decode("utf-8", "replace")) from None
        if not isinstance(body, Mapping):
            raise JudgeError("jev: odpowiedz nie do uzycia (to nie obiekt JSON)", kind="unusable",
                             status=status)
        try:
            return checked_reply("jev", body.get("answers"), body.get("model"), body.get("usage"), questions)
        except JudgeError as error:
            error.status = status
            raise
    detail = error_detail(payload)
    if status in (401, 403):
        raise JudgeError(f"jev odrzucil klucz {KEY_ENV} ({status})", kind="key_rejected", status=status,
                         detail=detail)
    if status in (400, 413, 415, 422):
        raise JudgeError(f"jev odrzucil pytanie ({status}: {detail})", kind="rejected", status=status,
                         detail=detail)
    if status in (408, 429) or 500 <= status < 600:
        raise JudgeError(f"jev chwilowo niedostepny ({status})", kind="overloaded", retryable=True,
                         status=status, detail=detail)
    raise JudgeError(f"jev odpowiedzial statusem {status}", kind="error", status=status, detail=detail)


def error_detail(payload: bytes) -> str:
    """What an error body says, short: a 422's `loc: msg` items, a 400's text, an object's message."""
    try:
        body = json.loads(payload)
    except ValueError:
        return payload[:QUOTE_LIMIT].decode("utf-8", "replace").strip()
    detail = body.get("detail") if isinstance(body, Mapping) else body
    if isinstance(detail, list):
        parts = []
        for item in detail:
            if isinstance(item, Mapping):
                where = ".".join(str(part) for part in item.get("loc") or [])
                message = str(item.get("msg", ""))
                parts.append(f"{where}: {message}" if where else message)
            else:
                parts.append(str(item))
        text = "; ".join(parts)
    elif isinstance(detail, Mapping):
        text = str(detail.get("message") or json.dumps(detail, sort_keys=True))
    else:
        text = "" if detail is None else str(detail)
    return text[:QUOTE_LIMIT]


def png_bytes(rgb: np.ndarray) -> bytes:
    """An RGB image (height, width, 3) as an 8-bit PNG, without filters."""
    image = np.ascontiguousarray(rgb, dtype=np.uint8)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"jev: an image must be (height, width, 3), is {image.shape}")
    height, width = image.shape[:2]
    rows = np.concatenate([np.zeros((height, 1), np.uint8), image.reshape(height, width * 3)], axis=1)

    def chunk(kind: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))

    header = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", header) + chunk(b"IDAT", zlib.compress(rows.tobytes()))
            + chunk(b"IEND", b""))


def frame_problem(shape: tuple[int, ...]) -> str:
    """Why Jev would refuse a frame of this shape, for the state and the panel; empty when it takes it."""
    if len(shape) != 3 or shape[2] != 3:
        return f"ksztalt {tuple(shape)}, nie RGB"
    low, high = IMAGE_SIDE_PX
    height, width = shape[:2]
    if low <= height <= high and low <= width <= high:
        return ""
    return f"{width}x{height} px, poza {low}..{high}"


class JevJudge:
    """Asks Jev over HTTPS. `transport` and `environ` are for tests: urllib and os.environ by default."""

    name = "jev"

    def __init__(self, model: str = DEFAULT_MODEL, *, transport: Transport = urllib_transport,
                 environ: Mapping[str, str] | None = None):
        self.model = model
        self._transport = transport
        self._environ = os.environ if environ is None else environ
        self._key = ""

    def __repr__(self) -> str:
        return f"JevJudge(model={self.model!r})"

    def check(self) -> None:
        """Reads the key from the environment; raises when there is none."""
        key = str(self._environ.get(KEY_ENV, "")).strip()
        if not key:
            raise RuntimeError(f"brak klucza jev: ustaw {KEY_ENV} w srodowisku twina "
                               "(uv run --env-file .env twin ui)")
        self._key = key

    def ask(self, state: Mapping[str, Any], questions: Questions, *, images: Mapping[str, np.ndarray],
            timeout_s: float, cancel: threading.Event) -> Reply:
        if not self._key:
            self.check()
        assets = {reference: {"type": "image", "format": "png",
                              "data_base64": base64.b64encode(png_bytes(image)).decode("ascii")}
                  for reference, image in images.items()}
        body = json.dumps(request_body(self.model, state, questions, assets), allow_nan=False,
                          separators=(",", ":")).encode("utf-8")
        headers = {"Authorization": f"Bearer {self._key}", "Content-Type": "application/json"}
        if assets:
            headers[IMAGES_HEADER] = "true"
        if cancel.is_set():
            raise JudgeError("jev: pytanie odwolane", kind="cancelled")
        try:
            status, payload = self._transport(SYSTEM_ONE_URL, headers, body, timeout_s)
        except (OSError, http.client.HTTPException) as exc:
            raise JudgeError(f"jev nieosiagalny ({type(exc).__name__})", kind="unreachable", retryable=True,
                             detail=f"{type(exc).__name__}: {exc}") from None
        return parse_reply(status, payload, questions)
