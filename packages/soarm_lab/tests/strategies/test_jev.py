"""The Jev client without Jev: the request it sends and how it reads each documented response.

A fake transport stands in for the network: it records what would go out and answers with a status and a
body shaped as TypeSafe's API documents them (OpenAPI 0.2.0). One test runs the urllib transport against
a local HTTP server. No test needs TYPESAFE_API_KEY.
"""

from __future__ import annotations

import base64
import contextlib
import http.server
import json
import math
import struct
import threading
import urllib.error
import urllib.request
import zlib
from collections.abc import Iterator, Mapping
from typing import Any

import numpy as np
import pytest

from soarm_lab.strategies.jev import (
    IMAGES_HEADER,
    KEY_ENV,
    SYSTEM_ONE_URL,
    JevJudge,
    error_detail,
    frame_problem,
    parse_reply,
    png_bytes,
    urllib_transport,
)
from soarm_lab.strategies.judges import JudgeError

KEY = "sk-test-SECRET-4f1c"
QUESTIONS = {
    "touched": {"type": "choice", "instructions": {"question": "Did it touch?"},
                "criteria": {"missed_object": {"what": "no"}, "touched_object": {"what": "yes"},
                             "blocked_above_object": {"what": "high"}}},
    "correction": {"type": "choice", "instructions": "Where next?",
                   "criteria": {"aim_at_latest_sighting": "there", "to_robot_left": "left"}},
}
STATE = {"attempt": 1, "descent": {"ended": "reached_touch_point_in_free_air", "drop_from_hover_mm": 60}}
#: The documented response shape, for our question names.
ANSWERED = {
    "model": "jev-1.14.0",
    "answers": {
        "touched": {"type": "choice", "choice": "missed_object", "confidence": 0.94,
                    "probabilities": {"missed_object": 0.96, "touched_object": 0.03,
                                      "blocked_above_object": 0.01}},
        "correction": {"type": "choice", "choice": "aim_at_latest_sighting", "confidence": 0.8,
                       "probabilities": {"aim_at_latest_sighting": 0.9, "to_robot_left": 0.1}},
    },
    "usage": {"input_tokens": 428, "output_tokens": 4},
}


class FakeTransport:
    """Records each request and answers from `responses`: (status, body) pairs or exceptions to raise."""

    def __init__(self, *responses: tuple[int, Any] | Exception):
        self.responses = list(responses)
        self.sent: list[dict[str, Any]] = []

    def __call__(self, url: str, headers: Mapping[str, str], body: bytes,
                 timeout_s: float) -> tuple[int, bytes]:
        self.sent.append({"url": url, "headers": dict(headers), "body": json.loads(body), "raw": body,
                          "timeout_s": timeout_s})
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        status, payload = response
        return status, payload if isinstance(payload, bytes) else json.dumps(payload).encode()


def judge(*responses: tuple[int, Any] | Exception) -> tuple[JevJudge, FakeTransport]:
    transport = FakeTransport(*responses)
    return JevJudge("jev-1.14.0", transport=transport, environ={KEY_ENV: KEY}), transport


def ask(jev: JevJudge, images: Mapping[str, np.ndarray] | None = None, state: Any = None,
        cancel: threading.Event | None = None):
    return jev.ask(STATE if state is None else state, QUESTIONS, images=images or {}, timeout_s=4.0,
                   cancel=cancel or threading.Event())


# ------------------------------------------------------------ the request
def test_the_request_is_the_documented_json_with_the_key_in_the_header_only():
    jev, transport = judge((200, ANSWERED))
    ask(jev)

    sent = transport.sent[0]
    assert sent["url"] == SYSTEM_ONE_URL == "https://api.typesafe.ai/v1/systemone"
    assert sent["headers"] == {"Authorization": f"Bearer {KEY}", "Content-Type": "application/json"}
    assert sent["timeout_s"] == 4.0
    assert sent["body"] == {"model": "jev-1.14.0", "state": STATE, "questions": QUESTIONS}
    assert list(sent["body"]["questions"]["touched"]["criteria"]) == ["missed_object", "touched_object",
                                                                     "blocked_above_object"]
    assert KEY.encode() not in sent["raw"]


def test_a_frame_goes_out_as_a_png_asset_with_the_images_header():
    image = np.random.default_rng(0).integers(0, 256, (300, 400, 3), dtype=np.uint8)
    jev, transport = judge((200, ANSWERED))
    ask(jev, images={"a1b2": image}, state={**STATE, "camera_views": {"front": "a1b2"}})

    sent = transport.sent[0]
    assert sent["headers"][IMAGES_HEADER] == "true"
    asset = sent["body"]["assets"]["a1b2"]
    assert set(asset) == {"type", "format", "data_base64"}
    assert asset["type"] == "image" and asset["format"] == "png"
    assert "\n" not in asset["data_base64"] and not asset["data_base64"].startswith("data:")
    assert decoded_png(base64.b64decode(asset["data_base64"])).tolist() == image.tolist()


def decoded_png(data: bytes) -> np.ndarray:
    """An 8-bit RGB PNG without filters, as `png_bytes` writes it."""
    assert data[:8] == b"\x89PNG\r\n\x1a\n"
    chunks, at = {}, 8
    while at < len(data):
        (length,), kind = struct.unpack(">I", data[at:at + 4]), data[at + 4:at + 8]
        body = data[at + 8:at + 8 + length]
        assert struct.unpack(">I", data[at + 8 + length:at + 12 + length])[0] == zlib.crc32(kind + body)
        chunks[kind] = body
        at += 12 + length
    width, height, depth, colour = struct.unpack(">IIBB", chunks[b"IHDR"][:10])
    assert (depth, colour) == (8, 2) and b"IEND" in chunks
    rows = np.frombuffer(zlib.decompress(chunks[b"IDAT"]), np.uint8).reshape(height, 1 + 3 * width)
    assert not rows[:, 0].any()
    return rows[:, 1:].reshape(height, width, 3)


def test_png_bytes_refuses_what_is_not_rgb():
    with pytest.raises(ValueError, match="height, width, 3"):
        png_bytes(np.zeros((10, 10), np.uint8))


@pytest.mark.parametrize("shape, problem", [
    ((300, 400, 3), ""),
    ((280, 1568, 3), ""),
    ((150, 200, 3), "200x150 px, poza 280..1568"),
    ((300, 1569, 3), "1569x300 px, poza 280..1568"),
    ((300, 400), "nie RGB"),
])
def test_jev_takes_a_frame_only_within_its_side_limits(shape: tuple[int, ...], problem: str):
    assert problem in frame_problem(shape) and (problem != "" or frame_problem(shape) == "")


def test_a_number_that_is_not_finite_never_goes_out():
    jev, transport = judge((200, ANSWERED))

    with pytest.raises(ValueError):
        ask(jev, state={"drop_mm": math.nan})
    assert transport.sent == []


# ------------------------------------------------------------ the key
def test_the_key_comes_from_the_environment_and_check_refuses_without_it():
    for environ in ({}, {KEY_ENV: "   "}):
        with pytest.raises(RuntimeError, match=f"brak klucza jev: ustaw {KEY_ENV}"):
            JevJudge(environ=environ).check()
    JevJudge(environ={KEY_ENV: KEY}).check()


@pytest.mark.parametrize("key", ["sk-live-SECRET\nsecond-line", "sk-live-SECRET\rx", "sk-live SECRET",
                                 "sk-live-SECRET\tx", "sk-live-SECRET-\u0142"])
def test_a_key_that_cannot_go_in_a_header_is_refused_without_being_quoted(key: str):
    """http.client would refuse such a header with a ValueError that quotes it whole."""
    with pytest.raises(RuntimeError) as caught:
        JevJudge(environ={"TYPESAFE_API_KEY": key}).check()
    assert "TYPESAFE_API_KEY" in str(caught.value) and "SECRET" not in str(caught.value)


def test_ask_reads_the_key_itself_when_check_was_not_called():
    jev, transport = judge((200, ANSWERED))
    reply = ask(jev)

    assert reply.answers["touched"].choice == "missed_object"
    assert transport.sent[0]["headers"]["Authorization"] == f"Bearer {KEY}"


def test_a_cancelled_call_sends_nothing():
    jev, transport = judge((200, ANSWERED))
    cancel = threading.Event()
    cancel.set()

    with pytest.raises(JudgeError) as caught:
        ask(jev, cancel=cancel)
    assert caught.value.kind == "cancelled" and transport.sent == []


# ------------------------------------------------------------ the response
def test_a_documented_reply_parses_with_the_model_that_answered_and_the_usage():
    reply = parse_reply(200, json.dumps(ANSWERED).encode(), QUESTIONS)

    assert reply.model == "jev-1.14.0" and reply.usage == {"input_tokens": 428, "output_tokens": 4}
    touched = reply.answers["touched"]
    assert touched.choice == "missed_object" and touched.confidence == 0.94
    assert touched.probabilities == ANSWERED["answers"]["touched"]["probabilities"]
    assert reply.answers["correction"].choice == "aim_at_latest_sighting"


NOT_AUTHENTICATED = {"detail": {"message": "Not authenticated"}}
MISSING_STATE = {"detail": [{"loc": ["body", "state"], "msg": "Field required", "type": "missing"}]}


@pytest.mark.parametrize("status, body, kind, retryable, reason", [
    (401, {"detail": "Invalid API key"}, "key_rejected", False, f"jev odrzucil klucz {KEY_ENV} (401)"),
    (403, NOT_AUTHENTICATED, "key_rejected", False, f"jev odrzucil klucz {KEY_ENV} (403)"),
    (422, MISSING_STATE, "rejected", False, "jev odrzucil pytanie (422: body.state: Field required)"),
    (400, {"detail": "max_tokens_exceeded"}, "rejected", False,
     "jev odrzucil pytanie (400: max_tokens_exceeded)"),
    (413, b"Request Entity Too Large", "rejected", False,
     "jev odrzucil pytanie (413: Request Entity Too Large)"),
    (429, {"detail": "rate limit"}, "overloaded", True, "jev chwilowo niedostepny (429)"),
    (500, b"", "overloaded", True, "jev chwilowo niedostepny (500)"),
    (529, {"detail": "Overloaded"}, "overloaded", True, "jev chwilowo niedostepny (529)"),
    (418, {}, "error", False, "jev odpowiedzial statusem 418"),
    (200, b"<html>gateway</html>", "unusable", False, "jev: odpowiedz nie do uzycia (to nie JSON)"),
    (200, [1, 2], "unusable", False, "jev: odpowiedz nie do uzycia (to nie obiekt JSON)"),
    (200, {"model": "jev-1.14.0", "answers": {}}, "unusable", False, "brak odpowiedzi na 'touched'"),
])
def test_every_failure_is_an_error_with_a_reason_never_an_answer(status: int, body: Any, kind: str,
                                                                  retryable: bool, reason: str):
    jev, _ = judge((status, body))

    with pytest.raises(JudgeError) as caught:
        ask(jev)
    error = caught.value
    assert error.kind == kind and error.retryable is retryable and reason in error.reason
    assert error.status == status
    assert KEY not in f"{error.reason} {error.detail} {error!r} {jev!r}"


@pytest.mark.parametrize("raised", [urllib.error.URLError("no route"), TimeoutError("timed out"),
                                    ConnectionResetError("reset")])
def test_no_response_at_all_is_unreachable_and_worth_another_try(raised: Exception):
    jev, _ = judge(raised)

    with pytest.raises(JudgeError) as caught:
        ask(jev)
    assert caught.value.kind == "unreachable" and caught.value.retryable
    assert caught.value.reason == f"jev nieosiagalny ({type(raised).__name__})"
    assert KEY not in caught.value.detail


def test_a_request_the_http_library_refuses_is_an_error_that_quotes_nothing():
    jev, _ = judge(ValueError(f"Invalid header value b'Bearer {KEY}'"))

    with pytest.raises(JudgeError) as caught:
        ask(jev)
    error = caught.value
    assert error.kind == "error" and not error.retryable and error.reason == "jev: zly naglowek zapytania"
    assert KEY not in f"{error.reason} {error.detail} {error!r}"


@pytest.mark.parametrize("response", [
    (401, {"detail": f"Invalid API key {KEY}"}),
    (422, {"detail": [{"loc": ["header", "authorization"], "msg": f"bad token {KEY}", "type": "value"}]}),
    (400, {"detail": "x" * 190 + KEY}),                                  # the quote's cut falls in the key
    (200, f"<html>{KEY}</html>".encode()),
    OSError(f"proxy refused Bearer {KEY}"),
])
def test_a_key_echoed_back_is_masked_in_every_error(response: tuple[int, Any] | Exception):
    jev, _ = judge(response)

    with pytest.raises(JudgeError) as caught:
        ask(jev)
    text = f"{caught.value.reason} {caught.value.detail} {caught.value!r}"
    assert KEY[:10] not in text and "***" in caught.value.detail


def test_error_details_are_short():
    assert error_detail(json.dumps({"detail": "x" * 1000}).encode()) == "x" * 200
    assert error_detail(json.dumps({"detail": [{"loc": ["body", "questions", "touched"], "msg": "bad"},
                                               "other"]}).encode()) == "body.questions.touched: bad; other"


# ------------------------------------------------------------ the urllib transport
class Recorder(http.server.BaseHTTPRequestHandler):
    """Answers POST /ok with a reply, /moved with a 302 to `location` and anything else with a 422, and
    keeps what it received."""

    received: list[dict[str, Any]] = []
    location = ""

    def do_POST(self) -> None:  # noqa: N802 - http.server's name
        body = self.rfile.read(int(self.headers["Content-Length"]))
        Recorder.received.append({"path": self.path, "headers": dict(self.headers), "body": json.loads(body)})
        if self.path == "/moved":
            self.send_response(302)
            self.send_header("Location", Recorder.location)
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        status, payload = (200, ANSWERED) if self.path == "/ok" else (422, MISSING_STATE)
        data = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, format: str, *args: Any) -> None:
        pass


class Elsewhere(http.server.BaseHTTPRequestHandler):
    """Another host, where a followed redirect would take the request and its headers."""

    received: list[dict[str, Any]] = []

    def do_GET(self) -> None:  # noqa: N802 - http.server's name
        Elsewhere.received.append({"path": self.path, "headers": dict(self.headers)})
        self.send_response(200)
        self.send_header("Content-Length", "0")
        self.end_headers()

    do_POST = do_GET  # noqa: N815 - http.server's name

    def log_message(self, format: str, *args: Any) -> None:
        pass


@contextlib.contextmanager
def serving(handler: type[http.server.BaseHTTPRequestHandler]) -> Iterator[str]:
    """A local HTTP server on a free port, by its base URL."""
    server = http.server.HTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}"
    finally:
        server.shutdown()
        server.server_close()


HEADERS = {"Authorization": "Bearer local", "Content-Type": "application/json"}


def test_the_urllib_transport_returns_error_statuses_with_their_body():
    Recorder.received.clear()
    with serving(Recorder) as base:
        ok = urllib_transport(f"{base}/ok", HEADERS, b'{"state": 1}', 5.0)
        refused = urllib_transport(f"{base}/refuse", HEADERS, b'{"state": 1}', 5.0)

    assert ok[0] == 200 and json.loads(ok[1]) == ANSWERED
    assert refused[0] == 422 and json.loads(refused[1]) == MISSING_STATE
    assert Recorder.received[0]["headers"]["Authorization"] == "Bearer local"
    assert Recorder.received[0]["body"] == {"state": 1}


def test_the_urllib_transport_never_follows_a_redirect_with_the_key():
    """urllib's default opener would follow a 302 to any host, Authorization header included."""
    Elsewhere.received.clear()
    with serving(Elsewhere) as elsewhere, serving(Recorder) as base:
        Recorder.location = f"{elsewhere}/stolen"
        urllib.request.urlopen(urllib.request.Request(f"{base}/moved", data=b"{}", headers=HEADERS,
                                                      method="POST"), timeout=5.0).close()
        followed = list(Elsewhere.received)
        Elsewhere.received.clear()
        status, _ = urllib_transport(f"{base}/moved", HEADERS, b"{}", 5.0)

    assert followed and followed[0]["headers"]["Authorization"] == "Bearer local"
    assert status == 302 and Elsewhere.received == []
    with pytest.raises(JudgeError, match="jev odpowiedzial statusem 302"):
        parse_reply(status, b"", QUESTIONS)
