"""The Meta gateway without the network: retries what the API says to retry, stops at the spend cap, caches answers."""

import json

import httpx
import pytest

from app import meta_api

OK = {"choices": [{"message": {"content": '{"a": 1}'}, "finish_reason": "stop"}],
      "usage": {"prompt_tokens": 1000, "completion_tokens": 1000}}


@pytest.fixture
def gateway(tmp_path, monkeypatch):
    monkeypatch.setattr(meta_api, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(meta_api, "LEDGER", tmp_path / "spend.jsonl")
    monkeypatch.setattr(meta_api, "_spent_usd", 0.0)
    monkeypatch.setattr(meta_api.time, "sleep", lambda s: None)
    monkeypatch.setenv("MODEL_API_KEY", "test-key")
    monkeypatch.setenv("MODEL_MODE", "live")
    return monkeypatch


def serve(gateway, statuses):
    calls = []

    def post(url, json, headers, timeout):
        calls.append(url)
        status = statuses[min(len(calls), len(statuses)) - 1]
        return httpx.Response(status, json=OK if status == 200 else {"error": "x"}, request=httpx.Request("POST", url))

    gateway.setattr(meta_api.httpx, "post", post)
    return calls


def ask(n=0):
    return meta_api.spark_json([{"role": "user", "content": f"q{n}"}], {"type": "object"}, "t")


def test_retries_503_then_succeeds_and_prices_the_call(gateway):
    calls = serve(gateway, [503, 200])
    answer = ask()
    assert answer.data == {"a": 1} and len(calls) == 2
    assert answer.usd == pytest.approx((1000 * 1.25 + 1000 * 4.25) / 1e6)


def test_does_not_retry_a_timeout(gateway):
    calls = serve(gateway, [504, 200])
    with pytest.raises(meta_api.ModelCallError):
        ask()
    assert len(calls) == 1


def test_second_identical_call_comes_from_the_cache(gateway):
    calls = serve(gateway, [200])
    ask(), ask()
    assert len(calls) == 1
    gateway.setenv("MODEL_MODE", "replay")
    assert ask().cached
    with pytest.raises(meta_api.ModelCallError):
        ask(1)  # not cached, and replay mode forbids the network


def test_spend_cap_stops_new_calls(gateway):
    serve(gateway, [200])
    gateway.setenv("USD_CAP", "0.001")
    ask(0)  # $0.0055: allowed, since nothing was spent yet
    with pytest.raises(meta_api.SpendCapReached):
        ask(1)
    assert len((meta_api.LEDGER).read_text().splitlines()) == 1
    assert json.loads(meta_api.LEDGER.read_text())["usd"] > 0


def test_sam_boxes_are_parsed_with_confidence_and_billed_per_image(gateway):
    text = ("<0f>0<|box;x1=12;y1=40;x2=30;y2=71;w=1000;h=1000;c=0.83|><|mask;x=0;y=0;c=0.83;data=1,2,!!|>\n"
            "<0f>1<|box;x1=500.5;y1=7;x2=520;y2=33;w=1000;h=1000;c=0.12|><|mask;x=0;y=0;c=0.12;data=1,2,!!|>")
    sam_reply = {"output": [{"type": "message", "content": [{"type": "output_text", "text": text}]}],
                 "usage": {"input_tokens": 0, "output_tokens": 0, "images_processed": 1}}
    gateway.setattr(meta_api.httpx, "post", lambda url, json, headers, timeout: httpx.Response(
        200, json=sam_reply, request=httpx.Request("POST", url)))
    segments, usd, cached = meta_api.sam_find(b"\xff\xd8jpeg", "person")
    assert [s.box for s in segments] == [(12, 40, 30, 71), (500.5, 7, 520, 33)]
    assert [s.confidence for s in segments] == [0.83, 0.12]
    assert usd == pytest.approx(0.0025) and not cached
