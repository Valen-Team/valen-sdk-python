import asyncio
import json
import time
import httpx
import pytest
from valen_sdk import AsyncValen, Valen, Noul, ValenTimeoutError, ValenResponseError
from valen_sdk.types import DecisionResponse

Q = {'flag': Noul(instructions='Flag?')}
def result():
    return {'model':'test', 'answers':{'flag':{'type':'noul','noul':0.8}}, 'usage':{}}


def test_borrowed_client_is_not_closed_and_controls_are_not_sent():
    requests, events = [], []
    def handle(request):
        requests.append(request)
        return httpx.Response(200, json=result())
    http = httpx.Client(transport=httpx.MockTransport(handle))
    with Valen.remote('https://test/v1', http_client=http, total_timeout=1, on_event=events.append) as client:
        assert client.system_one('private', Q, timeout=0.2).answers['flag'].noul == 0.8
    assert not http.is_closed
    assert set(json.loads(requests[0].content)) == {'state','questions'}
    assert [e['event'] for e in events] == ['start','attempt','complete']
    assert 'private' not in json.dumps(events)
    http.close()


def test_retry_after_longer_than_budget_is_not_retried():
    calls = []
    def busy(request):
        calls.append(request)
        return httpx.Response(429, json={}, headers={'retry-after':'60'})
    with Valen.remote('https://test/v1', transport=httpx.MockTransport(busy), total_timeout=.05) as client:
        started = time.monotonic()
        with pytest.raises(ValenTimeoutError): client.system_one('x', Q)
        assert time.monotonic()-started < .5
    assert len(calls) == 1


def test_protocol_failure_is_not_retried():
    calls = []
    def incomplete(request):
        calls.append(request)
        return httpx.Response(200, json={**result(), 'answers':{}}, headers={'x-request-id':'req-7'})
    with Valen.remote('https://test/v1', transport=httpx.MockTransport(incomplete)) as client:
        with pytest.raises(ValenResponseError) as error: client.system_one('x', Q)
    assert error.value.field == 'answers'
    assert error.value.request_id == 'req-7' and error.value.status_code == 200
    assert len(calls) == 1


def test_async_timeout_cancellation_and_client_ownership():
    async def run():
        active = asyncio.Event()
        cancelled = asyncio.Event()
        async def slow(request):
            active.set()
            try: await asyncio.sleep(10)
            except asyncio.CancelledError:
                cancelled.set()
                raise
        http = httpx.AsyncClient(transport=httpx.MockTransport(slow))
        async with AsyncValen('https://test/v1', http_client=http, total_timeout=.02) as client:
            with pytest.raises(ValenTimeoutError): await client.system_one('x', Q)
            assert cancelled.is_set()
            cancelled.clear()
            pending = asyncio.create_task(client.system_one('x', Q, total_timeout=1))
            await asyncio.sleep(0)
            pending.cancel()
            with pytest.raises(asyncio.CancelledError): await pending
            assert cancelled.is_set()
        assert not http.is_closed
        with pytest.raises(RuntimeError, match='closed'): await client.system_one('x', Q)
        await http.aclose()
    asyncio.run(run())


def test_missing_usage_is_unknown_and_missing_answer_is_clear():
    response = DecisionResponse.model_validate({'model':'test','answers':[], 'usage':{}})
    assert response.usage.input_tokens is None
    with pytest.raises(KeyError, match='available'): response.by_name('missing')


def test_retry_after_http_date_and_invalid_value():
    from datetime import datetime, timedelta, timezone
    from email.utils import format_datetime
    from valen_sdk.client import _delay
    date = format_datetime(datetime.now(timezone.utc)+timedelta(seconds=60))
    assert 58 < _delay(0, httpx.Response(429, headers={'retry-after': date})) <= 60
    assert 0 <= _delay(0, httpx.Response(429, headers={'retry-after':'invalid'})) <= .25


def test_bad_input_is_rejected_before_network():
    from valen_sdk import ValenInputError
    calls = []
    with Valen.remote('https://test/v1', transport=httpx.MockTransport(lambda r: calls.append(r))) as client:
        with pytest.raises(ValenInputError, match='unique'):
            client.decisions(input='x', questions=[{'name':'same','type':'predicate','instructions':'X?'},
                                                   {'name':'same','type':'predicate','instructions':'Y?'}])
        with pytest.raises(ValenInputError, match='nonempty'):
            client.system_one('x', {'flag':Noul(instructions='')})
    assert not calls


def test_local_close_waits_for_inference_and_warmup_reports_events():
    import threading
    from valen_sdk.onnx import ONNXBackend
    backend = ONNXBackend.__new__(ONNXBackend)
    backend.name, backend.variant_name, backend.providers = 'local-test', 'fp32', ['CPUExecutionProvider']
    backend.lock, backend.sessions, backend.closed = threading.RLock(), {'head':object()}, False
    backend.processor = object()
    events, admitted, release = [], threading.Event(), threading.Event()
    backend.on_event = events.append
    def inference(body, protocol, **options):
        with backend.lock:
            if backend.closed: raise RuntimeError('Valen client is closed')
            admitted.set()
            assert release.wait(2)
            return result()
    backend._predict = inference
    client = Valen(backend)
    task = threading.Thread(target=lambda: client.warmup({'state':'x','questions':Q}))
    task.start()
    assert admitted.wait(2)
    closed = threading.Event()
    closer = threading.Thread(target=lambda: (client.close(), closed.set()))
    closer.start()
    assert not closed.wait(.02)
    release.set()
    task.join(2); closer.join(2)
    assert closed.is_set() and client.info['closed']
    assert client.info['loaded_graphs'] == []
    assert [e['event'] for e in events] == ['start','complete']
    client.close()
