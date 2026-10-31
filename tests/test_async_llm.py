import asyncio
import json
import httpx
import pytest
from valen_sdk import AsyncValen, Noul, ValenTimeoutError, ValenAPIError


def reply(text):
    return {'choices':[{'message':{'content':text}, 'finish_reason':'stop'}], 'usage':{'prompt_tokens':2,'completion_tokens':3}}


def test_async_llm_concurrent_correction_and_diagnostics():
    async def run():
        active, peak = 0, 0
        async def handle(request):
            nonlocal active, peak
            active += 1; peak = max(peak, active)
            await asyncio.sleep(.01)
            active -= 1
            messages = json.loads(request.content)['messages']
            return httpx.Response(200, json=reply('invalid' if len(messages)==2 else '{"q1":{"true":3,"false":1}}'))
        async with AsyncValen.llm('openai','test', transport=httpx.MockTransport(handle), total_timeout=1) as client:
            responses = await asyncio.gather(*[client.system_one('x', {'flag':Noul(instructions='Flag?')}) for _ in range(3)])
        assert peak == 3
        for r in responses:
            assert r.answers['flag'].noul == .75
            assert r.usage.input_tokens == 4 and r.usage.n_retries_malformed == 1
            assert r.debug['normalized'] == {'flag':True}
            assert r.debug['probability_source'] == 'llm_self_reported'
    asyncio.run(run())


def test_async_llm_deadline_stops_provider_call():
    async def run():
        stopped = asyncio.Event()
        async def slow(request):
            try: await asyncio.sleep(10)
            finally: stopped.set()
        async with AsyncValen.llm('openai','test', transport=httpx.MockTransport(slow), total_timeout=.02) as client:
            with pytest.raises(ValenTimeoutError): await client.system_one('x', {'flag':Noul(instructions='Flag?')})
            assert stopped.is_set()
    asyncio.run(run())


def test_probability_normalization_can_be_disabled():
    async def run():
        transport = httpx.MockTransport(lambda request: httpx.Response(200, json=reply('{"q1":{"true":3,"false":1}}')))
        async with AsyncValen.llm('openai','test', transport=transport, normalize_probabilities=False, n_retry_malformed=0) as client:
            with pytest.raises(ValenAPIError, match='normalization disabled'): await client.system_one('x', {'flag':Noul(instructions='Flag?')})
    asyncio.run(run())


@pytest.mark.parametrize('provider', ['openai','anthropic','gemini'])
def test_native_async_providers_share_response_contract(provider):
    async def run():
        text = '{"q1":{"true":0.8,"false":0.2}}'
        bodies = {
            'openai':reply(text),
            'anthropic':{'content':[{'type':'text','text':text}],'stop_reason':'end_turn','usage':{'input_tokens':2,'output_tokens':3}},
            'gemini':{'candidates':[{'content':{'parts':[{'text':text}]},'finishReason':'STOP'}], 'usageMetadata':{'promptTokenCount':2,'candidatesTokenCount':3}},
        }
        async with AsyncValen.llm(provider,'test', transport=httpx.MockTransport(lambda request: httpx.Response(200,json=bodies[provider]))) as client:
            response = await client.system_one('x', {'flag':Noul(instructions='Flag?')})
            assert response.answers['flag'].noul == .8
            assert response.usage.input_tokens == 2
            assert response.debug['normalized'] == {'flag':False}
    asyncio.run(run())
