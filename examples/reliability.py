"""Offline example: deadlines, safe events, borrowed connections and async LLMs."""
import asyncio
import httpx
from valen_sdk import AsyncValen, Noul, Valen

QUESTIONS = {"arrived": Noul(instructions="The order arrived.")}

def main():
    def service(request):
        return httpx.Response(200, json={"model":"example", "answers":{
            "arrived":{"type":"noul", "noul":.9}}, "usage":{}})
    with httpx.Client(transport=httpx.MockTransport(service)) as http:
        with Valen.remote("https://example.test/v1", http_client=http,
                          total_timeout=5, on_event=print) as client:
            print(client.system_one("The order arrived.", QUESTIONS).nouls["arrived"].noul)
        assert not http.is_closed
    asyncio.run(compare())

async def compare():
    async def provider(request):
        return httpx.Response(200, json={"choices":[{"message":{
            "content":'{"q1":{"true":0.9,"false":0.1}}'}, "finish_reason":"stop"}]})
    # Mock provider: no credentials or network requests are used.
    async with AsyncValen.llm("openai", "example", api_key="",
                             transport=httpx.MockTransport(provider), total_timeout=5) as client:
        responses = await asyncio.gather(*[
            client.system_one(text, QUESTIONS) for text in ["The order arrived.", "It is here."]])
        print([r.nouls["arrived"].noul for r in responses])

if __name__ == "__main__":
    main()
