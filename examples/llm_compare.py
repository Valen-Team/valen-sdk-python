"""Ask the same questions to Valen and to an LLM, then compare answers, tokens and latency.

Needs OPENAI_API_KEY (or change the provider and model below). The free Valen API needs no key.
"""
import time

from valen_sdk import Choice, Noul, Valen

TICKETS = [
    "I was charged twice for my order. Please refund one of the payments.",
    "The app crashes every time I open the settings page.",
    "Can I change the delivery address for order 1042?",
    "Do you ship to Canada?",
]
QUESTIONS = {
    "queue": Choice(instructions="Which team should handle this ticket?", criteria={
        "billing": "Payments, charges and refunds",
        "technical": "Bugs, crashes and errors in the app",
        "orders": "Changes to an existing order or its delivery",
        "none": "None of these teams",
    }),
    "refund": Noul(instructions="The customer asks for money back."),
}

clients = {
    "Valen 2B": Valen.remote("https://api.valen-team.workers.dev/v1", model="valen-2b"),
    "gpt-4o-mini": Valen.llm("openai", "gpt-4o-mini"),
}
agree = 0
for ticket in TICKETS:
    print(ticket)
    queues = []
    for name, client in clients.items():
        started = time.perf_counter()
        result = client.system_one(ticket, QUESTIONS)
        seconds = time.perf_counter() - started
        queue = result.choices["queue"]
        queues.append(queue.choice)
        print(f"  {name:12} queue={queue.choice:10} confidence={queue.confidence:.2f} "
              f"refund={result.nouls['refund'].noul:.2f} tokens={result.usage.input_tokens} {seconds:.2f}s")
    agree += len(set(queues)) == 1
print(f"Queue agreement: {agree}/{len(TICKETS)}")
for client in clients.values():
    client.close()
