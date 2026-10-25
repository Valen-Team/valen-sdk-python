"""Call a local server started with `valen serve`."""
from valen_sdk import Valen

with Valen.remote("http://127.0.0.1:8000/v1", model="valen") as client:
    result = client.decisions(
        input="Please send my order to my new address.",
        questions=[{"name": "change_address", "type": "predicate",
                    "instructions": "Is the customer requesting a delivery address change?"}],
    )
    print("Address change probability:", result.by_name("change_address").probability)
