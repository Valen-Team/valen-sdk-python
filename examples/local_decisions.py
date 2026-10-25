"""Run three decisions on CPU; the SDK downloads and caches the model."""
from valen_sdk import Valen

with Valen.local("Valen-Team/Valen-0.8B-ONNX", variant="fp32") as client:
    result = client.decisions(
        input="I moved. Please ship my order to 12 Elm Street instead.",
        questions=[
            {"name": "intent", "type": "choice", "instructions": "What does the customer want?",
             "choices": [{"value": "address", "description": "Change the delivery address"},
                         {"value": "cancel", "description": "Cancel the order"}]},
            {"name": "change_address", "type": "predicate",
             "instructions": "The customer requests a delivery address change."},
            {"name": "explicitness", "type": "score", "instructions": "How explicit is the request?",
             "levels": [{"label": "none", "description": "No request"},
                        {"label": "possible", "description": "Asks whether a change is possible"},
                        {"label": "explicit", "description": "Explicitly requests a change"}]},
        ],
    )
    print("Intent:", result.by_name("intent").choice)
    print(f"Address change probability: {result.by_name('change_address').probability:.3f}")
    print(f"Explicitness score (0–2): {result.by_name('explicitness').score:.3f}")
