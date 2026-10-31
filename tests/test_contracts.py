"""The identical versioned fixture is consumed by the JS SDK tests."""
import json
from pathlib import Path
from types import SimpleNamespace
import pytest
from valen_sdk.validation import validate_response
from valen_sdk import ValenResponseError
from valen_sdk.responses import answer

CONTRACT = json.loads((Path(__file__).parents[1] / 'contracts/v1.json').read_text())

@pytest.mark.parametrize('case', CONTRACT['cases'], ids=lambda c: c['name'])
def test_response_contract(case):
    if case['valid']:
        result = validate_response(case['request'], case['response'])
        assert result == case['response']
        if 'expected_answer_names' in case:
            assert [a['name'] for a in result['answers']] == case['expected_answer_names']
    else:
        with pytest.raises(ValenResponseError) as error:
            validate_response(case['request'], case['response'])
        assert error.value.code == 'invalid_response' and not error.value.retryable
        assert error.value.field.startswith('answers')

@pytest.mark.parametrize('case', CONTRACT['formula_cases'], ids=lambda c: c['type'])
def test_formula_contract(case):
    result = answer(SimpleNamespace(kind=case['type'], keys=case['keys'], descriptions=case['descriptions']), case['logits'])
    for key, value in case['expected'].items():
        assert result[key] == pytest.approx(value) if isinstance(value, (dict, int, float)) else result[key] == value
