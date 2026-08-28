import os

import pytest
from openai import OpenAI



def test_openai_smoke():
    if os.environ.get("RUN_OPENAI_SMOKE", "false").lower() != "true":
        pytest.skip("Set RUN_OPENAI_SMOKE=true to run the live OpenAI smoke test")

    client = OpenAI()
    response = client.responses.create(
        model=os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),
        input="Write one simple Grade 9 linear functions question.",
    )
    assert response.output_text