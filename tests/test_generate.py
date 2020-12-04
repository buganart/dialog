import pickle

import pytest
from transformers import AutoModelForCausalLM, AutoTokenizer

from dialog import generate


@pytest.fixture(scope="session")
def pretrained_model():
    return "sshleifer/tiny-gpt2"  # A bit faster than gpt2-small.


@pytest.fixture(scope="session")
def model(request, pretrained_model):
    cache_str = request.config.cache.get("model", None)
    if cache_str is None:
        obj = AutoModelForCausalLM.from_pretrained(pretrained_model)
        cache_str = pickle.dumps(obj).decode("cp437")
        request.config.cache.set("model", cache_str)
    else:
        obj = pickle.loads(cache_str.encode("cp437"))
    return obj


@pytest.fixture(scope="session")
def tokenizer(request, pretrained_model):
    cache_str = request.config.cache.get("tokenizer", None)
    if cache_str is None:
        obj = AutoTokenizer.from_pretrained(
            pretrained_model,
            pad_token="[PAD]",  # TODO should we set this?
        )
        cache_str = pickle.dumps(obj).decode("cp437")
        request.config.cache.set("tokenizer", cache_str)
    else:
        obj = pickle.loads(cache_str.encode("cp437"))
    return obj


@pytest.mark.slow
def test_generate_answer(model, tokenizer, device):
    answer = generate.generate_answer(
        model,
        tokenizer,
        device=device,
        context=["How are you?"],
    )
    assert isinstance(answer, str)


@pytest.mark.slow
def test_generate(model, tokenizer, device):
    generate.generate(
        model=model,
        tokenizer=tokenizer,
        device=device,
        prefix="How are you?",
        steps=2,
    )
