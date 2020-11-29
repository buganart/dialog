import pytest

from dialog import finetune

REPLACE_CHAR = "�"


@pytest.fixture(scope="session")
def nlp():
    return finetune.create_nlp()


def write_to_file(path, text, encoding):
    with open(path, "w", encoding=encoding) as f:
        f.write(text)


@pytest.fixture
def text_file_path(tmpdir, encoding):
    path = tmpdir / "text.txt"
    write_to_file(path, "hello—world", encoding)
    return path


@pytest.mark.parametrize("encoding", ["utf8", "windows-1252"])
def test_read_file(text_file_path):
    text = finetune.read_file_try_encodings(text_file_path)
    assert text == "hello—world"


@pytest.mark.parametrize("encoding", ["windows-1252"])
def test_read_file_wrong_encoding_raises(text_file_path):
    with pytest.raises(ValueError):
        finetune.read_file_try_encodings(text_file_path, encodings=["utf8"])


@pytest.fixture
def document_text():
    return "This. Is?\n\nText!"


@pytest.fixture
def text_dir(tmpdir, document_text):
    path = tmpdir / "text.txt"
    write_to_file(path, document_text, encoding="utf8")
    return tmpdir


def test_extract_sentences(document_text, nlp):
    sentences = finetune.extract_sentences(document_text, nlp)
    assert sentences == ["This.", "Is?", "Text!"]


def test_read_text_dir(text_dir, nlp):
    sentences = finetune.read_text_dir(text_dir, nlp)
    assert sentences == [["This.", "Is?", "Text!"]]
