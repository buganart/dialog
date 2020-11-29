import pytest

from dialog import finetune

REPLACE_CHAR = "�"


@pytest.fixture
def text_file_path(tmpdir):
    path = tmpdir / "text.txt"
    with open(path, "wb") as f:
        f.write(b"\x97")
    return path


def test_unicode_errors(text_file_path):
    with pytest.raises(UnicodeDecodeError):
        finetune.read_lines(text_file_path)


def test_unicode_errors_replace(text_file_path):
    text = finetune.read_lines_warn_on_error(text_file_path)
    assert text == [REPLACE_CHAR]


@pytest.mark.usefixtures("text_file_path")
def test_read_text_dir(tmpdir):
    documents = finetune.read_text_dir(tmpdir)
    assert documents == [[REPLACE_CHAR]]
