import os
import pytest

from dialog.finetune import train


@pytest.fixture(autouse=True)
def disable_wandb():
    os.environ["WANDB_MODE"] = "dryrun"


@pytest.fixture
def text_dir(tmpdir):
    with open(tmpdir / "text.txt", "w") as f:
        for _ in range(5):
            print("This is some test text!", file=f)
    return tmpdir


@pytest.fixture
def save_dir(tmpdir):
    return tmpdir


def test_train(text_dir, save_dir):
    train(
        text_dir=text_dir,
        save_dir=save_dir,
        num_context=2,
        extra_trainer_args=dict(num_train_epochs=1),
    )
