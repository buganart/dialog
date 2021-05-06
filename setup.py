from setuptools import setup, find_packages

setup(
    name="dialog",
    version="0.1.0",
    url="https://github.com/buganart/dialog",
    author="buganart",
    description="Conversations with transformers.",
    packages=find_packages(),
    install_requires=[
        "click",
        "flask",
        "spacy",
        "torch",
        "tqdm",
        # XXX Switch back to "transformers" when transformers
        # v3.5.2 or higher is released
        # "transformers"
        "transformers @ https://github.com/huggingface/transformers/archive/e1f3156b218956d1c4b8904dfcffaa19a2138f6a.zip",  # noqa: E501
        "wandb",  # not in nixpkgs, assume manually installed
        "nltk",
    ],
    entry_points={
        "console_scripts": [
            "finetune = dialog.finetune:main",
            "generate = dialog.generate:main",
            "dialog-api = dialog.api:main",
        ]
    },
)
