from setuptools import setup, find_packages

setup(
    name="dialog",
    url="https://github.com/buganart/dialog",
    author="buganart",
    description="Conversations with transformers.",
    packages=find_packages(),
    install_requires=[
        "pandas",
        "torch",
        "tqdm",
        # XXX Switch back to "transformers" when transformers v3.5.2 or higher is released
        # "transformers"
        "transformers @ https://github.com/huggingface/transformers/archive/e1f3156b218956d1c4b8904dfcffaa19a2138f6a.zip",
        "wandb",
    ],
    use_scm_version=True,
    setup_requires=["setuptools_scm"],
    entry_points={
        "console_scripts": [
            "finetune = dialog.finetune:main",
            "generate = dialog.generate:main",
        ]
    },
)
