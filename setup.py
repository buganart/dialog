from setuptools import setup, find_packages

setup(
    name="dialog",
    url="https://github.com/buganart/dialog",
    author="buganart",
    description="Conversations with transformers.",
    packages=find_packages(),
    install_requires=[
        "pandas",
        # "torch",
        "tqdm",
        "transformers",
        # "wandb",
    ],
    use_scm_version=True,
    setup_requires=["setuptools_scm"],
    entry_points={
        "console_scripts": [
            "finetune = dialog.finetune:main",
        ]
    },

)
