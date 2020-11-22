# Dialog

Fine tune DialoGPT. Not working yet.

## Installation

Cuda, python,

    pip install -r requirements.txt

## Finetuning

    finetune --text-dir /path/to/text/files

## Generation

    generate --checkpoint-dir /path/to/checkpoint --prefix "I'm in the same boat."

## Development
Install dependencies

    pip install -r requirements.txt

Run tests

    pytest

## Todo

- [x] Training.
- [x] Save models.
- [x] Predictions.
- [ ] Colab.
