# Dialog

Fine tune DialoGPT.

## Installation

Cuda, python,

    pip install -r requirements.txt

## Finetuning

    finetune --text-dir /path/to/text/files

## Generation

    generate --checkpoint-dir /path/to/checkpoint --prefix "I'm in the same boat."

## API

Run it locally (requiring a working python installation) or inside docker.

### Locally
Install dependencies

    pip install -e .

Run the API server with local checkpoint directory

    dialog-api --checkpoint-dir /path/to/model/checkpoint/directory

Or with a pretrained model

    dialog-api --checkpoint-dir microsoft/DialoGPT-small

### Docker
Build docker container (*or* get the docker container from somewhere else).
Building requires the [nix](https://nixos.org/download.html) package manager to
be installed.

    nix-build container.nix

Import docker container

    docker load < ...docker-image-dialog-api.tar.gz

Run the API server with docker

    docker run -p 8080:8080 -e CHECKPOINT_DIR=/checkpoint -v /path/to/model/checkpoint/directory:/checkpoint -it dialog-api:0.1.0

Note the volume mount from the host file system to `/checkpoint` inside
the docker container. This can be used to choose which model to run (via a
different checkpoint).

#### Docker on Windows
Have not tested running anything on windows but I found this article about
running linux docker containers on windows:
https://hackernoon.com/how-to-run-docker-linux-containers-natively-on-windows-ti1i3uxr

### API Client

    curl -X POST -H "Content-Type: application/json" -d '{"context": ["Go Away."]}' 127.0.0.1:8080/generate

Returns a JSON object with an `answer` field.

    {"answer":"The greatest album of all time.","context":["Go Away."]}


## Development
Install dependencies

    pip install -r requirements.txt

Run tests

    pytest

## Todo

- [x] Training.
- [x] Save models.
- [x] Predictions.
- [x] Colab.
- [x] API.
