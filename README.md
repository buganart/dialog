# Dialog

Fine tune DialoGPT.

## Installation

Cuda, python,

    pip install -e .

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

    python api.py --checkpoint-dir /path/to/model/checkpoint/directory

Or with a pretrained model

    python api.py --checkpoint-dir microsoft/DialoGPT-small

### Docker
Build docker container with the Dockerfile.

    docker build -t <image name> .

Run the API server with docker using a pretrained model (get the uploaded docker container from buganart/)

    docker run -it --rm -p 8080:8080 buganart/dialog-api:0.1.2 python api.py --checkpoint-dir microsoft/DialoGPT-small

Note the volume mount from the host file system to `/checkpoint` inside
the docker container. This can be used to choose which model to run (via a
different checkpoint).

On windows with WSL2 the windows drives are accessible with the path format below, for example
    docker run -it --rm -p 8080:8080 -v d:/path/to/checkpoint_folder:/checkpoint buganart/dialog-api:0.1.2 python api.py --checkpoint-dir /checkpoint


#### Docker on Windows
Have not tested running anything on windows but I found this article about
running linux docker containers on windows:
https://hackernoon.com/how-to-run-docker-linux-containers-natively-on-windows-ti1i3uxr

### API Client

    curl -X POST -H "Content-Type: application/json" -d '{"context": ["Go Away."]}' 127.0.0.1:8080/generate

Returns a JSON object with an `answer` field.

    {"answer":"The greatest album of all time.","context":["Go Away."], "score":{"compound": 0.3612,"neg": 0.0,"neu": 0.0,"pos": 1.0}
