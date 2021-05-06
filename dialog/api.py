import os

import click
import torch
from flask import Flask, jsonify, request

from generate import generate_answer, load_model, load_tokenizer

import nltk

nltk.download("vader_lexicon")

from nltk.sentiment.vader import SentimentIntensityAnalyzer

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 5 * 1024 * 1024
app.config["SERVER_NAME"] = os.environ.get("SERVER_NAME")


@app.route("/generate", methods=["post"])
def generate():
    req = request.get_json(force=True)

    context = req["context"]
    print(f"Input: {context}")

    answer = generate_answer(
        app.config["model"],
        app.config["tokenizer"],
        app.config["device"],
        context=context,
    )

    print(f"Answer: {answer}")

    sid = SentimentIntensityAnalyzer()
    ss = sid.polarity_scores(answer)

    return jsonify(context=context, answer=answer, score=ss)


# @app.route("/version", methods=["GET"])
# def version():
#     with open(VERSION_PATH) as f:
#         return f.read().strip()


@app.route("/status", methods=["GET"])
def status():
    return "ok"


def initialize(checkpoint_dir):

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print(f"=> Loading model {checkpoint_dir}")
    model = load_model(checkpoint_dir).to(device)

    print(f"=> Loading tokenizer {checkpoint_dir}")
    tokenizer = load_tokenizer(checkpoint_dir)
    app.config.update(
        dict(
            model=model,
            tokenizer=tokenizer,
            device=device,
        )
    )


def setup(cli_checkpoint_dir=None):
    checkpoint_dir = os.environ.get("CHECKPOINT_DIR") or cli_checkpoint_dir
    if not checkpoint_dir:
        raise ValueError("Set --chekpoint-dir or CHECKPOINT_DIR")
    initialize(checkpoint_dir)
    return app


@click.command()
@click.option("--debug", "-d", is_flag=True)
@click.option("--checkpoint-dir", "-cp", required=True)
def main(debug, checkpoint_dir):
    app = setup(cli_checkpoint_dir=checkpoint_dir)
    app.run(debug=debug, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))


if __name__ == "__main__":
    main()
