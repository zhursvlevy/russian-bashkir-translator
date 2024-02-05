import sentencepiece as spm
from pathlib import Path
import click


@click.command()
@click.option("--root", "-r", help="root corpora directory")
@click.option("--subdirectory", "-s", help="directory with sentence-per-row .txt files", multiple=True)
@click.option("--model-prefix", "-p", help="name of the model")
@click.option("--vocab-size", "-v", default=32000, help="vocabulary size")
def main(root: str, subdirectory: str, model_prefix: str, vocab_size: int = 32000):
    root = Path(root)
    Path(model_prefix).parent.mkdir(exist_ok=True, parents=True)
    texts = []
    for subdir in subdirectory:
        texts.extend(list((root / subdir).rglob("*.txt")))
    spm.SentencePieceTrainer.train(
        input=texts, 
        model_prefix=model_prefix, 
        vocab_size=vocab_size)
    

if __name__ == "__main__":
    main()