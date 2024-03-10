import requests
import pandas as pd
from tqdm import tqdm
import click
from pathlib import Path


SERVICE_URL = "https://translate.api.cloud.yandex.net/translate/v2/translate"
IAM_TOKEN = "YOUR_IAM_TOKEN"
FOLDER_ID = "YOUR_FOLDER_ID"

@click.command()
@click.option("--source", "-s", help="source language (must be 'ru' or 'ba')")
@click.option("--target", "-t", help="target language (must be 'ru' or 'ba')")
@click.option("--length", "-l", default=300, help="number of sentences")
@click.option("--input-file", "-i", default="data/dataset/splits/test.parquet", help="file to sample texts")
@click.option("--save_dir", "-l", default="data/dataset/yandex_translate", help="save directory")
def get_translation(source: str, target: str, length: int, input_file: str, save_dir: str):

    test = pd.read_parquet(input_file)
    test = test.sample(length, random_state=42)
    translation = {"source": [], "translate": [], "target": []}
    for i, row in tqdm(test.iterrows()):
        text = row[source]
        body = {
            "targetLanguageCode": target,
            "texts": [text],
            "folderId": FOLDER_ID,
        }

        headers = {
            "Content-Type": "application/json",
            "Authorization": "Bearer {0}".format(IAM_TOKEN)
        }

        response = requests.post(SERVICE_URL,
            json = body,
            headers = headers
        )
        response.content
        translation["source"].append(text)
        translation["translate"].append(response.json()["translations"][0]["text"])
        translation["target"].append(row[target])

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=False)
    pd.DataFrame(translation).to_csv(save_dir / f"{source}_{target}_random_{length}.csv")


if __name__ == "__main__":
    get_translation()    
