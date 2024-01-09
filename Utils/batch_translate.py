import pandas as pd
from tqdm import tqdm

from deep_translator import GoogleTranslator
# https://deep-translator.readthedocs.io/en/latest/index.html

def translate_text(text, target_language='sv'):
    translation = GoogleTranslator(target=target_language).translate(text)
    return translation

def batch_translate_csv(input_csv, output_csv, col='text'):
    df = pd.read_csv(input_csv)

    if col not in df.columns:
        raise ValueError(f"'{col}' not found.")

    for i in tqdm(range(len(df)), desc='Translating rows'):
        row = df.iloc[i]
        row[col] = translate_text(row[col])

        row.to_frame().T.to_csv(output_csv, mode='a', header=not i, index=False)

if __name__ == "__main__":
    batch_translate_csv('./Labb/Data/emotions.csv', './Labb/Data/sv.csv', col='text')
