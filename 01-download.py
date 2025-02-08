import requests
import datasets


# Loads the MS MARCO dataset
ds = datasets.load_dataset('microsoft/ms_marco', 'v1.1') # Loads the MS MARCO dataset

docs = [passages
        for splits in ds
        for keys in ds[splits]
        for passages in keys['passages']['passage_text']] # Produces a list of all Passage Texts

qrys = [q
            for key in ds.keys()
            for q in ds[key]['query']] # Produces a list of all Queries

with open('./corpus/msmarco.txt', 'w', encoding='utf-8') as f: f.write('\n'.join(set(docs + qrys)))


# Loads the Text8 dataset
r = requests.get('https://huggingface.co/datasets/ardMLX/text8/resolve/main/text8')
with open('./corpus/text8.txt', 'wb') as f: f.write(r.content)

"""
Purpose of this script:

1. Download the MS MARCO dataset
2. Convert into text file of all passages and queries  
3. Download the Text8 dataset
4. Convert into text file of all words
5. Save the datasets to the ./corpus directory

"""