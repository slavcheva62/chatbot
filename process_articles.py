# -*- coding: utf-8 -*-

import pandas as pd
import re
import pickle

def clean_html(text):
    return re.sub(r'<.*?>', '', str(text))

def split_text(text, max_len=1000):
    chunks = []
    start = 0
    while start < len(text):
        end = start + max_len
        end = text.rfind(' ', start, end)
        if end == -1 or end <= start:
            end = start + max_len
        chunks.append(text[start:end].strip())
        start = end
    return chunks

# Зареждане на CSV файла с Joomla статиите (трябва да е качен предварително)
df = pd.read_csv('jos_content.csv')

all_chunks = []

for _, row in df.iterrows():
    intro = clean_html(row.get('introtext', ''))
    full = clean_html(row.get('fulltext', ''))
    combined = (intro + ' ' + full).strip()
    if combined:
        chunks = split_text(combined)
        all_chunks.extend(chunks)

# Запис във текстов файл
with open('joomla_clean_chunks.txt', 'w', encoding='utf-8') as f:
    for chunk in all_chunks:
        f.write(chunk + '\n')

# Ново: запис в pickle файл за чатбота
with open('chunks.pkl', 'wb') as f:
    pickle.dump(all_chunks, f)

print(f'Done. {len(all_chunks)} text chunks saved in text and pickle files.')
