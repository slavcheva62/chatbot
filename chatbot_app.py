# -*- coding: utf-8 -*-
import traceback
import datetime

def log(msg):
    with open("log.txt", "a", encoding="utf-8") as f:
        f.write(f"[{datetime.datetime.now().isoformat()}] {msg}\n")

log("🔄 Стартиране на чатбота...")

try:
    import pickle
    import numpy as np
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    log("✅ Успешно заредени библиотеки.")
except Exception as e:
    log("❌ Грешка при зареждане на библиотеки:")
    log(traceback.format_exc())
    raise

try:
    log("📂 Зареждам текстовете от joomla_clean_chunks.txt...")
    with open("joomla_clean_chunks.txt", "r", encoding="utf-8") as f:
        chunks = [line.strip() for line in f if line.strip()]
    log(f"✅ Заредени {len(chunks)} текста.")
except Exception as e:
    log("❌ Грешка при зареждане на текстовете:")
    log(traceback.format_exc())
    raise

try:
    log("⚙️ Векторизиране...")
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(chunks)
    log("✅ Векторизацията е готова.")
except Exception as e:
    log("❌ Грешка при векторизацията:")
    log(traceback.format_exc())
    raise

try:
    question = "Какво е HTML?"
    log(f"❓ Въпрос: {question}")
    question_vec = vectorizer.transform([question])
    similarities = cosine_similarity(question_vec, X)
    best_idx = similarities.argmax()
    answer = chunks[best_idx]
    log(f"💡 Отговор: {answer}")
except Exception as e:
    log("❌ Грешка при намиране на отговор:")
    log(traceback.format_exc())
    raise

log("🏁 Скриптът приключи успешно.")
