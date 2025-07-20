from flask import Flask, request, render_template_string
import os
from sentence_transformers import SentenceTransformer, util

app = Flask(__name__)

# Зареждане на модел
model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')

# Зареждане и векторизиране на текстовете
TEXT_FILE = "joomla_clean_chunks.txt"
if os.path.exists(TEXT_FILE):
    with open(TEXT_FILE, encoding="utf-8") as f:
        texts = list(set([line.strip() for line in f if line.strip()]))
    embeddings = model.encode(texts, convert_to_tensor=True)
else:
    texts = []
    embeddings = None

@app.route("/", methods=["GET", "POST"])
def home():
    answer = ""
    if request.method == "POST":
        user_question = request.form.get("question", "")
        if user_question.strip() and embeddings is not None:
            question_embedding = model.encode(user_question, convert_to_tensor=True)

            # Намиране на най-близките отговори
            cosine_scores = util.cos_sim(question_embedding, embeddings)[0]

            top_k = 3
            top_results = zip(cosine_scores.tolist(), texts)
            top_results = sorted(top_results, key=lambda x: x[0], reverse=True)[:top_k]

            if top_results[0][0] < 0.3:
                answer = "Извинявай, не можах да намеря подходящ отговор."
            else:
                seen = set()
                selected = []
                for score, text in top_results:
                    if text not in seen:
                        selected.append(text)
                        seen.add(text)
                answer = "\n\n".join(selected)
        else:
            answer = "Няма заредени текстове или въпросът е празен."

    html = """
    <h2>Чатбот Joomla</h2>
    <form method="post">
        <label>Питай чатбота:</label><br>
        <input name="question" style="width: 400px;" /><br><br>
        <input type="submit" value="Питай" />
    </form>
    {% if answer %}
        <h3>Отговор:</h3>
        <div style="border: 1px solid #ccc; padding: 10px; white-space: pre-wrap;">{{ answer }}</div>
    {% endif %}
    """
    return render_template_string(html, answer=answer)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
