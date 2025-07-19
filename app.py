from flask import Flask, request, render_template_string
import os
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# Зареждане на текстовете
TEXT_FILE = "joomla_clean_chunks.txt"
if os.path.exists(TEXT_FILE):
    with open(TEXT_FILE, encoding="utf-8") as f:
        texts = [line.strip() for line in f if line.strip()]
    vectorizer = TfidfVectorizer().fit(texts)
    vectors = vectorizer.transform(texts)
else:
    texts = []
    vectors = None

@app.route("/", methods=["GET", "POST"])
def home():
    answer = ""
    if request.method == "POST":
        user_question = request.form.get("question", "")
        if user_question.strip() and vectors is not None:
            question_vec = vectorizer.transform([user_question])
            similarity = cosine_similarity(question_vec, vectors)
            best_idx = np.argmax(similarity)
            answer = texts[best_idx]
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
        <div style="border: 1px solid #ccc; padding: 10px;">{{ answer }}</div>
    {% endif %}
    """
    return render_template_string(html, answer=answer)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
