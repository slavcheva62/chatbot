from flask import Flask, request, render_template_string
import os
import numpy as np
from sentence_transformers import SentenceTransformer, util

app = Flask(__name__)

# Зареждане на въпроси и отговори
QA_FILE = "joomla_qa.txt"
questions = []
answers = []

if os.path.exists(QA_FILE):
    with open(QA_FILE, encoding="utf-8") as f:
        for line in f:
            if "|" in line:
                q, a = line.strip().split("|", 1)
                questions.append(q)
                answers.append(a)

    model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
    question_embeddings = model.encode(questions, convert_to_tensor=True)
else:
    model = None
    question_embeddings = None

@app.route("/", methods=["GET", "POST"])
def home():
    response = ""
    if request.method == "POST":
        user_question = request.form.get("question", "").strip()
        if model and question_embeddings is not None and user_question:
            user_embedding = model.encode(user_question, convert_to_tensor=True)
            similarities = util.pytorch_cos_sim(user_embedding, question_embeddings)[0]
            best_idx = int(similarities.argmax())
            response = answers[best_idx]
        else:
            response = "Няма заредени въпроси и отговори или въпросът е празен."

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
    return render_template_string(html, answer=response)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
