from flask import Flask, request, render_template_string
from sentence_transformers import SentenceTransformer, util
import os
import torch

app = Flask(__name__)

# Зареждане на модела (по-малък, подходящ за Render free plan)
model = SentenceTransformer('paraphrase-MiniLM-L3-v2')

# Зареждане на текстовете
TEXT_FILE = "joomla_clean_chunks.txt"
if os.path.exists(TEXT_FILE):
    with open(TEXT_FILE, encoding="utf-8") as f:
        texts = [line.strip() for line in f if line.strip()]
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
            similarities = util.pytorch_cos_sim(question_embedding, embeddings)[0]
            best_idx = int(torch.argmax(similarities))
            answer = texts[best_idx]
        else:
            answer = "Няма заредени текстове или въпросът е празен."

    html = """
    <h2>Чатбот на училището</h2>
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
    app.run(host="0.0.0.0", port=5000, debug=False)
