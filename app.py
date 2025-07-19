from flask import Flask, request, render_template_string
import datetime

app = Flask(__name__)

# Тук замени с твоята логика за чатбота
def chatbot_answer(question):
    # Това е примерен отговор - после сложи твоя модел
    return f"Отговор на '{question}': Това е примерен отговор."

HTML = """
<!doctype html>
<title>Чатбот Joomla</title>
<h2>Чатбот Joomla</h2>
<form method="POST">
  <label for="question">Питай чатбота:</label><br>
  <input type="text" id="question" name="question" size="50" autofocus required><br><br>
  <input type="submit" value="Попитай">
</form>

{% if answer %}
<hr>
<p><b>Въпрос:</b> {{ question }}</p>
<p><b>Отговор:</b> {{ answer }}</p>
<p><i>Време: {{ time }}</i></p>
{% endif %}
"""

@app.route("/", methods=["GET", "POST"])
def home():
    answer = None
    question = None
    if request.method == "POST":
        question = request.form.get("question", "")
        # Тук викаш своя скрипт/функция, която дава отговор
        answer = chatbot_answer(question)
    return render_template_string(HTML, answer=answer, question=question, time=datetime.datetime.now())

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)
