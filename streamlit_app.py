import requests
import streamlit as st

API_URL = "http://localhost:8000/predict"

st.set_page_config(page_title="QA Next-Word Predictor", page_icon="🤖")

st.title("🤖 QA Next-Word Predictor")
st.markdown(
    """
This is a demo of a simple **RNN-based next-word prediction model**
trained on a small (100 unique) Question-Answer dataset.

⚠️ Note: the model currently predicts only the **first word** of the answer,
not a full multi-word response — so keep expectations modest, this is a
learning project, not a production chatbot.
"""
)

st.markdown(
    "📂 [Download the dataset the model was trained on](https://drive.google.com/file/d/1v7mHRpFQH5qrFivyKQG0iRzJrLZ32jg_/view?usp=sharing)"
)

SAMPLE_QUESTIONS = [
    "What is the capital of France?",
    "What is the capital of Germany?",
    "Who wrote 'To Kill a Mockingbird'?",
    "What is the largest planet in our solar system?",
    "What is the boiling point of water in Celsius?",
    "Who painted the Mona Lisa?",
    "What is the square root of 64?",
    "What is the chemical symbol for gold?",
    "Which year did World War II end?",
    "What is the longest river in the world?",
    "What is the capital of Japan?",
    "Who developed the theory of relativity?",
    "What is the freezing point of water in Fahrenheit?",
    "Which planet is known as the Red Planet?",
    "Who is the author of '1984'?",
    "What is the currency of the United Kingdom?",
    "What is the capital of India?",
    "Who discovered gravity?",
    "How many continents are there on Earth?",
    "Which gas do plants use for photosynthesis?",
    "What is the smallest prime number?",
    "Who invented the telephone?",
    "What is the capital of Australia?",
    "Which ocean is the largest?",
    "What is the speed of light in vacuum?",
    "Which language is spoken in Brazil?",
    "Who discovered penicillin?",
    "What is the capital of Canada?",
    "What is the largest mammal on Earth?",
    "Which element has the atomic number 1?",
    "What is the tallest mountain in the world?",
    "Which city is known as the Big Apple?",
    "How many planets are in the Solar System?",
    "Who painted 'Starry Night'?",
    "What is the chemical formula of water?",
    "What is the capital of Italy?",
    "Which country is famous for sushi?",
    "Who was the first person to step on the Moon?",
    "What is the main ingredient in guacamole?",
    "How many sides does a hexagon have?",
    "What is the currency of China?",
    "Who wrote 'Pride and Prejudice'?",
    "What is the chemical symbol for iron?",
    "What is the hardest natural substance on Earth?",
    "Which continent is the largest by area?",
    "Who was the first President of the United States?",
    "Which bird is known for its ability to mimic sounds?",
    "What is the longest-running animated TV show?",
    "What is the smallest country in the world?",
    "Which planet has the most moons?",
    "Who wrote 'Romeo and Juliet'?",
    "What is the main gas in Earth's atmosphere?",
    "How many bones are in the adult human body?",
    "Which metal is a liquid at room temperature?",
    "What is the capital of Russia?",
    "Who discovered electricity?",
    "Which is the second-largest country by land area?",
    "What is the color of a ripe banana?",
    "Which month has 28 days in a common year?",
    "What is the study of living organisms called?",
    "Which country is home to the Great Wall?",
    "What do bees collect from flowers?",
    "What is the opposite of 'day'?",
    "What is the capital of South Korea?",
    "Who invented the light bulb?",
    "Which gas do humans breathe in for survival?",
    "What is the square root of 144?",
    "Which country has the pyramids of Giza?",
    "Which sea creature has eight arms?",
    "Which holiday is celebrated on December 25?",
    "What is the currency of Japan?",
    "How many legs does a spider have?",
    "Which sport uses a net, ball, and hoop?",
    "Which country is famous for its kangaroos?",
    "Who was the first female Prime Minister of the UK?",
    "Which is the fastest land animal?",
    "What is the first element on the periodic table?",
    "What is the capital of Spain?",
    "Which planet is the closest to the Sun?",
    "Who is known as the father of computers?",
    "What is the capital of Mexico?",
    "How many colors are in a rainbow?",
    "Which musical instrument has black and white keys?",
    "Who discovered the Americas in 1492?",
    "Which Disney character has a long nose and grows it when lying?",
    "Who directed the movie 'Titanic'?",
    "Which superhero is also known as the Dark Knight?",
    "What is the capital of Brazil?",
    "Which fruit is known as the king of fruits?",
    "Which country is known for the Eiffel Tower?",
]

with st.expander("💡 Sample questions to try"):
    for q in SAMPLE_QUESTIONS:
        st.markdown(f"- {q}")

question = st.text_input("Ask a question:", placeholder="e.g. What is the largest planet in our solar system?")

threshold = st.slider("Confidence threshold", 0.0, 1.0, 0.5, 0.05)

if st.button("Predict") and question.strip():
    try:
        response = requests.post(
            API_URL,
            json={"question": question, "threshold": threshold},
            timeout=5,
        )
        response.raise_for_status()
        result = response.json()

        st.success(f"**Predicted next word:** {result['answer']}")
        st.caption(f"Confidence: {result['confidence']:.2%}")

    except requests.exceptions.ConnectionError:
        st.error("Could not connect to the API. Make sure it's running: `uvicorn api:app --reload --port 8000`")
    except Exception as e:
        st.error(f"Something went wrong: {e}")