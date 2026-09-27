import os
from pathlib import Path
import streamlit as st
import joblib
from dotenv import load_dotenv
from groq import Groq


# --------------------------------------------------
# Configuration
# --------------------------------------------------

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "model" / "multinomial_nb_model.pkl"

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    st.error("GROQ_API_KEY not found in .env file.")
    st.stop()


# --------------------------------------------------
# Load ML Model
# --------------------------------------------------

@st.cache_resource
def load_classifier():
    return joblib.load(MODEL_PATH)


model = load_classifier()


# --------------------------------------------------
# Groq Client
# --------------------------------------------------

client = Groq(api_key=GROQ_API_KEY)


# --------------------------------------------------
# Generate Smart Reply
# --------------------------------------------------

def generate_smart_reply(ticket_description, predicted_category):

    prompt = f"""
You are a professional customer support assistant.

Generate a short, helpful, and professional response to the
customer's support ticket.

Customer ticket:
{ticket_description}

Predicted category:
{predicted_category}

Instructions:
- Start the response directly with "Hi," or "Hello," without any placeholders, bracketed text, or customer names (e.g. do NOT use "Hi [Customer Name]").
- Address the customer's issue directly.
- Give practical next steps when appropriate.
- Do not invent company policies, refunds, account details, or technical facts that were not provided.
- Do not mention that an AI generated the response.
- Keep the response concise and customer-friendly.
- Do not include a subject line.
"""

    response = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a professional customer support assistant. "
                    "Provide concise, helpful and safe customer responses."
                )
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0.3,
        max_tokens=200
    )

    return response.choices[0].message.content.strip()


# --------------------------------------------------
# Streamlit UI
# --------------------------------------------------

st.set_page_config(
    page_title="Customer Support Classifier",
    page_icon="🎫",
    layout="centered"
)


st.title("🎫 Customer Support Classifier")

st.markdown("---")

st.subheader("Ticket Description")

ticket_description = st.text_area(
    "Enter the customer's support ticket:",
    placeholder="Example: The application is very slow today.",
    height=120
)


if st.button("Predict", type="primary"):

    if not ticket_description.strip():
        st.warning("Please enter a ticket description.")
        st.stop()

    # ----------------------------------------------
    # ML Prediction
    # ----------------------------------------------

    prediction = model.predict([ticket_description])[0]

    # Get confidence if classifier supports probabilities
    try:
        probabilities = model.predict_proba([ticket_description])
        confidence = probabilities.max() * 100
    except AttributeError:
        confidence = None

    st.markdown("---")

    st.subheader("Prediction")

    st.markdown(
        f"**Predicted Category:** `{prediction}`"
    )

    if confidence is not None:
        st.markdown(
            f"**Confidence:** `{confidence:.1f}%`"
        )

    # ----------------------------------------------
    # Groq Smart Reply
    # ----------------------------------------------

    st.markdown("---")

    st.subheader("🤖 LLM Reply")

    with st.spinner("Generating smart support response..."):

        try:
            reply = generate_smart_reply(
                ticket_description,
                prediction
            )

            st.success(reply)

        except Exception as e:
            st.error(
                f"Unable to generate LLM response: {e}"
            )