import os
import streamlit as st
from dotenv import load_dotenv
from agents import Agent, Runner

# 1. Load Environment Variables (.env)
load_dotenv()
env_api_key = os.getenv("OPENROUTER_API_KEY", "") or os.getenv("OPENAI_API_KEY", "")

# 2. Page Configuration
st.set_page_config(
    page_title="Royal Feast Hotel & Restaurant",
    page_icon="🍽️",
    layout="centered"
)

st.title("🍽️ Royal Feast Hotel & Restaurant")
st.subheader("Customer Service Desk (Powered by OpenAI-Agents SDK)")

# =========================================================================
# 👇 DEFAULT MODEL SETTING
DEFAULT_MODEL = "dots-studio/dots-3-note-preview:free"
# =========================================================================

# 3. Sidebar Configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    user_api_key = st.text_input(
        "OpenRouter API Key",
        value=env_api_key,
        type="password",
        help="Loaded automatically from .env if present"
    )

    custom_model = st.text_input(
        "OpenRouter Model Name",
        value=DEFAULT_MODEL,
        help="Paste model ID from openrouter.ai/models"
    )
    
    st.markdown("[Get OpenRouter Key](https://openrouter.ai/keys)")

api_key = user_api_key or env_api_key

# LiteLLM prefix handling
selected_model = custom_model.strip()
if not selected_model.startswith("openai/"):
    selected_model = f"openai/{selected_model}"

# 4. Strict System Instructions
RESTAURANT_INSTRUCTIONS = """
You are 'Aria', the Customer Lead Assistant for 'Royal Feast Hotel & Restaurant'.

STRICT GREETING RULES:
1. If the user says Salam / Asalamualikum:
   - First reply with: "Walaikum Assalam!"
   - Then immediately follow with: "WELCOME TO Royal Feast Hotel & Restaurant! How can I assist you today?"

2. If the user says Hi / Hello / Hey / Good Morning / Good Evening:
   - First reply with a friendly greeting: "Hello!" or "Hi there!"
   - Then immediately follow with: "WELCOME TO Royal Feast Hotel & Restaurant! How can I assist you today?"

3. If the user starts directly with a question without greeting:
   - Answer their question directly, concisely, and politely.

4. DO NOT dump the full menu or room prices on initial greetings! Wait until the user asks a specific question.

HOTEL & RESTAURANT KNOWLEDGE BASE (Use ONLY when customer asks):
- Breakfast (7:00 AM - 11:30 AM): Halwa Puri Thali ($8), Omelette Paratha ($6), Tea/Coffee ($3)
- Main Menu (12:00 PM - 11:30 PM): Chicken Biryani ($12), Mutton Karahi ($25), Butter Chicken ($15), Beef Burger ($10), Pizza ($18)
- Rooms: Deluxe Suite ($100/night), Executive Suite ($180/night)
- Services: 24/7 Room Service & Delivery
- Location: 45 Grand Avenue, City Center
"""

# 5. Session State
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display Past Chat History
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# 6. Normal Synchronous Function (No Async/Await)
def run_food_agent(user_prompt: str, key: str, model_name: str):
    os.environ["OPENAI_API_KEY"] = key
    os.environ["OPENAI_BASE_URL"] = "https://openrouter.ai/api/v1"

    food_lead_agent = Agent(
        name="Food & Hotel Lead Agent",
        instructions=RESTAURANT_INSTRUCTIONS,
        model=model_name
    )

    # Runner.run_sync() is used for synchronous execution
    result = Runner.run_sync(food_lead_agent, input=user_prompt)
    return result.final_output

# 7. User Input Logic
user_input = st.chat_input("Type your message (e.g., Salam, Hi, or What is on the menu?)...")

if user_input:
    if not api_key:
        st.error("⚠️ Please enter your OpenRouter API Key in the sidebar or .env file.")
    else:
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            with st.spinner("Aria is replying..."):
                try:
                    # Direct normal function call
                    response_text = run_food_agent(user_input, api_key, selected_model)
                    st.markdown(response_text)
                    st.session_state.messages.append({"role": "assistant", "content": response_text})
                except Exception as e:
                    st.error(f"Error: {e}")