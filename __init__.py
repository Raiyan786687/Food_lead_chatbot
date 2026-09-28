import os
import asyncio
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
st.subheader("AI Customer Lead Assistant (Powered by OpenAI-Agents SDK)")

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
    
    st.divider()
    st.markdown("### 📋 Quick Hotel Menu")
    with st.expander("View Full Menu & Pricing"):
        st.markdown("""
        **🍳 Breakfast (7:00 AM - 11:30 AM):**
        - Halwa Puri Thali ($8)
        - Omelette Paratha ($6)
        - Tea/Coffee ($3)
        
        **🍲 Main Menu (12:00 PM - 11:30 PM):**
        - Chicken Biryani ($12)
        - Mutton Karahi ($25)
        - Butter Chicken ($15)
        - Beef Burger ($10)
        - Pizza ($18)
        
        **🏨 Room Booking:**
        - Deluxe Suite ($100/night)
        - Executive Suite ($180/night)
        """)

api_key = user_api_key or env_api_key
selected_model = custom_model.strip()

# 4. Strict System Instructions (Optimized for Proactive Welcome & UX)
RESTAURANT_INSTRUCTIONS = """
You are 'Manager', the Customer Lead Assistant for 'Royal Feast Hotel & Restaurant'.

STRICT GREETING & WELCOME RULES:
1. If the user greets (e.g., Salam, Asalamualikum, Hi, Hello, Hey, Good Morning, Good Evening):
   - First, reply with appropriate greeting ("Walaikum Assalam!" or "Hello! / Hi there!").
   - Then warmly state: "WELCOME TO **Royal Feast Hotel & Restaurant**! 🍽️✨"
   - Immediately provide a brief, well-formatted breakdown of top categories so the user knows what we offer:
     
     *Sample Greeting Response Structure:*
     "How can I assist you today? Here is how we can help you:
     
     📋 **Our Top Offerings:**
     1. 🍲 **Dining & Food Menu** (Biryani, Karahi, Burgers, Pizza)
     2. 🍳 **Breakfast Specials** (Halwa Puri, Paratha, Tea)
     3. 🏨 **Room Reservations** (Deluxe & Executive Suites)
     4. 🚗 **24/7 Room Service & Home Delivery**
     
     Feel free to ask for prices, details, or place an order!"

2. If the user asks a specific question directly (without greeting):
   - Answer their question directly, accurately, and politely according to the Knowledge Base.

3. KNOWLEDGE BASE (Use when asked):
   - Breakfast: Halwa Puri Thali ($8), Omelette Paratha ($6), Tea/Coffee ($3)
   - Main Menu: Chicken Biryani ($12), Mutton Karahi ($25), Butter Chicken ($15), Beef Burger ($10), Pizza ($18)
   - Rooms: Deluxe Suite ($100/night), Executive Suite ($180/night)
   - Services: 24/7 Room Service & Delivery
   - Location: 45 Grand Avenue, City Center
"""

# 5. Session State Initializations
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display Past Chat History
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# 6. Async Execution Function for Single Model
async def run_food_agent_async(user_prompt: str, key: str, model_name: str):
    os.environ["OPENAI_API_KEY"] = key
    os.environ["OPENAI_BASE_URL"] = "https://openrouter.ai/api/v1"

    formatted_model = model_name.strip()
    if not formatted_model.startswith("openai/"):
        formatted_model = f"openai/{formatted_model}"

    food_lead_agent = Agent(
        name="Food & Hotel Lead Agent",
        instructions=RESTAURANT_INSTRUCTIONS,
        model=formatted_model
    )
    
    result = await Runner.run(food_lead_agent, input=user_prompt)
    return result.final_output

def execute_agent(user_prompt: str, key: str, model_name: str):
    try:
        loop = asyncio.get_event_loop()
        if loop.is_closed():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    return loop.run_until_complete(run_food_agent_async(user_prompt, key, model_name))

# Quick Test Prompt Buttons for Live Demo with Sir
st.markdown("##### 💡 Quick Test Prompts (Click to test):")
col1, col2, col3 = st.columns(3)

prompt_to_submit = None
if col1.button("💬 Send Salam"):
    prompt_to_submit = "Asalamualikum"
if col2.button("🍕 Ask Menu"):
    prompt_to_submit = "What is on the main food menu?"
if col3.button("🏨 Ask Room Rates"):
    prompt_to_submit = "What are the room charges for one night?"

# 7. User Input Logic
user_input = st.chat_input("Type your message (e.g., Salam, Hi, or What is on the menu?)...")

# Priority to button click or text input
final_prompt = prompt_to_submit or user_input

if final_prompt:
    if not api_key:
        st.error("⚠️ Please enter your OpenRouter API Key in the sidebar or .env file.")
    else:
        st.session_state.messages.append({"role": "user", "content": final_prompt})
        with st.chat_message("user"):
            st.markdown(final_prompt)

        with st.chat_message("assistant"):
            with st.spinner("Manager is replying..."):
                try:
                    response_text = execute_agent(final_prompt, api_key, selected_model)
                    st.markdown(response_text)
                    st.session_state.messages.append({"role": "assistant", "content": response_text})
                except Exception as e:
                    st.error(f"Error: {e}")
