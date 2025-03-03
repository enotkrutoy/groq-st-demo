"""
Groq Agent
"""
import time
import json
from datetime import datetime

import gspread
import streamlit as st
from google.oauth2 import service_account
from langchain import hub
from langchain_groq import ChatGroq
from langchain.agents import AgentExecutor, create_react_agent
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.callbacks import StreamlitCallbackHandler
from openai import OpenAI

#client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
GROQ_API_KEY = st.secrets.get("openai", {}).get("GROQ_API_KEY", None)
if not GROQ_API_KEY:
    st.error("Please add your Groq API key to the Streamlit secrets.toml file.")
    st.stop()

# Initialize Groq client
client = openai.OpenAI(
    api_key=GROQ_API_KEY,
    base_url="https://api.groq.com/openai/v1",
)
def execute_search_agent(query):
    """
    Execute the search agent
    """
    # Define Groq LLM Model
    llm = ChatGroq(temperature=0,
                groq_api_key=st.secrets["GROQ_API_KEY"],
                model_name="qwen-2.5-32b")

    # Web Search Tool
    tools = [TavilySearchResults(max_results=3)]

    # Pull prompt from LangChain Hub
    react_prompt = hub.pull("hwchase17/react")

    # Construct the ReAct agent
    agent = create_react_agent(llm, tools, react_prompt)

    # Create an agent executor by passing in the agent and tools
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

    return agent_executor.invoke({"input": query}, 
                                 {"callbacks": [st_callback]})

def check_text(text):
    """
    Проверьте текст
    """
    response = client.moderations.create(input=text)
    return response.results[0].flagged

def is_fake_question(text):
    """Check if the given text is safe for work using gpt4 zero-shot classifer."""
    response = client.chat.completions.create(
        model="deepseek-r1-distill-llama-70b",
        messages=[{"role": "system", "content": "Является ли данный текст фактический вопрос? Если да, верните `1`, иначе верните `0`"},
                  {"role": "user", "content": text}],
        max_tokens=1,
        temperature=0,
        seed=0,
        logit_bias={"15": 100,
                    "16": 100}
    )

    result = int(response.choices[0].message.content)

    if result == 1:
        return 0
    return 1

def append_to_sheet(prompt, generated, answer):
    """
    Add to GSheet
    """
    credentials = service_account.Credentials.from_service_account_info(
        json.loads(st.secrets["GCP_SERVICE_ACCOUNT"]),
        scopes=["https://www.googleapis.com/auth/spreadsheets", "https://www.googleapis.com/auth/drive"]
    )
    gc = gspread.authorize(credentials)
    sh = gc.open_by_url(st.secrets["PRIVATE_GSHEETS_URL"])
    worksheet = sh.get_worksheet(0) # Assuming you want to write to the first sheet
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    worksheet.append_row([current_time, prompt, generated, answer])

st.title("agents go brrrr with groq")
st.subheader("ReAct Search Agent")
st.write("powered by Groq, Mixtral, LangChain, and Tavily.")
query = st.text_input("Search Query", "Why is Groq so fast?")
button = st.empty()

if button.button("Search"):
    button.empty()
    with st.spinner("Checking your response..."):
        is_nsfw = check_text(query)
        # is_fake_qn = is_fake_question(query)
    # if is_nsfw or is_fake_qn:
    if is_nsfw:
        st.warning("Ваш запрос был помечен. Пожалуйста, обновите страницу, чтобы попробовать еще раз.", icon="🚫")
        append_to_sheet(query, False, "NIL")
        st.stop()
    
    start_time = time.time()
    st_callback = StreamlitCallbackHandler(st.container())
    try:
        results = execute_search_agent(query)
    except ValueError:
        st.error("An error occurred while processing your request. Please refresh the page to try again.")
        st.stop()
    st.success(f"Completed in {round(time.time() - start_time, 2)} seconds.")
    st.info(f""" ### QN: {results['input']}

**ANS:** {results['output']}""")
    append_to_sheet(results['input'], True, results['output'])
    st.info("Please refresh the page to try a new query.")
