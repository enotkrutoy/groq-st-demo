"""Groq Agent"""
import time
import json
from datetime import datetime

import gspread
import streamlit as st
from google.oauth2 import service_account
from langchain import hub
from langchain_groq import ChatGroQ
from langchain.agents import AgentExecutor, create_react_agent
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.callbacks import StreamlitCallbackHandler
from openai import OpenAI

# Получение API-ключа из секретов
groq_api_key = st.secrets.get("openai", {}).get("groq_api_key", None)
if not groq_api_key:
    st.error("Пожалуйста, добавьте свой API-ключ Groq в файл secrets.toml.")
    st.stop()

# Инициализация клиента Groq
client = OpenAI(
    api_key=groq_api_key,
    base_url="https://api.groq.com/openai/v1/",
)

def execute_search_agent(query):
    """Execute the search agent"""
    # Определение модели Groq LLM
    llm = ChatGroQ(
        temperature=0,
        model_name="qwen-2.5-32b"
    )

    # Инструменты веб-поиска
    tools = [TavilySearchResults(max_results=3)]

    # Получение промпта из LangChain Hub
    react_prompt = hub.pull("hwchase17/react")

    # Создание React-агента
    agent = create_react_agent(llm, tools, react_prompt)

    # Создание исполнителя агента
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

    # Обработка запроса
    st_callback = StreamlitCallbackHandler(st.container())
    results = agent_executor.invoke({"input": query}, {"callbacks": [st_callback]})

    return results

def check_text(text):
    """Проверка текста на соответствие"""
    response = client.moderations.create(
        body=text
    )
    return response.results[0].flagged

def is_fake_question(text):
    """Проверка, является ли текст фактическим вопросом"""
    response = client.chat.completions.create(
        model="deepseek-r1-distill-llama-70b",
        messages=[
            {"role": "system", "content": "является ли данный текст фактическим вопросом? если да, верните 1, иначе верните 0"},
            {"role": "user", "content": text}
        ],
        max_tokens=1,
        temperature=0,
        seed=0,
        logit_bias={"15": 100, "16": 100}
    )

    result = int(response.choices[0].message.content)
    return 0 if result == 1 else 1

def append_to_sheet(prompt, generated, answer):
    """Добавление данных в Google Sheets"""
    credentials = service_account.Credentials.from_service_account_info(
        json.loads(st.secrets["gcp_service_account"]),
        scopes=[
            "https://www.googleapis.com/auth/spreadsheets",
            "https://www.googleapis.com/auth/drive"
        ]
    )
    gc = gspread.authorize(credentials)
    sh = gc.open_by_url(st.secrets["private_gsheets_url"])
    worksheet = sh.get_worksheet(0)  # 假设 хотите записать в первый лист
    current_time = datetime.now().strftime("%y-%m-%d %H:%M:%S")
    worksheet.append_row([current_time, prompt, generated, answer])

# Основной интерфейс
st.title("Agents Go Brrrr with Groq")
st.subheader("React Search Agent")
st.write("Powered by Groq, Mixtral, LangChain, and Tavily.")

query = st.text_input("Search query", "Why is Groq so fast?")
button = st.empty()

if button.button("Search"):
    button.empty()
    with st.spinner("Checking your response..."):
        is_nsfw = check_text(query)
        # is_fake_qn = is_fake_question(query)
        # if is_nsfw or is_fake_qn:
        if is_nsfw:
            st.warning("Ваш запрос был помечен. Пожалуйста, обновите страницу, чтобы попробовать еще раз.", icon="🚫")
            append_to_sheet(query, False, "nil")
            st.stop()

    start_time = time.time()
    st_callback = StreamlitCallbackHandler(st.container())
    try:
        results = execute_search_agent(query)
    except ValueError:
        st.error("An error occurred while processing your request. Please refresh the page to try again.")
        st.stop()

    st.success(f"Completed in {round(time.time() - start_time, 2)} seconds.")
    st.info(f"""### Qn: {results['input']}

Ans: {results['output']}""")
    append_to_sheet(results['input'], True, results['output'])
    st.info("Please refresh the page to try a new query.")
