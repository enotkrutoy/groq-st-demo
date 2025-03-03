"""
app.py
"""
import time
import os

import streamlit as st
from groq import groq

from self_discover import (
    reasoning_modules,
    select_reasoning_modules,
    adapt_reasoning_modules,
    implement_reasoning_structure,
    execute_reasoning_structure
)

GROQ_API_KEY = st.secrets.get("openai", {}).get("GROQ_API_KEY", None)
if not GROQ_API_KEY:
    st.error("Please add your Groq API key to the Streamlit secrets.toml file.")
    st.stop()

# Initialize Groq client
client = openai.OpenAI(
    api_key=GROQ_API_KEY,
    base_url="https://api.groq.com/openai/v1",
)


st.set_page_config(page_title="groq demo", page_icon="⚡️", layout="wide")

st.title("groq demo")

tab1, tab2 = st.tabs(["генерация текста", "самопознание"])

with tab1:
    system_prompt = st.text_input("system prompt", "вы - являетесь экспертом для выполнения различных задач!")
    user_prompt = st.text_input("user prompt", "сформируйте ответ с учетом context, примерами кода и пояснениями.")
    context = st.text_input("context", "https://console.groq.com/docs/autogen")
    model_list = ["qwen-2.5-32b", "deepseek-r1-distill-llama-70b"]

    model = st.radio("select the llm", model_list, horizontal=True)

    button = st.empty()
    time_taken = st.empty()
    response = st.empty()

    if button.button("generate"):
        stream = client.chat.completions.create(
            model=model,
            messages=[{"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}],
            stream=True
        )

        start_time = time.time()

        streamed_text = ""

        for chunk in stream:
            chunk_content = chunk.choices[0].delta.content
            if chunk_content is not None:
                streamed_text += chunk_content
                response.info(streamed_text)

        time_taken.success(f"Время выполнения: {round(time.time() - start_time, 4)} секунд")

with tab2:
    st.info("""['самопознание: крупные языковые модели самостоятельные структуры рассуждений'](https://arxiv.org/abs/2402.03620) - zhou et al (2024)

this is based on mrcatid's [implementation](https://github.com/enotkrutoy/groq-st-demo/blob/main/self_discover.py). 
credits to [martin a](https://twitter.com/) for sharing this at [ml singapore](https://www.meetup.com/machine-learning-singapore/)
    """)
    task = st.text_input("какова ваша задача?")
    reasoning_model = st.radio("выберите свой llm для этой задачи рассуждения", model_list, horizontal=True, help="mixtral is recommended for better performance, but appears to be slower")

    button = st.empty()
    step4 = st.empty()
    step3 = st.empty() 
    step2 = st.empty()
    step1 = st.empty()

    if button.button("run"):
        prompt = select_reasoning_modules(reasoning_modules, task)
        select_reasoning_modules = ""
        stream_1 = client.chat.completions.create(
            model=reasoning_model,
            messages=[{"role": "system", "content": "вы - мировой эксперт в области рассуждений."},
                    {"role": "user", "content": prompt}],
            stream=True
        )

        for chunk in stream_1:
            chunk_content = chunk.choices[0].delta.content
            if chunk_content is not None:
                select_reasoning_modules += chunk_content
                step1.info("# шаг 1: выберите релевантные модули рассуждений для задачи \n \n" + select_reasoning_modules)

        prompt = adapt_reasoning_modules(select_reasoning_modules, task)
        stream_2 = client.chat.completions.create(
            model=reasoning_model,
            messages=[{"role": "system", "content": "вы - мировой эксперт в области рассуждений."},
                    {"role": "user", "content": prompt}],
            stream=True
        )

        adapted_modules = ""
        for chunk in stream_2:
            chunk_content = chunk.choices[0].delta.content
            if chunk_content is not None:
                adapted_modules += chunk_content
                step2.info("# шаг 2: адаптируйте выбранные модули рассуждений к конкретной задаче. \n \n " + adapted_modules)

        prompt = implement_reasoning_structure(adapted_modules, task)
        stream_3 = client.chat.completions.create(
            model=reasoning_model,
            messages=[{"role": "system", "content": "вы - мировой эксперт в области рассуждений."},
                    {"role": "user", "content": prompt}],
            stream=True
        )

        reasoning_structure = ""
        for chunk in stream_3:
            chunk_content = chunk.choices[0].delta.content
            if chunk_content is not None:
                reasoning_structure += chunk_content
                step3.info("# шаг 3: реализуйте адаптированные модули рассуждений в структуру действий. \n \n " + reasoning_structure)

        prompt = execute_reasoning_structure(reasoning_structure, task)
        stream_4 = client.chat.completions.create(
            model=reasoning_model,
            messages=[{"role": "system", "content": "вы - мировой эксперт в области рассуждений."},
                    {"role": "user", "content": prompt}],
            stream=True
        )

        result = ""
        for chunk in stream_4:
            chunk_content = chunk.choices[0].delta.content
            if chunk_content is not None:
                result += chunk_content
                step4.info("# шаг 4: выполните структуру рассуждений для решения конкретной задачи. \n \n " + result)
