"""
app.py
"""
import time
import os

import streamlit as st
from groq import Groq

from self_discover import (
    REASONING_MODULES,
    select_reasoning_modules,
    adapt_reasoning_modules,
    implement_reasoning_structure,
    execute_reasoning_structure
    )

API_KEY = os.environ.get("GROQ_API_KEY")
if API_KEY is None:
    try:
        API_KEY = st.secrets["GROQ_API_KEY"]
    except KeyError:
        # Handle the case where GROQ_API_KEY is neither in the environment variables nor in Streamlit secrets
        st.error("API key not found.")
        st.stop()

client = Groq(api_key=API_KEY)

st.set_page_config(page_title="Groq Demo", page_icon="⚡️", layout="wide")

st.title("Groq Demo")

tab1, tab2 = st.tabs(["Генерация текста", "Самопознание"])

with tab1:
   system_prompt = st.text_input("system prompt", "Вы - являетесь экспертом для выполнения различных задач!")
   user_prompt = st.text_input("user prompt", "Сформируйте ответ с учетом context, примерами кода и пояснениями.")
   context = st.text_input("context", "https://console.groq.com/docs/autogen")
    # model_list = client.models.list().data
    # model_list = [model.id for model in model_list]
    model_list = ["qwen-2.5-32b", "deepseek-r1-distill-llama-70b"]

    model = st.radio("Select the LLM", model_list, horizontal=True)

    button = st.empty()
    time_taken = st.empty()
    response = st.empty()

    if button.button("Generate"):
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
                streamed_text = streamed_text + chunk_content
                response.info(streamed_text)

        time_taken.success(f"Time taken: {round(time.time() - start_time,4)} seconds")

with tab2:
    st.info("""['Самопознание: крупные языковые модели самостоятельные структуры рассуждений'](https://arxiv.org/abs/2402.03620) - Zhou et al (2024)
    
This is based on MrCatID's [implementation](https://github.com/enotkrutoy/groq-st-demo/blob/main/self_discover.py). 
Credits to [Martin A](https://twitter.com/) for sharing this at [ML Singapore](https://www.meetup.com/machine-learning-singapore/)
    """)
    task = st.text_input("Какова ваша задача?")
    reasoning_model = st.radio("Выберите свой LLM для этой задачи рассуждения", model_list, horizontal=True, help="mixtral is recommended for better performance, but appears to be slower")

    button = st.empty()
    step4 = st.empty()
    step3 = st.empty() 
    step2 = st.empty()
    step1 = st.empty()

    if button.button("Run"):

        prompt = select_reasoning_modules(REASONING_MODULES, task)
        select_reasoning_modules = ""
        stream_1 = client.chat.completions.create(
            model=reasoning_model,
            messages=[{"role": "system", "content": "Вы - мировой эксперт в области рассуждений."},
                    {"role": "user", "content": prompt}],
            stream=True
        )
        
        for chunk in stream_1:
            chunk_content = chunk.choices[0].delta.content
            if chunk_content is not None:
                select_reasoning_modules = select_reasoning_modules + chunk_content
                step1.info("# Step 1: Выберите релевантные модули рассуждений для задачи \n \n"+select_reasoning_modules)
        

        prompt = adapt_reasoning_modules(select_reasoning_modules, task)
        stream_2 = client.chat.completions.create(
            model=reasoning_model,
            messages=[{"role": "system", "content": "Вы - мировой эксперт в области рассуждений."},
                    {"role": "user", "content": prompt}],
            stream=True
        )

        adapted_modules = ""
        for chunk in stream_2:
            chunk_content = chunk.choices[0].delta.content
            if chunk_content is not None:
                adapted_modules = adapted_modules + chunk_content
                step2.info("# Step 2: Адаптируйте выбранные модули рассуждений к конкретной задаче. \n \n " + adapted_modules)

        prompt = implement_reasoning_structure(adapted_modules, task)
        stream_3 = client.chat.completions.create(
            model=reasoning_model,
            messages=[{"role": "system", "content": "Вы - мировой эксперт в области рассуждений."},
                    {"role": "user", "content": prompt}],
            stream=True
        )

        reasoning_structure = ""
        for chunk in stream_3:
            chunk_content = chunk.choices[0].delta.content
            if chunk_content is not None:
                reasoning_structure = reasoning_structure + chunk_content
                step3.info("# Step 3: Реализуйте адаптированные модули рассуждений в структуру действий. \n \n " + reasoning_structure)

        prompt = execute_reasoning_structure(reasoning_structure, task)
        stream_4 = client.chat.completions.create(
            model=reasoning_model,
            messages=[{"role": "system", "content": "Вы - мировой эксперт в области рассуждений."},
                    {"role": "user", "content": prompt}],
            stream=True
        )

        result = ""
        for chunk in stream_4:
            chunk_content = chunk.choices[0].delta.content
            if chunk_content is not None:
                result = result + chunk_content
                step4.info("# Step 4: Выполните структуру рассуждений для решения конкретной задачи. \n \n " + result)
