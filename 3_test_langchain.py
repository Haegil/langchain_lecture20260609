from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

import os

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
print(api_key[:20])
model = os.getenv("MODEL_NAME")
print(model)

llm = ChatOpenAI(model=model, temperature=0.2, api_key=api_key)

prompt = PromptTemplate.from_template(
    "AI를 주제로 한 한국어 하이쿠를 작성하세요."
)

chain = prompt | llm | StrOutputParser()

result = chain.invoke({})
print(result)
