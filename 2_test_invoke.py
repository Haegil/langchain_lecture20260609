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

llm = ChatOpenAI(api_key=api_key, model=model, temperature=0.2)

print(llm.invoke("AI로 이용한 2행시 만들어줘.").content)