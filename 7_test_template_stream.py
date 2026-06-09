from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2, streaming=True)

prompt = PromptTemplate.from_template(
    "주제: {topic}\n요청: 한국어로 5문장 설명을 작성하세요."
)

chain = prompt | llm | StrOutputParser()

for chunk in chain.stream({"topic": "RAG"}):
    print(chunk, end="", flush=True)

print()
