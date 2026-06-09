from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda
import os

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
model = os.getenv("MODEL_NAME")
llm = ChatOpenAI(api_key=api_key, model=model, temperature=0.2)


# 프롬프트1 (base)
base_prompt = PromptTemplate.from_template(
    "AI 를 주제로 하이쿠를 작성해줘."
)
# 프롬프트2 (style)
style_prompt = PromptTemplate.from_template(
    "3행으로 구성해줘."
)

# 프롬프트3 (combined)
combined_prompt = PromptTemplate.from_template(
    "{base}\n{style}"
)
# 여러 프롬프트를 연결 하기 format(속성)
resultPrompt = combined_prompt.format(base=base_prompt.format(), style=style_prompt.format())

# chain은 체인은 Runnable만으로 구성. 문자열 x
chain = RunnableLambda(lambda _: resultPrompt) | llm | StrOutputParser()
result2 = chain.invoke({})
print(result2)

"""
[PromptTemplate의 메서드 역할]
- 생성: from_template()
- 문자열 변환: format(), 문자열 변환시 템플릿에 변수를 전달 가능.
PromptTemplate은 Runnable 클래스를 상속 받은 Runnable 타입.
다시 말해서 LangChain은 Runnable을 상속 받은 PromptTemplate 사용.
문자열은 체인(|)으로 묶는것 불가능. 
foramt 문자열은 체인에 묶으려면 RunnableLambda로 형 변환해야 한다.
그러나 권장하는 변수 전달 방법은 체인 실행시 invoke({})의 dict 인자로 전달 하는것.
JSON 문자열처럼 Key를 str 타입으로 만들어야 한다.
"""