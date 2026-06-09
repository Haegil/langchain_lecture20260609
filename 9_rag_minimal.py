from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

# 환경 변수 로드
load_dotenv()

# LLM 및 임베딩 모델
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)
embeddings = OpenAIEmbeddings()

# 문서 로드
with open("data/sample.txt", "r", encoding="utf-8") as f:
    text = f.read()

# 문서 분할
splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=30)
docs = splitter.create_documents([text])

# 벡터 저장소 생성
vectorstore = FAISS.from_documents(docs, embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

# RAG 프롬프트
prompt = ChatPromptTemplate.from_messages([
    ("system", "다음 문서 내용을 근거로만 답변합니다. 문서에 없으면 '문서 근거가 부족합니다'라고 답변합니다."),
    ("human", "질문: {question}\n\n문서:\n{context}")
])

parser = StrOutputParser()

# 실행
question = input("질문 입력: ").strip()
docs = retriever.invoke(question)
context = "\n\n".join(d.page_content for d in docs)

chain = prompt | llm | parser
print(chain.invoke({"question": question, "context": context}))
