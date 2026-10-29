import os
# #Huggingface国内镜像
# os.environ["HF_ENDPOINT"]="http://hf-mirror.com"
# #模型下载到项目内hf_models文件夹
# os.environ["HF-"]
from fastapi import FastAPI
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage,HumanMessage

from fastapi import UploadFile,File
from langchain_community.document_loaders import PyPDFLoader,TextLoader,Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
#from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
from langchain_community.embeddings import HuggingFaceEmbeddings

#加载同目录下的.env文件
load_dotenv("main.env")

app=FastAPI(title="内部文档Agent后端")

#向量库持久化目录
PERSIST_DIR="./chroma_knowledge"
#Embedding模型
embedding=HuggingFaceEmbeddings(
    model_name="shibing624/text2vec-base-chinese",
    model_kwargs={"device":"cpu"},
    encode_kwargs={"normalize_embeddings":True}
)
#初始化向量库
vector_db=Chroma(
    persist_directory=PERSIST_DIR,
    embedding_function=embedding
)
#文本切割器
text_splitter=RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=150,
    add_start_index=True
)
#文件上传接口
@app.post("/upload")
async def upload_file(file: UploadFile=File(...)):
    #1.临时保存文件
    temp_path=f"./temp_{file.filename}"
    with open(temp_path,"wb") as f:
        f.write(await file.read())

    #2.根据文件类型选择加载器
    if file.filename.endswith(".pdf"):
        loader=PyPDFLoader(temp_path)
    elif file.filename.endswith(".docx"):
        loader=Docx2txtLoader(temp_path)
    elif file.filename.endswith(".txt"):
        loader=TextLoader(temp_path)
    else:
        os.remove(temp_path)
        return{"msg":"仅支持pdf、docx、txt"}

    #3.加载+切分文档
    docs=loader.load()
    split_docs=text_splitter.split_documents(docs)

    #4.写入向量库并持久化
    vector_db.add_documents(split_docs)

    os.remove(temp_path)

    return{
        "filename":file.filename,
        "chunk_count":len(split_docs),
        "msg":"文档入库成功"
    }

#========DeepSeek配置区===========
llm=ChatOpenAI(
    base_url="https://api.deepseek.com/v1",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    model="deepseek-chat",
    temperature=0.1
)

#定义请求体模型:前端Post请求发过来的数据结构
class QuestionRequest(BaseModel):
    question:str #用户问题

#GET探活接口，浏览器直接访问测试服务是否启动
@app.get("/ping")
def ping():
    return{"msg":"Service ready"}

#问答接口,Post请求。接收用户问题+文档片段，调用大模型返回答案
@app.post("/ask")
async def ask(question:str):
    #1.向量检索:取出最相关的3个文本块
    retriever=vector_db.as_retriever(search_kwargs={"k":3})
    docs=retriever.invoke(question)
    #把检索的内容拼成参考上下文
    context="\n\n".join([d.page_content for d in docs])

    #组装消息：系统指令+参考文档+用户问题
    messages=[
        SystemMessage(content="你是公司内部知识库助手，只能基于提供的参考资料回答。资料没有就直接回复"\
                      "【文档中未查到该内容】,严禁编造信息。"),
         HumanMessage(content=f"参考资料:{context}\n用户问题:{question}")
    ]
    resp=await llm.ainvoke(messages)
    return{"answer":resp.content}
print("===代码执行到末尾 准备启动服务=====")
if __name__=="__main__":
    import uvicorn
    uvicorn.run("main:app",host="127.0.0.1",port=8000,reload=True)