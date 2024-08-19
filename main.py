from llama_index.llms.ollama import Ollama
from llama_parse import LlamaParse
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, PromptTemplate
from llama_index.core.embeddings import resolve_embed_model
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.core.agent import ReActAgent
from prompts import context , code_parser_template
from code_reader import code_reader

from pydantic import BaseModel
from llama_index.core.output_parsers import PydanticOutputParser
from llama_index.core.query_pipeline import QueryPipeline

from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings

from dotenv import load_dotenv
import os
import ast

load_dotenv()

llm = Ollama(model='llama3:8b', request_timeout=5000.0)

parser = LlamaParse(result_type="markdown")

file_extractor = {".pdf":parser}
documents = SimpleDirectoryReader("./data", file_extractor=file_extractor).load_data()

Settings.embed_model = HuggingFaceEmbedding(
    model_name="BAAI/bge-small-en-v1.5"
)

embed_model = resolve_embed_model("local:BAAI/bge-small-en-v1.5")
vector_index = VectorStoreIndex.from_documents(documents, embede_model=embed_model)
query_engine = vector_index.as_query_engine(llm=llm)

# query_engine.query("")
# result = query_engine.query("what are some of the routes in the api?")
# print(result)

tools = [
    QueryEngineTool(
        query_engine=query_engine,
        metadata=ToolMetadata(
            name = "flask_api_docs",
            description = "This gives documentation about code for an api"
        ),
    ),
    code_reader,
]

code_llm = Ollama(model="codellama")
agent = ReActAgent.from_tools(tools, llm=code_llm, verbose=True, context=context)

class CodeOutput(BaseModel):
    code: str
    description: str
    filename: str


parser = PydanticOutputParser(CodeOutput)
json_prompt_str = parser.format(code_parser_template)
json_prompt_tmpl = PromptTemplate(json_prompt_str)
output_pipeline = QueryPipeline(chain=[json_prompt_tmpl, llm])

while (prompt := input("Enter a prompt (q to quit): ")) != "q":

    retries = 0
    while retries < 3:
        try:
            result = agent.query(prompt)
            next_result = output_pipeline.run(response=result)
            cleaned_json = ast.literal_eval(str(next_result).replace("assistant:", ""))
            break
        except Exception as e:
            retries += 1
            print(f"Error occured, retry #{retries}:", e)
    if retries >= 3:
        print("Unable to process request, try again...")
        continue
            
    print("code generated")
    print(cleaned_json["code"])

    print("\n\n Description:", cleaned_json['descriotion'])

    filename = cleaned_json["filename"]

    try:
        with open(os.path.join("output", filename), "w") as f:
            f.write(cleaned_json["code"])
        print("Saved file", filename)
    except:
        print("Error saving file...")