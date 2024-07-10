from typing import List
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph, START
from typing_extensions import TypedDict
from semanticscholar import SemanticScholar
from semanticscholar.Paper import Paper
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
import pandas as pd
from langchain_core.output_parsers import StrOutputParser
import requests
from PyPDF2 import PdfReader
import io
from langchain.pydantic_v1 import BaseModel, Field
from tqdm import tqdm
from langsmith.schemas import Run, Example
from langsmith.evaluation import evaluate
from langsmith.client import Client

template = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful AI bot. Your name is {name}."),
    ("human", "Hello, how are you doing?"),
    ("ai", "I'm doing well, thanks!"),
    ("human", "{user_input}"),
])

llm = ChatOpenAI(model="gpt-4o")
sch = SemanticScholar()

class GraphState(TypedDict):
    excerpt: str
    cur_search_query: str
    documents: List[Paper]
    final_answer: str
    previous_queries: list[str]
    total_steps: int

query_prompt = ChatPromptTemplate.from_messages([
    ("system","You are a bot designed to help discover the value of an unknown citation in an academic excerpt. "
               "You will be given an excerpt with the unknown citation marked with the text [CITATION]. "
               "Please read the excerpt and try to find a short search query that cold help find the unkown "
               "citation. You should return a topic that is mentioned in the excerpt and follwed by "
               "the [CITATION]. Make sure you return an answer detailed enough to find the exact paper the unknown "
               "citation refers to. Include relevant details in the search query to find the exact paper. {previous_excerpt}"),
    ("human","Please help me determine a search query to discover what the unknown citation is in this excerpt: {excerpt}")
    ]
)

query_chain = query_prompt | llm | StrOutputParser()

def generate_query(state: GraphState):
    previous_excerpt = ""
    if state['previous_queries']:
        previous_excerpt = "Do not return one of the queries from this list: "+", ".join(list(state['previous_queries']))
    search_query = query_chain.invoke({"previous_excerpt":previous_excerpt,
                                                   "excerpt":state['excerpt']})
    
    if state['previous_queries']:
        previous_queries = state['previous_queries']+[search_query]
    else:
        previous_queries = [search_query]
    
    if state['total_steps']:
        total_steps = state['total_steps']+1
    else:
        total_steps = 1

    return {"cur_search_query":search_query,"previous_queries":previous_queries,"total_steps":total_steps}

def search_node(state: GraphState):
    try:
        docs = sch.search_paper(state['cur_search_query'],limit=5)
    except:
        return {"documents":[],"total_steps":state['total_steps']+1}
    max_index = 0
    while True:
        thing = False
        try:
            _ = docs[max_index]
            thing = True
        except:
            pass
        if thing == False:
            break
        max_index += 1

    return {"documents":[docs[i] for i in range(max_index)],"total_steps":state['total_steps']+1}

def route_after_search_node(state: GraphState):
    if state['total_steps'] > 15:
        return 'end'
    elif len(state["documents"]) == 0:
        return "generate_query"
    else:
        return "validate_search_results_node"

class Relevant(BaseModel):
    """Determine whether there is direct evidence in an academic paper for a search query"""

    is_relevant: str = Field(...,description="Whether the paper contains evidence for the query. Can only take on the values YES or NO.")
    reason: str = Field(...,description="Reason you believe the paper contains evidence for the search query")

validate_llm = llm.with_structured_output(Relevant)

validate_prompt = ChatPromptTemplate.from_messages([
    ("system","You are a helpful assistant tasked with finding a unique academic paper that supports the following statement: {search_query}. "
            "Please only return YES if the paper provides concrete, direct evidence of the search query."),
    ("human","Does this paper provide direct evidence for the statement in the query? {paper}")
])

validate_chain = validate_prompt | validate_llm 

def validate_search_results_node(state: GraphState):
    for doc in state['documents']:
        try:
            r = requests.get(doc['openAccessPdf']['url'])
            f = io.BytesIO(r.content)

            reader = PdfReader(f)
            paper_contents = ""
            for page in reader.pages:
                paper_contents += page.extract_text()
            relevant = validate_chain.invoke({"search_query":state['cur_search_query'],"paper":paper_contents})
            if relevant.is_relevant == "YES":
                return {"final_answer":doc['openAccessPdf']['url'],"total_steps":state['total_steps']+1}
        except:
            continue
    return {"final_answer":"NONE","total_steps":state['total_steps']+1}

def route_after_validation(state: GraphState):
    if state['final_answer'] != "NONE" or state['total_steps'] > 15:
        return  "end"
    else:
        return "generate_query"

workflow = StateGraph(GraphState)
workflow.add_node("validate_search_results_node",validate_search_results_node)
workflow.add_node("search_node",search_node)
workflow.add_node("generate_query",generate_query)
workflow.add_conditional_edges("validate_search_results_node",route_after_validation,{"generate_query":"generate_query","end":END})
workflow.add_conditional_edges("search_node",route_after_search_node,{"validate_search_results_node":"validate_search_results_node",
                                                                      "generate_query":"generate_query",
                                                                      "end":END})
workflow.add_edge("generate_query","search_node")
workflow.add_edge(START,"generate_query")
graph = workflow.compile()

if __name__=="__main__":
    client = Client()
    def predict(inputs: dict) -> dict:
        response = graph.invoke({"excerpt":inputs['excerpt']})
        target_paper_url = response['final_answer'] if 'final_answer' in response else "NONE"
        return {"output": target_paper_url}

    # Define evaluators
    def must_mention(run: Run, example: Example) -> dict:
        prediction = run.outputs.get("output") or ""
        required = example.outputs.get("target_paper_url") or []
        score = int(prediction==required)
        return {"key":"found_source", "score": score}

    #examples_list = [x for i,x in enumerate(client.list_examples(dataset_name="citeme-dataset")) if i < 1]

    experiment_results = evaluate(
        predict,
        data="citeme-dataset",
        evaluators=[must_mention], 
    )