import json
from openai import OpenAI
import os
from prompts import *
import sys

model = sys.argv[1]
dataset = sys.argv[2]

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

def call(query, model, system):
    response = client.chat.completions.create(
    model=model,
    messages=[
        {"role": "system", "content": system},
        {"role": "user", "content": query},
            ]
            )
    return response.choices[0].message.content

topk=9
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain


llm = ChatOpenAI(temperature=0,
             max_tokens=256,
             model_name=model
            )              

with open(f"retriever_results/{dataset}_{model}_res_top100_qaug.json", "r") as f:
    with open(f"reader_results/{dataset}_{model}_res.json", "w") as wf:
        for line in f:
            line = json.loads(line)
            q=line["query"].split("\n")[0]
            cot="\n".join(line["query"].split("\n")[1:])
            psgs = line["retrieved"][:topk]
            passages = ""
            for i in range(len(psgs)):
                passages += f"Passage #{i+1} " + ": " + psgs[i] + "\n"
            prompt = PromptTemplate(template=paug_template, input_variables=["question"])
            llm_chain = LLMChain(prompt=prompt, llm=llm)
            paug = llm_chain.run(question=q)
            prompt = PromptTemplate(template=qpaug_template, input_variables=["question" "cot", "passages"])
            llm_chain = LLMChain(prompt=prompt, llm=llm)
            base_answer = llm_chain.run(question=q, paug=paug.replace("[DONE]","").replace("[NONE]","").strip(), passages=passages)
            wf.write(json.dumps({"qpaug_grounded_predict": base_answer, "query": q, "answer": line["answer"], "passages": psgs, "paug": paug, "cot": cot})+"\n")