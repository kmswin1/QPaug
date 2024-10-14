import json
from openai import OpenAI
import os
import sys
API_KEY=os.environ["OPENAI_API_KEY"]
client = OpenAI(api_key=API_KEY)
model=sys.argv[1]
dataset = sys.argv[2]

def call(query, model):
    response = client.chat.completions.create(
    model=model,
    messages=[
        {"role": "system", "content": "You are an helpful assistant to explain step-by-step problem-solving process."},
        {"role": "user", "content": f"{query}\nLet's think step-by-step."},])
    return response.choices[0].message.content

with open(f"{dataset}_questions.json", "r") as f:
    with open(f"{dataset}_questions_cot_{model}.json", "w") as wf:
        for i, line in enumerate(f):
            line = json.loads(line)
            augmented_question = call(line["question"], model)
            line["augmented_questions"] = line["question"] + "\n" + augmented_question
            wf.write(json.dumps(line)+"\n")
