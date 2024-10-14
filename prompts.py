qpaug_template="""
{passages}
Your knowledge: : {paug}
Question: {question} Do not exceed 3 words
Answer:
"""

paug_template="""
Your job is to act as a subject matter expert. You will write a good-quality passage that can answer the question based on your factual knowledge. Do not write a passage if you don’t know accurate information about the question.
Now, let's start. After you write, please write [DONE] to indicate you are done. Write [NONE] if you cannot write a factual good passage.
Question: {question}
Passage:
"""