# QPaug
QPaug: Question and Passage Augmentation for Open-Domain Question Answering of LLMs <br> Empirical Methods in Natural Language Processing (EMNLP) 2024, Findings Accepted. <br>

#### Download data from [link](https://drive.google.com/file/d/1UKO2InSJkXNediwFQhXHs4BNV4Hul4GP/view?usp=sharing) and unzip in current directory.

```

## Question Augmentation
python question_augmentation.py {llm} {dataset}

## Retrieval by q_hat
python retrieval.py {retriever} {llm} {dataset}

## Passage Augmentation and Predict Answer
python predict.py {llm} {dataset}
```



## Citation
If you use any part of this code and pretrained weights for your own purpose, please cite our [paper](https://arxiv.org/pdf/2406.14277).
```
@InProceedings{
  title = 	 {QPaug: Question and Passage Augmentation for Open-Domain Question Answering of LLMs},
  author =       {Minsang Kim, CheonEum Park, Seungjun Baek},
  booktitle = 	 {Empirical Methods in Natural Language Processing (EMNLP), Findings},
  year = 	 {2024},
  series = 	 {Proceedings of Findings of EMNLP},
  month = 	 {12--16 Nov},
  publisher =    {Empirical Methods in Natural Language Processing (EMNLP), Findings}},
  pdf = 	 {https://arxiv.org/pdf/2406.14277},
  abstract = 	 {Retrieval-augmented generation (RAG) has received much attention for Open-domain question-answering (ODQA) tasks as a means to compensate for the parametric knowledge of large language models (LLMs). While previous approaches focused on processing retrieved passages to remove irrelevant context, they still rely heavily on the quality of retrieved passages which can degrade if the question is ambiguous or complex. In this paper, we propose a simple yet efficient method called question and passage augmentation (QPaug) via LLMs for open-domain QA. QPaug first decomposes the original questions into multiple-step sub-questions. By augmenting the original question with detailed sub-questions and planning, we are able to make the query more specific on what needs to be retrieved, improving the retrieval performance. In addition, to compensate for the case where the retrieved passages contain distracting information or divided opinions, we augment the retrieved passages with self-generated passages by LLMs to guide the answer extraction. Experimental results show that QPaug outperforms the previous state-of-the-art and achieves significant performance gain over existing RAG methods.}
  }
```
