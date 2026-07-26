from transformers import pipeline

classifier = pipeline("sentiment-analysis",model="bert-base-chinese")
result = classifier("今天天气很好,心情非常愉快!")
print(result)