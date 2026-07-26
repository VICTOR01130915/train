print("程序开始运行")

text ="今天天气很好"
print(f"输入文本:{text}")

if "好" in text or "愉快" in text:
	result ="positive"
if "坏" in text or "难过" in text:
	result ="negative"
else:
	result ="neutral"
print(f"判断结果:{result}")
print("程序结果")

