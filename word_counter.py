#day3 :词频统计
#统计英文文本中每个单词出现的次数
#示例文本
text ="""to be ,or not to be ,that is the question : whether 'tis nobler in the mind to suffer  The slings and arrows of outrageous fortune opr to take arms against a sea of troubles
"""

# 将标点换为空格，避免标点符号被当作单词一部分
for i in ", . ; : ! ?":
	text = text.replace(i," ")
#将所有字母转为小写，
words = text.lower().split()
word_count ={}
for word in words:
		word_count[word] = word_count.get(word,0)+1
for word ,count in sorted(word_count.items(),key=lambda x: x[1],reverse=True):
	print(f"{word}: {count}")
