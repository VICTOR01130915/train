#day2:猜数字游戏
import random

target = random.randint(1,100)
print("猜一个1-100之间的数字!")

while True:
	guess = int(input("输入你的猜测:"))
	if guess < target:
		print("太小了,再大一点")
	elif guess >target:
		print("太大了,再小一点")
	else:
		print("恭喜你猜对了!")
	break