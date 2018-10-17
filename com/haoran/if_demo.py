if 1:
    # 类型	False	        True
    # 布尔	False(与0等价)	True(与1等价)
    # 数值	0,0.0	        非零的数值
    # 字符串	'', ""(空字符串)	非空字符串
    # 容器	[],(),{},set()	至少有一个元素的容器对象
    # None	None	        非None对象
    print("if语句=====================")
    import random

    x = random.choice(range(100)) #0-99
    y = random.choice(range(200)) #0-199
    if x > y:
        print('x:',x)
    elif x == y:
        print('x+y:', x + y)
    else:
        print('y:',y)

