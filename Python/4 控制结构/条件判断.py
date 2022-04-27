"""
类型    False               True
布尔    False(与0等价)       True(与1等价)
数值    0,0.0               非零的数值
字符串   '', ""(空字符串)    非空字符串
容器    [],(),{},set()      至少有一个元素的容器对象
None    None                非None对象
"""
if __name__ == '__main__':
    x, y = eval(input("输入x,y："))
    if x > y:
        print('x:', x)
    elif x == y:
        print('x+y:', x + y)
    else:
        print('y:', y)
        