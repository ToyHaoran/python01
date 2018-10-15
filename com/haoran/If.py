if 1:
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