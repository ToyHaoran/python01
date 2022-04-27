import functools
import time

if __name__ == '__main__':
    # 装饰器：增强某函数的功能，但又不修改该函数的定义；
    def execute_time(func):
        @functools.wraps(func)  # 不改变原函数名
        def wrapper(*args, **kw):  # 可以接受任意参数的调用
            start = time.time()
            res = func(*args, **kw)  # 调用原函数，并返回结果
            stop = time.time()
            print('%s函数 executed in %s ms' % (func.__name__, stop - start))
            return res
        return wrapper

    @execute_time
    def now():
        print("现在")

    print(now())  # now = log(now)

