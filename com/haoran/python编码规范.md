
参考：

[Python代码样式指南](https://www.python.org/dev/peps/pep-0008/)

[某人总结](https://juejin.im/post/5afe94845188254267264da1)

[某人总结](https://blog.csdn.net/MrLevo520/article/details/69155636?utm_source=blogxgwz0)


# 文件编码
- 使用 4 空格缩进，禁用任何 TAB 符号
- 源码文件使用 UTF-8 无 BOM 编码格式
- 总是使用 Unix \n 风格换行符
- 在每一个 py 文件头，都添加如下内容：<br>
  ```
   #!/usr/bin/env python
   # -*- coding: utf-8 -*-
  ```
-


# 换行与缩进
- 每级缩进使用4个空格
- 续行与其包裹元素要对齐
- 限制所有行的最大长度为 79 个字符
- 空行
  - 顶层函数和类之间使用两个空行。
  - 类的方法之间使用一个空行。
  - 在函数中使用空行来表示不同的逻辑段落
  - if/for/while语句中，即使执行语句只有一句，也必须另起一行
- 导入不同模块，通常应当使用单独的行
- 避免多余空格
  - 紧贴着圆括号、方括号和花括号:`spam(ham[1], {eggs: 2})`
  - 紧贴在逗号，分号或冒号之前：
  - 紧贴在函数调用的参数列表的圆括号的开括号前
  - 紧贴在索引或切片的方括号的开括号前
  - 在赋值（或其他）语句的运算符周围，不要为了对齐而使用多个空格
  - 不要在一个关键字参数或者一个缺省参数值的 = 符号前后加一个空格：<br>
    `def complex(real, imag=0.0):return magic(r=real, i=imag)`

# 注释
- 不好理解的注释不如没有注释。注释要和代码保持同步！
- 为所有的共有模块、函数、类、方法写docstrings；非共有的没有必要，但是可以写注释（在def的下一行）

# 命名
- java中很多驼峰命名，而python中几乎看不到。
- 文件名(模块)：简短、小写、可以使用下划线。
- 包：简短、小写、**不能使用下划线**。如mypackage
- 类：使用CapWords的方式，模块内部使用的类采用_CapWords的方式
  - 类的属性(方法和变量)：全小写、下划线
  - 类的属性有3种作用域public、non-public和subclass API，可以理解成C++中的public、private、protected，non-public属性前，前缀一条下划线。
  - 类的属性若与关键字名字冲突，后缀一下划线，尽量不要使用缩略等其他方式
  - 总使用“self”作为实例方法的第一个参数。总使用“cls”作为类方法的第一个参数。
- 异常：使用CapWords+Error后缀的方式
- 全局变量：尽量只在模块内有效，类似C语言中的static。实现方法有两种，一是all机制;二是前缀一个下划线
- 变量：全小写、下划线
- 常量:全大写、下划线。
- 函数：全小写、下划线。如：my_example_function

# 其他
- 别用‘==’进行布尔值和 True 或者 False 的比较，直接`if ok`
- 尽可能使用‘is’‘is not’取代‘==’，比如if x is not None 要优于if x
- 异常中try的代码尽可能少
- 使用startswith() and endswith()代替切片进行序列前缀或后缀的检查 `foo.startswith('abc') and foo.endswith('xyz')`
- 使用isinstance()比较对象的类型:`print isinstance(foo,int)`
- 判断序列空或不空:不提倡`if len(foo)`,直接`if foo`
- 
- 
