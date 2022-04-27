class People:
    name = ''  # 定义基本属性  (_name类型的实例变量外部是可以访问的，但不推荐)
    age = 0
    __weight = 0  # 定义私有属性 (两个下划线开头) 无法从外部直接进行访问(或设置set和get方法)
    # 定义构造方法(用来初始化类，等同于java的构造方法)
    def __init__(self, n, a, w):
        self.name = n
        self.age = a
        self.__weight = w
    # 定义公共方法：类方法必须包含参数 self, 且为第一个参数，表示类的实例|对象
    def speak(self):
        print("%s 说: 我 %d 岁。" % (self.name, self.age))
        self.__speak()
    # 定义私有方法 (两个下划线开头) 只能在类的内部调用 self.__spark()
    def __speak(self):
        print("调用私有方法")

p = People('lisi', 10, 30)
p.speak()  # lisi 说: 我 10 岁。
p.score = 100  # 绑定实例属性
print(p.score)
# 单继承======================
class Student(People):
    grade = ''
    def __init__(self, n, a, w, g):
        People.__init__(self, n, a, w)
        self.grade = g
    # 覆写父类的方法
    def speak(self):
        print("%s 说: 我 %d 岁了，我在读 %d 年级" % (self.name, self.age, self.grade))
s = Student('ken', 10, 60, 3)
s.speak()  # ken 说: 我 10 岁了，我在读 3 年级
print(isinstance(s, People))
print(isinstance(s, Student))
# 多重继承===================
class Speaker():
    topic = ''
    name = ''
    def __init__(self, n, t):
        self.name = n
        self.topic = t
    def speak(self):
        print("我叫 %s，我是一个演说家，我演讲的主题是 %s" % (self.name, self.topic))
class Sample(Speaker, Student):  # 方法名同，默认调用的是在括号中排在前面的父类的方法，即Speaker方法
    a = ''
    def __init__(self, n, a, w, g, t):
        Student.__init__(self, n, a, w, g)
        Speaker.__init__(self, n, t)
test = Sample("Tim", 25, 80, 4, "Python")
test.speak()  # 我叫 Tim，我是一个演说家，我演讲的主题是 Python
# 多态========================
class Animal:
    def speak(self):
        print("我是动物")
class Dog(Animal):
    def speak(self):
        print("我是狗")
class Cat(Animal):
    def speak(self):
        print("我是猫")
dog = Dog()
cat = Cat()
def identity(animal):
    animal.speak()
identity(dog)
identity(cat)
# 多态的好处就是，当我们需要传入Dog、Cat、Tortoise……时，我们只需要接收Animal类型就可以了，因为Dog、Cat、Tortoise……都是Animal类型，
# 然后，按照Animal类型进行操作即可。由于Animal类型有speak()方法，因此，传入的任意类型，只要是Animal类或者子类，
# 就会自动调用实际类型的identity()方法，这就是多态的意思
# 方法重写================================
class Parent:  # 定义父类
    def myMethod(self):
        print('调用父类方法')
    def __init__(self):
        print('父类初始化')
class Child(Parent):  # 定义子类
    def myMethod(self):
        print('调用子类方法')
c = Child()  # 子类实例
c.myMethod()  # 子类调用重写方法
super(Child, c).myMethod()  # 用子类对象调用父类已被覆盖的方法
print("重写__init__方法===========")
"""
    子类不重写 __init__，实例化子类时，会自动调用父类定义的 __init__
    如果重写了__init__ 时，实例化子类，就不会调用父类已经定义的 __init__
    如果重写了__init__ 时，要继承父类的构造方法，可以使用 super 关键字：
"""
class Father(object):
    def __init__(self, name):
        self.name = name
        print("name: %s" % (self.name))
    def get_name(self):
        return 'Father ' + self.name
class Son(Father):
    def __init__(self, name):
        super(Son, self).__init__(name)
        print("hi")
        self.name = name
    def get_name(self):
        return 'Son ' + self.name
son = Son('runoob')
print(son.get_name())