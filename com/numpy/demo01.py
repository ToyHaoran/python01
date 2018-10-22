
import numpy as np


ndarray_对象简介 = 0
# https://www.tutorialspoint.com/numpy/numpy_ndarray_object.htm
if 0:
    print("ndarray对象简介=======")
    # 类似一个数组
    print(np.array([1, 2, 3]))
    print(np.array([1, 2, 3.0]))
    print(np.array([[1, 2], [3, 4]]))
    print(np.array([1, 2, 3], ndmin=2))

数据类型 = 0
# https://www.tutorialspoint.com/numpy/numpy_data_types.htm
if 0:
    # NumPy支持比Python更多种类的数值类型
    print("数据类型======")
    # NumPy数字类型是dtype（数据类型）对象的实例，每个对象都具有唯一的特征
    # 'b' - 布尔值
    # 'i' - （签名）整数
    # 'u' - 无符号整数
    # 'f' - 浮点数
    # 'c' - 复杂浮点
    # 'm' - timedelta
    # 'M' - 日期时间
    # 'O' - （Python）对象
    # 'S'，'a' - （byte-）字符串
    # 'U' - Unicode
    # 'V' - 原始数据（无效）

    print("数据类型对象dtype============")
    # numpy.dtype(object, align, copy) 对象、对齐
    # #int8, int16, int32, int64 can be replaced by equivalent string 'i1', 'i2','i4', etc
    print(np.dtype('i1'))
    print(np.dtype('i2'))
    print(np.dtype('i4'), np.dtype(np.int32)) # 乘以8的关系
    print(np.dtype('i8'))
    print(np.dtype('>i4'))

    print("结构化数据类型=======")
    dt = np.dtype([('age', np.int8)])
    print(dt)
    a = np.array([(10,), (20,), (30,)], dtype=dt)
    print(a)
    print(a['age'])
    print(a['age'].dtype)

    # 以下示例定义了一个名为student的结构化数据类型，其中包含字符串字段“name”，整数字段 “age”和浮点字段 “marks”。此dtype应用于ndarray对象
    student = np.dtype([('name', 'S20'), ('age', 'i1'), ('marks', 'f4')])
    print(student)
    a = np.array([('abc', 21, 50), ('xyz', 18, 75)], dtype=student)
    print(a)
    print(a['name'])
    print(a['name'].dtype)

ndarray_属性 = 0
# https://www.tutorialspoint.com/numpy/numpy_array_attributes.htm
if 0:
    print("ndarray的属性=======")

    print("shape========")
    # 此数组属性返回由数组维度组成的元组。它也可以用于调整阵列的大小
    a = np.array([[1, 2, 3], [4, 5, 6]])
    print(a)
    print(a.shape)
    a.shape = (3, 2) #或者 b = a.reshape(3, 2)
    print(a)

    print("ndim========")
    # 此数组属性返回数组维数
    a = np.arange(24)
    print(a.ndim)
    b = a.reshape(2, 4, 3)
    print(b.ndim)
    print(b)

    print("itemsize=======")
    # 此数组属性以字节为单位返回数组的每个元素的长度
    x = np.array([1, 2, 3, 4, 5], dtype=np.int8)
    print(x.itemsize)
    x = np.array([1, 2, 3, 4, 5], dtype=np.float32)
    print(x.itemsize)

    print("flags=====")
    # ndarray对象具有以下属性。其当前值由此函数返回。感觉没啥用
    x = np.array([1, 2, 3, 4, 5])
    print(x.flags)

创建_空数组 = 0
# https://www.tutorialspoint.com/numpy/numpy_array_creation_routines.htm
if 0:
    print("empty创建空数组=======")
    x2 = np.empty([3, 2], dtype=int)
    print(x2) # 数组中的元素显示随机值，因为它们未初始化(在Shell中显示正常，在Pycharm中大部分情况下显示为0，不知道为什么)

    print("zeros用0填充====")
    x = np.zeros((2, 2), dtype=[('x', 'i4'), ('y', 'i4')])
    print(x)
    print(x['x'])

    print("ones用1填充=====")
    # 返回指定大小和类型的新数组，用1填充。
    x = np.ones([2, 2], dtype=int)
    print(x)

创建_从现有数据 = 0
# https://www.tutorialspoint.com/numpy/numpy_array_from_existing_data.htm
if 0:
    print("从现有数据创建数组======")
    print("asarray========")
    # 将Python序列转换为ndarray
    x = [1, 2, 3]
    a = np.asarray(x, dtype=float)
    print(a)
    x = [(1, 2, 3), (4, 5)]
    a = np.asarray(x[0])
    print(a)
    a = np.asarray(x)
    print(a)

    print("frombuffer======")
    # 此函数将缓冲区解释为一维数组。暴露缓冲区接口的任何对象都用作返回ndarray的参数。
    s = b"Hello World" # 在Python3中，所有的字符串都是Unicode字符串,必须编码
    a = np.frombuffer(s, dtype='S1')
    print(a)

    print("fromiter==========")
    # 此函数从任何可迭代对象构建ndarray对象。此函数返回一个新的一维数组
    list = range(10)
    it = iter(list)
    x = np.fromiter(it, dtype=float)
    print(x.reshape(2, 5))

创建_从数值范围 = 0
# https://www.tutorialspoint.com/numpy/numpy_array_from_numerical_ranges.htm
if 0:
    print("arange======")
    # 此函数返回一个ndarray对象，该对象包含给定范围内的均匀间隔值。功能的格式如下
    # numpy.arange(start, stop, step, dtype)
    print(np.arange(10, 20, 2, np.int8))  # [10 12 14 16 18]

    print("linspace======")
    # 此函数类似于arange（）函数。在此函数中，指定间隔之间的均匀间隔值，而不是步长。该功能的用法如下
    # numpy.linspace(start, stop, num, endpoint, retstep, dtype)
    print(np.linspace(10, 20, 5, endpoint=True, dtype=np.float32, retstep=True))
    # [10.  12.5 15.  17.5 20. ] 均分为5个，包括结束点, 返回间隔2.5

    print("logspace=======")
    # 此函数返回一个ndarray对象，该对象包含在对数刻度上均匀分布的数字。开始和停止比例的终点是基数的索引，通常为10。
    # numpy.logspace(start, stop, num, endpoint, base, dtype)
    print(np.logspace(1, 10, num=10, base=2))
    print(np.logspace(1, 10, num=10, base=3, dtype=np.int64))

访问方式_切片和索引 = 0
# https://www.tutorialspoint.com/numpy/numpy_indexing_and_slicing.htm
if 0:
    print("切片和索引=======")
    a = np.arange(10)
    s = slice(2, 7, 2)
    print(a[s])
    print("或者：", a[2:7:2])
    print(a[:7])
    print(a[5])
    print(a[2:7])

    a = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    print(a[1:3])
    print(a[..., 1:]) # 生成与数组维度长度相同的选择元组
    print(a[1:3, :2])

访问方式_高级索引 = 0
# https://www.tutorialspoint.com/numpy/numpy_advanced_indexing.htm
if 0:
    print("ndarray访问方式_高级索引======")
    # 高级索引始终返回数据的副本。与此相反，切片仅呈现视图
    # 高级索引有两种类型 - 整数和布尔值
    print("整数索引=====")
    x = np.array([[1, 2], [3, 4], [5, 6]])
    print(x[[0, 1, 2], [0, 1, 0]])  # 该选择包括来自第一阵列的（0,0），（1,1）和（2,0）处的元素

    print("11=======")
    x = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]])
    rows = np.array([[0, 0], [3, 3]])
    cols = np.array([[0, 2], [0, 2]])
    # 选择的行索引是[0,0]和[3,3]，而列索引是[0,2]和[0,2]
    # 即二维坐标为：[00，02]，[30，32]
    print(x[rows, cols])  # 注意这个结果是二维的，和下面这个一维的不一样

    rows1 = np.array([0, 0, 3, 3])
    cols1 = np.array([0, 2, 0, 2])
    print(x[rows1, cols1])

    print("22=======")
    z = x[1:4, 1:3]
    y = x[1:4, [1, 2]]  # 等价
    print(z)
    print(y)

    print("布尔索引=====")
    print(x[x > 5])  # 返回大于5的项
    # 使用~（补码运算符）省略NaN（非数字）元素
    a = np.array([np.nan, 1, 2, np.nan,3,  2+6j])
    print(a[np.isnan(a)])
    print(a[~np.isnan(a)])
    print(a[np.iscomplex(a)])


广播 = 0
# https://www.tutorialspoint.com/numpy/numpy_broadcasting.htm
if 0:
    print("广播=====")
    # 广播是指NumPy在算术运算期间处理不同形状的阵列的能力。
    # 对数组的算术运算通常在相应的元素上完成。如果两个阵列具有完全相同的形状，则可以平滑地执行这些操作
    a = np.array([1, 2, 3, 4])
    b = np.array([10, 20, 30, 40])
    c = a * b
    print(c)
    # 如果两个数组的维度不同，则无法进行元素到元素的操作。
    # 但是，由于广播能力，在NumPy中仍然可以对非相似形状的阵列进行操作。
    # 较小的阵列被广播到较大阵列的大小，以便它们具有兼容的形状
    # 如果满足以下规则，则可以进行广播
    # Array with smaller ndim than the other is prepended with '1' in its shape.
    # Size in each dimension of the output shape is maximum of the input sizes in that dimension.
    # An input can be used in calculation, if its size in a particular dimension matches the output size or its value is exactly 1.
    # If an input has a dimension size of 1, the first data entry in that dimension is used for all calculations along that dimension.
    a = np.array([[1.0, 1.0, 1.0], [10.0, 10.0, 10.0], [20.0, 20.0, 20.0], [30.0, 30.0, 30.0]])
    b = np.array([1.0, 2.0, 3.0])
    print(a + b)

迭代数组 = 0
# https://www.tutorialspoint.com/numpy/numpy_iterating_over_array.htm
if 0:
    print("numpy.nditer======")
    a = np.arange(0, 60, 5)
    a = a.reshape(3, 4)
    print(a)
    for x in np.nditer(a):  # 等价于a.T
        print(x, end=" ")
    print()

    print("迭代顺序=========")
    # 默认是K：尽可能接近数组元素在内存中出现的顺序
    b = a.T
    print(b)
    c = b.copy(order='C')  # 横着
    c1 = b.copy(order='F')  # 竖着
    for x in np.nditer(b):
        print(x, end=" ")
    print()
    for x in np.nditer(c1, order='C'):  # 强制改一下顺序
        print(x, end=" ")
    print()

    print("迭代中修改值========")
    for x in np.nditer(a, op_flags=['readwrite']):
        x[...] = 2 * x
    print(a)

    print("外部循环==========")
    print(a)
    for x in np.nditer(a, flags=['external_loop'], order='F'):
        print(x)  # external_loop 每次迭代返回多个值的ndarray,而不是单个值
    print()

    print("广播迭代=======")
    # 如果两个数组是可广播的，则组合的nditer对象能够同时迭代它们。
    # 假设a尺寸3X4，并且存在另一个数组b尺寸1X4的，使用以下类型的迭代器（阵列b被广播到a的大小）。
    b = np.array([1, 2, 3, 4], dtype=int)
    print(a)
    for x, y in np.nditer([a, b], order="F"):
        print("%d:%d" % (x, y), end=" ")
    print()

ndarray_常用方法 = 0
# https://www.tutorialspoint.com/numpy/numpy_array_manipulation.htm
if 1:
    print("Changing Shape============")
    a = np.arange(8).reshape(4, 2)
    print(a)
    # 将多维降为一维数组，返回拷贝（copy），修改不会影响原来的。
    print("===")
    print(a.flatten(order="F"))  # 默认C
    a.flatten()[1] = 100
    print(a)
    print("===")
    # 同上面的功能是一样的。返回视图（view），修改会影响原来的。
    a.ravel()[1] = 100  # 不知道为什么order为C能改，'F'就改不了了。神经病
    print(a)

    print("Transpose Operations=======")




    print("Changing Dimensions=========")

    print("Joining Arrays=========")

    print("Splitting Arrays==========")

    print("Adding / Removing Elements========")




二元运算符 = 0
# https://www.tutorialspoint.com/numpy/numpy_binary_operators.htm











