

if __name__ == '__main__':
    # ``sorted``函数可以从任意序列的元素返回一个新的排好序的列表：
    # key指定的函数将作用于list的每一个元素上，并根据key函数返回的结果进行排序
    print(sorted([36, 5, -12, 9, -21], reverse=True))  # 默认升序排列
    print(sorted([36, 5, -12, 9, -21], key=abs))  # [5, 9, -12, -21, 36]

    # 字典排序
    students = [('Bob', 75), ('Adam', 92), ('Bart', 66), ('Lisa', 88)]
    print(sorted(students, key=lambda t: t[1]))  # Student未改变
    students.sort(key=lambda t: t[1])  # 返回None，Student已经改变
    print(students)

    # 序列函数 zip函数
    pitchers = [('Nolan', 'Ryan'), ('Roger', 'Clemens'), ('Schilling', 'Curt')]
    first_names, last_names = zip(*pitchers)  # zip既能压缩，也能解压
