if __name__ == '__main__':
    if 1:
        import os
        print("返回当前的工作目录==================")
        print(os.getcwd()) # H:\code\idea\python\com\haoran 绝对路径
        """
        关于相对路径和绝对路径
        常用'/'来表示相对路径，'\'来表示绝对路径，'\\'是转义的意思
        """
        # os.chdir('../test01')   # 修改当前的工作目录  相对路径
        # os.system('mkdir today')   # 执行系统命令 mkdir
        print(dir("xxx"))

