#!/usr/bin/env python3
# -*- coding: utf-8 -*-
if __name__ == '__main__':
    import sys
    print(sys.path)
    #'C:\\code\\project03python\\src\\com\\test2', 'C:\\code\\project03python',
    # 看到没有，路径只有本路径和项目路径，没有com路径，所以前面一定要加src才能导入。
    import src.com.test1.demo04
