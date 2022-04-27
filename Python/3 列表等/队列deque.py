"""
  在列表的最后添加或者弹出元素速度快，然而在列表里插入或者从头部弹出速度却不快（因为所有其他的元素都得一个一个地移动）
"""

if __name__ == '__main__':
    from collections import deque
    queue = deque(["aa", "bb", "cc"])
    queue.append("dd")
    queue.append("ee")
    print(queue.popleft())
    print(queue.popleft())
    print(queue.popleft())