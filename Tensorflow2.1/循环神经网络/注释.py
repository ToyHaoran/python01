from keras.layers import SimpleRNN

"""
layers.SimpleRNN(记忆体个数，activation='激活函数’,
    return_sequences= 是否每个时刻输出ht到下一层)
    activation='激活函数’ (不写， 默认使用tanh)
    return_sequences=True 各时间步都输出ht，适合中间层的循环核；
    return_sequences=False 仅在最后时间步输出ht(默认)，适合最后一层的循环核。
"""
SimpleRNN(3, return_sequences=True)  # 三个记忆体的循环核，在每个时间步输出ht


