"""
日期格式化符号%
%c 本地时间 Wed Aug 25 20:47:01 2021
%x 本地日期 08/25/21    %X 本地时间 20:44:26    %p AM PM
%y 两位数的年份(00-99)    %Y 四位数的年份(0000-9999)
%m 月份(01-12)    %b 月份简称(Mar) %B 全称    %d 月内中的一天(0-31)
%H 24小时制小时数(0-23)   %I 12小时制小时数(01-12)    %M 分钟数(00=59)   %S 秒(00-59)
%a 星期简称(Sat) %A 全称  %w 星期(0-6) 0是星期天
%j 今年第几天(001-366)
"""

if __name__ == '__main1__':
    import time
    # 获取当前时间戳 返回从1970/1/1 0:00 到 当前时间 经过的秒数(浮点数表示)
    now = time.time()  # 1629894604.6818604
    # time.struct_time(tm_year=2021, tm_mon=8, tm_mday=25, # tm_hour=19, tm_min=52, tm_sec=3,
    # tm_wday=2(0是周一), tm_yday=237(一年的第几天), tm_isdst=0)
    # 将 时间戳 转为 时间元组(上面一长串)
    local_time = time.localtime(now)
    # 将 时间元组 转为 时间戳
    now = time.mktime(local_time)
    time.mktime((2018, 9, 30, 9, 44, 31, 6, 273, 0))
    # 将 时间元组 转为 格式化字符串 '2021-08-25 20:17:45' (具体见日期格式化符号)
    now_str = time.strftime("%Y-%m-%d %H:%M:%S", local_time)
    # 将 格式化字符串 转换 时间元组
    now_struct_time = time.strptime(now_str, "%Y-%m-%d %H:%M:%S")

if __name__ == '__main1__':
    from datetime import datetime, timedelta
    # 基本的方法操作和time类似
    # 获取当前时间 datetime.datetime(2021, 8, 25, 20, 59, 28, 820097(毫秒))
    now = datetime.now()
    now.weekday()  # 0表示星期一
    # 获取10天后日期
    next_day = now + timedelta(days=10)  # 2021-09-04 21:09:53.861642
    # 获取10天后的日期 及那周的周一
    week_day = next_day - timedelta(days=next_day.weekday())  # 2021-08-30 21:13:40.952839

if __name__ == '__main__':
    import calendar
    import time
    calendar.isleap(2018)  # 是闰年返回True
    calendar.monthrange(2014, 11)  # (5, 30) 2014年11月份的第一天是周六，共有30天
    print(calendar.timegm(time.localtime()))  # 接受一个时间元组形式，返回该时刻的时间辍
    print(calendar.weekday(2018, 10, 27))  # 返回给定日期的日期码 0=星期一
