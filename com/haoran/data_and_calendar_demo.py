

import time
import calendar

def time模块():
    if 1:
        print("time模块==========")
        #Python 提供了一个 time 和 calendar 模块可以用于格式化日期和时间。
        now = time.time()  # 获取当前时间戳，是以秒为单位的浮点小数，从1970年1月1日午夜开始
        print("当前时间戳为:", now) # 1539401325.5900888

        localtime = time.localtime(now) # 从返回浮点数的时间戳方式向时间元组转换
        print(localtime.tm_year)
        print(localtime.tm_mon)
        print("本地时间：", localtime)

        print("格式化日期=========")
        # 将时间元组格式化成2016-03-20 11:45:39形式
        print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
        # 格式化成Sat Mar 28 22:24:24 2016形式
        print(time.strftime("%a %b %d %H:%M:%S %Y", localtime))
        # 将格式字符串转换时间元组,然后再转换为时间戳
        a = "Sat Mar 28 22:24:24 2016"
        print(time.mktime(time.strptime(a, "%a %b %d %H:%M:%S %Y")))
        print("线程睡眠3秒=====")
        # time.sleep(3)

        print("总结：相互之间的转化=============")
        # （1）当前时间戳
        # 1538271871.226226
        time.time()
        # （2）时间戳 → 时间元组，默认为当前时间
        # time.struct_time(tm_year=2018, tm_mon=9, tm_mday=3, tm_hour=9, tm_min=4, tm_sec=1, tm_wday=6, tm_yday=246, tm_isdst=0)
        time.localtime()
        time.localtime(1538271871.226226)
        # （3）时间戳 → 可视化时间
        # time.ctime(时间戳)，默认为当前时间
        time.ctime(1538271871.226226)
        # （4）时间元组 → 时间戳
        # 1538271871
        time.mktime((2018, 9, 30, 9, 44, 31, 6, 273, 0))
        # （5）时间元组 → 可视化时间
        # time.asctime(时间元组)，默认为当前时间
        time.asctime()
        time.asctime((2018, 9, 30, 9, 44, 31, 6, 273, 0))
        time.asctime(time.localtime(1538271871.226226))
        # （6）时间元组 → 可视化时间（定制）
        # time.strftime(要转换成的格式，时间元组)
        time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        # （7）可视化时间（定制） → 时间元祖
        # time.strptime(时间字符串，时间格式)
        print(time.strptime('2018-9-30 11:32:23', '%Y-%m-%d %H:%M:%S'))
        # （8）浮点数秒数，用于衡量不同程序的耗时，前后两次调用的时间差
        print("返回系统运行时间======")
        # 返回计时器的精准时间（系统的运行时间），包含整个系统的睡眠时间。
        # 由于返回值的基准点是未定义的，所以，只有连续调用的结果之间的差才是有效的。
        print(time.perf_counter())
        print("返回进程运行时间======")
        # 返回当前进程执行 CPU 的时间总和，不包含睡眠时间。
        # 由于返回值的基准点是未定义的，所以，只有连续调用的结果之间的差才是有效的
        print(time.process_time())


def calendar模块():
    if 0:
        print("calendar模块===========")
        # 返回一个多行字符串格式的year年年历，3个月一行，间隔距离为c。
        # 每日宽度间隔为w字符。每行长度为21*W+18+2*C。l是每星期行数。
        print(calendar.calendar(2018,w=2,l=1,c=6))

        # 返回当前每周起始日期的设置。默认情况下，首次载入calendar模块时返回0，即星期一。
        print(calendar.firstweekday())

        # 设置每周的起始日期码。0（星期一）到6（星期日）。
        print(calendar.setfirstweekday(0))

        # 是闰年返回True，否则为false。
        print(calendar.isleap(2018))

        # 返回在Y1，Y2两年之间的闰年总数。
        print(calendar.leapdays(2000, 2018))

        # 返回一个多行字符串格式的year年month月日历，两行标题，一周一行。
        # 每日宽度间隔为w字符。每行的长度为7* w+6。l是每星期的行数。
        print(calendar.month(2018,10,w=2,l=1))

        # 返回一个整数的单层嵌套列表。每个子列表装载代表一个星期的整数。
        # Year年month月外的日期都设为0;范围内的日子都由该月第几日表示，从1开始。
        print(calendar.monthcalendar(2018,10))

        # 返回两个整数。第一个是该月的星期几，第二个是该月有几天。星期几是从0（星期一）到 6（星期日）。
        print(calendar.monthrange(2014,11)) # (5, 30)解释：5 表示 2014 年 11 月份的第一天是周六，30 表示 2014 年 11 月份总共有 30 天。

        # calendar.prcal(year,w=2,l=1,c=6) 相当于 print calendar.calendar(year,w,l,c).
        # calendar.prmonth(year,month,w=2,l=1) # 相当于 print calendar.calendar（year，w，l，c）。

        # 和time.gmtime相反：接受一个时间元组形式，返回该时刻的时间辍（1970纪元后经过的浮点秒数）。
        print(calendar.timegm(time.localtime()))
        # 返回给定日期的日期码。0（星期一）到6（星期日）。月份为 1（一月） 到 12（12月）。
        print(calendar.weekday(2018,10,27))


if __name__ == '__main__':
    time模块()
    calendar模块()