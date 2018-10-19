#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# 推荐使用anaconda
import traceback

import mysql.connector
import sys

def connect_mysql_db():
    if 1:
        """  连接数据库

            参数：
                host，连接的数据库服务器主机名，默认为本地主机(localhost)。
                user，连接数据库的用户名，默认为当前用户。
                passwd，连接密码，没有默认值。
                db，连接的数据库名，没有默认值。
                conv，将文字映射到Python类型的字典。默认为MySQLdb.converters.conversions
                cursorclass，cursor()使用的种类，默认值为MySQLdb.cursors.Cursor。
                compress，启用协议压缩功能。
                named_pipe，在windows中，与一个命名管道相连接。
                init_command，一旦连接建立，就为数据库服务器指定一条语句来运行。
                read_default_file，使用指定的MySQL配置文件。
                read_default_group，读取的默认组。
                unix_socket，在unix中，连接使用的套接字，默认使用TCP。
                port，指定数据库服务器的连接端口，默认是3306
        """
        print("正在连接到mysql数据库......")
        db = mysql.connector.connect(
            host="localhost",
            port="3307",
            user="root2",
            passwd="root",
            database="taotao"
        )
        print("连上了!")
        return db

def create_table(db):
    if 0:
        # 使用cursor()方法获取操作游标
        cursor = db.cursor()

        # 如果存在表Sutdent先删除
        cursor.execute("DROP TABLE IF EXISTS Student")

        sql = """CREATE TABLE Student (
                ID CHAR(10) NOT NULL,
                Name CHAR(8),
                Grade INT )"""

        # 创建Sutdent表
        cursor.execute(sql)
        print("创建表成功")

def insert_rows(db):
    if 0:
        # 使用cursor()方法获取操作游标
        cursor = db.cursor()

        # SQL 插入语句
        sql = """INSERT INTO Student
             VALUES ('001', 'CZQ', 70),
                    ('002', 'LHQ', 80),
                    ('003', 'MQ', 90),
                    ('004', 'WH', 80),
                    ('005', 'HP', 70),
                    ('006', 'YF', 66),
                    ('007', 'TEST', 100)"""
        sql2 =  "INSERT INTO Student(ID, Name, Grade) VALUES (%s, %s, %s)" % ('Mac', 20, 'boy')

        # 批量插入
        sqlmany = "INSERT INTO Student(ID, Name, Grade) VALUES (%s, %s, %s)" # 这里是个占位符，不是格式化%，不要用%d
        val = [('001', 'CZQ', 70),
               ('002', 'LHQ', 80),
               ('003', 'MQ', 90),
               ('004', 'WH', 80),
               ('005', 'HP', 70),
               ('006', 'YF', 66),
               ('007', 'TEST', 100)]

        try:
            # 执行sql语句
            # cursor.execute(sql)
            # 批量插入
            cursor.executemany(sqlmany, val)

            # 提交到数据库执行
            db.commit()
            print("插入数据成功")
        except:
            # Rollback in case there is any error
            print("错误信息：", traceback.format_exc())
            print('插入数据失败!')
            db.rollback()

def delete_rows(db):
    if 0:
        # 使用cursor()方法获取操作游标
        cursor = db.cursor()

        # SQL 删除语句
        # sql = "DELETE FROM Student WHERE Grade = '%d'" % (100)
        sql = "DELETE FROM Student"

        # 为了防止数据库查询发生 SQL 注入的攻击，我们可以使用 %s 占位符来转义删除语句的条件：
        sql = "DELETE FROM Student WHERE ID = %s"
        na = ["001"]

        try:
            # 执行SQL语句
            cursor.execute(sql, na)

            # 可以用循环拼接删除字符串来批量删除。

            # 提交修改
            db.commit()
        except:
            print('删除数据失败!')
            print(traceback.format_exc())
            # 发生错误时回滚
            db.rollback()

def update_rows(db):
    if 0:
        # 使用cursor()方法获取操作游标
        cursor = db.cursor()

        # SQL 更新语句
        sql = "UPDATE Student SET Grade = Grade + 3 WHERE ID = '%s'" % ('003')

        try:
            # 执行SQL语句
            cursor.execute(sql)
            # 提交到数据库执行
            db.commit()
        except:
            print('更新数据失败!')
            # 发生错误时回滚
            db.rollback()

def query_rows(db):
    if 1:
        # 使用cursor()方法获取操作游标
        cursor = db.cursor()

        # SQL 查询语句
        #sql = "SELECT * FROM Student \
        #    WHERE Grade > '%d'" % (80)
        sql = "SELECT * FROM Student"
        try:
            # 执行SQL语句
            cursor.execute(sql)
            # 获取所有记录列表
            results = cursor.fetchall()
            print(type(results)) # <class 'list'>
            for row in results: # <class 'tuple'>
                print(row)

        except:
            print("Error: unable to fecth data")

def close_db(db):
    db.close()


if __name__ == '__main__':
    db = connect_mysql_db()
    create_table(db)
    insert_rows(db)
    delete_rows(db)
    update_rows(db)
    query_rows(db)
    close_db(db)


