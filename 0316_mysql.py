import pymysql
from contextlib import closing

connection = pymysql.connect(host='127.0.0.1', port=3306, charset='utf8mb4', user='root', password='chizuo7750380',
                             database='intership_spider')
connection.commit()

imdb = 'tt0046250'
targets = 'NULL'
sql = f"select * from douban_data where imdb='{imdb}'"
with closing(connection.cursor()) as cursor:
    res = cursor.execute(sql)
    res = cursor.fetchall()
    if len(res) > 0:
        res = res[0]
        print(res)
    sql = f"update douban_data set targets='{targets}' where imdb='{imdb}'"
    res = cursor.execute(sql)

    print(res)
    connection.commit()

    # import time

# query_str = '%测试%'
# sql = "select * from job_data where job_name like %s"
# params = ['%' + query_str + '%']
# # # res = cursor.execute(sql, params)
# with closing(connection.cursor()) as cursor:
#     res = cursor.execute(sql, [query_str])
#     res_set = cursor.fetchall()
#     print(res)
# for item in res_set:
#     print(item)
#
# cursor.close()
connection.close()
