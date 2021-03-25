import pymysql
from contextlib import closing
import json


class oh_my_database():
    def __init__(self, config, logger):
        if isinstance(config, str):
            self.__config = json.load(open(config))
        elif isinstance(config, dict):
            self.__config = config
        self.logger = logger
        self.__connection = None
        self.commit_interval = 10
        self.execute_time = 0

    def __set_connection(self):
        if self.__connection == None:
            try:
                self.__connection = pymysql.connect(**self.__config)
            except Exception as e:
                self.logger.error(f'database connect failed! {e}')
                self.close()
                raise e
            self.logger.info('database connected successfully')

    def __commit(self):
        self.execute_time += 1
        if self.execute_time >= self.commit_interval:
            self.execute_time = 0
            self.__set_connection()
            self.__connection.commit()

    def __force_commit(self):
        self.execute_time = self.commit_interval
        self.__commit()

    def execute(self, sql, params=None, is_fetch=False, is_update=False):
        self.__set_connection()
        with closing(self.__connection.cursor()) as cursor:
            try:
                if is_fetch:
                    self.__force_commit()
                    res = cursor.execute(sql, args=params)
                    return cursor.fetchall()
                if is_update:
                    res = cursor.execute(sql, args=params)
                    self.__force_commit()
                    return res
                else:
                    res = cursor.execute(sql, args=params)
                    self.__commit()
                    return res
            except Exception as e:
                self.logger.error(f'database execute error {e}')
                self.logger.error(f'database execute error with sql and params \n{sql}\n{params}')
                self.close()
                raise e

    def close(self):
        if self.__connection:
            self.__force_commit()
            self.__connection.close()
            self.__connection = None
        self.logger.info(f'database close successfully with remaining execute time {self.execute_time}')

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def __del__(self):
        self.close()

