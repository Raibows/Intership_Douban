from .base import oh_my_database
import re

class douban_db(oh_my_database):
    def __init__(self, config, logger):
        super().__init__(config, logger)
        self.text_limit = [50, 50, float, 50, 150, 150, 50, 10, 300, 110]
        self.table_name = 'douban_data'

    def check_params(self, params):
        if params:
            for i in range(len(params)):
                if isinstance(self.text_limit[i], int):
                    params[i] = params[i][:self.text_limit[i]]
                else:
                    params[i] = self.text_limit[i](params[i])
        return params

    def insert_item(self, params):
        params = self.check_params(params)

        sql = f"select * from {self.table_name} " \
              f"where imdb='{params[-3]}'"
        res = self.execute(sql, params=None, is_fetch=True)
        if len(res) == 0:
            sql = f"insert into  {self.table_name}" \
                  "(name, director, rate, genres, page_url, pic_url, region, imdb, brief, targets) " \
                  "values (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s);"

            return self.execute(sql, params)
        else:
            res = res[0]
            if res[-1] == 'NULL' and params[-1] != 'NULL':
                sql = f"update douban_data set targets='{params[-1]}' where imdb='{params[-3]}'"
                return self.execute(sql)
