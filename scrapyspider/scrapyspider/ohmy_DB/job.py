from .base import oh_my_database
import re


class job_db(oh_my_database):
    def __init__(self, config, logger):
        super().__init__(config, logger)
        self.text_max_len_limit = [50, 50, 50, 50, 20, 50, -1, 300]
        self.table_name = 'job_data'

    def check_params(self, params):
        if params:
            for i in range(len(params)):
                params[i] = params[i][:self.text_max_len_limit[i]]
            return params

    def process_salary(self, salary:str):
        # 全部将薪资转换为小时，默认22天，8小时工作制, return low, high, mean
        last_char_dict = {'月': 22 * 8, '年': 12 * 22 * 8, '日': 8, '天': 8, '时': 1}
        unit_char_dict = {'千': 1000, '万': 10000, '百': 100, '元': 1, '块': 1}

        if (not isinstance(salary, str)) or \
            salary.isspace() or \
            len(salary) <= 2 or \
            (not salary[-1] in last_char_dict):
            return [-1.0, -1.0, -1.0] # invalid data

        denominator = last_char_dict[salary[-1]]

        # find the numerator
        pattern = "[千万百元块]"
        unit_char = re.findall(pattern, salary)
        if len(unit_char) == 0:
            return [-2.0, -2.0, -2.0] # invalid data
        unit_char = unit_char[0]

        numerator = unit_char_dict[unit_char]

        #  find the real float salary
        pattern = "\d+(?:\.\d+)?"
        res = re.findall(pattern, salary)
        if len(res) == 0 or len(res) > 2:
            return [-3.0, -3.0, -3.0] # invalid data
        if len(res) == 1:
            res = float(res[0]) * numerator / denominator
            return [res, res, res]

        low = float(res[0]) * numerator / denominator
        high = float(res[1]) * numerator / denominator
        return [low, high, (low+high)/2]

    def process_job_city(self, job_address:str):
        return job_address.split('-')[0]

    def insert_item(self, params):
        # the salary index is 3
        params = self.check_params(params)
        params += self.process_salary(params[3])
        params.append(self.process_job_city(params[2]))

        sql = f"insert into  {self.table_name}" \
              "(job_name, job_company, job_address, job_salary, job_type, job_date, job_url, " \
              "job_info, job_salary_low, job_salary_high, job_salary_mean, job_city) " \
              "values (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s);"

        return self.execute(sql, params)

    def statistic_job_type_count(self):
        sql = f"select job_type, count(job_type) from {self.table_name} group by job_type;"
        return self.execute(sql, is_fetch=True)
    
    def statistic_job_type_salary(self, salary_type):
        assert salary_type in {'mean', 'low', 'high'}
        salary_type = f"job_salary_{salary_type}"
        sql = f"select job_type, avg({salary_type}) from {self.table_name} " \
              f"where {salary_type} > 0 group by job_type;"
        return self.execute(sql, is_fetch=True)

    def statistic_job_city_salary(self, salary_type):
        assert salary_type in {'mean', 'low', 'high'}
        salary_type = f"job_salary_{salary_type}"
        sql = f"select job_city, avg({salary_type}) from {self.table_name} " \
              f"where {salary_type} > 0 group by job_city;"
        return self.execute(sql, is_fetch=True)



if __name__ == '__main__':
    pass