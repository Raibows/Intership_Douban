'''
（必做）使用OOP编程实现用户登录业务流程。定义类UsersBiz，在类中定义一个实现登录的方法login(……)，
该方法返回bool类型数据。调用login方法传入相关数据进行业务逻辑验证。
说明：

 用户输入账号和密码
 进行非空验证
 账号密码正确验证（固定数据即可）
 若3次登录失败则退出（每次错误要提示用户还剩余几次机会）
'''

USERNAME = '123456'
PASSWORD = '123456'
MAX_TRY = 3


class UserBiz():
    def __init__(self, username:str, password:str, max_try:int):
        self.correct_username = username
        self.correct_password = password
        self.reserve_try_time = max_try

    def login(self, username:str, password:str) -> bool:
        if self.reserve_try_time <= 0: return False
        if username != '' and password != '':
            if self.correct_username == username and self.correct_password == password:
                return True
        self.reserve_try_time -= 1
        return False


userbiz = UserBiz(username=USERNAME, password=PASSWORD, max_try=MAX_TRY)

while userbiz.reserve_try_time > 0:
    username = input('please input the username: ')
    password = input('please input the password: ')
    if userbiz.login(username=username, password=password):
        print('login successfully!')
        break
    else:
        print(f'login failed, you have only {userbiz.reserve_try_time} times!')

