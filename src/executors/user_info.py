class UserInfo:
    def __init__(self, _i_net_id, _entry_num, _password, _p_ars_num, _rate):
        self.__i_net_id = _i_net_id
        self.__entry_num = _entry_num
        self.__password = _password
        self.__p_ars_num = _p_ars_num
        self.__rate = _rate
        self.__balance = 0

    @property
    def i_net_id(self):
        return self.__i_net_id
    @i_net_id.setter
    def i_net_id(self, value):
        self.__i_net_id = value
        
    @property
    def entry_num(self):
        return self.__entry_num
    @entry_num.setter
    def entry_num(self, value):
        self.__entry_num = value
        
    @property
    def password(self):
        return self.__password
    @password.setter
    def password(self, value):
        self.__password = value
        
    @property
    def p_ars_num(self):
        return self.__p_ars_num
    @p_ars_num.setter
    def p_ars_num(self, value):
        self.__p_ars_num = value
        
    @property
    def rate(self):
        return self.__rate
    @rate.setter
    def rate(self, value):
        self.__rate = value

    @property
    def balance(self):
        return self.__balance
    @balance.setter
    def balance(self, value):
        self.__balance = value
