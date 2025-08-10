class UserInfo:
    """ ユーザー情報を管理するクラス """
    def __init__(self, net_keiba_id: str, net_keiba_password: str,
                 i_net_id: str, entry_num: str, password: str, p_ars_num: str, rate: int):
        """ ユーザー情報を初期化する
            Args:
                net_keiba_id (str): ネット競馬ID
                net_keiba_password (str): ネット競馬パスワード
                i_net_id (str): I-net ID
                entry_num (str): エントリーナンバー
                password (str): パスワード
                p_ars_num (str): P-ARSナンバー
                rate (int): レート
        """
        self._net_keiba_id = net_keiba_id
        self._net_keiba_password = net_keiba_password
        self._i_net_id = i_net_id
        self._entry_num = entry_num
        self._password = password
        self._p_ars_num = p_ars_num
        self._rate = rate

    @property
    def net_keiba_id(self):
        return self._net_keiba_id
    @net_keiba_id.setter
    def net_keiba_id(self, value):
        self._net_keiba_id = value

    @property
    def net_keiba_password(self):
        return self._net_keiba_password
    @net_keiba_password.setter
    def net_keiba_password(self, value):
        self._net_keiba_password = value

    @property
    def i_net_id(self):
        return self._i_net_id
    @i_net_id.setter
    def i_net_id(self, value):
        self._i_net_id = value
        
    @property
    def entry_num(self):
        return self._entry_num
    @entry_num.setter
    def entry_num(self, value):
        self._entry_num = value
        
    @property
    def password(self):
        return self._password
    @password.setter
    def password(self, value):
        self._password = value
        
    @property
    def p_ars_num(self):
        return self._p_ars_num
    @p_ars_num.setter
    def p_ars_num(self, value):
        self._p_ars_num = value
        
    @property
    def rate(self):
        return self._rate
    @rate.setter
    def rate(self, value):
        self._rate = value

    def to_dict(self) -> dict:
        return {
            "net_keiba_id": self._net_keiba_id,
            "net_keiba_password": self._net_keiba_password,
            "i_net_id": self._i_net_id,
            "entry_num": self._entry_num,
            "password": self._password,
            "p_ars_num": self._p_ars_num,
            "rate": self._rate,
        }

    @classmethod
    def from_dict(cls, data: dict):
        obj = cls(
            net_keiba_id=data["net_keiba_id"],
            net_keiba_password=data["net_keiba_password"],
            i_net_id=data["i_net_id"],
            entry_num=data["entry_num"],
            password=data["password"],
            p_ars_num=data["p_ars_num"],
            rate=data["rate"]
        )
        return obj

