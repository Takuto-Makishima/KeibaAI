import requests


class Notification:
    """ 通知 """
    @staticmethod
    def send(message: str) -> None:
        """ 通知送信 
            Args:
                message (str): メッセージ
            Returns:
                None
        """
        url = 'https://discord.com/api/webhooks/1309626579579830344/SVyePgww0pAXVD6arRMIqWcTs3GxXxcHolYc4pEsULa0veTa4FqQ16q3S4G5HVqTJhah'
        data = {"content": message}
        try:
            requests.post(url, data=data)
        except Exception as e:
            print(f"エラー：{e}")
