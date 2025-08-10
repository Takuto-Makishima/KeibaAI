import json
from typing import Any

class JsonSerializer:
    """ JSON シリアライザ """
    @staticmethod
    def read(filepath: str) -> Any:
        """ JSON ファイルを読み込んで Python オブジェクトとして返す
            Args:
                filepath (str): 読み込む JSON ファイルのパス
            Returns:
                Any: 読み込んだデータ
        """
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"ファイルが見つかりません: {filepath}")
        except json.JSONDecodeError as e:
            print(f"JSON デコードエラー: {e}")
        return None

    @staticmethod
    def write(filepath: str, data: Any, indent: int = 4) -> None:
        """ Python オブジェクトを JSON ファイルとして書き込む
            Args:
                filepath (str): 書き込む JSON ファイルのパス
                data (Any): 書き込むデータ
                indent (int): インデントのスペース数
            Returns:
                None
        """
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=indent)
        except TypeError as e:
            print(f"シリアライズできません: {e}")
