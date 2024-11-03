import traceback
from functools import wraps

import customtkinter as ctk

from utils.CTKAlertDialog import CTkAlertDialog
from utils.message import *
from utils.settings import *


def exception_handler_decorator(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            # print(f"{func.__name__} Executing")
            a = func(*args, **kwargs)
            # print(f"{func.__name__} Executed Done")
            return a
        except Exception as e:
            tb_str = "".join(
                traceback.format_exception(type(e), value=e, tb=e.__traceback__)
            )

            # 打印文件名、行号和异常信息
            print(f"An error occurred: {e}")
            print(f"Traceback info: {tb_str}")

            # 假设你有一个 CTkAlertDialog 实例，用于显示错误信息
            CTkAlertDialog(
                title="HBIM Error",
                text=str(e),
                font=ctk.CTkFont(family="LXGW WenKai", size=22),
            )
            return None

    return wrapper
