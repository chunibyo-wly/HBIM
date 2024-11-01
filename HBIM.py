import os
import sys
from functools import wraps

WORKSPACE = os.path.dirname(os.path.abspath(__file__))
sys.path.append(WORKSPACE)

import tkinter as tk
from tkinter import font as tkFont
from tkinter import messagebox, ttk

import cccorelib
import nlopt
import pycc

# import sv_ttk
import customtkinter as ctk

# from semregpy.core import SemRegPy

CC = pycc.GetInstance()
ctk.set_default_color_theme("green")
ctk.FontManager.load_font(os.path.join(WORKSPACE, "LXGWWenKai.ttf"))


class CTkAlertDialog(ctk.CTkToplevel):
    """
    Dialog with extra window, message, cancel and ok button.
    For detailed information check out the documentation.
    """

    def __init__(
        self,
        fg_color=None,
        text_color=None,
        button_fg_color=None,
        button_hover_color=None,
        button_text_color=None,
        title: str = "CTkDialog",
        font=None,
        text: str = "CTkDialog",
    ):

        super().__init__(fg_color=fg_color)

        self._fg_color = (
            ctk.ThemeManager.theme["CTkToplevel"]["fg_color"]
            if fg_color is None
            else self._check_color_type(fg_color)
        )
        self._text_color = (
            ctk.ThemeManager.theme["CTkLabel"]["text_color"]
            if text_color is None
            else self._check_color_type(button_hover_color)
        )
        self._button_fg_color = (
            ctk.ThemeManager.theme["CTkButton"]["fg_color"]
            if button_fg_color is None
            else self._check_color_type(button_fg_color)
        )
        self._button_hover_color = (
            ctk.ThemeManager.theme["CTkButton"]["hover_color"]
            if button_hover_color is None
            else self._check_color_type(button_hover_color)
        )
        self._button_text_color = (
            ctk.ThemeManager.theme["CTkButton"]["text_color"]
            if button_text_color is None
            else self._check_color_type(button_text_color)
        )
        self._is_ok: bool = False
        self._running: bool = False
        self._title = title
        self._text = text
        self._font = font

        self.title(self._title)
        self.lift()  # lift window on top
        self.attributes("-topmost", True)  # stay on top
        self.protocol("WM_DELETE_WINDOW", self._on_closing)
        self.after(
            10, self._create_widgets
        )  # create widgets with slight delay, to avoid white flickering of background
        self.resizable(False, False)
        self.grab_set()  # make other windows not clickable

    def _create_widgets(self):
        self.grid_columnconfigure((0, 1), weight=1)
        self.rowconfigure(0, weight=1)

        self._label = ctk.CTkLabel(
            master=self,
            width=300,
            wraplength=300,
            fg_color="transparent",
            text_color=self._text_color,
            text=self._text,
            font=self._font,
        )
        self._label.grid(
            row=0, column=0, columnspan=2, padx=20, pady=20, sticky="ew"
        )

        self._ok_button = ctk.CTkButton(
            master=self,
            width=100,
            border_width=0,
            fg_color=self._button_fg_color,
            hover_color=self._button_hover_color,
            text_color=self._button_text_color,
            text="Ok",
            font=self._font,
            command=self._ok_event,
        )
        self._ok_button.grid(
            row=2,
            column=0,
            columnspan=1,
            padx=(20, 10),
            pady=(0, 20),
            sticky="ew",
        )

        self._cancel_button = ctk.CTkButton(
            master=self,
            width=100,
            border_width=0,
            fg_color=self._button_fg_color,
            hover_color=self._button_hover_color,
            text_color=self._button_text_color,
            text="Cancel",
            font=self._font,
            command=self._cancel_event,
        )
        self._cancel_button.grid(
            row=2,
            column=1,
            columnspan=1,
            padx=(10, 20),
            pady=(0, 20),
            sticky="ew",
        )

    def _ok_event(self, event=None):
        self._is_ok = True
        self.grab_release()
        self.destroy()

    def _on_closing(self):
        self.grab_release()
        self.destroy()

    def _cancel_event(self):
        self._is_ok = False
        self.grab_release()
        self.destroy()

    def get_state(self):
        self.master.wait_window(self)
        return self._is_ok


def exception_handler_decorator(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            print(f"{func.__name__} Executing")
            a = func(*args, **kwargs)
            print(f"{func.__name__} Executed Done")
            CC.freezeUI(False)
            return a
        except Exception as e:
            print(f"An error occurred: {e}")
            CTkAlertDialog(
                title="HBIM Error",
                text=str(e),
                font=ctk.CTkFont(family="LXGW WenKai", size=22),
            )
            return None

    return wrapper


class MainWindow:
    def __init__(self):
        # self.solver = SemRegPy()
        # self.solver.VERBOSE = False
        self.current_mesh = None
        self.json_results = []
        self.mesh_results = set()
        self.UI_REFRESH = 1000

        self.cmbParaAlg = None
        self.cmbParaIter = None
        self.cmbParaIter_complex = None
        self.cmbParaRefresh = None

        self.targetcloud = {}
        self.targetBIMcomp = {}
        # self.cc_mesh_result = pycc.ccHObject("Result")
        # CC.addToDB(self.cc_mesh_result)

        self.root_path = WORKSPACE
        self.initUI()

    @exception_handler_decorator
    def initUI(self):
        self.root = ctk.CTk()
        # 设置全局字体
        self.root.title("SemRegPy 动态建模插件演示 (c) HKU 2019-2022")
        self.root.resizable(True, True)
        self.root.iconbitmap(os.path.join(self.root_path, "hku_logo.ico"))
        self.root.geometry("800x300")
        self.font = ctk.CTkFont(family="LXGW WenKai", size=22)

        self.tabs = ctk.CTkTabview(self.root)
        self.tabs._segmented_button.configure(
            font=ctk.CTkFont(family="LXGW WenKai", size=16)
        )
        self.tabDemo = self.tabs.add("  流程演示  ")
        self.tabSetup = self.tabs.add("  参数设置  ")
        self.tabAPI = self.tabs.add("  API 调用说明  ")
        self.tabAbout = self.tabs.add("  关于  ")

        self.tabs.pack(expand=True, fill="both")

        # ! Tab 1 ========= tabDemo ==============
        # Step 1 : PCD load button // cur_row : 0
        cur_row = 0
        ctk.CTkLabel(
            self.tabDemo,
            text="开发环境测试文件加载演示",
            anchor="w",
            font=self.font,
        ).grid(row=cur_row, column=0, sticky="ew")
        ctk.CTkButton(
            self.tabDemo,
            text="加载测试文件",
            command=self.loadall,
            font=self.font,
        ).grid(row=cur_row, column=1, columnspan=3, sticky="ew")

        # Step 2 : select a target cloud // cur_row : 1
        cur_row += 1
        ctk.CTkLabel(
            self.tabDemo,
            text="1. 选择目标点云",
            anchor="w",
            font=self.font,
        ).grid(column=0, row=cur_row, sticky="ew")
        ctk.CTkButton(
            self.tabDemo,
            text="选择目标点云",
            command=self.selecttargetfile,
            font=self.font,
        ).grid(row=cur_row, column=1, columnspan=3, sticky="ew")

        for i in range(3):
            self.tabDemo.grid_rowconfigure(i, weight=1)  # 行随着窗口变化
        for i in range(4):
            self.tabDemo.grid_columnconfigure(i, weight=1)  # 列随着窗口变化
        # sv_ttk.set_theme("light")
        self.root.mainloop()

    @exception_handler_decorator
    def loadall(self):
        raise NotImplementedError("Not implemented yet")

    @exception_handler_decorator
    def selecttargetfile(self):
        raise NotImplementedError("Not implemented yet")


MainWindow()
