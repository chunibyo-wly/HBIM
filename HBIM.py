import os
import sys
import tkinter

from flask.cli import F

WORKSPACE = os.path.dirname(os.path.abspath(__file__))
sys.path.append(WORKSPACE)

import queue
import threading
import time
import traceback
from functools import wraps

import cccorelib
import customtkinter as ctk
import pycc

from semregpy.component.column import ColumnComponent
from semregpy.core import SemRegPy
from utils.settings import Settings

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


def longtask():
    pass


class MainWindow:
    def __init__(self, pipe):
        CC.freezeUI(False)

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

        self.pipe = pipe
        self.variable = None
        self.pcd_path = None
        self.CC_pcd = None
        self.mesh_path = None
        self.CC_mesh = None
        self.params = pycc.FileIOFilter.LoadParameters()
        self.params.parentWidget = CC.getMainWindow()
        self.params.alwaysDisplayLoadDialog = False
        self.initUI()

    @exception_handler_decorator
    def initUI(self):
        self.root = ctk.CTk()
        # 设置全局字体
        self.root.title("SemRegPy 动态建模插件演示 (c) HKU 2019-2022")
        self.root.resizable(True, True)
        self.root.iconbitmap(os.path.join(WORKSPACE, "hku_logo.ico"))
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

        self.initTabDemo()
        self.initTabSetup()
        self.initTabAPI()
        self.initTabAbout()
        self.root.mainloop()

    @exception_handler_decorator
    def send(self, msg):
        self.pipe.put(msg)

    def initTabDemo(self):
        # ! Step 1 : PCD load button // cur_row : 0
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
        ).grid(row=cur_row, column=1, columnspan=3, sticky="ew", pady=5, padx=2)
        # ctk.CTkButton(
        #     self.tabDemo,
        #     text="清除",
        #     command=self.clearDB,
        #     font=self.font,
        # ).grid(row=cur_row, column=2, columnspan=1, sticky="ew", pady=5, padx=2)
        # ctk.CTkButton(
        #     self.tabDemo,
        #     text="关闭 Pipe",
        #     command=lambda: self.send("exit"),
        #     font=self.font,
        # ).grid(row=cur_row, column=3, columnspan=1, sticky="ew", pady=5, padx=2)

        # ! Step 2 : select a target cloud // cur_row : 1
        cur_row += 1
        ctk.CTkLabel(
            self.tabDemo,
            text="1. 加载原始点云",
            anchor="w",
            font=self.font,
        ).grid(column=0, row=cur_row, sticky="ew")
        self.pcd_select_btn = ctk.CTkButton(
            self.tabDemo,
            text="本地点云选择",
            command=self.select_target_file,
            font=self.font,
        )
        self.pcd_select_btn.grid(
            row=cur_row, column=1, columnspan=3, sticky="ew", pady=5, padx=2
        )

        # ! Step 3 : select a target problem // cur_row : 2
        cur_row += 1
        ctk.CTkLabel(
            self.tabDemo,
            text="2. 加载 BIM 组件",
            anchor="w",
            font=self.font,
        ).grid(column=0, row=cur_row, sticky="ew")
        self.variable = ctk.StringVar()
        ctk.CTkComboBox(
            self.tabDemo,
            width=30,
            state="readonly",
            variable=self.variable,
            values=("Column", "Door", "Office"),
        ).grid(row=cur_row, column=1, columnspan=2, sticky="ew", padx=2)
        self.mesh_select_btn = ctk.CTkButton(
            self.tabDemo,
            text="本地 BIM 组件选择",
            command=self.select_target_BIM_compp,
            font=self.font,
        )
        self.mesh_select_btn.grid(
            row=cur_row, column=3, columnspan=1, sticky="ew", pady=5, padx=2
        )

        # ! Step 4 : match --> confirm or decline // cur_row : 3
        cur_row += 1
        ctk.CTkLabel(
            self.tabDemo,
            text="3. 执行三维重建",
            anchor="w",
            font=self.font,
        ).grid(column=0, row=cur_row, sticky="ew", padx=2)
        self.process_btn = ctk.CTkButton(
            self.tabDemo,
            text="快速配准",
            command=self.run_comp_callback,
            font=self.font,
            state="disabled",
        )
        self.process_btn.grid(
            row=cur_row, column=1, columnspan=1, sticky="ew", pady=5, padx=2
        )
        ctk.CTkButton(
            self.tabDemo,
            text="自动运行",
            command=self.iter_comp_callback,
            font=self.font,
        ).grid(row=cur_row, column=2, columnspan=1, sticky="ew", pady=5, padx=2)
        ctk.CTkButton(
            self.tabDemo,
            text="停止运行",
            command=self.iter_stop,
            font=self.font,
        ).grid(row=cur_row, column=3, columnspan=1, sticky="ew", pady=5, padx=2)

        # ! Step 5 : match --> export // cur_row : 4
        cur_row += 1
        ctk.CTkLabel(
            self.tabDemo,
            text="4. 存储结果",
            anchor="w",
            font=self.font,
        ).grid(column=0, row=cur_row, sticky="ew", padx=2)
        ctk.CTkButton(
            self.tabDemo,
            text="存储路径选择",
            command=self.export_json,
            font=self.font,
        ).grid(row=cur_row, column=1, columnspan=3, sticky="ew", pady=5, padx=2)

        for i in range(4):
            self.tabDemo.grid_columnconfigure(i, weight=1)  # 列随着窗口变化

    def initTabSetup(self):
        pass

    def initTabAPI(self):
        pass

    def initTabAbout(self):
        # Tab 4 ========= tabAbout ==============

        self.scrAbout = ctk.CTkTextbox(
            self.tabAbout,
            corner_radius=0,
            font=ctk.CTkFont(family="LXGW WenKai", size=16),
        )
        self.scrAbout.pack(expand=True, fill="both")
        self.scrAbout.insert(
            ctk.INSERT,
            """\
    动态建模插件 SemRegPy

    本插件以 Xue 等 (2019) 提出的语义配准 (Semantic Registration) 技术为基础，针对《面向粵港澳歷史文化保護傳承的虛擬現實技術研究與應用》(The applications of Virtual Reality technologies for cultural heritage conservation in the Guangdong-Hong Kong-Macao Greater Bay Area)项目而特别研发。

    通过 semregpy.core, semregpy.fitness, 和 semregpy.component 等 Python 接口，各类 GIS/BIM API 平台（例如 ArcGIS Pro 和 Revit）和应用提供动态建模的功能。（接口的参数类型和调用顺序，以本演示程序的内嵌流程为例）

    参考文献
    Xue, F., Lu, W., Chen, K., & Zetkulic, A. (2019). From semantic segmentation to semantic registration: Derivative-Free Optimization–based approach for automatic generation of semantically rich as-built Building Information Models from 3D point clouds. Journal of Computing in Civil Engineering, 33(4), 04019024.
    """,
        )
        self.scrAbout.configure(state=ctk.DISABLED)

    @exception_handler_decorator
    def get_entities(self):
        if not CC.haveSelection():
            entities = CC.dbRootObject()
            print(help(entities))
            entities = [
                (
                    entities.getChild(i)
                    if entities.getChild(i).isHierarchy()
                    else entities.getChild(i).getChild(0)
                )
                for i in range(entities.getChildrenNumber())
            ]
        else:
            entities = CC.getSelectedEntities()
        return entities

    @exception_handler_decorator
    def loadall(self):
        pcd_test_path = os.path.join(WORKSPACE, Settings.pcd_test_path)
        mesh_test_path = os.path.join(WORKSPACE, Settings.mesh_test_path)
        pcd_test = CC.loadFile(pcd_test_path, self.params)
        mesh_test = CC.loadFile(mesh_test_path, self.params)
        pcd_test.setEnabled(False)
        mesh_test.setEnabled(False)
        CC.updateUI()
        # entities = self.get_entities()
        # a = entities.getChildrenNumber()
        # print(a)
        return

        solver = SemRegPy()
        solver.VERBOSE = False
        solver.load_prob_file(pcd_test_path)
        bim_family = solver.load_mesh_file(mesh_test_path)
        if bim_family is None:
            bim_family = ColumnComponent()
        solver.solve(bim_family, max_eval=200)

    @exception_handler_decorator
    def select_target_file(self):
        tmp = ctk.filedialog.askopenfilename(
            filetypes=[
                ("PLY files", "*.ply"),
                ("PCD files", "*.pcd"),
            ],
            initialdir=WORKSPACE,
        )
        if tmp is None or tmp == "":
            return
        self.pcd_path = tmp
        self.pcd_select_btn.configure(text=os.path.basename(tmp))
        print(f"Selected file: {self.pcd_path}")
        self.CC_pcd = CC.loadFile(self.pcd_path, self.params)
        self.process_btn.configure(state="normal")

    @exception_handler_decorator
    def select_target_BIM_compp(self):
        tmp = ctk.filedialog.askopenfilename(
            filetypes=[("BIM component file", "*.obj")], initialdir=WORKSPACE
        )
        if tmp is None or tmp == "":
            return
        self.mesh_path = tmp
        self.mesh_select_btn.configure(text=os.path.basename(tmp))
        print(f"Selected file: {self.mesh_path}")

        # BIM family
        if "column" in self.mesh_path.lower():
            self.variable.set("Column")
        elif "door" in self.mesh_path.lower():
            self.variable.set("Door")
        elif "office" in self.mesh_path.lower():
            self.variable.set("Office")

    @exception_handler_decorator
    def run_comp_callback(self):
        raise NotImplementedError("Not implemented yet")

    @exception_handler_decorator
    def iter_comp_callback(self):
        raise NotImplementedError("Not implemented yet")

    @exception_handler_decorator
    def iter_stop(self):
        raise NotImplementedError("Not implemented yet")

    @exception_handler_decorator
    def export_json(self):
        raise NotImplementedError("Not implemented yet")


def initUI(pipe):
    MainWindow(pipe)


if __name__ == "__main__":
    pipe = queue.Queue()
    initUI(pipe)
