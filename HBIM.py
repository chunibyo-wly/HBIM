import math
import os
import sys

WORKSPACE = os.path.dirname(os.path.abspath(__file__))
sys.path.append(WORKSPACE)
import queue
import threading
import time

import cccorelib
import customtkinter as ctk
import pycc

CC = pycc.GetInstance()
ctk.set_default_color_theme("green")
ctk.FontManager.load_font(os.path.join(WORKSPACE, "LXGWWenKai.ttf"))

from semregpy import ColumnComponent, DoorComponent, OfficeComponent, SemRegPy
from utils import (
    EXIT,
    REGISTER_MULTI,
    Settings,
    SignalMessage,
    exception_handler_decorator,
)


def CC_translate(entity, translation=(0.0, 0.0, 0.0)):
    x, y, z = translation
    glMat = entity.getGLTransformation()
    translation = glMat.getTranslationAsVec3D()
    translation.x += x
    translation.y += y
    translation.z += z
    glMat.setTranslation(translation)

    entity.setGLTransformation(glMat)
    entity.applyGLTransformation_recursive()
    CC.redrawAll()


class MainWindow:
    def __init__(self, pipe_in, pipe_out):
        CC.freezeUI(False)

        # self.solver = SemRegPy()
        # self.solver.VERBOSE = False
        self.current_mesh = None
        self.json_results = []
        self.mesh_results = set()
        self.UI_REFRESH = 10

        self.cmbParaAlg = None
        self.cmbParaIter = None
        self.cmbParaIter_complex = None
        self.cmbParaRefresh = None

        self.targetcloud = {}
        self.targetBIMcomp = {}
        # self.cc_mesh_result = pycc.ccHObject("Result")
        # CC.addToDB(self.cc_mesh_result)

        self.pipe_in = pipe_in
        self.pipe_out = pipe_out
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
        self.pipe_in.put(msg)

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
            text="加载预设测试文件",
            command=self.loadall,
            font=self.font,
        ).grid(row=cur_row, column=1, columnspan=3, sticky="ew", pady=5, padx=2)

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
            text="单步配准",
            command=self.register_single,
            font=self.font,
            state="disabled",
        )
        self.process_btn.grid(
            row=cur_row, column=1, columnspan=1, sticky="ew", pady=5, padx=2
        )
        ctk.CTkButton(
            self.tabDemo,
            text="自动配准",
            command=self.register_multi,
            font=self.font,
        ).grid(row=cur_row, column=2, columnspan=1, sticky="ew", pady=5, padx=2)
        ctk.CTkButton(
            self.tabDemo,
            text="停止运行",
            command=self.register_multi_stop,
            font=self.font,
            state="disabled",
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

        cur_row += 1
        self.progress_bar = ctk.CTkProgressBar(
            self.tabDemo, orientation="horizontal", height=10
        )
        self.progress_bar.set(0)
        self.progress_bar.grid(
            row=cur_row,
            column=0,
            columnspan=4,
            sticky="ew",
            pady=5,
            padx=2,
        )

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
        self._set_pcd_path(os.path.join(WORKSPACE, Settings.pcd_test_path))
        self._set_mesh_path(os.path.join(WORKSPACE, Settings.mesh_test_path))
        CC.updateUI()

    @exception_handler_decorator
    def _set_pcd_path(self, pcd_path):
        self.pcd_path = pcd_path
        self.pcd_select_btn.configure(text=os.path.basename(pcd_path))
        print(f"Selected file: {self.pcd_path}")
        self.CC_pcd = CC.loadFile(self.pcd_path, self.params)
        self.process_btn.configure(state="normal")

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
        self._set_pcd_path(tmp)

    @exception_handler_decorator
    def _set_mesh_path(self, mesh_path):
        self.mesh_path = mesh_path
        self.mesh_select_btn.configure(text=os.path.basename(mesh_path))
        print(f"Selected file: {self.mesh_path}")
        # BIM family
        if "column" in self.mesh_path.lower():
            self.variable.set("Column")
        elif "door" in self.mesh_path.lower():
            self.variable.set("Door")
        elif "office" in self.mesh_path.lower():
            self.variable.set("Office")

    @exception_handler_decorator
    def select_target_BIM_compp(self):
        tmp = ctk.filedialog.askopenfilename(
            filetypes=[("BIM component file", "*.obj")], initialdir=WORKSPACE
        )
        if tmp is None or tmp == "":
            return
        self._set_mesh_path(tmp)

    @exception_handler_decorator
    def register_single(self):
        raise NotImplementedError("Not implemented yet")

    @exception_handler_decorator
    def _progress_fake(self):
        # 缓动函数：指数衰减（先快后慢）
        def ease_out_expo(t):
            return 1 - math.pow(2, -10 * t) if t < 0.95 else 0.95

        elapsed = time.time() - self.process_start_time
        progress_ratio = min(elapsed / 2, 0.95)  # 进度时间比例
        eased_progress = ease_out_expo(progress_ratio)  # 应用缓动函数
        return eased_progress

    @exception_handler_decorator
    def _query_result(self):
        if self.pipe_out.empty():
            value = self._progress_fake()
            self.progress_bar.set(value)
            self.root.after(self.UI_REFRESH, self._query_result)
        else:
            result = self.pipe_out.get()
            self.progress_bar.set(1)
            print(result)
            mesh_candidate = CC.loadFile(self.mesh_path, self.params)
            CC_translate(
                mesh_candidate, (result["t"][0], result["t"][1], result["t"][2])
            )

    @exception_handler_decorator
    def register_multi(self):
        msg = SignalMessage(REGISTER_MULTI)
        msg.message = {
            "pcd_path": self.pcd_path,
            "mesh_path": self.mesh_path,
            "bim_family": self.variable,
        }
        self.pipe_in.put(msg)
        self.progress_bar.set(0)
        self.root.update()
        self.process_start_time = time.time()
        self.root.after(self.UI_REFRESH, self._query_result)

    @exception_handler_decorator
    def register_multi_stop(self):
        raise NotImplementedError("Not implemented yet")

    @exception_handler_decorator
    def export_json(self):
        raise NotImplementedError("Not implemented yet")


@exception_handler_decorator
def execute_register_multi(params):
    pcd_path = params["pcd_path"]
    mesh_path = params["mesh_path"]
    bim_family_type = params["bim_family"]

    solver = SemRegPy()
    solver.VERBOSE = False
    solver.load_prob_file(pcd_path)
    bim_family = None
    if bim_family_type == "Column":
        bim_family = ColumnComponent()
    elif bim_family_type == "Door":
        bim_family = DoorComponent()
    elif bim_family_type == "Office":
        bim_family = OfficeComponent()

    tmp = solver.load_mesh_file(mesh_path)
    if bim_family is None:
        bim_family = tmp if tmp is not None else ColumnComponent()
    result = solver.solve(bim_family, max_eval=200)

    mesh_center = solver.mesh.pcd.origin.get_center()
    translation = (
        result["best_c"][0] - mesh_center[0],
        result["best_c"][1] - mesh_center[1],
        result["best_c"][2] - mesh_center[2],
    )
    rotation = result["best_rz"]
    return {
        "t": translation,
        "R": rotation,
    }


def background_task(pipe_in, pipe_out):
    """
    Always running in the background without blocking
    both the CloudCompare and the TKinter mainloop

    Handles the Long-running tasks:
        Register multiple BIM components
    """
    pycc.ccLog.Warning("Background task started")
    while True:
        try:
            msg = pipe_in.get_nowait()
            msg_text = msg.signal
            if msg_text == EXIT:
                break
            elif msg_text == REGISTER_MULTI:
                params = msg.message
                result = execute_register_multi(params)
                pipe_out.put(result)
        except queue.Empty:
            pass
        time.sleep(0.1)
    pycc.ccLog.Warning("Background task stopped")


if __name__ == "__main__":
    """
    tkinter    ---> pipe_in  ---> background (request)
    background ---> pipe_out ---> tkinter    (result)
    """

    pipe_in = queue.Queue()
    pipe_out = queue.Queue()
    t = threading.Thread(
        target=background_task,
        args=(
            pipe_in,
            pipe_out,
        ),
        daemon=True,
    )
    t.start()
    MainWindow(pipe_in, pipe_out)
    pipe_in.put(SignalMessage(EXIT))
    t.join()
