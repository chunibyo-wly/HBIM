import customtkinter as ctk


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
