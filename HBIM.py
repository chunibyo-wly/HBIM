import tkinter
from tkinter import ttk

import sv_ttk

root = tkinter.Tk()

button = ttk.Button(root, text="Toggle theme", command=sv_ttk.toggle_theme)
button.pack()

root.mainloop()
