import subprocess
import os
import sys
from tkinter import messagebox
from tkinter.filedialog import askopenfilename
from pathlib import Path
import time
import webbrowser
from tkinter import Tk, Canvas, Frame, Label, Button, Entry, Listbox, Scrollbar
#import tkinter.ttk as ttk
import json
#from MEEGbuddy import MEEGbuddy

class TMS_EEG_GUI(Frame):

    EXP = None
    BUDDY = None

    def __init__(self, root):
        self.root = root
        self.root.title('TMS-EEG GUI')
        self.init_data_dir()
        width = self.root.winfo_screenwidth()
        height = self.root.winfo_screenheight()
        self.size = min([height,width])
        self.header_font = ('Helvetica',int(self.size*24/1080))
        self.text_font = ('Helvetica',int(self.size*14/1080))
        Frame.__init__(self, root, width=self.size, height=self.size)
        self.load_or_init_screen()
        self.pack()

    def init_data_dir(self):
        self.data_dir = '../TMS_EEG_GUI_data'
        if not os.path.isdir(self.data_dir):
            os.makedirs(self.data_dir)
        sub_dirs_names = ['saved_exp']
        self.sub_dirs = {}
        for sub_dir_name in sub_dirs_names:
            sub_dir_full_path = os.path.join(self.data_dir,sub_dir_name)
            if not os.path.isdir(sub_dir_full_path):
                os.makedirs(sub_dir_full_path)
            self.sub_dirs[sub_dir_name] = sub_dir_full_path

    def load_or_init_screen(self):
        def getFrame(y0,y1,x0,x1):
            frame = Frame(self.root,
                          height=(y1-y0)*self.size-self.size/100,
                          width=(x1-x0)*self.size-self.size/100)
            frame.pack_propagate(0)
            frame.place(x=x0*self.size,y=y0*self.size)
            return frame
        wraplength = 0.1*self.size
        # Headers
        Label(getFrame(0,0.2,0,0.5),text='Create New',
              font=self.header_font).pack(fill='both',expand=1)
        Label(getFrame(0,0.2,0.5,1),text='Load',
              font=self.header_font).pack(fill='both',expand=1)
        # File loaders
        Label(getFrame(0.2,0.3,0,0.1),text='Load new raw file',
              wraplength=wraplength,font=self.text_font
              ).pack(fill='both',expand=1)
        new_file = Entry(getFrame(0.2,0.3,0.1,0.4),font=self.header_font)
        new_file.pack(fill='both',expand=1)
        new_file.focus_set()
        load_new_file_button = Button(getFrame(0.2,0.3,0.4,0.5),text='Browse',
                                      font=self.text_font)
        load_new_file_button.pack(fill='both',expand=1)
        load_files = [f for f in os.listdir(self.sub_dirs['saved_exp'])
                      if f.split('.')[-1] == 'json']
        Label(getFrame(0.2,0.3,0.5,0.6),text='Load previous',wraplength=wraplength,
              font=self.text_font).pack(fill='both',expand=1)
        load_file_list = Listbox(getFrame(0.2,0.3,0.6,0.85),height=3,
                                 font=self.header_font)
        load_file_list.pack(fill='both',expand=1)
        for lf in [''] + load_files:
            load_file_list.insert('end',lf)
        scrollbar = Scrollbar(getFrame(0.2,0.3,0.85,0.9), orient='vertical')
        scrollbar.config(command=load_file_list.yview)
        scrollbar.pack(fill='both',expand=1)
        load_previous_button = Button(getFrame(0.2,0.3,0.9,1),text='Load',
                                      font=self.text_font)
        load_previous_button.pack(fill='both',expand=1)
        # experiment data
        Label(getFrame(0.3,0.4,0,0.1),text='Name',
              font=self.text_font).pack(fill='both',expand=1)
        name_entry = Entry(getFrame(0.3,0.4,0.1,0.5),font=self.header_font)
        name_entry.pack(fill='both',expand=1)
        name_entry.focus_set()
        Label(getFrame(0.3,0.4,0.5,0.6),text='Date',
              font=self.text_font).pack(fill='both',expand=1)

if __name__ == '__main__':
    root = Tk()
    gui = TMS_EEG_GUI(root)
    root.mainloop()