#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Главное меню для запуска различных модулей системы классификации лояльности клиентов.
"""

import os
import sys
import subprocess
import tkinter as tk
from tkinter import ttk, messagebox, PhotoImage
import importlib.util

class MainGUI:
    """Главное меню для запуска различных модулей системы"""
    
    def __init__(self, root):
        self.root = root
        self.root.title("Система классификации лояльности клиентов Acoola")
        self.root.geometry("700x450")
        self.root.minsize(650, 400)
        
        # Создание стиля
        self.style = ttk.Style()
        self.style.configure("TFrame", background="#f0f0f0")
        self.style.configure("TButton", font=("Arial", 11))
        self.style.configure("TLabel", font=("Arial", 11), background="#f0f0f0")
        self.style.configure("Header.TLabel", font=("Arial", 16, "bold"), background="#f0f0f0")
        self.style.configure("Description.TLabel", font=("Arial", 10), background="#f0f0f0")
        self.style.configure("BigButton.TButton", font=("Arial", 12, "bold"))
        
        # Создание главного фрейма
        self.main_frame = ttk.Frame(root, padding=20)
        self.main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Заголовок
        ttk.Label(self.main_frame, text="Система классификации лояльности клиентов Acoola", 
                  style="Header.TLabel").pack(pady=10)
        
        ttk.Label(self.main_frame, text="Выпускная квалификационная работа", 
                  style="Description.TLabel").pack(pady=0)
        
        # Фрейм с кнопками модулей
        self.modules_frame = ttk.Frame(self.main_frame)
        self.modules_frame.pack(fill=tk.BOTH, expand=True, pady=20)
        
        # Настройка сетки
        self.modules_frame.columnconfigure(0, weight=1)
        self.modules_frame.columnconfigure(1, weight=1)
        
        # Кнопки модулей
        self.create_module_button(0, 0, "Обучение моделей", 
                                 "Запуск процесса обучения моделей классификации", 
                                 self.launch_train_gui)
        
        self.create_module_button(0, 1, "Демонстрация моделей", 
                                 "Визуализация и сравнение результатов обученных моделей", 
                                 self.launch_demo_gui)
        
        self.create_module_button(1, 0, "Препроцессинг", 
                                 "Предобработка данных и формирование признаков", 
                                 self.launch_preprocessing)
        
        self.create_module_button(1, 1, "Создание отчетов", 
                                 "Генерация отчетов о результатах классификации", 
                                 self.launch_reports)
        
        # Фрейм для кнопок внизу
        self.bottom_frame = ttk.Frame(self.main_frame)
        self.bottom_frame.pack(fill=tk.X, pady=10)
        
        # Кнопка выхода
        self.exit_button = ttk.Button(self.bottom_frame, text="Выход", command=self.root.quit)
        self.exit_button.pack(side=tk.RIGHT, padx=5)
        
        # Проверка зависимостей при запуске
        self.check_dependencies()
    
    def create_module_button(self, row, col, title, description, command):
        """Создание карточки модуля с кнопкой запуска"""
        frame = ttk.Frame(self.modules_frame, borderwidth=2, relief=tk.GROOVE)
        frame.grid(row=row, column=col, padx=10, pady=10, sticky=tk.NSEW)
        
        ttk.Label(frame, text=title, font=("Arial", 12, "bold")).pack(pady=(10, 5))
        ttk.Label(frame, text=description, wraplength=250).pack(pady=(0, 10))
        
        ttk.Button(frame, text="Запустить", command=command, style="BigButton.TButton").pack(pady=(0, 10))
    
    def launch_train_gui(self):
        """Запуск GUI для обучения моделей"""
        self.launch_module("train_models_gui.py", "Обучение моделей")
    
    def launch_demo_gui(self):
        """Запуск GUI для демонстрации моделей"""
        self.launch_module("model_demo_gui.py", "Демонстрация моделей")
    
    def launch_preprocessing(self):
        """Запуск модуля предобработки данных"""
        messagebox.showinfo("Препроцессинг", 
                          "Модуль препроцессинга запускается из GUI обучения моделей.\n\n"
                          "Выберите 'Обучение моделей' и в параметрах обучения выберите "
                          "'Полный пайплайн'.")
    
    def launch_reports(self):
        """Запуск генерации отчетов"""
        messagebox.showinfo("Отчеты", 
                          "Модуль создания отчетов запускается из GUI демонстрации моделей.\n\n"
                          "Выберите 'Демонстрация моделей' и с помощью параметров визуализации "
                          "сгенерируйте необходимые отчеты.")
    
    def launch_module(self, module_filename, module_name):
        """Запуск Python-модуля"""
        if not os.path.exists(module_filename):
            messagebox.showerror("Ошибка", f"Файл {module_filename} не найден!")
            return
        
        try:
            subprocess.Popen([sys.executable, module_filename])
        except Exception as e:
            messagebox.showerror("Ошибка", f"Не удалось запустить {module_name}:\n{str(e)}")
    
    def check_dependencies(self):
        """Проверка наличия необходимых зависимостей"""
        # Проверка наличия обязательных Python-модулей
        required_modules = ["numpy", "pandas", "sklearn", "matplotlib", "tkinter"]
        missing_modules = []
        
        for module in required_modules:
            if module != "tkinter":  # tkinter уже импортирован
                try:
                    importlib.util.find_spec(module)
                except ImportError:
                    missing_modules.append(module)
        
        if missing_modules:
            missing_str = ", ".join(missing_modules)
            messagebox.showwarning("Предупреждение", 
                                 f"Отсутствуют следующие модули Python: {missing_str}\n\n"
                                 f"Установите их с помощью pip:\n"
                                 f"pip install {' '.join(missing_modules)}")
        
        # Проверка наличия файла данных
        if not os.path.exists("dataset/Concept202408.csv"):
            messagebox.showwarning("Предупреждение", 
                                 "Файл данных dataset/Concept202408.csv не найден!\n\n"
                                 "Для работы системы необходимо поместить файл с данными "
                                 "в каталог dataset.")
        
        # Проверка наличия исходников скриптов
        if not os.path.exists("src/modeling/model_demo.py"):
            messagebox.showwarning("Предупреждение", 
                                 "Не найдены некоторые модули в каталоге src!\n\n"
                                 "Пожалуйста, убедитесь, что структура каталогов соответствует описанию в README.md.")
        
        # Проверка наличия GUI-модулей
        if not os.path.exists("train_models_gui.py") or not os.path.exists("model_demo_gui.py"):
            messagebox.showwarning("Предупреждение", 
                                 "Не найдены модули GUI для обучения или демонстрации моделей!\n\n"
                                 "Некоторые функции могут быть недоступны.")


if __name__ == "__main__":
    try:
        # Запуск GUI
        root = tk.Tk()
        app = MainGUI(root)
        root.mainloop()
    except Exception as e:
        print(f"Критическая ошибка: {str(e)}")
        if "tkinter" in str(e):
            print("Убедитесь, что в вашей системе установлен модуль tkinter")
        sys.exit(1) 