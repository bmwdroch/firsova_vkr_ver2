#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Графический интерфейс для запуска обучения моделей классификации лояльности клиентов.
"""

import os
import sys
import subprocess
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from tkinter.scrolledtext import ScrolledText
import threading
import queue
import time

class RedirectText:
    """Класс для перенаправления вывода в текстовый виджет"""
    def __init__(self, text_widget, queue):
        self.text_widget = text_widget
        self.queue = queue

    def write(self, string):
        self.queue.put(string)

    def flush(self):
        pass

class TrainingGUI:
    """Графический интерфейс для обучения моделей"""
    
    def __init__(self, root):
        self.root = root
        self.root.title("Обучение моделей классификации лояльности")
        self.root.geometry("800x600")
        self.root.minsize(650, 500)
        
        # Создание стиля
        self.style = ttk.Style()
        self.style.configure("TFrame", background="#f0f0f0")
        self.style.configure("TButton", font=("Arial", 10))
        self.style.configure("TLabel", font=("Arial", 10), background="#f0f0f0")
        self.style.configure("TRadiobutton", font=("Arial", 10), background="#f0f0f0")
        self.style.configure("Header.TLabel", font=("Arial", 12, "bold"), background="#f0f0f0")
        
        # Создание главного фрейма
        self.main_frame = ttk.Frame(root, padding=10)
        self.main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Заголовок
        ttk.Label(self.main_frame, text="Обучение моделей классификации лояльности клиентов", 
                  style="Header.TLabel").pack(pady=10)
        
        # Фрейм для параметров
        self.params_frame = ttk.LabelFrame(self.main_frame, text="Параметры запуска этапов")
        self.params_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Новые чекбоксы для управления пропусками этапов
        self.skip_preprocessing_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(self.params_frame, text="Пропустить предобработку данных", 
                         variable=self.skip_preprocessing_var).grid(row=0, column=0, sticky=tk.W, padx=5, pady=2)

        self.skip_loyalty_features_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(self.params_frame, text="Пропустить создание признаков лояльности", 
                         variable=self.skip_loyalty_features_var).grid(row=1, column=0, sticky=tk.W, padx=5, pady=2)

        self.skip_model_training_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(self.params_frame, text="Пропустить обучение (использовать сохраненные модели)", 
                         variable=self.skip_model_training_var).grid(row=2, column=0, sticky=tk.W, padx=5, pady=2)
        
        # Фрейм для дополнительных опций
        self.options_frame = ttk.LabelFrame(self.main_frame, text="Дополнительные опции")
        self.options_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Опции
        self.save_intermediate = tk.BooleanVar(value=True)
        ttk.Checkbutton(self.options_frame, text="Сохранять промежуточные результаты", 
                         variable=self.save_intermediate).grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        
        self.balance_classes = tk.BooleanVar(value=False)
        ttk.Checkbutton(self.options_frame, text="Балансировка классов", 
                         variable=self.balance_classes).grid(row=1, column=0, sticky=tk.W, padx=5, pady=5)
        
        self.tune_hyperparams_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(self.options_frame, text="Настроить гиперпараметры",
                        variable=self.tune_hyperparams_var).grid(row=0, column=1, sticky=tk.W, padx=5, pady=5)

        self.create_ensemble_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(self.options_frame, text="Создать ансамблевые модели",
                        variable=self.create_ensemble_var).grid(row=1, column=1, sticky=tk.W, padx=5, pady=5)
        
        # Путь к датасету
        self.dataset_frame = ttk.Frame(self.main_frame)
        self.dataset_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(self.dataset_frame, text="Файл данных:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        
        self.dataset_path = tk.StringVar(value="dataset/Concept202408.csv")
        self.dataset_entry = ttk.Entry(self.dataset_frame, textvariable=self.dataset_path, width=50)
        self.dataset_entry.grid(row=0, column=1, padx=5, pady=5, sticky=tk.W+tk.E)
        
        self.browse_button = ttk.Button(self.dataset_frame, text="Обзор...", command=self.browse_dataset)
        self.browse_button.grid(row=0, column=2, padx=5, pady=5)
        
        # Путь для сохранения результатов
        ttk.Label(self.dataset_frame, text="Папка для результатов:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=5)
        
        self.output_path = tk.StringVar(value="output")
        self.output_entry = ttk.Entry(self.dataset_frame, textvariable=self.output_path, width=50)
        self.output_entry.grid(row=1, column=1, padx=5, pady=5, sticky=tk.W+tk.E)
        
        self.output_button = ttk.Button(self.dataset_frame, text="Обзор...", command=self.browse_output)
        self.output_button.grid(row=1, column=2, padx=5, pady=5)
        
        # Фрейм для кнопок управления
        self.buttons_frame = ttk.Frame(self.main_frame)
        self.buttons_frame.pack(fill=tk.X, pady=10)
        
        self.run_button = ttk.Button(self.buttons_frame, text="Запустить обучение", command=self.run_training)
        self.run_button.pack(side=tk.LEFT, padx=5)
        
        self.stop_button = ttk.Button(self.buttons_frame, text="Остановить", command=self.stop_training, state=tk.DISABLED)
        self.stop_button.pack(side=tk.LEFT, padx=5)
        
        self.exit_button = ttk.Button(self.buttons_frame, text="Выход", command=self.root.quit)
        self.exit_button.pack(side=tk.RIGHT, padx=5)
        
        # Фрейм для вывода логов
        self.log_frame = ttk.LabelFrame(self.main_frame, text="Лог выполнения")
        self.log_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Фрейм для кнопок управления логами
        self.log_buttons_frame = ttk.Frame(self.log_frame)
        self.log_buttons_frame.pack(fill=tk.X, side=tk.BOTTOM, padx=5, pady=2)
        
        # Кнопки для работы с логами
        self.copy_log_button = ttk.Button(self.log_buttons_frame, text="Копировать лог", 
                                         command=self.copy_log_to_clipboard)
        self.copy_log_button.pack(side=tk.LEFT, padx=5)
        
        self.save_log_button = ttk.Button(self.log_buttons_frame, text="Сохранить лог в файл", 
                                         command=self.save_log_to_file)
        self.save_log_button.pack(side=tk.LEFT, padx=5)
        
        self.clear_log_button = ttk.Button(self.log_buttons_frame, text="Очистить лог", 
                                          command=self.clear_log)
        self.clear_log_button.pack(side=tk.LEFT, padx=5)
        
        # Текстовое поле для вывода логов
        self.log_text = ScrolledText(self.log_frame, height=10)
        self.log_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.log_text.config(state=tk.DISABLED)
        
        # Настройка контекстного меню для лога
        self.log_context_menu = tk.Menu(self.log_text, tearoff=0)
        self.log_context_menu.add_command(label="Копировать", command=self.copy_selected_log)
        self.log_context_menu.add_command(label="Копировать всё", command=self.copy_log_to_clipboard)
        self.log_context_menu.add_separator()
        self.log_context_menu.add_command(label="Сохранить в файл", command=self.save_log_to_file)
        self.log_context_menu.add_command(label="Очистить", command=self.clear_log)
        
        # Привязка правой кнопки мыши к контекстному меню
        self.log_text.bind("<Button-3>", self.show_log_context_menu)
        
        # Очередь для передачи вывода между потоками
        self.queue = queue.Queue()
        self.running = False
        self.process = None
        
        # Обновление вывода
        self.update_output()
        
    def browse_dataset(self):
        """Выбор файла с данными"""
        filepath = filedialog.askopenfilename(
            title="Выберите файл с данными",
            filetypes=[("CSV файлы", "*.csv"), ("Все файлы", "*.*")]
        )
        if filepath:
            self.dataset_path.set(filepath)
    
    def browse_output(self):
        """Выбор папки для сохранения результатов"""
        dirpath = filedialog.askdirectory(title="Выберите папку для сохранения результатов")
        if dirpath:
            self.output_path.set(dirpath)
    
    def run_training(self):
        """Запуск обучения моделей"""
        # Проверка наличия файла с данными
        if not os.path.exists(self.dataset_path.get()):
            messagebox.showerror("Ошибка", f"Файл данных не найден: {self.dataset_path.get()}")
            return
        
        # Создание выходной директории, если она не существует
        os.makedirs(self.output_path.get(), exist_ok=True)
        os.makedirs(os.path.join(self.output_path.get(), 'models'), exist_ok=True)
        os.makedirs(os.path.join(self.output_path.get(), 'results'), exist_ok=True)
        
        # Формирование команды
        cmd = [sys.executable, "src/run_pipeline.py", 
               "--input_file", self.dataset_path.get(),
               "--output_dir", self.output_path.get()]
        
        # Добавление опций в зависимости от выбранного режима
        if self.skip_preprocessing_var.get():
            cmd.append("--skip_preprocessing")
        
        if self.skip_loyalty_features_var.get():
            cmd.append("--skip_loyalty_features")

        if self.skip_model_training_var.get():
            cmd.append("--skip_model_training")

        # Дополнительные опции
        if self.save_intermediate.get():
            cmd.append("--save_intermediate")
        
        if self.balance_classes.get():
            cmd.append("--balance_classes")

        if self.tune_hyperparams_var.get():
            cmd.append("--tune_hyperparams")

        if self.create_ensemble_var.get():
            cmd.append("--create_ensemble")
        
        # Вывод информации о запуске
        self.log_text.config(state=tk.NORMAL)
        self.log_text.delete(1.0, tk.END)
        self.log_text.insert(tk.END, "Запуск обучения моделей...\n")
        self.log_text.insert(tk.END, f"Команда: {' '.join(cmd)}\n\n")
        self.log_text.config(state=tk.DISABLED)
        
        # Изменение состояния кнопок
        self.run_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.NORMAL)
        
        # Запуск в отдельном потоке
        self.running = True
        threading.Thread(target=self.run_process, args=(cmd,), daemon=True).start()
    
    def run_process(self, cmd):
        """Выполнение процесса в отдельном потоке"""
        try:
            # Создание и запуск процесса
            self.process = subprocess.Popen(
                cmd, 
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                bufsize=1
            )
            
            # Чтение вывода
            for line in self.process.stdout:
                if self.running:
                    self.queue.put(line)
                else:
                    break
            
            # Ожидание завершения процесса
            self.process.wait()
            
            # Вывод информации о завершении
            if self.process.returncode == 0:
                self.queue.put("\n\nОбучение успешно завершено.\n")
            else:
                self.queue.put(f"\n\nПроцесс завершился с ошибкой (код {self.process.returncode}).\n")
        
        except Exception as e:
            self.queue.put(f"\n\nОшибка при выполнении процесса: {str(e)}\n")
        
        finally:
            # Восстановление состояния кнопок
            self.root.after(0, self.reset_buttons)
    
    def stop_training(self):
        """Остановка процесса обучения"""
        if self.process and self.process.poll() is None:
            self.running = False
            self.process.terminate()
            self.queue.put("\n\nПроцесс был остановлен пользователем.\n")
            self.reset_buttons()
    
    def reset_buttons(self):
        """Восстановление состояния кнопок"""
        self.run_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)
    
    def update_output(self):
        """Обновление вывода из очереди"""
        try:
            while True:
                line = self.queue.get_nowait()
                self.log_text.config(state=tk.NORMAL)
                self.log_text.insert(tk.END, line)
                self.log_text.see(tk.END)
                self.log_text.config(state=tk.DISABLED)
                self.queue.task_done()
        except queue.Empty:
            pass
        
        self.root.after(100, self.update_output)
    
    def copy_selected_log(self):
        """Копирование выделенного текста из лога в буфер обмена"""
        try:
            self.log_text.config(state=tk.NORMAL)
            selected_text = self.log_text.selection_get()
            self.root.clipboard_clear()
            self.root.clipboard_append(selected_text)
            self.log_text.config(state=tk.DISABLED)
            messagebox.showinfo("Копирование", "Выделенный текст скопирован в буфер обмена")
        except tk.TclError:
            messagebox.showinfo("Копирование", "Нет выделенного текста")
        except Exception as e:
            messagebox.showerror("Ошибка", f"Ошибка при копировании: {str(e)}")
    
    def copy_log_to_clipboard(self):
        """Копирование всего лога в буфер обмена"""
        try:
            self.log_text.config(state=tk.NORMAL)
            all_text = self.log_text.get(1.0, tk.END)
            self.root.clipboard_clear()
            self.root.clipboard_append(all_text)
            self.log_text.config(state=tk.DISABLED)
            messagebox.showinfo("Копирование", "Весь лог скопирован в буфер обмена")
        except Exception as e:
            messagebox.showerror("Ошибка", f"Ошибка при копировании: {str(e)}")
    
    def save_log_to_file(self):
        """Сохранение лога в файл"""
        try:
            filename = filedialog.asksaveasfilename(
                title="Сохранить лог в файл",
                filetypes=[("Текстовые файлы", "*.txt"), ("Все файлы", "*.*")],
                defaultextension=".txt"
            )
            if not filename:
                return
            
            self.log_text.config(state=tk.NORMAL)
            log_text = self.log_text.get(1.0, tk.END)
            self.log_text.config(state=tk.DISABLED)
            
            with open(filename, "w", encoding="utf-8") as f:
                f.write(log_text)
            
            messagebox.showinfo("Сохранение", f"Лог сохранен в файл:\n{filename}")
        except Exception as e:
            messagebox.showerror("Ошибка", f"Ошибка при сохранении: {str(e)}")
    
    def clear_log(self):
        """Очистка лога"""
        self.log_text.config(state=tk.NORMAL)
        self.log_text.delete(1.0, tk.END)
        self.log_text.config(state=tk.DISABLED)
    
    def show_log_context_menu(self, event):
        """Показ контекстного меню для лога"""
        try:
            # Временно включаем состояние NORMAL для выделения текста
            self.log_text.config(state=tk.NORMAL)
            self.log_context_menu.post(event.x_root, event.y_root)
            # Возвращаем состояние DISABLED после показа меню
            self.log_text.config(state=tk.DISABLED)
        except Exception:
            pass


if __name__ == "__main__":
    # Проверка наличия Python модулей
    try:
        import tkinter
    except ImportError:
        print("Ошибка: модуль tkinter не найден!")
        print("Пожалуйста, установите Python с поддержкой GUI (tkinter).")
        sys.exit(1)
    
    # Проверка наличия файла run_pipeline.py
    if not os.path.exists("src/run_pipeline.py"):
        print("Ошибка: файл src/run_pipeline.py не найден!")
        print("Пожалуйста, убедитесь, что вы запускаете скрипт из корневой директории проекта.")
        sys.exit(1)
    
    # Запуск GUI
    root = tk.Tk()
    app = TrainingGUI(root)
    root.mainloop() 