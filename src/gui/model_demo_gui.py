#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Графический интерфейс для демонстрации работы моделей классификации лояльности клиентов.
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

class DemoGUI:
    """Графический интерфейс для демонстрации моделей"""
    
    def __init__(self, root):
        self.root = root
        self.root.title("Демонстрация моделей классификации лояльности")
        self.root.geometry("800x650")
        self.root.minsize(700, 550)
        
        # Создание стиля
        self.style = ttk.Style()
        self.style.configure("TFrame", background="#f0f0f0")
        self.style.configure("TButton", font=("Arial", 10))
        self.style.configure("TLabel", font=("Arial", 10), background="#f0f0f0")
        self.style.configure("TCheckbutton", font=("Arial", 10), background="#f0f0f0")
        self.style.configure("Header.TLabel", font=("Arial", 12, "bold"), background="#f0f0f0")
        
        # Создание главного фрейма
        self.main_frame = ttk.Frame(root, padding=10)
        self.main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Заголовок
        ttk.Label(self.main_frame, text="Демонстрация моделей классификации лояльности клиентов", 
                  style="Header.TLabel").pack(pady=10)
        
        # Фрейм для выбора моделей
        self.models_frame = ttk.LabelFrame(self.main_frame, text="Выбор моделей")
        self.models_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Выбор моделей
        self.models_checkboxes = {}
        self.models_vars = {}
        models_list = [
            ("Логистическая регрессия", "logistic_regression"),
            ("Случайный лес", "random_forest"),
            ("XGBoost", "xgboost"),
            ("LightGBM", "lightgbm"),
            ("Ансамблевая модель", "ensemble")
        ]
        
        for i, (text, value) in enumerate(models_list):
            var = tk.BooleanVar(value=True)
            cb = ttk.Checkbutton(self.models_frame, text=text, variable=var)
            cb.grid(row=i//3, column=i%3, sticky=tk.W, padx=20, pady=5)
            self.models_vars[value] = var
            self.models_checkboxes[value] = cb
        
        # Кнопка выбора всех/ни одной модели
        models_select_frame = ttk.Frame(self.models_frame)
        models_select_frame.grid(row=(len(models_list)-1)//3 + 1, column=0, columnspan=3, pady=5)
        
        ttk.Button(models_select_frame, text="Выбрать все", 
                   command=lambda: self.select_models(True)).pack(side=tk.LEFT, padx=5)
        ttk.Button(models_select_frame, text="Снять выбор", 
                   command=lambda: self.select_models(False)).pack(side=tk.LEFT, padx=5)
        
        # Фрейм для визуализаций
        self.viz_frame = ttk.LabelFrame(self.main_frame, text="Параметры визуализации")
        self.viz_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Опции визуализации
        self.viz_vars = {}
        viz_options = [
            ("Матрицы ошибок", "confusion_matrices", True),
            ("ROC-кривые", "roc_curves", True),
            ("PR-кривые", "pr_curves", True),
            ("Важность признаков", "feature_importance", True),
            ("Сравнение метрик", "metrics_comparison", True),
            ("SHAP анализ", "shap_analysis", False)
        ]
        
        for i, (text, value, default) in enumerate(viz_options):
            var = tk.BooleanVar(value=default)
            cb = ttk.Checkbutton(self.viz_frame, text=text, variable=var)
            cb.grid(row=i//3, column=i%3, sticky=tk.W, padx=20, pady=5)
            self.viz_vars[value] = var
        
        # Путь к данным и папкам
        self.paths_frame = ttk.Frame(self.main_frame)
        self.paths_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Путь к датасету
        ttk.Label(self.paths_frame, text="Файл данных:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        
        self.dataset_path = tk.StringVar(value="dataset/Concept202408.csv")
        self.dataset_entry = ttk.Entry(self.paths_frame, textvariable=self.dataset_path, width=50)
        self.dataset_entry.grid(row=0, column=1, padx=5, pady=5, sticky=tk.W+tk.E)
        
        self.browse_dataset_button = ttk.Button(self.paths_frame, text="Обзор...", command=self.browse_dataset)
        self.browse_dataset_button.grid(row=0, column=2, padx=5, pady=5)
        
        # Путь к моделям
        ttk.Label(self.paths_frame, text="Папка с моделями:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=5)
        
        self.models_dir = tk.StringVar(value="output/models")
        self.models_dir_entry = ttk.Entry(self.paths_frame, textvariable=self.models_dir, width=50)
        self.models_dir_entry.grid(row=1, column=1, padx=5, pady=5, sticky=tk.W+tk.E)
        
        self.browse_models_button = ttk.Button(self.paths_frame, text="Обзор...", command=self.browse_models_dir)
        self.browse_models_button.grid(row=1, column=2, padx=5, pady=5)
        
        # Путь для вывода результатов
        ttk.Label(self.paths_frame, text="Папка для результатов:").grid(row=2, column=0, sticky=tk.W, padx=5, pady=5)
        
        self.output_dir = tk.StringVar(value="output/demo")
        self.output_dir_entry = ttk.Entry(self.paths_frame, textvariable=self.output_dir, width=50)
        self.output_dir_entry.grid(row=2, column=1, padx=5, pady=5, sticky=tk.W+tk.E)
        
        self.browse_output_button = ttk.Button(self.paths_frame, text="Обзор...", command=self.browse_output_dir)
        self.browse_output_button.grid(row=2, column=2, padx=5, pady=5)
        
        # Дополнительные параметры
        self.options_frame = ttk.LabelFrame(self.main_frame, text="Дополнительные параметры")
        self.options_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Сохранение результатов
        self.save_results = tk.BooleanVar(value=True)
        ttk.Checkbutton(self.options_frame, text="Сохранять результаты", 
                        variable=self.save_results).grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        
        # Показать подробные метрики
        self.show_detailed_metrics = tk.BooleanVar(value=True)
        ttk.Checkbutton(self.options_frame, text="Показать подробные метрики", 
                        variable=self.show_detailed_metrics).grid(row=0, column=1, sticky=tk.W, padx=5, pady=5)
        
        # Тест на новых данных
        self.test_new_data = tk.BooleanVar(value=False)
        ttk.Checkbutton(self.options_frame, text="Тестировать на новых данных", 
                        variable=self.test_new_data).grid(row=1, column=0, sticky=tk.W, padx=5, pady=5)
        
        # Фрейм для кнопок управления
        self.buttons_frame = ttk.Frame(self.main_frame)
        self.buttons_frame.pack(fill=tk.X, pady=10)
        
        self.run_button = ttk.Button(self.buttons_frame, text="Запустить демонстрацию", command=self.run_demo)
        self.run_button.pack(side=tk.LEFT, padx=5)
        
        self.stop_button = ttk.Button(self.buttons_frame, text="Остановить", command=self.stop_demo, state=tk.DISABLED)
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
    
    def select_models(self, select_all=True):
        """Выбрать все или ни одной модели"""
        for var in self.models_vars.values():
            var.set(select_all)
    
    def browse_dataset(self):
        """Выбор файла с данными"""
        filepath = filedialog.askopenfilename(
            title="Выберите файл с данными",
            filetypes=[("CSV файлы", "*.csv"), ("Все файлы", "*.*")]
        )
        if filepath:
            self.dataset_path.set(filepath)
    
    def browse_models_dir(self):
        """Выбор папки с моделями"""
        dirpath = filedialog.askdirectory(title="Выберите папку с моделями")
        if dirpath:
            self.models_dir.set(dirpath)
    
    def browse_output_dir(self):
        """Выбор папки для вывода результатов"""
        dirpath = filedialog.askdirectory(title="Выберите папку для вывода результатов")
        if dirpath:
            self.output_dir.set(dirpath)
    
    def run_demo(self):
        """Запуск демонстрации моделей"""
        # Проверка наличия файла с данными
        if not os.path.exists(self.dataset_path.get()):
            messagebox.showerror("Ошибка", f"Файл данных не найден: {self.dataset_path.get()}")
            return
        
        # Проверка наличия папки с моделями
        if not os.path.exists(self.models_dir.get()):
            messagebox.showerror("Ошибка", f"Папка с моделями не найдена: {self.models_dir.get()}")
            return
        
        # Проверка, что хотя бы одна модель выбрана
        selected_models = [k for k, v in self.models_vars.items() if v.get()]
        if not selected_models:
            messagebox.showerror("Ошибка", "Необходимо выбрать хотя бы одну модель для демонстрации")
            return
        
        # Создание выходной директории, если она не существует
        os.makedirs(self.output_dir.get(), exist_ok=True)
        
        # Формирование команды
        cmd = [sys.executable, "src/modeling/model_demo.py", 
               "--input_file", self.dataset_path.get(),
               "--models_dir", self.models_dir.get(),
               "--output_dir", self.output_dir.get(),
               "--models", ",".join(selected_models)]
        
        # Добавление опций визуализации
        viz_options = [k for k, v in self.viz_vars.items() if v.get()]
        if viz_options:
            cmd.extend(["--visualizations", ",".join(viz_options)])
        
        # Дополнительные опции
        if self.save_results.get():
            cmd.append("--save_results")
        
        if self.show_detailed_metrics.get():
            cmd.append("--detailed_metrics")
        
        if self.test_new_data.get():
            cmd.append("--test_new_data")
        
        # Вывод информации о запуске
        self.log_text.config(state=tk.NORMAL)
        self.log_text.delete(1.0, tk.END)
        self.log_text.insert(tk.END, "Запуск демонстрации моделей...\n")
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
                self.queue.put("\n\nДемонстрация успешно завершена.\n")
                self.queue.put("Результаты сохранены в указанной папке.\n")
                
                # Дополнительная информация о визуализациях
                viz_files = [
                    (self.viz_vars["confusion_matrices"].get(), "confusion_matrices.png", "Матрицы ошибок"),
                    (self.viz_vars["roc_curves"].get(), "roc_curves.png", "ROC-кривые"),
                    (self.viz_vars["pr_curves"].get(), "precision_recall_curves.png", "PR-кривые"),
                    (self.viz_vars["metrics_comparison"].get(), "metrics_comparison.png", "Сравнение метрик"),
                    (self.viz_vars["feature_importance"].get(), "feature_importance.png", "Важность признаков")
                ]
                
                self.queue.put("\nСозданные визуализации:\n")
                for enabled, filename, desc in viz_files:
                    if enabled:
                        file_path = os.path.join(self.output_dir.get(), filename)
                        if os.path.exists(file_path):
                            self.queue.put(f"- {desc}: {filename}\n")
            else:
                self.queue.put(f"\n\nПроцесс завершился с ошибкой (код {self.process.returncode}).\n")
        
        except Exception as e:
            self.queue.put(f"\n\nОшибка при выполнении процесса: {str(e)}\n")
        
        finally:
            # Восстановление состояния кнопок
            self.root.after(0, self.reset_buttons)
    
    def stop_demo(self):
        """Остановка процесса демонстрации"""
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
    
    # Проверка наличия файла model_demo.py
    if not os.path.exists("src/modeling/model_demo.py"):
        print("Ошибка: файл src/modeling/model_demo.py не найден!")
        print("Пожалуйста, убедитесь, что вы запускаете скрипт из корневой директории проекта.")
        sys.exit(1)
    
    # Запуск GUI
    root = tk.Tk()
    app = DemoGUI(root)
    root.mainloop() 