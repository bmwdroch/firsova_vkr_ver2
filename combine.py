import os

# --- НАСТРОЙКИ ---

# Список файлов для объединения в правильном порядке.
# Убедитесь, что все пути указаны корректно относительно места запуска скрипта
# (предполагается, что скрипт запускается из корневой директории проекта firsova_vkr).
files_to_concatenate = [
    # Глава 2
    'docs/thesis_chapters/ch2_00_introduction.md',
    'docs/thesis_chapters/ch2_01_data_description.md',
    'docs/thesis_chapters/ch2_02_data_preprocessing.md',
    'docs/thesis_chapters/ch2_03_ml_models_development.md',
    'docs/thesis_chapters/ch2_04_testing_and_optimization.md',
    'docs/thesis_chapters/ch2_05_conclusions.md',

    # Глава 3
    'docs/thesis_chapters/ch3_00_introduction.md',
    'docs/thesis_chapters/ch3_01_application_architecture_design.md',
    'docs/thesis_chapters/ch3_02_functional_blocks_design.md',
    'docs/thesis_chapters/ch3_03_testing_strategy.md',
    'docs/thesis_chapters/ch3_04_conclusions.md',

    # Общее заключение по ВКР
    'docs/thesis_chapters/conclusion.md',
]

# Имя и путь для итогового объединенного файла
# Он будет создан в корневой директории проекта, если не указать другой путь.
output_filename = 'ПОЛНЫЙ_ТЕКСТ_ВКР.md'

# Разделитель, который будет вставлен между содержимым каждого файла.
# '---' создаст горизонтальную линию в Markdown.
# '\n\n\\pagebreak\n\n' может быть использован для некоторых конвертеров MD -> PDF для создания разрыва страницы.
# '<div style="page-break-after: always;"></div>' может сработать для HTML -> PDF.
# Для простого разделения можно использовать просто '\n\n'
separator = '\n\n<br><hr><br>\n\n' # Горизонтальная линия с отступами для лучшей визуальной сепарации

# --- КОНЕЦ НАСТРОЕК ---

def combine_markdown_files(file_list, output_file, sep):
    """
    Объединяет содержимое указанных markdown файлов в один выходной файл.
    """
    successful_files = 0
    missing_files = []

    # Убедимся, что директория для выходного файла существует, если указан путь
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        try:
            os.makedirs(output_dir)
            print(f"Создана директория: {output_dir}")
        except Exception as e:
            print(f"Не удалось создать директорию {output_dir}: {e}")
            return

    try:
        with open(output_file, 'w', encoding='utf-8') as outfile:
            print(f"Начало объединения файлов в '{output_file}'...")
            for i, filepath in enumerate(file_list):
                try:
                    with open(filepath, 'r', encoding='utf-8') as infile:
                        content = infile.read()
                        outfile.write(content)
                        successful_files += 1
                        print(f"Успешно добавлен: '{filepath}'")
                        
                        # Добавляем разделитель, если это не последний файл
                        if i < len(file_list) - 1:
                            outfile.write(sep)
                            
                except FileNotFoundError:
                    print(f"ПРЕДУПРЕЖДЕНИЕ: Файл '{filepath}' не найден и будет пропущен.")
                    missing_files.append(filepath)
                except Exception as e:
                    print(f"ОШИБКА при чтении файла '{filepath}': {e}")
            
            # Добавляем информацию о каждом файле в начало, если нужно
            # for filepath in file_list:
            #     if filepath not in missing_files:
            #         outfile.write(f"\n\n<!-- Содержимое из файла: {filepath} -->\n\n")


        print(f"\n--- Завершено ---")
        print(f"Успешно обработано и добавлено файлов: {successful_files}")
        if missing_files:
            print(f"Пропущено из-за отсутствия (не найдены): {len(missing_files)} файлов")
            for mf in missing_files:
                print(f"  - {mf}")
        print(f"Итоговый файл сохранен как: '{output_file}'")

    except Exception as e:
        print(f"КРИТИЧЕСКАЯ ОШИБКА при записи в выходной файл '{output_file}': {e}")

if __name__ == '__main__':
    # Проверка, что мы находимся в ожидаемой директории (опционально, для удобства)
    # current_dir_name = os.path.basename(os.getcwd())
    # if current_dir_name != 'firsova_vkr':
    #     print(f"ПРЕДУПРЕЖДЕНИЕ: Скрипт, возможно, запущен не из корневой директории проекта ('firsova_vkr'). Текущая директория: {os.getcwd()}")
    #     print("Пожалуйста, убедитесь, что пути в 'files_to_concatenate' указаны корректно.")

    combine_markdown_files(files_to_concatenate, output_filename, separator)