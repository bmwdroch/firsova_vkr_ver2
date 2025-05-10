from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles # Импорт для статических файлов
from src.api.routes import ui_routes # Импортируем наш новый роутер
import os # Для работы с путями

# Определяем путь к директории со статическими файлами
# Предполагаем, что main.py находится в src, а static - в src/static
# Для большей надежности можно использовать абсолютные пути или pathlib
current_dir = os.path.dirname(os.path.abspath(__file__))
static_dir = os.path.join(current_dir, "static")

app = FastAPI(
    title="API для классификации лояльности клиентов Acoola",
    description="Это API предоставляет эндпоинты для взаимодействия с ML моделью классификации клиентов и демонстрационным UI.",
    version="0.1.0",
)

# Монтируем директорию static для раздачи статических файлов (CSS, JS, изображения)
# Первый аргумент - это URL-путь, по которому будут доступны файлы
# Второй - имя директории, которую нужно создать, если ее нет (необязательно здесь)
# Третий - имя директории на диске, которую монтируем
if not os.path.exists(static_dir):
    os.makedirs(static_dir)
    print(f"Создана директория для статики: {static_dir}")
    # Можно также создать css поддиректорию, если нужно
    css_dir = os.path.join(static_dir, "css")
    if not os.path.exists(css_dir):
        os.makedirs(css_dir)
        print(f"Создана директория для CSS: {css_dir}")
        # Можно сразу создать пустой styles.css, если его еще нет
        # open(os.path.join(css_dir, "styles.css"), 'a').close()

app.mount("/static", StaticFiles(directory=static_dir), name="static")

# Подключаем роутер для UI
app.include_router(ui_routes.router, prefix="/ui", tags=["User Interface"])

@app.get("/", tags=["Root"])
async def read_root():
    return {"message": "Добро пожаловать в API классификации лояльности клиентов Acoola!"}

# Сюда можно будет добавлять другие роутеры, если они появятся

if __name__ == "__main__":
    import uvicorn
    # Запуск uvicorn сервера. 
    # Для продакшена лучше использовать gunicorn или другой ASGI-сервер.
    uvicorn.run(app, host="0.0.0.0", port=8000) 