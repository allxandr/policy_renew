# policy renew
# Пролонгация полисов

- Был проанализирован файл data.txt
- Небольшой анализ представлен в тетрадке ml.ipynb
- На основе анализа был создан класс Model в файлике mlmodel.py
- В классе реализованы методы предобработки данных, обучения и предсказания (для дальнейшего внедренеия в прод)
- Реализовано сохранение модели на диск (файл model.ml - хранит обученую GBClassifier модель)
- Файл app.py реализует Flask приложение 
- В приложении есть небольшой фронт (в папке static) - страница с формой для пользователей
- В приложении реализован web json api для метода predict
- В файле tests.py протестировано api веб приложения данными из файла data.txt с типом данных 'TEST'
