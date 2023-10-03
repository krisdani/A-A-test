# A-A-test
A/A tests and quality checks of split systems.

## Описание задания.
А/А-тестирование мобильного приложения. Необходимо посчитать результаты A/A-теста, проверяя метрику качества FPR (будем проверять на конверсии в покупку). Известно, что сплит-система сломана. Требуется проверить утверждение о поломке и найти ее причины, если сплит-система действительно сломана.

## Описание колонок:
experimentVariant – вариант эксперимента
version – версия приложения
purchase – факт покупки

## Задача.
Запустите A/A-тест.
Посчитайте FPR на уровне альфа = 0.05 (подвыборки без возвращения объемом 1000). Вы увидите, что FPR > альфа! Нам нужно наоборот – чтобы было меньше.
Найдите причины поломки сплит-системы, ориентируясь на результаты эксперимента.
Напишите выводы, которые можно сделать на основе анализа результатов A/A-теста.
