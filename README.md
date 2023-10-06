# Mobile app testing
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

## Дополнения.
Для того, чтобы решить задачу, я объявляю 2 функции.

Функция metric_fpr будет считать метрику качества FPR с помощью А/А тестирований. 

Здесь я подаю на вход несколько параметров:

df_x, df_y - значения для варианта 1 и варианта 2

metric_col - будет передавать значения для колонки, которая является целевой метрикой для проверки (purchase)

n_sim - количество симуляций, которые будут расчитываться на А/А тестах

n_s_perc - процент наблюдений от исходной, которые я буду брать для подвыборок для n_s_min

n_s_min - граница, выше которой не берутся количественные наблюдения

estimator - сюда я записываю статистический оценщик

*args, **kwargs - аргументы для оценщиков.

Во второй функции я использую первую для того, чтобы прогоняться по всем группам внутри данных эксперимента.
Группировкой является версия приложения, в которой я ищу разницу между средними.

В процессе я вижу, что в группе v2.8.0 сильно отличаются средние, высокий FPR, следовательно, в этой версии я ищу поломку.
