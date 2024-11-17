# spbu_deep_learning
Практики и домашние задания по курсу "Нейросетевые технологии", ПМ-ПУ СПбГУ
Чтобы начать: делаете форк этого репозитория себе и отправляете мне в личку свои имя-фамилию и ссылку на реп.

## Ссылки
**Здесь будет ссылка на таблицу с оценками и ссылки на доп.материалы**

# Практики
**План является предварительным и будет меняться**

1. Введение в Pytorch
2. Введение в свертки, простейшая сверточная сеть
3. Transfer learning, ResNet, визуализация
4. Детекция/сегментация
5. Few-shot/zero-shot
6. RNN/LSTM
7. Transformers

# Домашние задания
**Список выданных заданий и сроков**

1. Введение в pytorch и пайплайн обучения. Выдан 14.09 - мягкий дедлайн 28.09 - жесткий дедлайн - 08.10
2. Свертки, дальнейший пайплайн обучения. Выдан 28.09 - мягкий дедлайн 12.10 - жесткий дедлайн 22.10
3. Классификация. Второе соревнование. TBA.

# Правила выполнения домашних работ
** Правила могут дополняться и слегка меняться в процессе. Исключение будет прописано в задаче,если такое будет.
## Виды заданий
Будет три вида заданий:
- "Задача" - задание, в котором надо написать код, протестировать, проделать вычисления итп, на защите будет тестирование от меня итп.
- "Тест" - более простое задание, в котором надо ответить на вопрос, выбрать вариант или написать формулу
- "Соревнование" - соревнование на Kaggle, где вы соревнуетесь друг с другои и, если мне станет интересно, со мной. Можно делать что угодно с учетом ограничений, которые будут указаны в тексте.

Решение задач можно оформить в виде ноутбука или .py файлов. Тесты будут организованы как google form.
Соревнования проходят на kaggle, дедлайн соревнований не привязан явно к мягкому или жесткому дедлайну, пока соренования будут длиться от 2 до 4 недель.

## Сроки

Мягкий дедлайн по каждому заданию - 14 день после его публикации, время защиты - ближайшая пара. Если задание опубликовано в другой день, то время на выполнение - те же 14 дней, но защита будет после ближайшего занятия после.
Примеры: 
- Задание  опубликовано в субботу до занятия или на нем, дедалайн - следующее занятие (последняя минута занятия). В конце следующего занятия происходит отсечка и начинается защита.
- Задание опубликовано в среду. Дедлайн на сдачу - среда через неделю (время пуша на гитхаб + 1 минута). Ждем до субботы, в субботу защита.

После мягкого дедлайна каждый день просрочки уменьшает оценку на 10 %. Итого жесткий дедлайн - 24 день. Защита также на ближайшей паре после него.
Расчет будет проходить следующим образом:
- Считается балл за задачи (как будто штрафа нет)
- Результат умножается на 1 - 0.1 * на количество просроченных дней
- Просрочка начинается, как только заканчивается время на сдачу. Т.е. если срок - 23.59 среды, в 00.00 четверга множитель уже 0.9.

Сдачей задачи считается коммит и пуш в именной репозиторий на гитхаб. Время коммита будет считаться временем сдачи.
Можно делать сколько угодно коммитов до финальной сдачи. Я буду учитывать только последний их них. Даже если вы сделали сколько угодно коммитов, если финальный будет после дедлайна, за него будет штраф. В случае исправлений в процессе - балл считается после всех исправлений (до дедлайна). При желании можно сделать попытку сдачи задания до дедлайна - я при наличии свободного времени посмотрю код и сделаю замечания. При этом максимальное число исправлений и сроки ответов ограничены моим свободным временем.

Сдачей теста считается отправка формы. Сдачей соревнования считается отправка предсказания на kaggle.

Для **соревнований**: дедлайн один (жесткий) - конец соревнования. После публикации результата необходимо прислать мне описание решения. Подойдет:
- Ноутбук с обучением модели + краткий отчет об идеях
- Набор ноутбуков с разными идеями 
- Набор .py файлов + краткий отчет об идеях

Также я планирую устраивать небольшие обсуждения идей на паре в процессе соревнования (и после).

**Исключение** для всех заданий - зачетный период, тогда срок будет min(дедлайн, зачет).

## Семинарские ноутбуки
За прорешенные семинарские ноутбуки будет добавляться (максимум) по 10 баллов. Они будут считаться особыми дополнительными заданиями.
Я выкладываю свое решение после следующего занятия. Их можно будет обсудить также, как домашку.
Таким образом, **дедлайн** по семинарам есть только один - жесткий.
Эта часть опциональная, однако работа на парах может повлиять на оценку. 
К тому же, я буду довольна активной работой, и следовательно, буду меньше налегать на разработку домашек.

## Оценки и штрафы
Учет списывания будет вестись по коммитам. Кто первый закоммитил - тот и молодец. 

Неэффективный, плохо читаемый код или непонятные графики могут снизить оценку. 
Советую пользоваться линтерами (isort, black, pylint, mypy). 
Интересные находки - библиотеки для визуализации, работы с данными, отслеживанием экспериментов, продвинутые модели - все это может повысить оценку, даже немного сверх максимума.

## Защита заданий
На защите я буду задавать вопросы по работе - на понимание формул, кода - и другие уточняющие вопросы. Также я планирую прогонять тесты, так что лучше отправлять задачи до занятия, чтобы не тратить время.
До дедлайна решения можно будет исправлять, я буду присылать в личку инофрмацию  непройденных тестах или иных ошибках.
Если у вас были проблемы при выполнении задачи - при защите можно улучшить свою оценку, продемонстрировав понимание :) Или ухудшить, если окажется, что работа списана.
Если вы использовали какие-то источники, туториалы, видео итп при выполнении заданий, то мне нужно будет прислать отдельный файл с их списком.

# Финальная оценка
Оценка будет основана на баллах, заработанных в течение семестра.
Минимальное число баллов для того, чтобы претендовать на 3 - 60% от базовых баллов.
За 4+ можно будет бороться для более высоких сумм баллов. Баллы за дополнительные задачи ({*}) не будут учитываться при расчете процента выполнения.
Пример: В задачах в сумме можно получить 100 баллов без звездочек и 60 со звездочками. Петя набрал 58 баллов за все задачи (со звездочками  и без). Бедный Петя - у него будет 2.
Вася набрал 60 баллов - он допущен к зачету и может получить 3. Катя набрала 120 баллов - она допущена к зачету и скорее всего получит 4 или 5.
Игорь набрал 160 баллов - возможно, он даже сможет получить 5 автоматом. 

Все баллы выше минимума будут влиять на вашу оценку, но уже по динамической шкале. 

На зачете будет небольшой тест и/или опрос по материалам практик. 
Также на зачете будет последний шанс защитить домашки, если они будут открыты.
Проблемы организации будут резолвиться в вашу пользу (если перенеслась дата, аудитория, что-то не заработало, я ошиблась в оценивании, забыла что-то проверить итп).
Окончательная система оценивания будет опубликована ближе к зачету.
