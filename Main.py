import re
import csv
import sys
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer

#Основные сущности - Филиповский Виталий
#Словарь из 5 основных сущностей: urls, emails, phones, dates, prices. Используется и для извлечения, и для обезличивания.
RE = {
    'urls':   re.compile(r'(?:https?://|www\.)[^\s<>"\')\]]+', re.I),
    'emails': re.compile(r'\b[\w._%+\-]+@[\w.\-]+\.[A-Za-z]{2,}\b'),
    'phones': re.compile(r'(?:\+?\d[\s\-\(\)]?){7,15}\d'),
    'dates':  re.compile(r'\b(?:\d{1,2}[./\-]\d{1,2}[./\-]\d{2,4}|\d{4}[./\-]\d{1,2}[./\-]\d{1,2})\b'),
    'prices': re.compile(
        r'\b\d{1,3}(?:[\s\u00A0]?\d{3})*(?:[.,]\d+)?\s*'
        r'(?:руб\w*|₽|rub|usd|eur|р\.?|\$|€)\b', re.I),
}


#Автоматические стоп слова Ветин Константин
#Считает частоту всех слов в корпусе. Возвращает множество стоп-слов = топ-N самых частых + все слова длиной ≤ min_len.
def build_stopwords(texts, top_n=60, min_len=2):
    counter = Counter()
    for t in texts:
        for w in re.findall(r'[A-Za-zА-Яа-яЁё]+', str(t).lower()):
            counter[w] += 1

    stop = {w for w in counter if len(w) <= min_len}
    stop |= {w for w, _ in counter.most_common(top_n)}
    return stop, counter


#Автоматические окончания - Ветин Константин
#Для каждого слова ≥5 символов берёт хвосты длиной 2–4 символа и считает их частоту.
#Возвращает множество хвостов, встретившихся ≥ min_count раз — это «выученные» окончания языка.
def build_endings(texts, min_count=5, min_suffix_len=2, max_suffix_len=4):
    tails = Counter()
    for t in texts:
        for w in re.findall(r'[а-яё]+', str(t).lower()):
            if len(w) < 5:
                continue
            for L in range(min_suffix_len, max_suffix_len + 1):
                tails[w[-L:]] += 1
    return {s for s, c in tails.items() if c >= min_count}

#Лемматизация - Моторин Михаил
#Отсекает от слова самое длинное окончание из endings.
#Не трогает слова короче 5 символов и не-кириллицу.
def lemmatize_word(word, endings):
    w = word.lower()
    if not re.match(r'^[а-яё]+$', w):
        return w
    for L in (4, 3, 2):
        if w[-L:] in endings and len(w) - L >= 4:
            return w[:-L]
    return w

#Базовая очистка и анонимизация - Моторин Михаил
#Приводит к нижнему регистру, убирает эмодзи, схлопывает !!!→!, удаляет мусорные символы, нормализует пробелы.
def clean(text):
    text = str(text).lower()
    text = re.sub(r'[\u2600-\u27BF\U0001F300-\U0001FAFF]+', ' ', text)  # эмодзи
    text = re.sub(r'([!?.,])\1+', r'\1', text)                          # !!! -> !
    text = re.sub(r'[^\w\s.,!?+\-@:/\u0400-\u04FF]', ' ', text)
    return re.sub(r'\s+', ' ', text).strip()

#Заменяет найденные сущности маркерами: [URL], [EMAIL], [PHONE], [PRICE], [DATE].
def anonymize(text):
    for marker, key in [('[URL]','urls'), ('[EMAIL]','emails'),
                        ('[PHONE]','phones'), ('[PRICE]','prices'), ('[DATE]','dates')]:
        text = RE[key].sub(marker, text)
    return re.sub(r'\s+', ' ', text).strip()

#Возвращает словарь {urls, emails, phones, dates, prices} со списками найденных значений.
def extract(text):
    return {k: rx.findall(text) for k, rx in RE.items()}


#Автоматизация - Тагильцева Дарья
#Создаёт CountVectorizer с универсальным token_pattern и переданным списком стоп-слов.
def make_vectorizer(stopwords):
    return CountVectorizer(
        token_pattern=r'(?u)\b\w\w+\b',
        stop_words=list(stopwords),
        lowercase=True,
    )

#Автоматизация для 1 сообщения - Воронов Арсений
#Полный процесс обработки для 1 сообщения из csv: извлечь сущности - очистить - обезличить - токенизировать - лемматизировать.
#Возвращает словарь со всеми полями.
def process(text, vec, endings):
    ents = extract(text)
    anon = anonymize(clean(text))
    tokens = vec.build_analyzer()(anon)
    lemmas = [lemmatize_word(t, endings) for t in tokens]
    return {
        'clean_text': ' '.join(tokens),
        'tokens': '; '.join(tokens),
        'lemmas': '; '.join(lemmas),
        **{k: '; '.join(v) for k, v in ents.items()},
    }

#Строит CountVectorizer, считает суммарные частоты слов по корпусу и возвращает топ-N пар (слово, частота).
def top_words(texts, n=20):
    vec = CountVectorizer(token_pattern=r'(?u)\b\w\w+\b')
    m = vec.fit_transform(texts)
    freqs = m.sum(axis=0).A1
    return sorted(zip(vec.get_feature_names_out(), freqs),
                  key=lambda x: -x[1])[:n]

#Начало работы с файлом - Васильев Андрей
#1) Читает CSV.
#2) Автоматически определяет текстовую колонку (среднее число слов > 3).
#3) Строит стоп-слова и окончания из датасета.
#4) Печатает топ-20 до/после очистки и топ-20 лемм.
#5) Сохраняет clean_messages.csv.
def main(path):
    with open(path, encoding='utf-8-sig') as f:
        rows = list(csv.DictReader(f))
    fields = list(rows[0].keys())
    text_col = None
    for col in fields:
        avg = sum(len(str(r[col]).split()) for r in rows) / len(rows)
        if avg > 3:
            text_col = col
            break
    if text_col is None:
        text_col = fields[-1]

    texts = [r[text_col] for r in rows]

    #Авто-стоп-слова
    stopwords, raw_counter = build_stopwords(texts, top_n=60)

    #Авто-окончания
    endings = build_endings(texts, min_count=5)

    #TOP-20 до очистки
    print("TOP-20 до очистки")
    for i, (w, c) in enumerate(top_words(texts), 1):
        print(f"{i:>2}. {w:<22} {int(c)}")

    #Обработка
    vec = make_vectorizer(stopwords)
    processed = [process(t, vec, endings) for t in texts]

    meta_cols = [c for c in fields if c != text_col]
    result = [{**{k: r[k] for k in meta_cols}, **p}
              for r, p in zip(rows, processed)]

    #TOP-20 после очистки
    print("TOP-20 после очистки")
    for i, (w, c) in enumerate(top_words([r['clean_text'] for r in result]), 1):
        print(f"{i:>2}. {w:<22} {int(c)}")

    #Сохранение
    out_fields = meta_cols + ['clean_text','tokens','lemmas',
                              'urls','emails','phones','dates','prices']
    with open('clean_messages.csv', 'w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=out_fields)
        w.writeheader()
        w.writerows(result)
    print("\nСохранено: clean_messages.csv")

#Запуск скрипта
if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else '05_tehnicheskaya_podderzhka.csv')
