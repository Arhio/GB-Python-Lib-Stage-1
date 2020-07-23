#!/usr/bin/env python
# coding: utf-8

# Библиотеки Python для Data Science: Numpy, Matplotlib, Scikit-learn
# 
# Урок 1

# Тема “Вычисления с помощью Numpy”

# Задание 1

# Импортируйте библиотеку Numpy и дайте ей псевдоним np.
# Создайте массив Numpy под названием a размером 5x2, то есть состоящий из 5 строк и 2 столбцов. Первый столбец должен содержать числа 1, 2, 3, 3, 1, а второй - числа 6, 8, 11, 10, 7. Будем считать, что каждый столбец - это признак, а строка - наблюдение. Затем найдите среднее значение по каждому признаку, используя метод mean массива Numpy. Результат запишите в массив mean_a, в нем должно быть 2 элемента.

# In[24]:


import numpy as np


a = np.array([[1,6], [2,8], [3,11], [3,10], [1,7]])
a
mean_a = np.mean(A, axis=0)
mean_a


# Задание 2

# Вычислите массив a_centered, отняв от значений массива “а” средние значения соответствующих признаков, содержащиеся в массиве mean_a. Вычисление должно производиться в одно действие. Получившийся массив должен иметь размер 5x2.

# In[25]:


a_centered = a - mean_a
a_centered


# Задание 3

# Найдите скалярное произведение столбцов массива a_centered. В результате должна получиться величина a_centered_sp. Затем поделите a_centered_sp на N-1, где N - число наблюдений.

# In[32]:


a_centered_sp = a_centered[:,0].dot(a_centered[:, 1]) / (np.shape(a)[0] - 1)
a_centered_sp


# Задание 4**

# Число, которое мы получили в конце задания 3 является ковариацией двух признаков, содержащихся в массиве “а”. В задании 4 мы делили сумму произведений центрированных признаков на N-1, а не на N, поэтому полученная нами величина является несмещенной оценкой ковариации.
# Подробнее узнать о ковариации можно здесь:
# https://studopedia.ru/9_153900_viborochnaya-kovariatsiya-i-viborochnaya-dispersiya.html
# В этом задании проверьте получившееся число, вычислив ковариацию еще одним способом - с помощью функции np.cov. В качестве аргумента m функция np.cov должна принимать транспонированный массив “a”. В получившейся ковариационной матрице (массив Numpy размером 2x2) искомое значение ковариации будет равно элементу в строке с индексом 0 и столбце с индексом 1.

# In[35]:


np.cov(a.T)[0, 1]


# Тема “Работа с данными в Pandas”

# Задание 1

# Импортируйте библиотеку Pandas и дайте ей псевдоним pd. Создайте датафрейм authors со столбцами author_id и author_name, в которых соответственно содержатся данные: [1, 2, 3] и ['Тургенев', 'Чехов', 'Островский'].
# Затем создайте датафрейм book cо столбцами author_id, book_title и price, в которых соответственно содержатся данные: 
# [1, 1, 1, 2, 2, 3, 3], 
# ['Отцы и дети', 'Рудин', 'Дворянское гнездо', 'Толстый и тонкий', 'Дама с собачкой', 'Гроза', 'Таланты и поклонники'],
# [450, 300, 350, 500, 450, 370, 290].

# In[41]:


import pandas as pd


authors = pd.DataFrame({
    'author_id': [1, 2, 3],
    'author_name': ['Тургенев', 'Чехов', 'Островский'],
})

book = pd.DataFrame({
    'author_id': [1, 1, 1, 2, 2, 3, 3],
    'book_title': ['Отцы и дети', 'Рудин', 'Дворянское гнездо', 'Толстый и тонкий', 'Дама с собачкой', 'Гроза', 'Таланты и поклонники'],
    'price': [450, 300, 350, 500, 450, 370, 290]
})

print(authors)
print(book)


# Задание 2

# Получите датафрейм authors_price, соединив датафреймы authors и books по полю author_id.

# In[46]:


authors_price = book.merge(authors, how = 'left', left_on='author_id', right_on='author_id')
authors_price


# Задание 3

# Создайте датафрейм top5, в котором содержатся строки из authors_price с пятью самыми дорогими книгами.

# In[51]:


top5 = authors_price.sort_values(by=['price', 'author_name'], ascending=False).reset_index(drop = True).head(5)
top5


# Задание 4

# Создайте датафрейм authors_stat на основе информации из authors_price. В датафрейме authors_stat должны быть четыре столбца:
# author_name, min_price, max_price и mean_price,
# в которых должны содержаться соответственно имя автора,минимальная, максимальная и средняя цена на книги этого автора.
# 

# In[89]:


authors_mean = authors_price.groupby('author_name').mean()
authors_mean.rename(columns={'price' : 'mean_price'}, inplace=True)

authors_min = authors_price.groupby('author_name').min()
authors_min.rename(columns={'price' : 'min_price'}, inplace=True)

authors_max = authors_price.groupby('author_name').max()
authors_max.rename(columns={'price' : 'max_price'}, inplace=True)

authors_stat = authors.merge(authors_mean, how = 'left', left_on='author_name', right_on='author_name').merge(authors_min, how = 'left', left_on='author_name', right_on='author_name').merge(authors_max, how = 'left', left_on='author_name', right_on='author_name')[['author_name', 'mean_price', 'min_price', 'max_price']]
authors_stat


# Задание 5**

# Создайте новый столбец в датафрейме authors_price под названием cover, в нем будут располагаться данные о том, какая обложка у данной книги - твердая или мягкая. В этот столбец поместите данные из следующего списка:
# ['твердая', 'мягкая', 'мягкая', 'твердая', 'твердая', 'мягкая', 'мягкая'].
# Просмотрите документацию по функции pd.pivot_table с помощью вопросительного знака.Для каждого автора посчитайте суммарную стоимость книг в твердой и мягкой обложке. Используйте для этого функцию pd.pivot_table. При этом столбцы должны называться "твердая" и "мягкая", а индексами должны быть фамилии авторов. Пропущенные значения стоимостей заполните нулями, при необходимости загрузите библиотеку Numpy.
# Назовите полученный датасет book_info и сохраните его в формат pickle под названием "book_info.pkl". Затем загрузите из этого файла датафрейм и назовите его book_info2. Удостоверьтесь, что датафреймы book_info и book_info2 идентичны.

# In[94]:


authors_price['cover'] = ['твердая', 'мягкая', 'мягкая', 'твердая', 'твердая', 'мягкая', 'мягкая']

authors_price


# In[103]:


book_info = pd.pivot_table(authors_price, values=['price'],index=['author_name', 'cover'], fill_value=0)
print(book_info)
print('--------------------------')

book_info.to_pickle('book_info')
book_info = pd.read_pickle('book_info')
print(book_info)


# In[ ]:




