import numpy as np
arr = np.random.randint(0, 101, size=100)

def dispersia(x):
    return np.var(x)

def otklon(x):
    return np.std(x)

def mediana(x):
    return np.median(x)

variance = dispersia(arr)
std_dev = otklon(arr)
median = mediana(arr)

print("Массив:", arr)
print("Дисперсия:", variance)
print("Среднее квадратичное отклонение:", std_dev)
print("Медиана:", median)