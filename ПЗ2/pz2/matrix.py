import random


class Matrix:

    # Конструктор класса
    def __init__(self, rows, columns=None, gen=True, point=1, unit=False):

        if columns is None and isinstance(rows, int):
            columns = rows

        if not isinstance(rows, int) or not isinstance(columns, int):
            raise TypeError("rows и columns должны быть int")

        self.rows = rows
        self.columns = columns
        self.data = None

        if not unit:
            if gen:
                # Генерация произвольных значений элементов матрицы в промежутке от -100 до 100
                self.data = [
                    [random.randint(-100, 100) for j in range(self.columns)]
                    for i in range(self.rows)
                ]
            else:
                # Заполнение всех элементов матрицы фиксированным значением переданным в point
                self.data = [
                    [point for j in range(self.columns)] for i in range(self.rows)
                ]
        else:
            # Генерация еденичной матрицы
            if self.rows == self.columns:
                self.data = [
                    [1 if i == j else 0 for j in range(self.columns)]
                    for i in range(self.rows)
                ]
            else:
                raise ValueError(
                    "Возможно создание только квадратной еденичной матрицы!"
                )

    # Перегрузка оператора сложения
    def __add__(self, other):

        if not isinstance(other, Matrix):
            return NotImplemented

        assert isinstance(other, Matrix)

        if self.rows != other.rows or self.columns != other.columns:
            raise ValueError("Размеры матриц не совпадают")

        result = Matrix(self.rows, self.columns, False)
        result.data = [
            [a + b for a, b in zip(r1, r2)] for r1, r2 in zip(self.data, other.data)
        ]

        return result

    # Перегрузка оператора вычитания
    def __sub__(self, other):
        if not isinstance(other, Matrix):
            return NotImplemented
        if self.rows != other.rows or self.columns != other.columns:
            raise ValueError("Размеры матриц не совпадают")
        m = Matrix(self.rows, self.columns, False, 0)
        for i in range(self.rows):
            for j in range(self.columns):
                m.data[i][j] = self.data[i][j] - other.data[i][j]
        return m

    # Перегрузка оператора умножения
    def __mul__(self, other):
        if isinstance(other, Matrix) and self.columns == other.rows:
            m = Matrix(self.rows, other.columns, False, 0)
            for i in range(self.rows):
                for j in range(other.columns):
                    for k in range(self.columns):
                        m.data[i][j] += self.data[i][k] * other.data[k][j]
            return m

        elif isinstance(other, (int, float)):
            m = Matrix(self.rows, self.columns, False, 0)
            for i in range(self.rows):
                for j in range(self.columns):
                    m.data[i][j] = self.data[i][j] * other
            return m

        return NotImplemented

    # Перегрузка оператора умножения (справа)
    def __rmul__(self, other):
        if isinstance(other, (int, float)):
            return self * other
        return NotImplemented

    # Копирование матрицы
    def copy(self):
        m = Matrix(self.rows, self.columns, False)
        m.data = [
            [self.data[i][j] for j in range(self.columns)] for i in range(self.rows)
        ]
        return m

    # Вывод матрицы
    def printm(self):
        for i in range(self.rows):
            for j in range(self.columns):
                print(f"{self.data[i][j]:.2f}", end=" ")
                if j == self.columns - 1:
                    print()

    # Операция транспонирования матрицы
    def transposition(self):
        m = Matrix(self.columns, self.rows, False, 0)
        for i in range(self.rows):
            for j in range(self.columns):
                m.data[j][i] = self.data[i][j]
        return m

    # Поиск вектора средних значений
    def find_vct_of_avg(self):
        vct = Matrix(self.columns, 1, False, 0)
        for k in range(self.columns):
            for m in range(self.rows):
                vct.data[k][0] += self.data[m][k]
        return (vct * (1 / self.rows)).transposition()

    # Поиск матрицы центрирования данных
    def matr_of_cnt_data(self):
        B = self - Matrix(self.rows, 1, False, 1) * self.find_vct_of_avg()
        return B

    # Поиск ковариационной матрицы
    def covariance_matr(self):
        B = self.matr_of_cnt_data()
        C = (1 / (self.rows - 1)) * B.transposition() * B
        return C

    # QR-разложение методом Гивенса
    def Givens_Ortoganalization(self, EPS=0.01):

        # Выделяем переменные под хранение синусов и косинусов
        c, s = 0, 0

        # Сохраняем размеры матрицы в отдельные переменные для удобства
        M = self.rows
        N = self.columns

        # Создаем еденичную матрицу Q
        Q = Matrix(self.rows, self.columns, unit=True)

        # Копируем C в матрицу R
        R = self.copy()

        # Вспомогательные переменные
        help1, help2 = 0, 0

        for j in range(N - 1):

            for i in range(j + 1, M):

                if abs(R.data[i][j]) > EPS:
                    help1 = (R.data[i][j] ** 2 + R.data[j][j] ** 2) ** 0.5
                    c = R.data[j][j] / help1
                    s = R.data[i][j] / help1

                    for k in range(j, N):
                        help1 = c * R.data[j][k] + s * R.data[i][k]
                        help2 = c * R.data[i][k] - s * R.data[j][k]
                        R.data[j][k] = help1
                        R.data[i][k] = help2

                    for k in range(M):
                        help1 = c * Q.data[k][j] + s * Q.data[k][i]
                        help2 = c * Q.data[k][i] - s * Q.data[k][j]
                        Q.data[k][j] = help1
                        Q.data[k][i] = help2

        return Q, R
