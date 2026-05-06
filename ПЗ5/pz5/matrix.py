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
                print(f"{self.data[i][j]:.15f}", end=" ")
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

    def Hessenberg(self, EPS=0.01):

        H = self.copy()
        # Количество строк
        M = self.rows
        # Количество столбцов
        N = self.columns
        # Введем вектор p и вспомогательные переменные
        p = Matrix(M, 1, False, 0)
        print(p.data[0][0])
        s, beta, mu = 0, 0, 0
        # Создаем еденичную матрицу Q
        Q = Matrix(M, unit=True)

        for i in range(N - 2):

            for I in range(i + 2, M):

                s += H.data[I][i] ** 2

            if s**0.5 > EPS:

                s += H.data[i + 1][i] ** 2

                # Выбор знака слагаемого beta
                if H.data[i + 1][i] < 0:
                    beta = s**0.5
                else:
                    beta = -(s**0.5)

                mu = 1.0 / beta / (beta - H.data[i + 1][i])

                for I in range(M):
                    p.data[I][0] = 0
                    if I >= i + 1:
                        p.data[I][0] = H.data[I][i]

                p.data[i + 1][0] -= beta

                for m in range(i, N):
                    s = 0
                    for n in range(i, M):
                        s += H.data[n][m] * p.data[n][0]
                    s *= mu
                    for n in range(i, M):
                        H.data[n][m] -= s * p.data[n][0]

                for m in range(M):
                    Ap, Qp = 0, 0
                    for n in range(M):
                        Ap += H.data[m][n] * p.data[n][0]
                        Qp += Q.data[m][n] * p.data[n][0]
                    Ap *= mu
                    Qp *= mu
                    for n in range(i, M):
                        H.data[m][n] -= Ap * p.data[n][0]
                        Q.data[m][n] -= Qp * p.data[n][0]
        return H, Q

    def Hausholder_Ortogonalization(self, EPS=0.01):

        # Фиксируем количество строк и столбцов матрицы
        M = self.rows
        N = self.columns
        # Копируем текущую матрицу в матрицу R
        R = self.copy()
        # Создаем еденичную матрицу Q
        Q = Matrix(M, unit=True)
        # Создаем вектор p
        p = Matrix(M, 1, False, 0)

        s, beta, mu = 0, 0, 0

        # Осуществляем цикл зануления по всем столбцам кроме последнего
        for i in range(N - 1):
            s = 0
            # Накодим квадрат нормы всех элементов, которые мы хотим обнулить
            for I in range(i + 1, M):
                s += R.data[I][i] ** 2
            # Сравниваем со значением машинного нуля, и в случае если соответствующее значение нормы больше EPS начинаем процесс обнуления
            if s**0.5 > EPS:
                # Добавляем к норме первый элемент столбца
                s += R.data[i][i] ** 2
                if R.data[i][i] < 0:
                    beta = s**0.5
                else:
                    beta = -(s**0.5)

                mu = 1.0 / beta / (beta - R.data[i][i])

                for I in range(M):
                    p.data[I][0] = 0
                    if I >= i:
                        p.data[I][0] = R.data[I][i]

                p.data[i][0] -= beta

                for m in range(i, N):
                    s = 0
                    for n in range(i, M):
                        s += R.data[n][m] * p.data[n][0]
                    s *= mu
                    for n in range(i, M):
                        R.data[n][m] -= s * p.data[n][0]

                for m in range(M):
                    s = 0
                    for n in range(i, M):
                        s += Q.data[m][n] * p.data[n][0]
                    s *= mu
                    for n in range(i, M):
                        Q.data[m][n] -= s * p.data[n][0]

        return Q, R

    def QR_iterations(self, EPS=0.01):

        M = self.columns

        RQ = self.copy()

        evec = Matrix(M, unit=True)
        eval = []

        while True:
            Num = 0
            for i in range(1, M):
                if abs(RQ.data[i][i - 1]) < EPS:
                    Num += 1
            if Num == M - 1:
                break

            Q, R = RQ.Hausholder_Ortogonalization(EPS=EPS)

            RQ = R * Q

            evec = evec * Q

        for i in range(M):
            eval.append(RQ.data[i][i])

        return eval, evec

    # зануление i строки с j позиции
    def Hausholder_raw_transf(self, V: Matrix, i: int, j: int, EPS: float):
        # Фиксируем количество строк и столбцов матрицы
        M = self.rows
        N = self.columns
        # Создаем вектор p
        p = Matrix(N, 1, False, 0)

        s, beta, mu = 0, 0, 0

        s = 0
        # Накодим квадрат нормы всех элементов, которые мы хотим обнулить
        for I in range(j + 1, N):
            s += self.data[i][I] ** 2

        # Сравниваем со значением машинного нуля, и в случае если соответствующее значение нормы больше EPS начинаем процесс обнуления
        if s**0.5 > EPS:
            # Добавляем к норме первый элемент столбца
            s += self.data[i][j] ** 2
            if self.data[i][j] < 0:
                beta = s**0.5
            else:
                beta = -(s**0.5)

            mu = 1.0 / beta / (beta - self.data[i][j])

            for I in range(N):
                p.data[I][0] = 0
                if I >= j:
                    p.data[I][0] = self.data[i][I]

            p.data[j][0] -= beta

            for m in range(M):
                s = 0
                for n in range(j, N):
                    s += self.data[m][n] * p.data[n][0]
                s *= mu
                for n in range(j, N):
                    self.data[m][n] -= s * p.data[n][0]

            for m in range(N):
                s = 0
                for n in range(j, N):
                    s += V.data[m][n] * p.data[n][0]
                s *= mu
                for n in range(j, N):
                    V.data[m][n] -= s * p.data[n][0]

        return self, V

    # столбец i с позиции j
    def Hausholder_column_transf(self, U: Matrix, i: int, j: int, EPS: float):
        # Фиксируем количество строк и столбцов матрицы
        M = self.rows
        N = self.columns
        # Создаем вектор p
        p = Matrix(M, 1, False, 0)

        s, beta, mu = 0, 0, 0

        s = 0
        # Накодим квадрат нормы всех элементов, которые мы хотим обнулить
        for I in range(j + 1, M):
            s += self.data[I][i] ** 2

        # Сравниваем со значением машинного нуля, и в случае если соответствующее значение нормы больше EPS начинаем процесс обнуления
        if s**0.5 > EPS:
            # Добавляем к норме первый элемент столбца
            s += self.data[j][i] ** 2
            if self.data[j][i] < 0:
                beta = s**0.5
            else:
                beta = -(s**0.5)

            mu = 1.0 / beta / (beta - self.data[j][i])

            for I in range(M):
                p.data[I][0] = 0
                if I >= j:
                    p.data[I][0] = self.data[I][i]

            p.data[j][0] -= beta

            for m in range(N):
                s = 0
                for n in range(j, M):
                    s += self.data[n][m] * p.data[n][0]
                s *= mu
                for n in range(j, M):
                    self.data[n][m] -= s * p.data[n][0]

            for m in range(M):
                s = 0
                for n in range(j, M):
                    s += U.data[m][n] * p.data[n][0]
                s *= mu
                for n in range(j, M):
                    U.data[m][n] -= s * p.data[n][0]

        return self, U
