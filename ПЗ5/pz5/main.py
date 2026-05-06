from matrix import Matrix


def gen_gilbert_matrix(N: int, a: float = 2.124, b: float = 3.498):
    M = Matrix(N, gen=False)
    for i in range(N):
        for j in range(N):
            M.data[i][j] = 1 / (1 + a * i + b * j)
    print(f"Матрица Гильберта размером {N}*{N}:")
    M.printm()
    print("///////////////////////////")
    return M


# ////////////////////////// LU-разложение /////////////////////////////////


def find_main_element(LU: Matrix, j: int):
    max_ind = j
    for i in range(j + 1, LU.rows):
        if abs(LU.data[i][j]) > (LU.data[max_ind][j]):
            max_ind = i
    return max_ind


def LU_decomposition(A: Matrix):

    M = A.rows

    # создаем копию матрицы A и записываем ее в LU
    LU = A.copy()

    # создаем массив перестановок - список
    P = [i for i in range(M)]

    # построим верхне треугольную матрицу U
    # Gauss first
    for i in range(M - 1):
        I = find_main_element(LU, i)
        # если это не диагональ переставляем I и i строки местами
        if I != i:

            LU.data[i], LU.data[I] = LU.data[I], LU.data[i]

            ind = P[i]
            P[i] = P[I]
            P[I] = ind

        # для оставшихся строк выполняем гауссово исключение
        for j in range(i + 1, M):
            help = LU.data[j][i] / LU.data[i][i]

            # зануляем внутренние компоненты для уменьшения ошибок в вычислениях
            LU.data[j][i] = 0
            # вычитаем элементы i строки из строк от i+1 до M
            for k in range(i + 1, M):
                LU.data[j][k] -= help * LU.data[i][k]

    # строим нижнетреугольную матрицу L
    for i in range(M):
        for j in range(i):
            sum_LikUkj = 0
            for k in range(j):
                sum_LikUkj += LU.data[i][k] * LU.data[k][j]
            LU.data[i][j] = (A.data[P[i]][j] - sum_LikUkj) / LU.data[j][j]

    return LU, P


# прямой ход для LU разложения Ly=f
def LU_direct_way(LU: Matrix, f: Matrix):
    M = LU.rows
    F = f.copy()

    for i in range(M):
        for j in range(i):
            F.data[i][0] -= LU.data[i][j] * F.data[j][0]
    return F


# обратный ход для LU разложения Ux=y
def LU_back_way(LU: Matrix, f: Matrix):
    M = LU.rows
    F = f.copy()

    for i in range(M - 1, -1, -1):
        for j in range(i + 1, M):
            F.data[i][0] -= LU.data[i][j] * F.data[j][0]
        F.data[i][0] /= LU.data[i][i]
    return F


def LU_solver(A: Matrix, f: Matrix):

    # количество строк в матрице
    M = A.rows

    if M == 0:
        raise ValueError("Передайте на вход не пустую матрицу")
    if M != A.columns:
        raise ValueError(
            "Для реализации данного метода необходимо передать квадратную матрицу"
        )

    LU, P = LU_decomposition(A)

    res = f.copy()

    for i in range(M):
        res.data[i][0] = f.data[P[i]][0]

    # прямой ход
    F = LU_direct_way(LU, res)
    # обратный ход
    F_fin = LU_back_way(LU, F)

    return F_fin


# //////////////////////////QR разложение на базе процедуры Грамма-Шмидта/////////////////////////////////


# функция копирования k-го столбца матрицы
def copy_column(A: Matrix, k: int):

    # количество строк матрицы А
    M = A.rows
    # количество столбцов матрицы А
    N = A.columns

    clm = Matrix(M, 1, gen=False)

    if M == 0:
        raise ValueError("Нулевое число строк матрицы")

    if k >= N:
        raise ValueError("Выход за пределы матрицы по индексу столбца")

    for i in range(A.rows):
        clm.data[i][0] = A.data[i][k]

    return clm


# норма вектора - евклидова
def norm_ev(vec: Matrix):

    if vec.columns == 1:
        norm = 0
        for i in range(vec.rows):
            norm += vec.data[i][0] ** 2

        return norm**0.5
    else:
        raise ValueError("Не корректный формат вектора, необходим формат вида N*1")


def Gram_Schmindt_Procedure(A: Matrix, q: Matrix, r: Matrix, eps: float = 1e-15):

    A_c = A.copy()

    # число строк
    M = A_c.rows
    # число столбцов
    N = A_c.columns

    R = r.copy()
    Q = q.copy()

    # rij=(xj,qi)
    for j in range(N):
        for i in range(j):
            for k in range(M):
                R.data[i][j] += A_c.data[k][j] * Q.data[k][i]

        # копируем j-й столбец из матрицы А
        _q = copy_column(A_c, j)
        # вычисляем новый векторо q матрицы Q, но пока без нормировки
        for i in range(j):
            for k in range(M):
                _q.data[k][0] -= Q.data[k][i] * R.data[i][j]

        # вычисляем норму j-го построенного ортогонального веткора
        R.data[j][j] = norm_ev(_q)

        if R.data[j][j] < eps:
            return Q, R
        # нормируем полученный вектор
        for i in range(M):
            Q.data[i][j] = _q.data[i][0] / R.data[j][j]

    return Q, R


# метод для QR-разложения мпатрицы А
def QR_decomp_Gram_Sch(A: Matrix):
    M = A.rows
    N = A.columns
    Q = Matrix(M, unit=True)
    R = Matrix(M, N, gen=False, point=0)

    Q_f, R_f = Gram_Schmindt_Procedure(A, Q, R)

    return Q_f, R_f


# прямой ход y = Q^Tf
def QR_direct_way(Q: Matrix, f: Matrix):

    res = f.copy()
    Q_t = Q.transposition()
    res = Q_t * f

    return res


# обратный ход Rx=y
def QR_back_way(R: Matrix, res: Matrix):

    M = R.rows
    F = res.copy()

    for i in range(M - 1, -1, -1):
        for j in range(i + 1, M):
            F.data[i][0] -= R.data[i][j] * F.data[j][0]
        F.data[i][0] /= R.data[i][i]
    return F


def QR_solver(A: Matrix, f: Matrix):
    # количество строк в матрице
    M = A.rows

    f_c = f.copy()

    if M == 0:
        raise ValueError("Передайте на вход не пустую матрицу")
    if M != A.columns:
        raise ValueError(
            "Для реализации данного метода необходимо передать квадратную матрицу"
        )

    Q, R = QR_decomp_Gram_Sch(A)

    res = QR_direct_way(Q, f_c)
    fin = QR_back_way(R, res)

    return fin


# ////////////////////////// Усеченное SVD разложение /////////////////////////////////


def SVD(A: Matrix, eps: float = 1e-15):

    A_c = A.copy()

    M = A_c.rows
    N = A_c.columns

    if M == 0:
        raise ValueError("Передайте на вход не пустую матрицу")
    if N == 0:
        raise ValueError("Передайте на вход не пустую матрицу")

    # наименьшее измерение
    min_size = min(M, N)

    # размеры нижней и верхней внешних диагоналей
    up_size = min_size - 1
    down_size = min_size - 1

    # матрица левых сингулярных векторов
    U = Matrix(M, gen=False, unit=True)
    # матрица сингулярных чисел
    Sigma = A_c.copy()
    # матрица правых сингулярных векторов
    V = Matrix(N, gen=False, unit=True)

    # 1 этап - бидигоанализация
    for i in range(min_size - 1):
        Sigma.Hausholder_column_transf(U, i, i, eps)
        Sigma.Hausholder_raw_transf(V, i, i + 1, eps)
    if M > N:
        Sigma.Hausholder_column_transf(U, N - 1, N - 1, eps)
        down_size += 1
    if M < N:
        Sigma.Hausholder_raw_transf(V, M - 1, M, eps)
        up_size += 1

    # 2 этап - преследование

    while True:
        # число ненулевых элементов над главной диагональю
        countupelements = 0
        # обнуление верхней диагонали
        for i in range(up_size):
            if abs(Sigma.data[i][i + 1]) > eps:
                Sigma.Hausholder_raw_transf(V, i, i, eps)
            else:
                countupelements += 1

        # если все элементы выше диагонали приняты за 0, то завершаем итерационный процесс
        if countupelements == up_size:
            break
        # обнуление нижней диагонали
        for i in range(down_size):
            if abs(Sigma.data[i + 1][i]) > eps:
                Sigma.Hausholder_column_transf(U, i, i, eps)
    # далее обрабатываем две особые ситцации
    # убираем отрицательные сингулярные числа
    for i in range(min_size):
        if Sigma.data[i][i] < 0:
            Sigma.data[i][i] = -Sigma.data[i][i]
            for j in range(M):
                U.data[j][i] = -U.data[j][i]

    # сортируем по возрастанию сингулярные числа
    for I in range(min_size):
        max_elem = Sigma.data[I][I]
        ind = I
        for i in range(I + 1, min_size):
            if Sigma.data[i][i] > max_elem:
                max_elem = Sigma.data[i][i]
                ind = i
        # если нашли такой элемент переставляем столбцы в матрицах U V
        if I != ind:
            Sigma.data[ind][ind] = Sigma.data[I][I]
            Sigma.data[I][I] = max_elem
            for r in range(M):
                elem = U.data[r][I]
                U.data[r][I] = U.data[r][ind]
                U.data[r][ind] = elem
            for r in range(N):
                elem = V.data[r][I]
                V.data[r][I] = V.data[r][ind]
                V.data[r][ind] = elem

    return U, Sigma, V


def SVD_solver(A: Matrix, f: Matrix, eps: float = 1e-15):

    M = A.rows
    N = A.columns

    # осущетсвляем проверку матрицы на факт того, что она является квадратной для
    # дальнейшего корректного вычисления обратной матрицы от sigma (поскокльу в рамках данной работы взаимодейтсвие осущетсвляется с квадратными матрицами)
    if M == N:

        U, Sigma, V = SVD(A, eps)
        print("Sigma")
        Sigma.printm()
        print("-------------------")

        # для того чтобы решить систему воспользуемся соотношением A^(-1)=V*sigma^(-1)*U^T

        # осуществим реализацию усеченной svd

        count_of_delete = 0

        max_cond_num = Sigma.data[0][0]
        min_cond_num = Sigma.data[0][0]

        for i in range(M):

            if Sigma.data[i][i] > 1e-10:
                if Sigma.data[i][i] < min_cond_num:
                    min_cond_num = Sigma.data[i][i]
                Sigma.data[i][i] = 1.0 / Sigma.data[i][i]
            else:
                count_of_delete += 1
                Sigma.data[i][i] = 0.0
                print("прошло")

        # определяем количество оставшихся столбцов
        M_red = M - count_of_delete

        U_new = Matrix(M, M_red, gen=False, point=0)
        Sigma_new = Matrix(M_red, M_red, gen=False, point=0)
        V_new = Matrix(M, M_red, gen=False, point=0)

        for i in range(M):
            for j in range(M_red):
                U_new.data[i][j] = U.data[i][j]

        for i in range(M_red):
            Sigma_new.data[i][i] = Sigma.data[i][i]

        for i in range(M):
            for j in range(M_red):
                V_new.data[i][j] = V.data[i][j]

        A_m = V_new * Sigma_new * U_new.transposition()

        res = A_m * f

        cond = max_cond_num / min_cond_num

        return res, cond
    else:
        raise ValueError("Используейте квадратную матрицу")


# //////////////////////////////////////////////
# функция для рассчета относительной погрешности
def delta(x_f: Matrix, x: Matrix):
    subs = x - x_f
    nrm = norm_ev(subs)
    nrm_d = norm_ev(x_f)
    nrm_res = nrm / nrm_d
    return nrm_res


# //////////////////////////////////////////////
# количество элементов в матрице Гилбьерта
N = 20

print("Исходный результат:")
x_target = Matrix(N, 1, gen=False)
x_target.printm()

# генерируем матрицу Гилберта необходимой размерности
M = gen_gilbert_matrix(N)

f = M * x_target

# //////////////////////////////////////////////
# осуществляем LU-разложение

f1 = LU_solver(M, f)

print("Результат полученный с помощью LU разложения:")

f1.printm()
print(f"Относительная погрешность: {delta(x_target, f1)}")
# //////////////////////////////////////////////
# осуществляем QR-разложение на базе процедуры Грамма-Шмидта
# A = Matrix(3, gen=False)
# A.data = [[-2, -2, -1], [1, 0, -2], [0, 1, 2]]
# A.data = [[-2, -2, -2], [1, 0, 0], [0, 1, 1]]

f2 = QR_solver(M, f)

print("Результат полученный с помощью QR разложения методом Грамма-Шмидта:")

f2.printm()
print(f"Относительная погрешность: {delta(x_target, f2)}")
# //////////////////////////////////////////////
# разложение посредством SVD
print("Результат полученный с помощью SVD:")
# A = Matrix(2, 3, gen=False)
# A.data = [[2, 1, 0], [1, 2, 0]]
f3, cond = SVD_solver(M, f)

f3.printm()
print(f"Относительная погрешность: {delta(x_target, f3)}")
print(f"Обсусловленность матрицы: {cond}")

# if __name__ == "__main__":
#     main()
