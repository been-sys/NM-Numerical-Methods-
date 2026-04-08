from matrix import Matrix


def main():
    # Пункты 1 и 2 (возможна случайная генерация элементов таблицы, соотвествующий механизм описан в конструкторе класса)
    m = Matrix(4, 2, False, 0)
    m.data = [[7, 6], [8, 12], [15, 17], [3, 5]]
    # Пункты 3 4 5 (промежуточные шаги полностью реализованы в методах класса)
    C = m.covariance_matr()
    # Пункт 6
    Q, R = C.Givens_Ortoganalization()
    C.printm()
    print()
    Q.printm()
    print()
    R.printm()


if __name__ == "__main__":
    main()
