from matrix import Matrix


def main():
    M = Matrix(2, 2, False, 0)
    # M.data = [[2, 0, 1], [0, 2, 0], [1, 0, 2]]
    M.data = [[24.92, 26], [26, 31.33]]

    H, Q = M.Hessenberg(EPS=0.01)

    eval, evec = H.QR_iterations(EPS=0.01)

    print("Eigen values:")
    print(*eval)
    print("Eigen vectors:")
    (Q * evec).printm()


if __name__ == "__main__":
    main()
