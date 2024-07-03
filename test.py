for j in range(2,5):  # Adjusting for 0-indexing
        for i in range(1,9-j):  #
            if i + j + 1 <= m + 2 * r:
                if kts[i + j] - kts[i] != 0:
                    a1 = (x - kts[i]) / (kts[i + j] - kts[i])
                else:
                    a1 = np.zeros(N)
                if kts[i + j + 1] - kts[i + 1] != 0:
                    a2 = (x - kts[i + 1]) / (kts[i + j + 1] - kts[i + 1])
                else:
                    a2 = np.zeros(N)
                BB[:, i, j] = a1 * BB[:, i, j - 1] + (1 - a2) * BB[:, i + 1, j - 1]
            elif i + j + 1 <= m + 2 * r:
                if kts[i + j + 1] - kts[i] != 0:
                    a1 = (x - kts[i]) / (kts[i + j + 1] - kts[i])
                else:
                    a1 = np.zeros(N)
            BB[:, i, j] = a1 * BB[:, i, j - 1]

    XX = BB[:, :2**l + r - 2, r - 2]