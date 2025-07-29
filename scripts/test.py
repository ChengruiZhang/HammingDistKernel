import math
import matplotlib.pyplot as plt

def Cube_Vec(hq, hk, s, d, BG, C_vec, C_cube, k, block, c):
    term1 = max(
        ((hq / hk + s) * d) / (8 * BG),
        ((2 * hq / hk + s) * d) / C_vec,
        ((s + 1) * d) / (2 * BG)
    )

    term2 = max(
        ((s + 1) * d) / (2 * BG),
        (2 * s * d) / (C_cube / 16),
        (2 * s) / BG
    )

    term3 = max(
        (2 * s) / BG,
        (4 * k) / BG,
        (c * (s / block) * math.log(s / block) + s) / C_vec
    )

    total_cost = term1 + term2 + term3
    return total_cost


def Vec(hq, hk, s, d, BG, k, c, block, C_vec):
    term1 = ((hq + s) * d) / (8 * BG)
    term2 = (4 * k) / BG
    term3 = ((d + 4 + 15 + c * (1 / block) * math.log(s / block)) * s) / C_vec

    return max(term1, term2, term3)

def figure_plt(hq, hk, s, d, BG, C_vec, C_cube, k, block, c, s_values):

    # costs = [Vec(32, 8, s, 4096, 40, 1024, 5, 8, 128) for s in s_values]

    total_costs = [Cube_Vec(hq, hk, s, d, BG, C_vec, C_cube, k, block, c) for s in s_values]
    costs = [Vec(hq, hk, s, d, BG, k, c, block, C_vec) for s in s_values]
    
    plt.figure(figsize=(10, 6))
    plt.plot(s_values, total_costs, label='Cube_Vec', linewidth=2)
    plt.plot(s_values, costs, label='Vec', linewidth=2, linestyle='--')
    plt.xlabel('s (Sequence Length)', fontsize=12)
    plt.ylabel('Cost', fontsize=12)
    plt.title('Comparison of Cube_Vec vs Vec over s', fontsize=14)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    # plt.show()
    plt.savefig("1.png")

if __name__ == "__main__":
    # Example parameters
    hq = 32  # Hamming query length
    hk = 8   # Hamming key length
    s = 4096  # Number of Seq
    d = 576   # Dimension of the hash
    BG = 40   # Bandwidth GM
    C_vec = 32  # Vector cache size
    C_cube = 8192  # Cube cache size
    k = 1024   # Some constant related to the algorithm
    block = 64  # Block size
    c = 5    # Constant for sort

    total_cost = Cube_Vec(hq, hk, s, d, BG, C_vec, C_cube, k, block, c)
    cost = Vec(hq, hk, s, d, BG, k, c, block, C_vec)

    s_values = list(range(128, 8192 + 1, 128))
    figure_plt(hq, hk, s, d, BG, C_vec, C_cube, k, block, c, s_values)

    print("Total Cost:", total_cost)
    print("Cost:", cost)
