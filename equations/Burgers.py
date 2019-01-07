import numpy as np
import math
import matplotlib.pyplot as plt

#   WENOスキームを用いて解く
#   周期境界条件
def Burgers(u, dx, dt):
    #   まずは、セル左の境界での値を補間
    #   ENOの補間式
    p1 = [1.0 / 3.0 * u[i - 2] - 7.0 / 6.0 * u[i - 1] + 11.0 / 6.0 * u[i] for i in range(len(u))]
    p2 = [-1.0 / 6.0 * u[i - 1] + 5.0 / 6.0 * u[i - 1] + 1.0 / 3.0 * u[(i + 1)%len(u)] for i in range(len(u))]
    p3 = [1.0 / 3.0 * u[i] + 5.0 / 6.0 * u[(i + 1)%len(u)] - 1.0 / 6.0 * u[(i + 2)%len(u)] for i in range(len(u))]

    #   Smoothness indicator
    S1 = [13.0 / 12.0 * (u[i - 2] - 2 * u[i - 1] + u[i]) ** 2 + 1.0 / 4.0 * (u[i - 2] - 4 * u[i - 1] + 3 * u[i]) ** 2
          for i in range(len(u))]
    S2 = [13.0 / 12.0 * (u[i - 1] - 2 * u[i] + u[(i + 1)%len(u)]) ** 2 + 1.0 / 4.0 * (u[i - 1] - u[(i + 1)%len(u)]) ** 2 for i in
          range(len(u))]
    S3 = [13.0 / 12.0 * (u[i] - 2 * u[(i + 1)%len(u)] + u[(i + 2)%len(u)]) ** 2 + 1.0 / 4.0 * (3 * u[i] - 4 * u[(i + 1)%len(u)] + u[(i + 2)%len(u)]) ** 2
          for i in range(len(u))]

    #   parameter alpha
    #   epsilon from "http://www.scholarpedia.org/article/WENO_methods"
    epsilon = 1.0e-6
    alpha1 = [0.1 / (S1[i] + epsilon) for i in range(len(u))]
    alpha2 = [0.6 / (S2[i] + epsilon) for i in range(len(u))]
    alpha3 = [0.3 / (S3[i] + epsilon) for i in range(len(u))]

    #   Weight
    w1 = [alpha1[i] / (alpha1[i] + alpha2[i] + alpha3[i]) for i in range(len(u))]
    w2 = [alpha2[i] / (alpha1[i] + alpha2[i] + alpha3[i]) for i in range(len(u))]
    w3 = [alpha3[i] / (alpha1[i] + alpha2[i] + alpha3[i]) for i in range(len(u))]

    #   u^L_[i+1/2]
    u_l = [w1[i] * p1[i] + w2[i] * p2[i] + w3[i] * p3[i] for i in range(len(u))]

    """   続いて、同様にしてセル右側での値を補間   """
    #   上の手続きをインデックス逆転させただけ
    #   ENOの補間式
    p1 = [1.0 / 3.0 * u[(i + 2)%len(u)] - 7.0 / 6.0 * u[(i + 1)%len(u)] + 11.0 / 6.0 * u[i] for i in range(len(u))]
    p2 = [-1.0 / 6.0 * u[(i + 1)%len(u)] + 5.0 / 6.0 * u[(i + 1)%len(u)] + 1.0 / 3.0 * u[i - 1] for i in range(len(u))]
    p3 = [1.0 / 3.0 * u[i] + 5.0 / 6.0 * u[i - 1] - 1.0 / 6.0 * u[i - 2] for i in range(len(u))]

    #   Smoothness indicator
    S1 = [13.0 / 12.0 * (u[(i + 2)%len(u)] - 2 * u[(i + 1)%len(u)] + u[i]) ** 2 + 1.0 / 4.0 * (u[(i + 2)%len(u)] - 4 * u[(i + 1)%len(u)] + 3 * u[i]) ** 2
          for i in range(len(u))]
    S2 = [13.0 / 12.0 * (u[(i + 1)%len(u)] - 2 * u[i] + u[i - 1]) ** 2 + 1.0 / 4.0 * (u[(i + 1)%len(u)] - u[i - 1]) ** 2 for i in
          range(len(u))]
    S3 = [13.0 / 12.0 * (u[i] - 2 * u[i - 1] + u[i - 2]) ** 2 + 1.0 / 4.0 * (3 * u[i] - 4 * u[i - 1] + u[i - 2]) ** 2
          for i in range(len(u))]

    #   parameter alpha
    #   epsilon from "http://www.scholarpedia.org/article/WENO_methods"
    epsilon = 1.0e-6
    alpha1 = [0.1 / (S1[i] + epsilon) for i in range(len(u))]
    alpha2 = [0.6 / (S2[i] + epsilon) for i in range(len(u))]
    alpha3 = [0.3 / (S3[i] + epsilon) for i in range(len(u))]

    #   Weight
    w1 = [alpha1[i] / (alpha1[i] + alpha2[i] + alpha3[i]) for i in range(len(u))]
    w2 = [alpha2[i] / (alpha1[i] + alpha2[i] + alpha3[i]) for i in range(len(u))]
    w3 = [alpha3[i] / (alpha1[i] + alpha2[i] + alpha3[i]) for i in range(len(u))]

    #   u^R_[i+1/2]
    u_r = [w1[i] * p1[i] + w2[i] * p2[i] + w3[i] * p3[i] for i in range(len(u))]

    ###################
    #   数値流速の計算  #
    ###################
    f_half = [0.5 * (0.5 * u_r[i] ** 2 + 0.5 * u_l[i] ** 2) + 0.5 * abs(0.5 * (u_r[i] + u_l[i])) * (u_r[i] - u_l[i]) for
              i in range(len(u))]
    new_u = [u[i] - dt / dx * (f_half[i] - f_half[i - 1]) for i in range(len(u))]

    return new_u

def main():
    n = int(1000)
    x = [math.sin(2*math.pi*float(i)/float(n)) for i in range(n)]
    for i in range(100):
        print((x))
        x =Burgers(x,0.1,0.1)
    plt.plot(x)
    plt.show()

if __name__=='__main__':
    main()
