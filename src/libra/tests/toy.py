x11 = -0.31*x01 + 0.99*x02 -0.63
x12 = -1.25*x01 - 0.64*x02 + 1.88
#
ReLU(x11)
ReLU(x12)

x21 = 0.40*x11 + 1.21*x12 + 0.00
x22 = 0.64*x11 + 0.69*x12 - 0.39
#
ReLU(x21)
ReLU(x22)

x31 = 0.26*x21 + 0.33*x22 + 0.45
x32 = 1.42*x21 + 0.40*x22 - 0.45
x34 = 0.4*x21 + 1.30*x22 - 0.31
x35 = 0.72*x21 + 1.03*x22 - 0.23
#