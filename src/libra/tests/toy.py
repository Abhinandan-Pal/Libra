x11 = 1*x01 + 1*x02 + 0
x12 = 1*x01 - 1*x02 + 0
#
ReLU(x11)
ReLU(x12)

x21 = 1*x11 + 1*x12 - 0.5
x22 = 1*x11 - 1*x12 + 0.0
#
ReLU(x21)
ReLU(x22)

x31 = -1*x21 + 1*x22 + 3
x32 = 0*x21 + 1*x22 - 0.0
#
