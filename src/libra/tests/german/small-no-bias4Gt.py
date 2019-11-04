
assume (x05 > 0.04)

x10 = (0.028009)*x00 + (-0.280901)*x01 + (0.191003)*x02 + (-0.303536)*x03 + (0.025699)*x04 + (0.032034)*x05 + (-0.403571)*x06 + (-0.062023)*x07 + (0.484698)*x08 + (-0.161644)*x09 + (0.507273)*x010 + (0.151547)*x011 + (0.295406)*x012 + (0.410221)*x013 + (-0.351037)*x014 + (0.446027)*x015 + (-0.987182)*x016 + (0.062733)
x11 = (-0.196836)*x00 + (0.181963)*x01 + (0.008990)*x02 + (-0.396747)*x03 + (0.227628)*x04 + (0.207187)*x05 + (0.599359)*x06 + (-0.030937)*x07 + (-0.186594)*x08 + (0.340489)*x09 + (-0.260150)*x010 + (0.526372)*x011 + (0.653996)*x012 + (0.281655)*x013 + (0.152867)*x014 + (-0.148558)*x015 + (-0.244733)*x016 + (0.052417)
x12 = (-0.416333)*x00 + (-0.744860)*x01 + (0.410567)*x02 + (-0.262446)*x03 + (0.151132)*x04 + (0.192337)*x05 + (0.169373)*x06 + (0.219673)*x07 + (0.232567)*x08 + (0.235408)*x09 + (0.339077)*x010 + (0.072960)*x011 + (0.254763)*x012 + (-0.490889)*x013 + (0.369108)*x014 + (0.090126)*x015 + (0.379968)*x016 + (-0.032487)
x13 = (-0.617348)*x00 + (-0.115182)*x01 + (-0.120522)*x02 + (-0.298584)*x03 + (0.471799)*x04 + (0.100145)*x05 + (-0.278672)*x06 + (-0.157985)*x07 + (-0.059666)*x08 + (0.102096)*x09 + (0.561216)*x010 + (-0.693093)*x011 + (0.183128)*x012 + (0.350307)*x013 + (0.012143)*x014 + (-0.483934)*x015 + (0.472037)*x016 + (-0.033304)
x14 = (0.330402)*x00 + (-0.174827)*x01 + (-0.111634)*x02 + (0.094053)*x03 + (-0.487691)*x04 + (-0.129903)*x05 + (-0.095243)*x06 + (0.446838)*x07 + (0.054211)*x08 + (-0.190476)*x09 + (-0.079359)*x010 + (0.323857)*x011 + (0.285883)*x012 + (-0.398511)*x013 + (0.100215)*x014 + (-0.192808)*x015 + (0.232349)*x016 + (-0.072457)
#
ReLU(x10)
ReLU(x11)
ReLU(x12)
ReLU(x13)
ReLU(x14)

x20 = (0.374822)*x10 + (-0.769535)*x11 + (-1.011873)*x12 + (-0.425558)*x13 + (-0.971703)*x14 + (0.093996)
x21 = (0.739728)*x10 + (0.015289)*x11 + (0.265360)*x12 + (-0.653519)*x13 + (-0.365594)*x14 + (0.152610)
x22 = (-0.336611)*x10 + (-0.574578)*x11 + (-0.266664)*x12 + (-0.441964)*x13 + (-0.481683)*x14 + (0.000000)
x23 = (-0.370698)*x10 + (-0.623022)*x11 + (0.145756)*x12 + (0.837724)*x13 + (0.489852)*x14 + (-0.117604)
x24 = (-0.632274)*x10 + (-0.519331)*x11 + (-0.083431)*x12 + (-0.700962)*x13 + (0.506413)*x14 + (0.071242)
#
ReLU(x20)
ReLU(x21)
ReLU(x22)
ReLU(x23)
ReLU(x24)

x30 = (-0.559990)*x20 + (-0.608594)*x21 + (0.378340)*x22 + (-0.673664)*x23 + (0.277722)*x24 + (0.153905)
x31 = (-0.824973)*x20 + (0.485527)*x21 + (0.357409)*x22 + (0.370006)*x23 + (-0.597390)*x24 + (0.055736)
x32 = (0.504774)*x20 + (0.108021)*x21 + (0.523490)*x22 + (-0.816985)*x23 + (-0.332495)*x24 + (0.051303)
x33 = (-0.616772)*x20 + (0.216898)*x21 + (-0.210932)*x22 + (0.947673)*x23 + (0.452021)*x24 + (-0.029592)
x34 = (-0.620989)*x20 + (0.855777)*x21 + (-0.366247)*x22 + (0.843074)*x23 + (-0.696562)*x24 + (0.026424)
#
ReLU(x30)
ReLU(x31)
ReLU(x32)
ReLU(x33)
ReLU(x34)

x40 = (0.136584)*x30 + (0.465654)*x31 + (-0.589021)*x32 + (0.586656)*x33 + (-0.235543)*x34 + (-0.058084)
x41 = (-0.935386)*x30 + (0.827103)*x31 + (0.730416)*x32 + (0.978635)*x33 + (0.495401)*x34 + (0.073184)
x42 = (0.385904)*x30 + (0.330302)*x31 + (-0.529907)*x32 + (0.677606)*x33 + (0.719135)*x34 + (0.037234)
x43 = (-0.870421)*x30 + (0.296299)*x31 + (0.303225)*x32 + (0.220115)*x33 + (-0.511983)*x34 + (-0.070804)
x44 = (1.022420)*x30 + (-0.383544)*x31 + (-0.380989)*x32 + (-0.541589)*x33 + (-0.636047)*x34 + (0.188337)
#
ReLU(x40)
ReLU(x41)
ReLU(x42)
ReLU(x43)
ReLU(x44)

x50 = (-0.992637)*x40 + (-0.540870)*x41 + (0.379415)*x42 + (-0.286286)*x43 + (0.512012)*x44 + (-0.034540)
x51 = (0.530165)*x40 + (0.893148)*x41 + (0.825319)*x42 + (-0.823145)*x43 + (-1.016319)*x44 + (0.034540)
