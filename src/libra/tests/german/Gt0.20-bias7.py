
assume (x05 > 0.04)

x10 = (-0.280160)*x00 + (-0.230443)*x01 + (0.597130)*x02 + (0.031230)*x03 + (-0.017688)*x04 + (-0.254305)*x05 + (0.430026)*x06 + (-0.027808)*x07 + (-0.292561)*x08 + (-0.023500)*x09 + (-0.019670)*x010 + (0.039149)*x011 + (-0.184604)*x012 + (0.037639)*x013 + (-0.486760)*x014 + (0.357496)*x015 + (0.479039)*x016 + (0.018418)
x11 = (0.072912)*x00 + (0.273247)*x01 + (-0.521361)*x02 + (-0.149477)*x03 + (-0.244512)*x04 + (0.043708)*x05 + (-0.303463)*x06 + (0.390037)*x07 + (0.142257)*x08 + (0.588276)*x09 + (0.551082)*x010 + (-0.033840)*x011 + (0.433517)*x012 + (0.270956)*x013 + (-0.387843)*x014 + (0.066515)*x015 + (-0.037667)*x016 + (0.079036)
x12 = (0.009279)*x00 + (0.361709)*x01 + (-1.023214)*x02 + (-0.591827)*x03 + (0.368390)*x04 + (0.443208)*x05 + (0.588277)*x06 + (0.199391)*x07 + (-0.716333)*x08 + (0.324398)*x09 + (0.200583)*x010 + (0.424657)*x011 + (0.145449)*x012 + (-0.482834)*x013 + (0.236141)*x014 + (0.046168)*x015 + (0.463641)*x016 + (0.026416)
x13 = (-0.255692)*x00 + (0.144344)*x01 + (-0.004864)*x02 + (-0.480603)*x03 + (0.271657)*x04 + (0.585552)*x05 + (0.147812)*x06 + (0.434856)*x07 + (0.106361)*x08 + (-0.103687)*x09 + (0.232763)*x010 + (-0.169305)*x011 + (-0.162327)*x012 + (-0.349759)*x013 + (0.090531)*x014 + (-0.263047)*x015 + (-0.141834)*x016 + (0.033244)
x14 = (0.049968)*x00 + (0.803874)*x01 + (-0.390656)*x02 + (0.157834)*x03 + (-0.024508)*x04 + (0.435445)*x05 + (0.493129)*x06 + (-0.184197)*x07 + (0.011891)*x08 + (0.554493)*x09 + (-0.280086)*x010 + (0.287803)*x011 + (0.288338)*x012 + (-0.422141)*x013 + (0.228356)*x014 + (0.330109)*x015 + (0.419117)*x016 + (0.133375)
#
ReLU(x10)
ReLU(x11)
ReLU(x12)
ReLU(x13)
ReLU(x14)

x20 = (-0.926278)*x10 + (-0.785429)*x11 + (-0.354894)*x12 + (0.791651)*x13 + (0.766285)*x14 + (0.088997)
x21 = (-0.328040)*x10 + (-0.578944)*x11 + (-0.406393)*x12 + (0.620099)*x13 + (0.404475)*x14 + (0.063797)
x22 = (-0.757123)*x10 + (-0.219783)*x11 + (-0.664724)*x12 + (0.205297)*x13 + (0.031592)*x14 + (-0.013655)
x23 = (-0.301369)*x10 + (0.824374)*x11 + (-0.513471)*x12 + (-0.876266)*x13 + (0.651355)*x14 + (0.018807)
x24 = (-0.678208)*x10 + (0.436363)*x11 + (0.996166)*x12 + (0.508515)*x13 + (0.650177)*x14 + (0.102907)
#
ReLU(x20)
ReLU(x21)
ReLU(x22)
ReLU(x23)
ReLU(x24)

x30 = (0.421788)*x20 + (0.699562)*x21 + (0.621708)*x22 + (-0.815259)*x23 + (0.366491)*x24 + (0.130063)
x31 = (0.510558)*x20 + (0.266606)*x21 + (-0.615755)*x22 + (0.000331)*x23 + (-0.660094)*x24 + (-0.029713)
x32 = (-0.317690)*x20 + (-0.012082)*x21 + (0.161765)*x22 + (-0.361840)*x23 + (-0.534973)*x24 + (0.000000)
x33 = (0.555502)*x20 + (-0.450860)*x21 + (0.427699)*x22 + (-0.131301)*x23 + (-0.682597)*x24 + (0.000000)
x34 = (-0.185462)*x20 + (-0.437983)*x21 + (-0.398872)*x22 + (-0.704439)*x23 + (-0.298607)*x24 + (0.000000)
#
ReLU(x30)
ReLU(x31)
ReLU(x32)
ReLU(x33)
ReLU(x34)

x40 = (-0.307049)*x30 + (0.744699)*x31 + (-0.565443)*x32 + (-0.344662)*x33 + (-0.563485)*x34 + (-0.006038)
x41 = (-0.044348)*x30 + (-0.693776)*x31 + (-0.182796)*x32 + (-0.648276)*x33 + (0.501347)*x34 + (0.000000)
x42 = (-0.296975)*x30 + (-0.577508)*x31 + (-0.458916)*x32 + (0.610815)*x33 + (-0.563613)*x34 + (0.000000)
x43 = (0.892314)*x30 + (0.040994)*x31 + (-0.307384)*x32 + (0.405860)*x33 + (0.553578)*x34 + (-0.000069)
x44 = (-0.529037)*x30 + (-0.570893)*x31 + (-0.036921)*x32 + (0.689564)*x33 + (-0.177243)*x34 + (0.000000)
#
ReLU(x40)
ReLU(x41)
ReLU(x42)
ReLU(x43)
ReLU(x44)

x50 = (0.713789)*x40 + (0.205651)*x41 + (-0.370138)*x42 + (-0.275031)*x43 + (0.646314)*x44 + (-0.472154)
x51 = (-0.556771)*x40 + (0.905806)*x41 + (-0.451489)*x42 + (-1.245101)*x43 + (0.146641)*x44 + (0.472154)
