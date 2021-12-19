assume(-1 <= x00 <= 1)      # spurious
assume(-1 <= x01 <= 1)      # gross weight
assume(-1 <= x02 <= 1)      # temperature
assume(-1 <= x03 <= 1)      # altitude pressure
assume(-1 <= x04 <= 1)      # speed
assume(-1 <= x05 <= 1)      # wind
assume(-1 <= x06 <= 1)      # slope



x10 = (0.042868)*x00 + (0.358656)*x01 + (-0.042266)*x02 + (0.044351)*x03 + (1.089207)*x04 + (-0.094103)*x05 + (0.030209)*x06 + (0.813460)
x11 = (-0.000181)*x00 + (0.023415)*x01 + (-0.173008)*x02 + (-0.189846)*x03 + (2.324846)*x04 + (-0.481679)*x05 + (-0.018099)*x06 + (0.721989)
x12 = (-0.002704)*x00 + (0.305991)*x01 + (0.257663)*x02 + (0.382174)*x03 + (1.207401)*x04 + (0.269870)*x05 + (-0.237693)*x06  + (-0.089845)
x13 = (0.022004)*x00 + (0.842049)*x01 + (0.222931)*x02 + (0.421870)*x03 + (1.709608)*x04 + (0.152055)*x05 + (-0.137482)*x06  + (-0.703942)
x14 = (-0.001252)*x00 + (0.187511)*x01 + (0.080904)*x02 + (0.145405)*x03 + (1.796169)*x04 + (0.187181)*x05 + (-0.063689)*x06 + (0.745336)
x15 = (0.009304)*x00 + (1.219303)*x01 + (0.346605)*x02 + (0.449413)*x03 + (2.422393)*x04 + (0.569870)*x05 + (-0.152754)*x06  + (0.448015)
x16 = (0.015865)*x00 + (0.236149)*x01 + (0.139553)*x02 + (0.179294)*x03 + (2.214066)*x04 + (0.451206)*x05 + (-0.044815)*x06  + (1.071367)
x17 = (0.021715)*x00 + (0.702090)*x01 + (0.428681)*x02 + (0.577222)*x03 + (-2.244044)*x04 + (0.437589)*x05 + (0.081774)*x06 + (1.625044)
x18 = (0.010054)*x00 + (3.692974)*x01 + (0.091981)*x02 + (0.237195)*x03 + (-0.383203)*x04 + (0.202910)*x05 + (-0.048303)*x06 + (1.026329)
x19 = (-0.010964)*x00 + (0.621551)*x01 + (0.235219)*x02 + (0.417779)*x03 + (1.653007)*x04 + (0.215808)*x05 + (-0.143462)*x06 + (0.767937)
x110 = (-0.026458)*x00 + (0.930249)*x01 + (0.385343)*x02 + (0.294321)*x03 + (0.507272)*x04 + (0.281220)*x05 + (-0.019293)*x06 + (1.359362)
x111 = (0.003056)*x00 + (0.141511)*x01 + (0.034918)*x02 + (0.024875)*x03 + (2.049487)*x04 + (-0.015903)*x05 + (0.008293)*x06  + (1.435690)
#
ReLU(x10)
ReLU(x11)
ReLU(x12)
ReLU(x13)
ReLU(x14)
ReLU(x15)
ReLU(x16)
ReLU(x17)
ReLU(x18)
ReLU(x19)
ReLU(x110)
ReLU(x111)

x20 = (-0.120257)*x10 + (1.500900)*x11 + (-0.819523)*x12 + (-0.726754)*x13 + (-0.385895)*x14 + (-0.794264)*x15 + (-0.916295)*x16 + (0.451613)*x17 + (0.140222)*x18 + (-0.333339)*x19 + (0.009119)*x110 + (2.025482)*x111 + (1.002011)
x21 = (0.628255)*x10 + (-0.370735)*x11 + (-0.243467)*x12 + (0.571194)*x13 + (-0.138143)*x14 + (-0.151729)*x15 + (0.402771)*x16 + (-0.888947)*x17 + (2.085976)*x18 + (0.966229)*x19 + (-0.238646)*x110 + (0.172652)*x111 + (-0.821574)
x22 = (0.323096)*x10 + (-0.587485)*x11 + (-0.710373)*x12 + (-0.982887)*x13 + (1.585009)*x14 + (-1.229504)*x15 + (1.230497)*x16 + (0.473705)*x17 + (-0.622586)*x18 + (-0.312683)*x19 + (0.150929)*x110 + (0.193172)*x111 + (2.463772)
x23 = (0.393751)*x10 + (0.447198)*x11 + (0.248001)*x12 + (1.479162)*x13 + (0.514711)*x14 + (1.624043)*x15 + (0.690912)*x16 + (0.220317)*x17 + (-0.133916)*x18 + (1.372528)*x19 + (0.972119)*x110 + (0.650765)*x111 + (0.283497)
x24 = (0.201546)*x10 + (-0.765675)*x11 + (0.719614)*x12 + (1.144337)*x13 + (0.685304)*x14 + (1.385870)*x15 + (0.605917)*x16 + (0.311277)*x17 + (-1.975036)*x18 + (0.752055)*x19 + (0.249741)*x110 + (-0.426070)*x111 + (-0.083149)
x25 = (0.170170)*x10 + (-0.959308)*x11 + (-1.047559)*x12 + (0.281859)*x13 + (-0.432001)*x14 + (0.192115)*x15 + (-0.361495)*x16 + (1.593109)*x17 + (-0.176822)*x18 + (0.130276)*x19 + (1.222024)*x110 + (-0.316731)*x111 + (0.975884)
x26 = (0.923617)*x10 + (-0.055252)*x11 + (0.443316)*x12 + (1.790374)*x13 + (-0.768812)*x14 + (1.359212)*x15 + (-0.365545)*x16 + (-0.501663)*x17 + (0.435643)*x18 + (-0.051980)*x19 + (0.184193)*x110 + (-0.253772)*x111 + (-1.742661)
x27 = (-1.689763)*x10 + (-0.516423)*x11 + (0.197860)*x12 + (0.779089)*x13 + (0.360796)*x14 + (-0.111963)*x15 + (-0.111482)*x16 + (-0.452423)*x17 + (0.730053)*x18 + (-1.178242)*x19 + (1.132623)*x110 + (0.366151)*x111 + (0.433818)
x28 = (-0.251693)*x10 + (0.370036)*x11 + (0.215716)*x12 + (0.985809)*x13 + (1.087343)*x14 + (0.741220)*x15 + (1.145067)*x16 + (0.443841)*x17 + (-0.197615)*x18 + (0.520991)*x19 + (1.133853)*x110 + (0.478344)*x111 + (0.636909)
x29 = (0.505050)*x10 + (0.890525)*x11 + (0.316414)*x12 + (0.305449)*x13 + (1.180471)*x14 + (0.986672)*x15 + (0.959270)*x16 + (0.076420)*x17 + (-0.022742)*x18 + (0.748662)*x19 + (0.486021)*x110 + (1.066265)*x111 + (0.434025)
x210 = (-0.059213)*x10 + (0.244953)*x11 + (0.053130)*x12 + (1.171430)*x13 + (1.210620)*x14 + (1.060386)*x15 + (0.760330)*x16 + (-0.183878)*x17 + (-0.425581)*x18 + (0.706911)*x19 + (0.549841)*x110 + (0.455087)*x111 + (0.210529)
x211 = (0.015029)*x10 + (0.043089)*x11 + (0.241906)*x12 + (0.029222)*x13 + (0.396657)*x14 + (0.458003)*x15 + (0.948973)*x16 + (-0.479642)*x17 + (-0.799695)*x18 + (0.918800)*x19 + (0.398349)*x110 + (0.253957)*x111 + (0.397906)
#
ReLU(x20)
ReLU(x21)
ReLU(x22)
ReLU(x23)
ReLU(x24)
ReLU(x25)
ReLU(x26)
ReLU(x27)
ReLU(x28)
ReLU(x29)
ReLU(x210)
ReLU(x211)

x30 = (0.103174)*x20 + (-0.356487)*x21 + (0.310974)*x22 + (-0.170567)*x23 + (-0.447869)*x24 + (-0.382032)*x25 + (0.127222)*x26 + (-0.267952)*x27 + (-0.358838)*x28 + (-0.460369)*x29 + (-0.398777)*x210 + (-0.150915)*x211 + (0.000000)
x31 = (-1.004022)*x20 + (0.607952)*x21 + (-0.402378)*x22 + (0.372449)*x23 + (0.762377)*x24 + (-2.636132)*x25 + (0.952052)*x26 + (-0.810108)*x27 + (0.432434)*x28 + (0.568109)*x29 + (0.360753)*x210 + (1.236572)*x211 + (-0.155229)
x32 = (-0.650611)*x20 + (0.553993)*x21 + (-0.535159)*x22 + (0.732942)*x23 + (0.773739)*x24 + (-2.332463)*x25 + (0.613686)*x26 + (-0.941793)*x27 + (0.540778)*x28 + (0.903764)*x29 + (0.424145)*x210 + (1.055903)*x211 + (0.819221)
x33 = (0.848416)*x20 + (-0.375935)*x21 + (2.693231)*x22 + (-0.612701)*x23 + (0.076097)*x24 + (0.447860)*x25 + (-0.611863)*x26 + (0.460297)*x27 + (0.261743)*x28 + (-0.170165)*x29 + (-0.142718)*x210 + (0.218386)*x211 + (1.996024)
x34 = (-1.078582)*x20 + (-1.475278)*x21 + (-1.787722)*x22 + (0.928821)*x23 + (1.778959)*x24 + (-0.792623)*x25 + (0.624376)*x26 + (-1.731876)*x27 + (-0.073439)*x28 + (-0.025535)*x29 + (0.340970)*x210 + (-0.190909)*x211 + (-0.658233)
x35 = (-0.281869)*x20 + (1.242041)*x21 + (-0.101333)*x22 + (1.253384)*x23 + (0.540989)*x24 + (-1.074786)*x25 + (1.141573)*x26 + (-0.866635)*x27 + (0.801399)*x28 + (0.872080)*x29 + (1.056429)*x210 + (0.411568)*x211 + (1.197408)
x36 = (-2.716872)*x20 + (0.398061)*x21 + (1.240863)*x22 + (-0.282524)*x23 + (0.632441)*x24 + (2.582144)*x25 + (1.221135)*x26 + (-0.743645)*x27 + (0.135235)*x28 + (-0.701527)*x29 + (-0.056838)*x210 + (-0.005725)*x211 + (-0.092679)
x37 = (-1.352088)*x20 + (0.394207)*x21 + (-1.080538)*x22 + (0.345248)*x23 + (0.738104)*x24 + (-2.330176)*x25 + (0.844771)*x26 + (-1.859859)*x27 + (0.663421)*x28 + (0.019837)*x29 + (0.944846)*x210 + (0.197361)*x211 + (-0.615072)
x38 = (-0.330433)*x20 + (0.356122)*x21 + (-0.333489)*x22 + (-0.394847)*x23 + (-0.235649)*x24 + (-0.335791)*x25 + (0.079153)*x26 + (0.274537)*x27 + (-0.185026)*x28 + (-0.213745)*x29 + (-0.412558)*x210 + (0.368088)*x211 + (0.032052)
x39 = (-0.269930)*x20 + (0.611984)*x21 + (-0.200308)*x22 + (1.158284)*x23 + (0.646028)*x24 + (-0.430536)*x25 + (0.579655)*x26 + (-0.267135)*x27 + (0.954335)*x28 + (0.528210)*x29 + (0.243228)*x210 + (0.422131)*x211 + (1.423392)
x310 = (-1.276605)*x20 + (0.473304)*x21 + (-2.449260)*x22 + (0.590638)*x23 + (0.065990)*x24 + (-0.263909)*x25 + (1.557579)*x26 + (-2.712982)*x27 + (0.083632)*x28 + (-0.224389)*x29 + (0.118491)*x210 + (0.911421)*x211 + (-0.322383)
x311 = (0.012037)*x20 + (0.579870)*x21 + (0.343755)*x22 + (1.141776)*x23 + (0.612871)*x24 + (-1.271455)*x25 + (1.182078)*x26 + (-0.908573)*x27 + (1.069973)*x28 + (0.837008)*x29 + (0.958990)*x210 + (0.351176)*x211 + (1.049228)
#
ReLU(x30)
ReLU(x31)
ReLU(x32)
ReLU(x33)
ReLU(x34)
ReLU(x35)
ReLU(x36)
ReLU(x37)
ReLU(x38)
ReLU(x39)
ReLU(x310)
ReLU(x311)

x40 = (-0.047217)*x30 + (0.247297)*x31 + (-0.168150)*x32 + (1.754197)*x33 + (0.531040)*x34 + (0.731759)*x35 + (-2.427425)*x36 + (0.323511)*x37 + (0.339887)*x38 + (-0.015793)*x39 + (1.002399)*x310 + (0.813237)*x311 + (0.676751)
x41 = (0.476208)*x30 + (0.568299)*x31 + (1.003872)*x32 + (2.646505)*x33 + (1.052537)*x34 + (1.221260)*x35 + (-2.138870)*x36 + (0.875034)*x37 + (0.276275)*x38 + (0.861859)*x39 + (0.786027)*x310 + (1.010112)*x311 + (1.310946)
x42 = (0.262450)*x30 + (-0.066043)*x31 + (-1.488005)*x32 + (0.822095)*x33 + (0.357037)*x34 + (-0.735064)*x35 + (-0.761307)*x36 + (0.229135)*x37 + (0.164859)*x38 + (-1.298344)*x39 + (-0.337172)*x310 + (-0.983344)*x311 + (1.494027)
x43 = (0.056648)*x30 + (-0.476194)*x31 + (0.224577)*x32 + (-0.684796)*x33 + (0.323761)*x34 + (-0.553229)*x35 + (-0.023460)*x36 + (-0.507930)*x37 + (0.415670)*x38 + (-0.487932)*x39 + (-0.387900)*x310 + (0.347776)*x311 + (-0.071960)
x44 = (-0.222548)*x30 + (-0.514491)*x31 + (0.219671)*x32 + (-0.298418)*x33 + (-0.485789)*x34 + (-0.043730)*x35 + (0.178684)*x36 + (0.099509)*x37 + (-0.129575)*x38 + (-0.170562)*x39 + (0.168444)*x310 + (-0.120567)*x311 + (-0.062450)
x45 = (-0.103760)*x30 + (0.453116)*x31 + (1.004716)*x32 + (1.424995)*x33 + (1.074103)*x34 + (0.929108)*x35 + (-4.015090)*x36 + (0.564959)*x37 + (-0.100701)*x38 + (0.748865)*x39 + (0.995521)*x310 + (1.205008)*x311 + (0.781495)
x46 = (0.291801)*x30 + (0.807415)*x31 + (0.989458)*x32 + (2.209660)*x33 + (1.158934)*x34 + (0.888037)*x35 + (-1.583795)*x36 + (0.559027)*x37 + (0.518273)*x38 + (0.555511)*x39 + (0.761874)*x310 + (1.224745)*x311 + (1.351128)
x47 = (0.298438)*x30 + (0.995660)*x31 + (0.865403)*x32 + (2.244227)*x33 + (0.508233)*x34 + (0.953064)*x35 + (-2.084457)*x36 + (0.696808)*x37 + (0.306662)*x38 + (0.900982)*x39 + (0.792872)*x310 + (0.496161)*x311 + (1.333931)
x48 = (0.088229)*x30 + (-0.181383)*x31 + (0.020328)*x32 + (-0.541000)*x33 + (0.333512)*x34 + (-0.237548)*x35 + (0.024448)*x36 + (0.139371)*x37 + (-0.340728)*x38 + (-0.570341)*x39 + (0.349874)*x310 + (0.262200)*x311 + (-0.106706)
x49 = (0.244058)*x30 + (0.161830)*x31 + (0.131013)*x32 + (-0.358149)*x33 + (0.323931)*x34 + (-0.096254)*x35 + (0.304025)*x36 + (0.010849)*x37 + (0.155557)*x38 + (0.031854)*x39 + (0.191634)*x310 + (-0.645626)*x311 + (-0.150910)
x410 = (0.461961)*x30 + (-0.205940)*x31 + (-0.027640)*x32 + (-0.691637)*x33 + (0.468750)*x34 + (-0.368203)*x35 + (2.564429)*x36 + (-0.249060)*x37 + (0.302502)*x38 + (0.296885)*x39 + (0.166206)*x310 + (0.074834)*x311 + (-0.490279)
x411 = (0.030860)*x30 + (-2.382241)*x31 + (-0.288287)*x32 + (0.748677)*x33 + (0.221349)*x34 + (-0.646686)*x35 + (-1.761246)*x36 + (0.089636)*x37 + (0.275884)*x38 + (1.052053)*x39 + (0.392365)*x310 + (-0.820184)*x311 + (1.735650)
#
ReLU(x40)
ReLU(x41)
ReLU(x42)
ReLU(x43)
ReLU(x44)
ReLU(x45)
ReLU(x46)
ReLU(x47)
ReLU(x48)
ReLU(x49)
ReLU(x410)
ReLU(x411)

x50 = (-0.520984)*x40 + (-1.248203)*x41 + (2.467129)*x42 + (0.374833)*x43 + (0.598691)*x44 + (-0.890586)*x45 + (-0.805343)*x46 + (-1.133413)*x47 + (0.138004)*x48 + (0.095554)*x49 + (0.491149)*x410 + (1.731321)*x411 + (51.369723)
x51 = (0.520984)*x40 + (1.248203)*x41 + (-2.467129)*x42 + (-0.374833)*x43 + (-0.598691)*x44 + (0.890586)*x45 + (0.805343)*x46 + (1.133413)*x47 + (-0.138004)*x48 + (-0.095554)*x49 + (-0.491149)*x410 + (-1.731321)*x411 + (-48.630277)