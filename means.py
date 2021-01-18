import numpy as np

mean_l_h = 0
LS = [224,230,232,225]
LV = [237,245,242,240]
UH = [40,105,85,82]
mean_u_s = 255
mean_u_v = 255

mean_l_s = np.mean(LS)
mean_l_v = np.mean(LV)
mean_u_h = np.mean(UH)

print(f'LH:{mean_l_s} LS:{mean_l_s} LV:{mean_l_v} UH:{mean_u_h} LS:{mean_u_s} LV:{mean_u_v} ')
'''
Obj1: mean=480.52324037 327.4873838 
Lh:0
Ls:224
Lv:237
Uh:40
Us:255
Uv:255

Obj2: mean=615.16331658 250.83417085
Lh:0
Ls:230
Lv:245
Uh:105
Us:255
Uv:255

Obj3: mean=409.30617284 342.99012346
Lh:0
Ls:232
Lv:242
Uh:85
Us:255
Uv:255

Obj4: mean=525.56470588 224.36470588
Lh:0
Ls:225
Lv:240
Uh:82
Us:255
Uv:255
'''