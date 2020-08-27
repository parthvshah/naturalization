import numpy as np
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(7, 3), subplot_kw=dict(aspect="equal"))

# bigram
real = 79+131
gen = 127+102
cd = 48+21

# hybrid
# real = 135+119
# gen = 91+88
# cd = 28+47

total = real+gen+cd

realp = str(round(real/total*100,1))
genp = str(round(gen/total*100,1))
cdp = str(round(cd/total*100,1))

recipe = ["Real: "+realp+"%", "Generated: "+genp+"%", "Cannot Determine: "+cdp+"%"]

data = [real, gen, cd]

wedges, texts = ax.pie(data, wedgeprops=dict(width=0.5), startangle=-40)

bbox_props = dict(boxstyle="square,pad=0.3", fc="w", ec="k", lw=0.72)
kw = dict(arrowprops=dict(arrowstyle="-"),
          bbox=bbox_props, zorder=0, va="center")

for i, p in enumerate(wedges):
    ang = (p.theta2 - p.theta1)/2. + p.theta1
    y = np.sin(np.deg2rad(ang))
    x = np.cos(np.deg2rad(ang))
    horizontalalignment = {-1: "right", 1: "left"}[int(np.sign(x))]
    connectionstyle = "angle,angleA=0,angleB={}".format(ang)
    kw["arrowprops"].update({"connectionstyle": connectionstyle})
    ax.annotate(recipe[i], xy=(x, y), xytext=(1.35*np.sign(x), 1.4*y),
                horizontalalignment=horizontalalignment, **kw)

plt.savefig('1.png')

