import numpy as np
import matplotlib.pyplot as plt

font = {'size': 11.5}
plt.rc('font', **font)

fig, ax = plt.subplots(figsize=(7, 3), subplot_kw=dict(aspect="equal"))

# bigram
real = 84+140
gen = 138+113
cd = 54+23

# hybrid
# real = 144+126
# gen = 99+100
# cd = 33+50

total = real+gen+cd

realp = str(round(real/total*100,1))
genp = str(round(gen/total*100,1))
cdp = str(round(cd/total*100,1))

recipe = ["Real: "+realp+"%", "Generated: "+genp+"%", "Cannot Determine: "+cdp+"%"]

data = [real, gen, cd]

wedges, texts = ax.pie(data, colors=["white", "white", "white"], wedgeprops={"width":0.5, "edgecolor": "black"}, startangle=-40)

bbox_props = dict(boxstyle="square,pad=0.3", fc="w", ec="k", lw=0.72)
kw = dict(arrowprops=dict(arrowstyle="-"),
          bbox=bbox_props, zorder=0, va="center")

patterns = ["-" , "+" , "o"]

for i, p in enumerate(wedges):
    p.set_hatch(patterns[(i)%len(patterns)])
    ang = (p.theta2 - p.theta1)/2. + p.theta1
    y = np.sin(np.deg2rad(ang))
    x = np.cos(np.deg2rad(ang))
    horizontalalignment = {-1: "right", 1: "left"}[int(np.sign(x))]
    connectionstyle = "angle,angleA=0,angleB={}".format(ang)
    kw["arrowprops"].update({"connectionstyle": connectionstyle})
    ax.annotate(recipe[i], xy=(x, y), xytext=(1.35*np.sign(x), 1.4*y),
                horizontalalignment=horizontalalignment, **kw)

plt.savefig('1.png')

