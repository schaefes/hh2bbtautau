# encoding: utf-8

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.colors import LogNorm, Normalize
import mplhep

sns.set(font_scale=3)
fig, ax = plt.subplots(figsize=(20,20))
br_label = ['bb', r'$\tau\tau$', r'$\mu\mu$', 'cc', 'gg', r'$\gamma\gamma$', r'Z$\gamma$', 'WW', 'ZZ']
br = np.array([0.5824, 0.06272, 0.0002176, 0.02891, 0.08187, 0.002270, 0.001533, 0.2137, 0.02619])
br_prod = np.zeros((len(br), len(br)))
im = plt.imshow(br_prod, vmin=np.min(br_prod), vmax=np.max(br_prod))
for i, b in enumerate(br):
    br_prod[i,:] = b * br
br_prod = np.flip(br_prod, axis=1)
for i in range(len(br_label)):
    for j in range(len(br_label)):
        if i < len(br_label) - j -1:
            br_prod[i, j] = 0.0000000000009
up_triang = np.triu(np.ones_like(br_prod)).astype(bool)
sns.heatmap(br_prod, cmap="Greens", square=True, norm=LogNorm(), xticklabels=br_label[::-1],
            yticklabels=br_label, cbar_kws={"shrink": 0.823, 'drawedges':True})
fig.axes[1].get_frame_on()
for i in range(len(br_label)):
    for j in range(len(br_label)):
        if i < len(br_label) - j -1:
            continue
        num = br_prod[i, j]
        if num < 0.0001:
            num, exp = "{:.2E}".format(num).split('E')
            exp = f'-{exp[-1]}'
            num1 = num
            num2 = f"$\cdot$ $10^{exp[0]}$$^{exp[1]}$"
            text = plt.text(j+.5, i+0.35, num1, ha="center", va="center", color="black", fontsize=23)
            text = plt.text(j+.5, i+0.65, num2, ha="center", va="center", color="black", fontsize=23)
        else:
            num = np.round(br_prod[i, j], 4)
            text = plt.text(j+.5, i+0.5, num, ha="center", va="center", color="black", fontsize=23)
print(*dir(fig),sep="\n")
plt.yticks(rotation=0)

# Drawing the frame
plt.axhline(y = 0, color='k',linewidth = 3)
plt.axhline(y = br_prod.shape[1], color = 'k',
            linewidth = 3)

plt.axvline(x = 0, color = 'k',
            linewidth = 3)

plt.axvline(x = br_prod.shape[0],
            color = 'k', linewidth = 3)
plt.ylabel(r'$\it{BR(H \rightarrow XX)}$')
plt.xlabel(r'$\it{BR(H \rightarrow YY)}$')
plt.text(0.6, 0.35, 'Branching Ratio', fontsize=50, verticalalignment='top')
plt.text(0.6, 0.95, r'$\it{HH \rightarrow XXYY}$', fontsize=50, verticalalignment='top')
for spine in ax.spines.values():
    spine.set(visible=True, lw=1, edgecolor="black")
