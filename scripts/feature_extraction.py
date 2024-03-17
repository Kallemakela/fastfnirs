#%%
from fastfnirs.classification import extract_features_from_raw, get_feature_names, concatenate_features
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

T = 600
ex_x_1 = np.sin(np.linspace(0, 7, T))
ex_x_2 = np.cos(np.linspace(0, 7, T))
ex_x = np.vstack([ex_x_1, ex_x_2])[None, :]
ex_x = np.tile(ex_x, (4, 1, 1))
ex_x = ex_x * 1

print(f'{ex_x.shape=}')
ex_X = {
    'sub-01': ex_x,
}
channels = ['1 hbo', '2 hbo']
n_windows=6
features=['MV', 'MAV', 'STD', 'PZN', 'polyfit_coef_1', 'PMN']
ex_Xf = extract_features_from_raw(
    ex_X,
    n_windows=n_windows,
    features=features,
)
ex_Xf = concatenate_features(ex_Xf)
xf = ex_Xf['sub-01'][0]
feature_names = get_feature_names(channels, features, n_windows)

pf = defaultdict(list)
for fi, f in enumerate(feature_names):
    fn, _ = f.rsplit('_', 1)
    pf[fn].append(xf[fi])

plot_chs = ['1 hbo']#, '2 hbo']
wl = T // n_windows
px = []
for wi in range(n_windows):
    px.append(wi*wl + wl//2)
#%%


feature_order = [
    'MV',
    'MAV',
    'polyfit_coef_1',
    'STD',
    'PZN',
    'PMN',
]
# eliminate channels that are not in plot_chs
pf_ch = {k:v for k,v in pf.items() if k[:5] in plot_chs}
# eliminate features that are not in feature_order
pf_ch = {k:v for k,v in pf_ch.items() if k.split()[-1] in feature_order}
# sort by feature_order
pf_ch = dict(sorted(pf_ch.items(), key=lambda x: feature_order.index(x[0].split()[-1])))
#%%
plt.figure(figsize=(7, 15))
for pi, (fn, f) in enumerate(pf_ch.items()):
    plt.subplot(6, 1, pi+1)
    for ci, ch in enumerate(plot_chs):
        plt.plot(ex_x[0, ci], label='original')
    
    plt.axvline(0, color='black', linestyle='--', alpha=0.5, linewidth=1)
    plt.axhline(0, color='black', linestyle='--', alpha=0.5, linewidth=1)

    for wi in range(n_windows):
        # plt.axvline(wi*wl + wl//2, color='r')
        plt.axvline(wi*wl + wl, color='black', linestyle='--', alpha=0.5, linewidth=1)
    fch = fn[:5]
    scale = 1
    if 'polyfit_coef_1' in fn:
        scale = 100
    if fch in plot_chs:
        plt.plot(px, np.array(f)*scale, marker='o', label=fn[6:])

    plt.xticks([])
    plt.yticks([])
    plt.legend(loc='lower left')

plt.suptitle('Feature extraction', fontsize=16)
plt.tight_layout()
plt.savefig('fig/feature_extraction.png', dpi=300)
plt.show()
# %%
