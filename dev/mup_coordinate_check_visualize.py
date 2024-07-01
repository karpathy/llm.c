from copy import copy
import os
import struct

import pandas as pd


# Taken from: https://github.com/microsoft/mup/blob/main/mup/coord_check.py#L472
def display_and_save_plot(df, y='l1', save_to=None, suptitle=None, x='width', hue='module',
                    legend='full', name_contains=None, name_not_contains=None, module_list=None,
                    loglog=True, logbase=2, face_color=None, subplot_width=5,
                    subplot_height=4):
    '''Plot coord check data `df` obtained from `get_coord_data`.

    Input:
        df:
            a pandas DataFrame obtained from `get_coord_data`
        y:
            the column of `df` to plot on the y-axis. Default: `'l1'`
        save_to:
            path to save the resulting figure, or None. Default: None.
        suptitle:
            The title of the entire figure.
        x:
            the column of `df` to plot on the x-axis. Default: `'width'`
        hue:
            the column of `df` to represent as color. Default: `'module'`
        legend:
            'auto', 'brief', 'full', or False. This is passed to `seaborn.lineplot`.
        name_contains, name_not_contains:
            only plot modules whose name contains `name_contains` and does not contain `name_not_contains`
        module_list:
            only plot modules that are given in the list, overrides `name_contains` and `name_not_contains`
        loglog:
            whether to use loglog scale. Default: True
        logbase:
            the log base, if using loglog scale. Default: 2
        face_color:
            background color of the plot. Default: None (which means white)
        subplot_width, subplot_height:
            The width and height for each timestep's subplot. More precisely,
            the figure size will be
                `(subplot_width*number_of_time_steps, subplot_height)`.
            Default: 5, 4

    Output:
        the `matplotlib` figure object
    '''
    ### preprocessing
    df = copy(df)
    # nn.Sequential has name '', which duplicates the output layer
    df = df[df.module != '']
    if module_list is not None:
        df = df[df['module'].isin(module_list)]
    else:
        if name_contains is not None:
            df = df[df['module'].str.contains(name_contains)]
        if name_not_contains is not None:
            df = df[~(df['module'].str.contains(name_not_contains))]
    # for nn.Sequential, module names are numerical
    try:
        df['module'] = pd.to_numeric(df['module'])
    except ValueError:
        pass

    ts = df.t.unique()

    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set()

    def tight_layout(plt):
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    ### plot
    fig = plt.figure(figsize=(subplot_width * len(ts), subplot_height))
    hue_order = sorted(set(df['module']))
    if face_color is not None:
        fig.patch.set_facecolor(face_color)
    ymin, ymax = min(df[y]), max(df[y])
    for t in ts:
        t = int(t)
        plt.subplot(1, len(ts), t)
        sns.lineplot(x=x, y=y, data=df[df.t == t], hue=hue, hue_order=hue_order, legend=legend if t == 1 else None)
        plt.title(f't={t}')
        if t != 1:
            plt.ylabel('')
        if loglog:
            plt.loglog(base=logbase)
        ax = plt.gca()
        ax.set_ylim([ymin, ymax])
    if suptitle:
        plt.suptitle(suptitle)
    tight_layout(plt)
    if save_to is not None:
        plt.savefig(save_to)
        print(f'coord check plot saved to {save_to}')

    return fig

def get_num_bytes_for_dtype(dtype):
    if dtype == 'f':
        return 4
    elif dtype == 'i':
        return 4
    elif dtype == 'bf16':
        return 2
    else:
        raise ValueError(f'Unknown dtype: {dtype}')

def get_list_from_c_binary_file(filename, n, dtype = 'f'):
    dtype_size = get_num_bytes_for_dtype(dtype)
    num_bytes = n * dtype_size
    data = open(filename, "rb").read(num_bytes)
    if dtype == "f":
        num_format = f'{n}{dtype}'
        num_list = list(struct.unpack(num_format, data))
    else:
        raise ValueError(f'Unknown dtype: {dtype}')
    return num_list

module_names_hardcoded = [
    "enc_fwd",
    "ln1_fwd",
    "qkv_1",
    "l_att_y_1",
    "l_attproj_1",
    "l_residual2_1",
    "l_fch_1",
    "l_fch_gelu_1",
    "l_cproj_1",
    "l_residual3_1",
    "qkv_2",
    "l_att_y_2",
    "l_attproj_2",
    "l_residual2_2",
    "l_fch_2",
    "l_fch_gelu_2",
    "l_cproj_2",
    "l_residual3_2",
    "logits"
]

def get_coord_check_data(fpath):
    data = []
    num_steps = 4
    num_activations = len(module_names_hardcoded)
    for f in fpath:
        name = os.path.basename(f)  # file format example: `usemup=0_width=256_coord_check_data.bin`
        width = int(name.split('_')[1].split('=')[1])
        l1_activations = get_list_from_c_binary_file(f, num_steps * num_activations, 'f')
        for i, l1_val in enumerate(l1_activations):
            data.append({
                't': i // num_activations + 1,
                'l1': l1_val,
                'module': module_names_hardcoded[i % num_activations],
                'width': width,
            })

    return pd.DataFrame(data)

def save_plot(file_paths, mup):
    legend=False
    optimizer = 'adam'
    lr = 0.006
    nseeds = 4
    df = get_coord_check_data(file_paths)
    prm = 'μP' if mup else 'SP'
    save_to = os.path.join(root_dir, f'{prm.lower()}_trsfmr_{optimizer}_coord.png')
    suptitle=f'{prm} Transformer {optimizer} lr={lr} nseeds={nseeds}'

    display_and_save_plot(df, legend=legend, save_to=save_to, suptitle=suptitle, face_color='xkcd:light grey' if not mup else None)

root_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir))

file_paths_mup = [os.path.join(root_dir, file) for file in os.listdir(root_dir) if file.startswith("usemup=1")]
file_paths_no_mup = [os.path.join(root_dir, file) for file in os.listdir(root_dir) if file.startswith("usemup=0")]

save_plot(file_paths_mup, mup=True)
save_plot(file_paths_no_mup, mup=False)
