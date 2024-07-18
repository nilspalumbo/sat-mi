## File adapated from https://github.com/mechanistic-interpretability-grokking/progress-measures-paper

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import einops
import tqdm.notebook as tqdm

import random
import time

from pathlib import Path
import pickle
import os

import plotly.express as px
import plotly.io as pio

import plotly.graph_objects as go

from torch.utils.data import DataLoader

from functools import *
import pandas as pd
import gc

import itertools
import copy
import re


# Key Helpers
def to_numpy(tensor, flat=False):
    if isinstance(tensor, np.ndarray):
        return tensor
    elif isinstance(tensor, list):
        return np.array(tensor)
    elif isinstance(tensor, torch.Tensor):
        if flat:
            return tensor.flatten().detach().cpu().numpy()
        else:
            return tensor.detach().cpu().numpy()
    elif type(tensor) in [int, float, bool, str]:
        return np.array(tensor)
    else:
        raise ValueError(f"Input to to_numpy has invalid type: {type(tensor)}")


def melt(tensor):
    arr = to_numpy(tensor)
    n = arr.ndim
    grid = np.ogrid[tuple(map(slice, arr.shape))]
    out = np.empty(arr.shape + (n+1,), dtype=np.result_type(arr.dtype, int))
    offset = 1

    for i in range(n):
        out[..., i+offset] = grid[i]
    out[..., -1+offset] = arr
    out.shape = (-1, n+1)

    df = pd.DataFrame(out, columns=['value']+[str(i)
                      for i in range(n)], dtype=float)
    return df.convert_dtypes([float]+[int]*n)


def broadcast_up(array, shape, axis_str=None):
    n = len(shape)
    m = len(array.shape)
    if axis_str is None:
        axis_str = " ".join([f"x{i}" for i in range(n-m, n)])
    return einops.repeat(array, f"{axis_str}->({' '.join([f'x{i}' for i in range(n)])})", **{f"x{i}": shape[i] for i in range(n)})



# Defining Kwargs
DEFAULT_KWARGS = dict(
    xaxis="x",  # Good
    yaxis="y",  # Good
    range_x=None,  # Good
    range_y=None,  # Good
    animation_name="snapshot",  # Good
    color_name="Color",  # Good
    color=None,
    log_x=False,  # Good
    log_y=False,  # Good
    toggle_x=False,  # Good
    toggle_y=False,  # Good
    legend=True,  # Good
    hover=None,  # Good
    hover_name="data",  # GOod
    return_fig=True,  # Good
    animation_index=None,  # Good
    line_labels=None,  # Good
    markers=False,  # Good
    frame_rate=None,  # Good
    facet_labels=None,
    debug=False,
    transition="none",  # TODO: Work it out
)


def split_kwargs(kwargs):
    custom = dict(DEFAULT_KWARGS)
    plotly = {}
    for k, v in kwargs.items():
        if k in custom:
            custom[k] = v
        else:
            plotly[k] = v
    return custom, plotly


# Figure Editing
## Specific Helper Functions
def update_play_button(button, custom_kwargs):
    button.args[1]['transition']['easing'] = custom_kwargs['transition']
    if custom_kwargs['frame_rate'] is not None:
        button.args[1]['transition']['duration'] = custom_kwargs['frame_rate']
        button.args[1]['frame']['duration'] = custom_kwargs['frame_rate']


def update_hovertemplate(data, string):
    if data.hovertemplate is not None:
        data.hovertemplate = data.hovertemplate[:-
                                                15]+"<br>"+string+"<extra></extra>"


def add_button(layout, button, pos=None):
    if pos is None:
        num_prev_buttons = len(layout.updatemenus)
        button['y'] = 1 - num_prev_buttons * 0.15
    else:
        button['y'] = pos
    if 'x' not in button:
        button['x'] = -0.1
    layout.updatemenus = layout.updatemenus + (button,)


def add_axis_toggle(layout, axis, pos=None):
    assert axis in "xy", f"Invalid axis: {axis}"
    is_already_log = layout[f"{axis}axis"].type == 'log'
    toggle_axis = dict(
        type="buttons",
        active=0 if is_already_log else -1,
        buttons=[dict(
            label=f"Log {axis}-axis",
            method="relayout",
            args=[{f"{axis}axis.type": "log"}],
            args2=[{f"{axis}axis.type": "linear"}],
        )]
    )
    add_button(layout, toggle_axis, pos=pos)

## Global Helpers
def update_data(data, custom_kwargs, index):
    if custom_kwargs['hover'] is not None and isinstance(data, go.Heatmap):
        # Assumption -
        hover = custom_kwargs['hover']
        hover_name = custom_kwargs['hover_name']
        hover = to_numpy(hover)
        data.customdata = hover
        update_hovertemplate(data, f"{hover_name}=%{{customdata}}")
    if custom_kwargs['markers']:
        data['mode'] = 'lines+markers'
    if custom_kwargs['line_labels'] is not None:
        data['name'] = custom_kwargs['line_labels'][index]
        data['hovertemplate'] = re.sub(
            f"={index}", f"={data['name']}", data['hovertemplate'])
    return


def update_data_list(data_list, custom_kwargs):
    for c, data in enumerate(data_list):
        update_data(data, custom_kwargs, c)
    return


def update_frame(frame, custom_kwargs, frame_index):
    update_data_list(frame['data'], custom_kwargs)
    return


def update_layout(layout, custom_kwargs, is_animation):
    if custom_kwargs['debug']:
        print(layout, is_animation)
    layout.xaxis.title.text = custom_kwargs['xaxis']
    layout.yaxis.title.text = custom_kwargs['yaxis']
    if custom_kwargs['range_x'] is not None:
        layout.xaxis.range = custom_kwargs['range_x']
    if custom_kwargs['range_y'] is not None:
        layout.yaxis.range = custom_kwargs['range_y']
    if custom_kwargs['log_x']:
        layout.xaxis.type = 'log'
    if custom_kwargs['log_y']:
        layout.yaxis.type = 'log'
    if custom_kwargs['toggle_x']:
        add_axis_toggle(layout, 'x')
    if custom_kwargs['toggle_y']:
        add_axis_toggle(layout, 'y')
    if not custom_kwargs['legend']:
        layout.showlegend = False
    if custom_kwargs['facet_labels']:
        for i, label in enumerate(custom_kwargs['facet_labels']):
            layout.annotations[i]['text'] = label
            if i > 0:
                layout[f"xaxis{i+1}"].title = layout["xaxis"].title

    if is_animation:
        for updatemenu in layout.updatemenus:
            if "buttons" in updatemenu:
                for button in updatemenu['buttons']:
                    if button.label == "&#9654;":
                        update_play_button(button, custom_kwargs)
            if button.label == "&#9654;":

                button.transition.easing = custom_kwargs['transition']
                button.transition.easing = custom_kwargs['transition']
        layout.sliders[0].currentvalue.prefix = custom_kwargs['animation_name']+"="
        if custom_kwargs['animation_index'] is not None:
            steps = layout.sliders[0].steps
            for c, step in enumerate(steps):
                step.label = custom_kwargs['animation_index'][c]


def update_fig(fig, custom_kwargs, inplace=True):
    if custom_kwargs['debug']:
        print(fig.frames == tuple())
    if not inplace:
        fig = copy.deepcopy(fig)
    update_data_list(fig['data'], custom_kwargs)
    is_animation = 'frames' in fig and fig.frames != tuple()
    if is_animation:
        for frame_index, frame in enumerate(fig['frames']):
            update_frame(frame, custom_kwargs, frame_index)
    update_layout(fig.layout, custom_kwargs, is_animation)
    return fig

# Plotting Functions
def line_or_scatter(tensor, plot_type, x=None, mode='multi', squeeze=True, **kwargs):
    custom_kwargs, plotly_kwargs = split_kwargs(kwargs)
    array = to_numpy(tensor)
    animation_name = custom_kwargs['animation_name']
    xaxis = custom_kwargs['xaxis']
    yaxis = custom_kwargs['yaxis']
    color_name = custom_kwargs['color_name']
    color = custom_kwargs['color']
    if custom_kwargs['debug']:
        print(color, color_name)
    if squeeze:
        array = array.squeeze()

    if animation_name:
        mode = "animate"

    df = melt(array)

    if plot_type == 'line':
        if len(df.columns) == 2:
            _x_name = '0'
            _color_name = None
            _animation_name = None
        elif len(df.columns) == 3:
            _x_name = '1'
            if mode == 'multi':
                _color_name = '0'
                _animation_name = None
            elif mode == 'animate':
                _color_name = None
                _animation_name = '0'
        elif len(df.columns) == 4:
            _x_name = '2'
            _color_name = '1'
            _animation_name = '0'
        else:
            raise ValueError(
                f"Input tensor has too many dimensions: {array.shape}")

    else:
        if len(df.columns) == 2:
            _x_name = '0'
            _color_name = None
            _animation_name = None
        elif len(df.columns) == 3:
            _x_name = '1'
            _color_name = None
            _animation_name = '0'
        else:
            raise ValueError(
                f"Input tensor has too many dimensions: {array.shape}")

        if color is not None:
            _color_name = color_name
            color = to_numpy(color)
            color = broadcast_up(color, array.shape)
            df[_color_name] = color.flatten()
    if x is not None:
        x = to_numpy(x)
        x = broadcast_up(x, array.shape)
        df[_x_name] = x.flatten()
    if custom_kwargs['debug']:
        display(df)

    if custom_kwargs['hover'] is not None:
        hover_data = to_numpy(custom_kwargs['hover'])
        df[custom_kwargs['hover_name']] = broadcast_up(hover_data, array.shape)
        hover_names = [custom_kwargs['hover_name']]
    else:
        hover_names = []

    if custom_kwargs['debug']:
        display(df)

    if plot_type == 'line':
        plot_fn = px.line
    elif plot_type == 'scatter':
        plot_fn = px.scatter

    fig = plot_fn(
        df,
        x=_x_name,
        y='value',
        color=_color_name,
        animation_frame=_animation_name,
        hover_data=hover_names,
        labels={_x_name: xaxis, 'value': yaxis, _color_name: color_name, _animation_name: animation_name}, **plotly_kwargs)
    update_fig(fig, custom_kwargs)

    if custom_kwargs['return_fig']:
        return fig
    else:
        fig.show()


scatter = partial(line_or_scatter, plot_type='scatter')
line = partial(line_or_scatter, plot_type='line')

def imshow_base(array, **kwargs):
    custom_kwargs, plotly_kwargs = split_kwargs(kwargs)
    fig = px.imshow(to_numpy(array), **plotly_kwargs)
    update_fig(fig, custom_kwargs)
    if custom_kwargs['return_fig']:
        return fig
    else:
        fig.show()


imshow = partial(imshow_base, color_continuous_scale='RdBu', color_continuous_midpoint=0.0, aspect='auto')
imshow_pos = partial(imshow_base, color_continuous_scale='Blues', aspect='auto')
inputs_heatmap = partial(imshow, xaxis='Context', yaxis='Current',
                         color_continuous_scale='RdBu', color_continuous_midpoint=0.0)

legend_in_plot_dict = dict(
    xanchor='right',
    x=0.95,
    title='',
    orientation='h',
    y=1.,
    yanchor='top',
    bgcolor='rgba(255, 255, 255, 0.3)',
)


def put_legend_in_plot(fig):
    fig.update_layout(legend=legend_in_plot_dict)


def histogram(array, **kwargs):
    custom_kwargs, plotly_kwargs = split_kwargs(kwargs)
    array = to_numpy(array)
    df = melt(array)
    fig = px.histogram(df, x='value', **plotly_kwargs)
    update_fig(fig, custom_kwargs)
    if custom_kwargs['return_fig']:
        return fig
    else:
        fig.show()

def unflatten_first(tensor, p):
    if tensor.shape[0]==p*p:
        return einops.rearrange(tensor, '(x y) ... -> x y ...', x=p, y=p)
    else: 
        return tensor

def lines(lines_list, x=None, mode='lines', labels=None, xaxis='', yaxis='', title = '', log_y=False, hover=None, **kwargs):
    '''Helper function to plot multiple lines'''
    if type(lines_list)==torch.Tensor:
        lines_list = [lines_list[i] for i in range(lines_list.shape[0])]
    if x is None:
        x=np.arange(len(lines_list[0]))
    fig = go.Figure(layout={'title':title})
    fig.update_xaxes(title=xaxis)
    fig.update_yaxes(title=yaxis)
    for c, line in enumerate(lines_list):
        if type(line)==torch.Tensor:
            line = to_numpy(line)
        if labels is not None:
            label = labels[c]
        else:
            label = c
        fig.add_trace(go.Scatter(x=x, y=line, mode=mode, name=label, hovertext=hover, **kwargs))
    if log_y:
        fig.update_layout(yaxis_type="log")
    fig.show()

def line_marker(x, **kwargs):
    lines([x], mode='lines+markers', **kwargs)


def animate_lines(lines_list, snapshot_index = None, snapshot='snapshot', hover=None, xaxis='x', yaxis='y', **kwargs):
    if type(lines_list)==list:
        lines_list = torch.stack(lines_list, axis=0)
    lines_list = to_numpy(lines_list, flat=False)
    if snapshot_index is None:
        snapshot_index = np.arange(lines_list.shape[0])
    if hover is not None:
        hover = [i for j in range(len(snapshot_index)) for i in hover]
    print(lines_list.shape)
    rows=[]
    for i in range(lines_list.shape[0]):
        for j in range(lines_list.shape[1]):
            rows.append([lines_list[i][j], snapshot_index[i], j])
    df = pd.DataFrame(rows, columns=[yaxis, snapshot, xaxis])
    px.line(df, x=xaxis, y=yaxis, animation_frame=snapshot, range_y=[lines_list.min(), lines_list.max()], hover_name=hover,**kwargs).show()


def imshow_fourier(tensor, p, fourier_basis_names, title='', animation_name='snapshot', facet_labels=[], **kwargs):
    # Set nice defaults for plotting functions in the 2D fourier basis
    # tensor is assumed to already be in the Fourier Basis
    if tensor.shape[0]==p*p:
        tensor = unflatten_first(tensor, p)
    tensor = torch.squeeze(tensor)
    fig=px.imshow(to_numpy(tensor),
            x=fourier_basis_names, 
            y=fourier_basis_names, 
            labels={'x':'x Component', 
                    'y':'y Component', 
                    'animation_frame':animation_name},
            title=title,
            color_continuous_midpoint=0., 
            color_continuous_scale='RdBu', 
            **kwargs)
    fig.update(data=[{'hovertemplate':"%{x}x * %{y}y<br>Value:%{z:.4f}"}])
    if facet_labels:
        for i, label in enumerate(facet_labels):
            fig.layout.annotations[i]['text'] = label
    fig.show()

def animate_multi_lines(lines_list, y_index=None, snapshot_index = None, snapshot='snapshot', hover=None, swap_y_animate=False, **kwargs):
    # Can plot an animation of lines with multiple lines on the plot.
    if type(lines_list)==list:
        lines_list = torch.stack(lines_list, axis=0)
    lines_list = to_numpy(lines_list, flat=False)
    if swap_y_animate:
        lines_list = lines_list.transpose(1, 0, 2)
    if snapshot_index is None:
        snapshot_index = np.arange(lines_list.shape[0])
    if y_index is None:
        y_index = [str(i) for i in range(lines_list.shape[1])]
    if hover is not None:
        hover = [i for j in range(len(snapshot_index)) for i in hover]
    print(lines_list.shape)
    rows=[]
    for i in range(lines_list.shape[0]):
        for j in range(lines_list.shape[2]):
            rows.append(list(lines_list[i, :, j])+[snapshot_index[i], j])
    df = pd.DataFrame(rows, columns=y_index+[snapshot, 'x'])
    px.line(df, x='x', y=y_index, animation_frame=snapshot, range_y=[lines_list.min(), lines_list.max()], hover_name=hover, **kwargs).show()

def animate_scatter(lines_list, snapshot_index = None, snapshot='snapshot', hover=None, yaxis='y', xaxis='x', color=None, color_name = 'color', **kwargs):
    # Can plot an animated scatter plot
    # lines_list has shape snapshot x 2 x line
    if type(lines_list)==list:
        lines_list = torch.stack(lines_list, axis=0)
    lines_list = to_numpy(lines_list, flat=False)
    if snapshot_index is None:
        snapshot_index = np.arange(lines_list.shape[0])
    if hover is not None:
        hover = [i for j in range(len(snapshot_index)) for i in hover]
    if color is None:
        color = np.ones(lines_list.shape[-1])
    if type(color)==torch.Tensor:
        color = to_numpy(color)
    if len(color.shape)==1:
        color = einops.repeat(color, 'x -> snapshot x', snapshot=lines_list.shape[0])
    print(lines_list.shape)
    rows=[]
    for i in range(lines_list.shape[0]):
        for j in range(lines_list.shape[2]):
            rows.append([lines_list[i, 0, j].item(), lines_list[i, 1, j].item(), snapshot_index[i], color[i, j]])
    print([lines_list[:, 0].min(), lines_list[:, 0].max()])
    print([lines_list[:, 1].min(), lines_list[:, 1].max()])
    df = pd.DataFrame(rows, columns=[xaxis, yaxis, snapshot, color_name])
    px.scatter(df, x=xaxis, y=yaxis, animation_frame=snapshot, range_x=[lines_list[:, 0].min(), lines_list[:, 0].max()], range_y=[lines_list[:, 1].min(), lines_list[:, 1].max()], hover_name=hover, color=color_name, **kwargs).show()

