from IPython import get_ipython
from IPython.display import set_matplotlib_formats

import matplotlib
import numpy as np


def configure_plotting():
    # Configure matplotlib
    ip = get_ipython()
    ip.config['InlineBackend']['figure_format'] = 'svg'
    # set_matplotlib_formats('svg')
    
    # Fix for https://stackoverflow.com/a/36021869/2217463
    ip.config['InlineBackend']['rc'] = {}
    ip.enable_matplotlib(gui='inline')

    matplotlib.rcParams['text.usetex'] = False
    matplotlib.rcParams['figure.figsize'] = (10, 7)
    matplotlib.rcParams['font.size'] = 16


from tempfile import TemporaryDirectory
from pathlib import Path

def to_svg_jshtml(anim, fps=None, embed_frames=True, default_mode=None):
    """Generate a JavaScript HTML representation of the animation but using the svg
       file format
    """
        
    if fps is None and hasattr(anim, '_interval'):
        # Convert interval in ms to frames per second
        fps = 1000 / anim._interval

    # If we're not given a default mode, choose one base on the value of
    # the repeat attribute
    if default_mode is None:
        default_mode = 'loop' if anim.repeat else 'once'

    # Can't open a NamedTemporaryFile twice on Windows, so use a
    # TemporaryDirectory instead.
    with TemporaryDirectory() as tmpdir:
        path = Path(tmpdir, "temp.html")
        writer = matplotlib.animation.HTMLWriter(fps=fps,
                                                 embed_frames=embed_frames,
                                                 default_mode=default_mode)
        writer.frame_format = 'svg'
        anim.save(str(path), writer=writer)
        jshtml = path.read_text()
            
        # the mime type of svg is wrong in matplotlib
        jshtml = jshtml.replace("data:image/svg", "data:image/svg+xml")
        
    return jshtml

    
def draw_classic_axes(ax, x=0, y=0, xlabeloffset=.1, ylabeloffset=.07):
    ax.set_axis_off()
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    ax.annotate(
        ax.get_xlabel(), xytext=(x1, y), xy=(x0, y),
        arrowprops=dict(arrowstyle="<-"), va='center'
    )
    ax.annotate(
        ax.get_ylabel(), xytext=(x, y1), xy=(x, y0),
        arrowprops=dict(arrowstyle="<-"), ha='center'
    )
    for pos, label in zip(ax.get_xticks(), ax.get_xticklabels()):
        ax.text(pos, y - xlabeloffset, label.get_text(),
                ha='center', va='bottom')
    for pos, label in zip(ax.get_yticks(), ax.get_yticklabels()):
        ax.text(x - ylabeloffset, pos, label.get_text(),
                ha='right', va='center')
