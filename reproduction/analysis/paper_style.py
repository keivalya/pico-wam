"""Figure typography at final TMLR print size (6.5-inch text width)."""
import matplotlib as mpl
from matplotlib.text import Text
WIDTH = 6.5
BLUE, ORANGE, TEAL, GRAY = '#0072B2', '#D55E00', '#168079', '#666666'
def use_style():
    mpl.rcParams.update({'font.family':'STIXGeneral','mathtext.fontset':'stix',
        'font.size':9, 'axes.labelsize':9, 'axes.titlesize':9,
        'xtick.labelsize':8.5, 'ytick.labelsize':8.5, 'legend.fontsize':8.5,
        'pdf.fonttype':42,'ps.fonttype':42,'axes.spines.top':False,
        'axes.spines.right':False,'axes.linewidth':.7,'lines.linewidth':1.2,
        'lines.markersize':4,'axes.grid':False,'savefig.dpi':300,
        'figure.constrained_layout.h_pad':.07,'figure.constrained_layout.w_pad':.07,
        'axes.titlepad':9,'axes.labelpad':4,'legend.frameon':False})
def save(fig,path):
    """Keep physical dimensions fixed and reject clipped visible text."""
    fig.canvas.draw(); renderer=fig.canvas.get_renderer(); bounds=fig.bbox
    outside_ticks=set()
    for ax in fig.axes:
        for axis in [ax.xaxis,ax.yaxis]:
            low,high=sorted(axis.get_view_interval())
            for tick in axis.get_major_ticks()+axis.get_minor_ticks():
                if not low <= tick.get_loc() <= high:
                    outside_ticks.update([tick.label1,tick.label2])
    for text in fig.findobj(Text):
        if text in outside_ticks:continue
        if not text.get_visible() or not text.get_text().strip():continue
        # Hidden tick labels may still be returned by an artist traversal.
        if text.axes is not None and not text.axes.get_visible():continue
        box=text.get_window_extent(renderer)
        if box.width and box.height and (box.x0 < -1 or box.y0 < -1 or box.x1 > bounds.x1+1 or box.y1 > bounds.y1+1):
            raise ValueError(f'Text outside figure: {text.get_text()!r}')
    fig.savefig(path)
