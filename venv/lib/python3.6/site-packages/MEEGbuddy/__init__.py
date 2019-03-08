""" MEEG Resources from MGH Division of Neurotherapeutics """

__version__ = '0.0.dev0'

from .MEEGbuddy import MEEGbuddy,create_demi_events
from .MBComparator import MBComparator
from . import pci
from .psd_multitaper_plot_tools import DraggableResizeableRectangle,ButtonClickProcessor
from . import gif_combine #import combine_gifs