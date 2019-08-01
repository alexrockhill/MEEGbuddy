""" MEEGbuddy Alex Rockhill aprockhill206@gmail.com """

__version__ = '0.0.dev0'

from .MEEGbuddy import (MEEGbuddy, create_demi_events, loadMEEGbuddy,
						loadMEEGbuddies, getMEEGbuddiesBySubject,
						BIDS2MEEGbuddies, recon_subject, 
						setup_source_space)
from .MBComparator import MBComparator
from . import pci
from .psd_multitaper_plot_tools import (DraggableResizeableRectangle,
                                        ButtonClickProcessor)
from . import gif_combine #import combine_gifs
from . import bv2fif