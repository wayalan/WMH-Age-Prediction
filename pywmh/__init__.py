"""
pywmh - A Lightweight, MATLAB-Free Pipeline for WMH Segmentation & Age Estimation

Modules:
    lst_lga    : Pure Python implementation of LST-LGA (Lesion Growth Algorithm)
    fsl_engine : FSL automation for coregistration, skull-stripping, FAST, FNIRT
    age_model  : PVWMH / DWMH atlas parcellation & WMH-predicted age estimation
"""

from .lst_lga import segment_wmh_lga
from .age_model import calculate_wmh_volumes_and_age
from .fsl_engine import check_fsl_installed

__version__ = "1.0.0"
