"""
pywmh - A Lightweight, MATLAB-Free Pipeline for WMH Segmentation & Age Estimation

Modules:
    lst_lga    : Pure Python implementation of LST-LGA (Lesion Growth Algorithm)
    fsl_engine : FSL automation for coregistration, skull-stripping, FAST, FNIRT
    age_model  : PVWMH / DWMH atlas parcellation & WMH-predicted age estimation
"""

def __getattr__(name):
    # Keep independent age correction usable without imaging dependencies.
    from importlib import import_module
    modules = {'segment_wmh_lga': 'lst_lga',
               'calculate_wmh_volumes_and_age': 'age_model',
               'check_fsl_installed': 'fsl_engine'}
    if name not in modules:
        raise AttributeError(name)
    return getattr(import_module('.' + modules[name], __name__), name)

__version__ = "1.0.0"
