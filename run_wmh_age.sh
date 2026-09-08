#!/usr/bin/env bash
# ==============================================================================
# run_wmh_age.sh - Headless (No GUI) CLI Runner for AutomatedWMHAge
#
# Usage:
#   ./run_wmh_age.sh <SourceFolder> [kappa] [voxel_size] [outdir]
#
# Examples:
#   ./run_wmh_age.sh /path/to/subject_dicom_or_nii
#   ./run_wmh_age.sh /path/to/subject_dicom_or_nii 0.3 1.5 /path/to/output
# ==============================================================================

set -euo pipefail

if [ "$#" -lt 1 ]; then
    echo "Usage: $0 <SourceFolder> [kappa=0.3] [voxel_size=1.5] [outdir]"
    exit 1
fi

SRC_DIR="$1"
KAPPA="${2:-0.3}"
VX_SIZE="${3:-1.5}"
OUT_DIR="${4:-}"

# Check for MATLAB binary
if command -v matlab &> /dev/null; then
    MATLAB_BIN="matlab"
elif [ -f "/Applications/MATLAB_R2022a.app/bin/matlab" ]; then
    MATLAB_BIN="/Applications/MATLAB_R2022a.app/bin/matlab"
else
    # Search common Mac locations
    MATLAB_CANDIDATE=$(ls -d /Applications/MATLAB_*.app/bin/matlab 2>/dev/null | tail -n 1 || true)
    if [ -n "$MATLAB_CANDIDATE" ] && [ -x "$MATLAB_CANDIDATE" ]; then
        MATLAB_BIN="$MATLAB_CANDIDATE"
    else
        echo "Error: MATLAB executable not found in PATH or /Applications." >&2
        exit 1
    fi
fi

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

echo "=================================================="
echo " Starting WMH-Age Pipeline (Headless Mode)        "
echo " MATLAB Binary : $MATLAB_BIN"
echo " Source Dir    : $SRC_DIR"
echo " Kappa         : $KAPPA"
echo " Voxel Size    : $VX_SIZE mm"
if [ -n "$OUT_DIR" ]; then
    echo " Output Dir    : $OUT_DIR"
fi
echo "=================================================="

if [ -n "$OUT_DIR" ]; then
    MATLAB_CMD="addpath('$SCRIPT_DIR'); try, [vol, age] = AutomatedWMHAge('$SRC_DIR', $KAPPA, $VX_SIZE, '$OUT_DIR'); catch ME, disp(getReport(ME)); exit(1); end; exit(0);"
else
    MATLAB_CMD="addpath('$SCRIPT_DIR'); try, [vol, age] = AutomatedWMHAge('$SRC_DIR', $KAPPA, $VX_SIZE); catch ME, disp(getReport(ME)); exit(1); end; exit(0);"
fi

# Run MATLAB with no display, no splash, no desktop, and no figure windows
"$MATLAB_BIN" -nodisplay -nosplash -nodesktop -r "$MATLAB_CMD"
