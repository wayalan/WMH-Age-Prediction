function [WMHvolume, PredAge] = AutomatedWMHAge(SourceFolder, varargin)
%% A Linear Regression Model Using Periventricular WMH and Deep WMH to Predict Individual's Age
% This is a toolbox for PVWMH DWMH segmentation and WMH-Age prediction.
% This tool requires SPM12 with CAT12 and LST toolboxes installed.
%
% Usage:
%   [WMHvolume, PredAge] = AutomatedWMHAge(SourceFolder)
%   [WMHvolume, PredAge] = AutomatedWMHAge(SourceFolder, 'kappa', 0.3, 'voxel_size', 1.5, 'outdir', '/path/to/out')
%
% Inputs:
%   SourceFolder : Path containing T1 and FLAIR data.
%                  Supports:
%                  1) Folder with subdirectories containing "T1" and "T2"/"FLAIR" DICOMs.
%                  2) Folder containing 'T1w.nii' and 'T2f.nii' directly.
%
% Optional Parameters (Name-Value or Positional):
%   'kappa'      : Threshold for LST LGA lesion segmentation [default = 0.3]
%   'voxel_size' : Voxel size in MNI space (1.5 or 1.0) [default = 1.5]
%   'outdir'     : Directory for output files [default = fullfile(SourceFolder, 'wmh_age_output')]
%   'nproc'      : Number of parallel processes for CAT12 [default = min(4, feature('numcores'))]
%   'force'      : Force re-run completed steps [default = false]
%
% Reference:
%   Huang CC, Chou KH, Lee WJ, Yang AC, Tsai SJ, Chen LK, Chung CP, Lin CP.
%   Brain white matter hyperintensities-predicted age reflects neurovascular
%   health in middle-to-old aged subjects. Age Ageing. 2022 May 1;51(5):afac106.
%
% Chu-Chung Huang 2022 / Updated 2026
%%

fprintf('=======================================================\n');
fprintf('       Automated WMH-Age Estimation Pipeline           \n');
fprintf('=======================================================\n');

%% 1. Parameter Parsing
p = inputParser;
addRequired(p, 'SourceFolder', @(x) ischar(x) || isstring(x));
addParameter(p, 'kappa', 0.3, @(x) isnumeric(x) && x > 0 && x < 1);
addParameter(p, 'voxel_size', 1.5, @(x) isnumeric(x) && ismember(x, [1.0, 1.5]));
addParameter(p, 'outdir', '', @(x) ischar(x) || isstring(x));
addParameter(p, 'nproc', min(4, feature('numcores')), @isnumeric);
addParameter(p, 'force', false, @islogical);

% Support legacy positional arguments: AutomatedWMHAge(SourceFolder, kappa, vx_size, outdir)
if ~isempty(varargin) && isnumeric(varargin{1})
    if length(varargin) >= 1, p.addOptional('pos_kappa', varargin{1}); end
    if length(varargin) >= 2, p.addOptional('pos_vx', varargin{2}); end
    if length(varargin) >= 3, p.addOptional('pos_out', varargin{3}); end
    parse(p, SourceFolder, varargin{:});
    kappa_value = p.Results.pos_kappa;
    if isfield(p.Results, 'pos_vx'), vx_size = p.Results.pos_vx; else, vx_size = 1.5; end
    if isfield(p.Results, 'pos_out'), OutDir = char(p.Results.pos_out); else, OutDir = ''; end
    nproc = min(4, feature('numcores'));
    force_rerun = false;
else
    parse(p, SourceFolder, varargin{:});
    kappa_value = p.Results.kappa;
    vx_size = p.Results.voxel_size;
    OutDir = char(p.Results.outdir);
    nproc = p.Results.nproc;
    force_rerun = p.Results.force;
end

SourceFolder = char(SourceFolder);
if ~isfolder(SourceFolder)
    error('Source folder does not exist: %s', SourceFolder);
end

if isempty(OutDir)
    OutDir = fullfile(SourceFolder, 'wmh_age_output');
end
if ~isfolder(OutDir)
    mkdir(OutDir);
end

fprintf('Source Folder : %s\n', SourceFolder);
fprintf('Output Folder : %s\n', OutDir);
fprintf('LST Kappa     : %.2f\n', kappa_value);
fprintf('MNI Resolution: %.1f mm\n', vx_size);
fprintf('CPU Workers   : %d\n', nproc);

%% 2. Check SPM & Toolboxes
spm_path = which('spm');
if isempty(spm_path)
    error('SPM12 is not found in MATLAB path. Please add SPM12 to your path.');
end
[spm_hpath, ~, ~] = fileparts(spm_path);

cat_path = which('cat12');
if isempty(cat_path)
    error('CAT12 toolbox is not found in SPM. Please install CAT12 in spm12/toolbox/cat12.');
end
[cat_hpath, ~, ~] = fileparts(cat_path);

% Suppress GUI popup windows for SPM batch runs
spm('defaults', 'FMRI');
spm_get_defaults('cmdline', true);
spm_jobman('initcfg');

%% 3. Image Preparation (DICOM or direct NIfTI)
T1w_target = fullfile(OutDir, 'T1w.nii');
T2f_target = fullfile(OutDir, 'T2f.nii');

if exist(T1w_target, 'file') == 0 || exist(T2f_target, 'file') == 0 || force_rerun
    % Check if NIfTI files already exist in SourceFolder
    src_t1 = dir(fullfile(SourceFolder, '*T1*.nii*'));
    src_t2 = dir(fullfile(SourceFolder, '*T2*.nii*'));
    if isempty(src_t2)
        src_t2 = dir(fullfile(SourceFolder, '*FLAIR*.nii*'));
    end

    if ~isempty(src_t1) && ~isempty(src_t2)
        fprintf('Found existing NIfTI files in source folder. Preparing copies...\n');
        copy_or_unzip_nii(fullfile(src_t1(1).folder, src_t1(1).name), T1w_target);
        copy_or_unzip_nii(fullfile(src_t2(1).folder, src_t2(1).name), T2f_target);
    else
        % Convert from DICOM
        fprintf('Converting DICOM to NIfTI...\n');
        
        % Locate T1 DICOM folder
        d_all = dir(SourceFolder);
        d_dirs = d_all([d_all.isdir] & ~ismember({d_all.name}, {'.', '..'}));
        
        t1_dir_idx = find(contains({d_dirs.name}, 'T1', 'IgnoreCase', true), 1);
        t2_dir_idx = find(contains({d_dirs.name}, 'T2', 'IgnoreCase', true) | ...
                          contains({d_dirs.name}, 'FLAIR', 'IgnoreCase', true), 1);
        
        if isempty(t1_dir_idx) || isempty(t2_dir_idx)
            error('Could not automatically identify T1 and T2/FLAIR folders in %s. Please ensure folder names contain "T1" and "T2" or "FLAIR".', SourceFolder);
        end
        
        t1_dcm_dir = fullfile(SourceFolder, d_dirs(t1_dir_idx).name);
        t2_dcm_dir = fullfile(SourceFolder, d_dirs(t2_dir_idx).name);
        
        t1_files = dir(fullfile(t1_dcm_dir, '*'));
        t1_files = t1_files(~[t1_files.isdir]);
        t1_dcm_paths = fullfile({t1_files.folder}', {t1_files.name}');
        
        t2_files = dir(fullfile(t2_dcm_dir, '*'));
        t2_files = t2_files(~[t2_files.isdir]);
        t2_dcm_paths = fullfile({t2_files.folder}', {t2_files.name}');
        
        % Convert T1
        t1_temp_dir = fullfile(OutDir, 'dcm_t1_tmp');
        if ~isfolder(t1_temp_dir), mkdir(t1_temp_dir); end
        clear matlabbatch;
        matlabbatch{1}.spm.util.import.dicom.data = t1_dcm_paths;
        matlabbatch{1}.spm.util.import.dicom.root = 'flat';
        matlabbatch{1}.spm.util.import.dicom.outdir = {t1_temp_dir};
        matlabbatch{1}.spm.util.import.dicom.protfilter = '.*';
        matlabbatch{1}.spm.util.import.dicom.convopts.format = 'nii';
        matlabbatch{1}.spm.util.import.dicom.convopts.meta = 0;
        matlabbatch{1}.spm.util.import.dicom.convopts.icedims = 0;
        spm_jobman('run', matlabbatch);
        
        t1_converted = dir(fullfile(t1_temp_dir, '*.nii'));
        if isempty(t1_converted)
            error('Failed to convert T1 DICOM to NIfTI.');
        end
        movefile(fullfile(t1_temp_dir, t1_converted(1).name), T1w_target);
        rmdir(t1_temp_dir, 's');
        
        % Convert T2
        t2_temp_dir = fullfile(OutDir, 'dcm_t2_tmp');
        if ~isfolder(t2_temp_dir), mkdir(t2_temp_dir); end
        clear matlabbatch;
        matlabbatch{1}.spm.util.import.dicom.data = t2_dcm_paths;
        matlabbatch{1}.spm.util.import.dicom.root = 'flat';
        matlabbatch{1}.spm.util.import.dicom.outdir = {t2_temp_dir};
        matlabbatch{1}.spm.util.import.dicom.protfilter = '.*';
        matlabbatch{1}.spm.util.import.dicom.convopts.format = 'nii';
        matlabbatch{1}.spm.util.import.dicom.convopts.meta = 0;
        matlabbatch{1}.spm.util.import.dicom.convopts.icedims = 0;
        spm_jobman('run', matlabbatch);
        
        t2_converted = dir(fullfile(t2_temp_dir, '*.nii'));
        if isempty(t2_converted)
            error('Failed to convert T2/FLAIR DICOM to NIfTI.');
        end
        movefile(fullfile(t2_temp_dir, t2_converted(1).name), T2f_target);
        rmdir(t2_temp_dir, 's');
    end
else
    fprintf('Step 1: T1w.nii and T2f.nii already exist in output directory. Skipping DICOM conversion.\n');
end

%% 4. WMH Segmentation (SPM LST LGA)
wmh_pattern = fullfile(OutDir, sprintf('ples_lga_*.nii'));
wmh_files = dir(wmh_pattern);

if isempty(wmh_files) || force_rerun
    fprintf('Step 2: Running LST LGA WMH segmentation (kappa = %.2f)...\n', kappa_value);
    clear matlabbatch;
    matlabbatch{1}.spm.tools.LST.lga.data_T1 = {[T1w_target, ',1']};
    matlabbatch{1}.spm.tools.LST.lga.data_F2 = {[T2f_target, ',1']};
    matlabbatch{1}.spm.tools.LST.lga.opts_lga.initial = kappa_value;
    matlabbatch{1}.spm.tools.LST.lga.opts_lga.mrf = 1;
    matlabbatch{1}.spm.tools.LST.lga.opts_lga.maxiter = 50;
    matlabbatch{1}.spm.tools.LST.lga.html_report = 0; % Headless: do not pop up browser
    
    spm_jobman('run', matlabbatch);
    clear matlabbatch;
    
    wmh_files = dir(wmh_pattern);
    if isempty(wmh_files)
        error('LST segmentation did not generate lesion probability map: %s', wmh_pattern);
    end
else
    fprintf('Step 2: LST WMH segmentation already exists (%s). Skipping.\n', wmh_files(1).name);
end
WMHSeg_file = fullfile(wmh_files(1).folder, wmh_files(1).name);

%% 5. T1 Segmentation & Normalization (CAT12)
def_field = fullfile(OutDir, 'mri', 'y_T1w.nii');

if exist(def_field, 'file') == 0 || force_rerun
    fprintf('Step 3: Running CAT12 tissue segmentation and deformation field estimation...\n');
    
    % Locate template
    dartel_tpm = fullfile(cat_hpath, 'templates_1.50mm', 'Template_1_IXI555_MNI152.nii');
    if ~exist(dartel_tpm, 'file')
        % Fallback for newer CAT12 versions (e.g. CAT12.8 / CAT12.9)
        tpm_search = dir(fullfile(cat_hpath, 'templates_*', 'Template_1_*.nii'));
        if ~isempty(tpm_search)
            dartel_tpm = fullfile(tpm_search(1).folder, tpm_search(1).name);
        else
            dartel_tpm = fullfile(cat_hpath, 'templates_MNI152NLin2009cAsym', 'Template_1_Dartel.nii');
        end
    end
    
    clear matlabbatch;
    matlabbatch{1}.spm.tools.cat.estwrite.data = {[T1w_target, ',1']};
    matlabbatch{1}.spm.tools.cat.estwrite.nproc = nproc;
    matlabbatch{1}.spm.tools.cat.estwrite.opts.affreg = 'mni';
    matlabbatch{1}.spm.tools.cat.estwrite.opts.biasstr = 0.5;
    matlabbatch{1}.spm.tools.cat.estwrite.extopts.registration.vox = vx_size;
    matlabbatch{1}.spm.tools.cat.estwrite.extopts.registration.bb = 12;
    matlabbatch{1}.spm.tools.cat.estwrite.extopts.segmentation.APP = 1070;
    matlabbatch{1}.spm.tools.cat.estwrite.extopts.segmentation.LASstr = 0.5;
    matlabbatch{1}.spm.tools.cat.estwrite.extopts.segmentation.gcutstr = 2;
    matlabbatch{1}.spm.tools.cat.estwrite.extopts.segmentation.cleanupstr = 0.5;
    matlabbatch{1}.spm.tools.cat.estwrite.extopts.segmentation.WMHC = 1;
    matlabbatch{1}.spm.tools.cat.estwrite.extopts.admin.ignoreErrors = 0;
    matlabbatch{1}.spm.tools.cat.estwrite.extopts.admin.verb = 1;
    matlabbatch{1}.spm.tools.cat.estwrite.extopts.admin.print = 0;
    matlabbatch{1}.spm.tools.cat.estwrite.output.surface = 0;
    matlabbatch{1}.spm.tools.cat.estwrite.output.ROImenu.noROI = struct([]);
    matlabbatch{1}.spm.tools.cat.estwrite.output.GM.native = 1;
    matlabbatch{1}.spm.tools.cat.estwrite.output.GM.mod = 0;
    matlabbatch{1}.spm.tools.cat.estwrite.output.WM.native = 1;
    matlabbatch{1}.spm.tools.cat.estwrite.output.WM.mod = 0;
    matlabbatch{1}.spm.tools.cat.estwrite.output.CSF.native = 1;
    matlabbatch{1}.spm.tools.cat.estwrite.output.warps = [1 0];
    
    spm_jobman('run', matlabbatch);
    clear matlabbatch;
else
    fprintf('Step 3: CAT12 deformation field already exists (%s). Skipping.\n', def_field);
end

%% 6. Apply Deformation (Warp & Modulate Lesion Map to MNI Space)
mWMH_files = dir(fullfile(OutDir, 'm0wples*.nii'));

% FIX: Robust existence check avoiding empty struct index crash
if isempty(mWMH_files) || force_rerun
    fprintf('Step 4: Applying CAT12 deformation to WMH lesion map (Native -> MNI with Modulation)...\n');
    clear matlabbatch;
    matlabbatch{1}.spm.tools.cat.tools.defs.field1 = {[def_field, ',1']};
    matlabbatch{1}.spm.tools.cat.tools.defs.images = {[WMHSeg_file, ',1']};
    matlabbatch{1}.spm.tools.cat.tools.defs.interp = 1;
    matlabbatch{1}.spm.tools.cat.tools.defs.modulate = 2; % Modulate affine + non-linear (volume preserving)
    
    spm_jobman('run', matlabbatch);
    clear matlabbatch;
    
    mWMH_files = dir(fullfile(OutDir, 'm0wples*.nii'));
    if isempty(mWMH_files)
        error('Failed to produce modulated MNI lesion map m0wples*.nii.');
    end
else
    fprintf('Step 4: Modulated MNI lesion map already exists (%s). Skipping.\n', mWMH_files(1).name);
end

mWMH_file = fullfile(mWMH_files(1).folder, mWMH_files(1).name);

%% 7. WMH Volume & Age Estimation
fprintf('Step 5: Quantifying PVWMH / DWMH volumes and predicting age...\n');

script_dir = fileparts(mfilename('fullpath'));
if isempty(script_dir), script_dir = pwd; end
atlaspath = fullfile(script_dir, 'WMH_Atlas', sprintf('%.1fmm', vx_size));
if ~isfolder(atlaspath)
    % Fallback without trailing decimal if needed (e.g. 1mm instead of 1.0mm)
    atlaspath = fullfile(script_dir, 'WMH_Atlas', [num2str(vx_size) 'mm']);
end

pv_atlas_file = fullfile(atlaspath, '1_PVWMH_1-10mm.nii.gz');
dw_atlas_file = fullfile(atlaspath, '2_DWMH_10mmup.nii.gz');

if ~exist(pv_atlas_file, 'file') || ~exist(dw_atlas_file, 'file')
    error('Atlas files not found in: %s', atlaspath);
end

% Load volumes safely without external load_untouch_nii dependency
PV_data  = read_nii_safe(pv_atlas_file);
DW_data  = read_nii_safe(dw_atlas_file);
WMH_data = read_nii_safe(mWMH_file);

voxel_vol_cc = (vx_size^3) / 1000.0; % mm^3 to cc (ml)

WMHvolume.PV = sum(WMH_data(PV_data == 1)) * voxel_vol_cc;
WMHvolume.D  = sum(WMH_data(DW_data == 1)) * voxel_vol_cc;
WMHvolume.Total = sum(WMH_data(WMH_data > 0)) * voxel_vol_cc;

% Numerical floor protection against log10(0) = -Inf
epsilon = 1e-4;
if WMHvolume.PV <= 0
    warning('Periventricular WMH volume is 0 or negative. Clamping to %.1e for log10 calculation.', epsilon);
    pv_val_for_log = epsilon;
else
    pv_val_for_log = WMHvolume.PV;
end

if WMHvolume.D <= 0
    warning('Deep WMH volume is 0 or negative. Clamping to %.1e for log10 calculation.', epsilon);
    d_val_for_log = epsilon;
else
    d_val_for_log = WMHvolume.D;
end

% Paper formula: Huang et al., Age Ageing 2022
PredAge = 11.069 * log10(pv_val_for_log) + 1.624 * log10(d_val_for_log) + 64.159;

fprintf('\n----------------- Results -----------------\n');
fprintf('Total WMH Volume : %.3f cc\n', WMHvolume.Total);
fprintf('PVWMH Volume     : %.3f cc\n', WMHvolume.PV);
fprintf('DWMH Volume      : %.3f cc\n', WMHvolume.D);
fprintf('Predicted WMH Age: %.2f years old\n', PredAge);
fprintf('-------------------------------------------\n\n');

%% 8. Save Structured Results
[~, subj_id] = fileparts(SourceFolder);
res_table = table({subj_id}, WMHvolume.Total, WMHvolume.PV, WMHvolume.D, PredAge, kappa_value, vx_size, ...
    'VariableNames', {'SubjectID', 'Total_WMH_cc', 'PVWMH_cc', 'DWMH_cc', 'Predicted_Age', 'Kappa', 'Resolution_mm'});

csv_path = fullfile(OutDir, 'WMH_Age_Results.csv');
writetable(res_table, csv_path);

mat_path = fullfile(OutDir, 'WMH_Age_Results.mat');
save(mat_path, 'WMHvolume', 'PredAge', 'res_table');

fprintf('Results saved to:\n  %s\n  %s\n', csv_path, mat_path);
fprintf('Processing completed successfully.\n');

end


%% Helper Functions
function copy_or_unzip_nii(src, dst)
    if endsWith(src, '.gz', 'IgnoreCase', true)
        tmp = gunzip(src, tempdir);
        movefile(tmp{1}, dst);
    else
        copyfile(src, dst);
    end
end

function img_data = read_nii_safe(file_path)
    % Reads NIfTI (.nii or .nii.gz) using SPM without external toolbox dependencies
    is_gz = endsWith(file_path, '.gz', 'IgnoreCase', true);
    if is_gz
        extracted = gunzip(file_path, tempdir);
        read_target = extracted{1};
    else
        read_target = file_path;
    end
    
    try
        V = spm_vol(read_target);
        img_data = spm_read_vols(V);
        img_data(isnan(img_data)) = 0;
    catch ME
        % Fallback to MATLAB builtin niftiread if SPM fails
        if exist('niftiread', 'file')
            img_data = double(niftiread(read_target));
            img_data(isnan(img_data)) = 0;
        else
            rethrow(ME);
        end
    end
    
    if is_gz && exist(read_target, 'file')
        delete(read_target);
    end
end