function result = CorrectWMHAge(ChronologicalAge, varargin)
% Independent linear-reference-v1 post-processing; supply raw age OR raw PAD.
% Example: r = CorrectWMHAge(65, 'raw_wmh_age', 70)
p = inputParser;
validScalar = @(x) isnumeric(x) && isscalar(x) && isreal(x) && isfinite(x);
addRequired(p, 'ChronologicalAge', @(x) validScalar(x) && x > 0);
addParameter(p, 'raw_wmh_age', [], @(x) isempty(x) || validScalar(x));
addParameter(p, 'raw_pad', [], @(x) isempty(x) || validScalar(x));
parse(p, ChronologicalAge, varargin{:});
assert(xor(isempty(p.Results.raw_wmh_age), isempty(p.Results.raw_pad)), ...
    'Provide exactly one of raw_wmh_age or raw_pad.');
raw = p.Results.raw_wmh_age;
if isempty(raw), raw = ChronologicalAge + p.Results.raw_pad; end
result.Chronological_Age = ChronologicalAge;
result.Raw_WMH_Age = raw;
result.Raw_PAD = raw - ChronologicalAge;
result.Corrected_PAD = result.Raw_PAD - (17.52549262674548 - 0.30849983973570466 * ChronologicalAge);
result.Corrected_WMH_Age = ChronologicalAge + result.Corrected_PAD;
result.Age_Correction_Method = 'linear-reference-v1';
end
