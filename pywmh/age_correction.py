"""Optional post-processing of existing WMH age estimates; no volume changes."""
import argparse
import json
import math

METHOD = 'linear-reference-v1'
INTERCEPT = 17.52549262674548
AGE_SLOPE = -0.30849983973570466


def correct_age(chronological_age, *, raw_wmh_age=None, raw_pad=None, method=METHOD):
    """Supply chronological age and exactly one raw age or PAD, in years."""
    if method != METHOD:
        raise ValueError(f'Unknown correction method: {method}')
    if (raw_wmh_age is None) == (raw_pad is None):
        raise ValueError('Provide exactly one of raw_wmh_age or raw_pad.')
    age = float(chronological_age)
    raw = float(raw_wmh_age if raw_wmh_age is not None else age + float(raw_pad))
    if not math.isfinite(age) or age <= 0 or not math.isfinite(raw):
        raise ValueError('Chronological age must be finite and positive; raw input must be finite.')
    gap = raw - age
    corrected = gap - (INTERCEPT + AGE_SLOPE * age)
    return dict(Chronological_Age=age, Raw_WMH_Age=raw, Raw_PAD=gap,
                Corrected_PAD=corrected, Corrected_WMH_Age=age + corrected,
                Age_Correction_Method=method)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--age', type=float, required=True)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--raw-wmh-age', type=float)
    group.add_argument('--raw-pad', type=float)
    parser.add_argument('--out', help='Optional output JSON file')
    args = parser.parse_args()
    try:
        result = correct_age(args.age, raw_wmh_age=args.raw_wmh_age, raw_pad=args.raw_pad)
    except (ValueError, TypeError) as exc:
        parser.error(str(exc))
    content = json.dumps(result, indent=2, allow_nan=False)
    if args.out:
        with open(args.out, 'w') as stream:
            stream.write(content + '\n')
    print(content)


if __name__ == '__main__':
    main()
