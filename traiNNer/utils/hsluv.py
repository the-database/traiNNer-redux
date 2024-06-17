"""Port of https://github.com/hsluv/hsluv-python/blob/master/hsluv.py to PyTorch,
modified to support float32 precision.
"""

import torch

_m = [[3.240969941904521, -1.537383177570093, -0.498610760293],
      [-0.96924363628087, 1.87596750150772, 0.041555057407175],
      [0.055630079696993, -0.20397695888897, 1.056971514242878]]

_m_inv = [[0.41239079926595, 0.35758433938387, 0.18048078840183],
          [0.21263900587151, 0.71516867876775, 0.072192315360733],
          [0.019330818715591, 0.11919477979462, 0.95053215224966]]

_ref_y = 1.0
_ref_u = 0.19783000664283
_ref_v = 0.46831999493879
_kappa = 903.2962962
_epsilon = 0.0088564516


def _y_to_l(y):
    return torch.where(y > _epsilon, 116 * torch.pow(y / _ref_y, 1 / 3) - 16, y / _ref_y * _kappa)


def hsluv_to_lch(h, s, l):
    l = torch.clamp(l, 0, 100)
    max_chroma = _max_chroma_for_lh(l, h)
    c = max_chroma * s / 100
    return torch.stack([l, c, h], dim=-1)


def lch_to_hsluv(l, c, h):
    _hx_max = torch.clamp(_max_chroma_for_lh(l, h), 1e-12)
    s = c / _hx_max * 100

    s = torch.where((l > 100 - 1e-5) | (l < 1e-8), 0, s)  # was: l > 100 - 1e-7
    l = torch.clamp(l, 0, 100)

    return torch.stack([h, torch.clamp(s, 0, 100), l], dim=-1)


def _length_of_ray_until_intersect(theta, line):
    denominator = (torch.sin(theta) - line["slope"] * torch.cos(theta))
    clamped_denominator = torch.where(torch.abs(denominator) < 1e-5, 1e-12, denominator)
    # assert not torch.isnan(line['intercept'] / clamped_denominator).any()
    return line["intercept"] / clamped_denominator


def _get_bounds(l):
    result = []  # TODO vectorize
    sub1 = ((l + 16) ** 3) / 1560896
    sub2 = torch.where(sub1 > _epsilon, sub1, l / _kappa)
    mt = torch.tensor(_m).to(l)
    for c in range(3):
        m1, m2, m3 = mt[c]
        for t in range(2):
            top1 = (284517 * m1 - 94839 * m3) * sub2
            top2 = (838422 * m3 + 769860 * m2 + 731718 * m1) * l * sub2 - (769860 * t) * l
            bottom = (632260 * m3 - 126452 * m2) * sub2 + 126452 * t
            slope = top1 / bottom
            intercept = top2 / bottom
            result.append({"slope": slope, "intercept": intercept})
    return result


def _max_chroma_for_lh(l, h):
    hrad = torch.deg2rad(h)
    bounds = _get_bounds(l)
    # TODO vectorize
    lengths = [_length_of_ray_until_intersect(hrad, bound) for bound in bounds]
    lengths = torch.stack(lengths)

    # Mask out negative lengths
    non_negative_lengths = torch.where(lengths >= 0, lengths, torch.max(lengths).to(l))

    return non_negative_lengths.min(dim=0).values


def rgb_to_xyz(rgb):
    rgbl = torch.where(rgb <= 0.04045, rgb / 12.92,
                       torch.pow((rgb + 0.055) / 1.055, 2.4))
    xyz = torch.matmul(rgbl, torch.tensor(_m_inv).to(rgbl).transpose(0, 1))
    return xyz


def xyz_to_luv(xyz):
    x = xyz[..., 0]
    y = xyz[..., 1]
    z = xyz[..., 2]

    l = _y_to_l(y)

    divider = x + 15 * y + 3 * z

    var_u = 4 * x / divider
    var_v = 9 * y / divider

    u = 13 * l * (var_u - _ref_u)
    v = 13 * l * (var_v - _ref_v)

    u = torch.where(l == 0, 0, u)
    v = torch.where(l == 0, 0, v)

    return torch.stack([l, u, v], dim=-1)


def luv_to_lch(luv):
    l = luv[..., 0]

    u = luv[..., 1]
    u_eps = torch.full_like(u, 1e-12)
    u_eps = torch.where(u >= 0, u_eps, -u_eps)
    clamped_u = torch.where(u == 0, u_eps, u)

    v = luv[..., 2]
    v_eps = torch.full_like(v, 1e-12)
    v_eps = torch.where(v >= 0, v_eps, -v_eps)
    clamped_v = torch.where(v == 0, v_eps, v)

    c = torch.sqrt(clamped_u ** 2 + clamped_v ** 2)
    h = torch.atan2(clamped_v, clamped_u) * 180 / torch.pi
    h = torch.where(h < 0, h + 360, h)
    # max c among valid grayscale in float32: 6.69640576234087347984e-05
    # min c among valid barely saturated colors in float32: 0.028972066058486234
    # use midpoint as threshold
    h = torch.where(c < 0.028972067, 0, h)  # was: c < 1e-8 for float64
    return torch.stack([l, c, h], dim=-1)


def rgb_to_hsluv(rgb_tensor):
    rgb_tensor = rgb_tensor.permute(0, 2, 3, 1).clamp(1e-12, 1)
    xyz = rgb_to_xyz(rgb_tensor)
    luv = xyz_to_luv(xyz)
    lch = luv_to_lch(luv)
    l = lch[..., 0]
    c = lch[..., 1]
    h = lch[..., 2]
    return lch_to_hsluv(l, c, h).permute(0, 3, 1, 2)
