#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""A small module to kick out color values for q-values."""

from bisect import bisect_left

CUTOFFS = (0.0001, 0.001, 0.01, 0.05, 0.2)

UPREGULATED_COLORS = (
        # (R, G, B)
        (165, 0, 38), # <= 0.0001
        (215, 48, 39), # <= 0.001
        (244, 109, 67), # <= 0.01
        (253, 174, 97), # <= 0.05
        (254, 224, 139), # <= 0.2
        (255, 255, 255), # <= 1
)
DOWNREGULATED_COLORS = (
        #(R, G, B)
        (0, 104, 55), # <= 0.0001
        (26, 152, 80), # <= 0.001
        (102, 189, 99), # <= 0.01
        (166, 217, 106), # <= 0.05
        (217, 239, 139), # <= 0.2
        (255, 255, 255), # <= 1
)

rgb_vals_to_hex = lambda rgb_vals: '#' + ''.join(('%02x' % value for
        value in rgb_vals))

UPREGULATED_HEX_CODES = [rgb_vals_to_hex(rgb_ints) for rgb_ints in
        UPREGULATED_COLORS]
DOWNREGULATED_HEX_CODES = [rgb_vals_to_hex(rgb_ints) for rgb_ints in
        DOWNREGULATED_COLORS]


def value_to_rdylgn(value, sign):
    """
    Calculates the appropriate ColorBrewer RdYlGn color for a given
    value, returned as an RGB hex string.

    This is intended to be used with p- or q-values.

    Returns a string of hexadecimal of the form `'RRGGBB'` of the
    respective hex values for red, green, and blue.

    :Parameters:
    - `value`: a value between 0 and 1 (inclusive)
    - `sign`: the "sign" of the value, either `'+'` or `'-'`;  `'+'` if
      up-regulated, `'-'` if down-regulated

    """

    if sign == '+':
        hex_codes = UPREGULATED_HEX_CODES
    elif sign == '-':
        hex_codes = DOWNREGULATED_HEX_CODES
    else:
        raise ValueError('sign should be \'+\' or \'-\'')

    index = bisect_left(CUTOFFS, value)
    hex_value = hex_codes[index]

    return hex_value

