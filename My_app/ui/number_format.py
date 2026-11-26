"""How a metric value is written on screen: up to two decimals."""
# Display only. The exports carry the full-precision number, never this text.
import math

#: Below this magnitude two decimals would read 0.00 or keep a single digit,
#: so the value is written in scientific notation instead.
SCIENTIFIC_BELOW = 0.01


def format_value(value):
    """`value` with at most two decimals; scientific when smaller than 0.01.

    0.123456 -> '0.12', 3.14159 -> '3.14', 45.67 -> '45.67', 123.4 -> '123.4',
    12345 -> '12345', 0.0099 -> '9.9e-3', 0.00004 -> '4.0e-5'.
    Trailing zeros are dropped (3.10 -> '3.1'). NaN and infinity are written
    as 'NaN' and 'inf'.
    """
    value = float(value)
    if math.isnan(value):
        return "NaN"
    if math.isinf(value):
        return "inf" if value > 0 else "-inf"
    if value == 0:
        return "0"

    if abs(value) < SCIENTIFIC_BELOW:
        # Two significant figures, exponent without padding: 9.9e-3, 4.0e-5.
        mantissa, exponent = ("%.1e" % value).split("e")
        return "%se%d" % (mantissa, int(exponent))

    text = "%.2f" % value
    text = text.rstrip("0").rstrip(".")
    return text
