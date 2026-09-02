_COMPRESSOR2LINEINFO = [
    ("jpeg2000", ("#EE7733", "-", "o")),
    ("sperr", ("#117733", ":", "s")),
    ("zfp-round", ("#DDAA33", "--", "D")),
    ("zfp", ("#EE3377", "--", "^")),
    ("sz3-abs", ("#CC3311", "-.", "p")),
    ("sz3", ("#CC3311", "-.", "v")),
    ("bitround-pco", ("#0077BB", ":", "P")),
    ("bitround", ("#33BBEE", "-", "X")),
    ("stochround-pco", ("#BBBBBB", "--", "d")),
    ("stochround", ("#009988", "--", "h")),
    ("tthresh", ("#882255", "-.", "<")),
    ("ebcc-abs", ("#AA4444", "-.", ">")),
    ("ebcc", ("#AA4444", "-.", "8")),
]


def _get_lineinfo(compressor: str) -> tuple[str, str, str]:
    """Get the line color, style, and marker for a given compressor."""
    for comp, (color, linestyle, marker) in _COMPRESSOR2LINEINFO:
        if compressor.startswith(comp):
            return color, linestyle, marker
    raise ValueError(f"Unknown compressor: {compressor}")


_COMPRESSOR2LEGEND_NAME = [
    ("jpeg2000", "JPEG2000"),
    ("sperr", "SPERR"),
    ("zfp-round", "ZFP-ROUND"),
    ("zfp", "ZFP"),
    ("sz3-abs", "SZ3-Abs"),
    ("sz3", "SZ3"),
    ("bitround-pco", "BitRound + PCO"),
    ("bitround", "BitRound + Zstd"),
    ("stochround-pco", "StochRound + PCO"),
    ("stochround", "StochRound + Zstd"),
    ("tthresh", "TTHRESH"),
    ("ebcc-abs", "EBCC-Abs"),
    ("ebcc", "EBCC"),
]

DISTORTION2LEGEND_NAME = {
    "Relative MAE": "Mean Absolute Error",
    "Relative DSSIM": "DSSIM",
    "Relative MaxAbsError": "Max Absolute Error",
    "Relative SpectralError": "Spectral Error",
}


def _get_compressor_legend_name(compressor: str) -> str:
    """Get the legend name for a given compressor."""
    for comp, name in _COMPRESSOR2LEGEND_NAME:
        if compressor.startswith(comp):
            return name

    return compressor  # Fallback to the compressor name if not found in the mapping.
