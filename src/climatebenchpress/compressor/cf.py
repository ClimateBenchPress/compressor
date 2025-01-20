import cf_xarray as cfxr


_ENSEMBLE_CRITERIA = dict(
    standard_name=("realization",),
    axis=("E",),
    cartesian_axis=("E",),
    grads_dim=("e",),
)

cfxr.options.set_options(
    custom_criteria=dict(
        realization=_ENSEMBLE_CRITERIA,
        E=_ENSEMBLE_CRITERIA,
    )
)
