# Evaluating the Benchmark Results

To evaluate the benchmark results, ensure you have data at `path/to/data-loader/datasets`, which should be downloaded using the [data loader](https://github.com/ClimateBenchPress/data-loader).

On a high-level the benchmark evaluation pipeline progresses in the following steps:

1. Compute three error bound levels for each input dataset.
2. Compress each input dataset with all the benchmark compressors for all three error bounds.
3. Compute compressor performance metrics for evaluation purposes.
4. Optional: Create summary plots of the benchmark results.

We will now go through each of these steps in more detail. As you work through these steps, the pipeline will progressively populate the directories `datasets-error-bounds`, `compressed-datasets`, `metrics`, and `plots`.

## Create Error Bounds

Begin by creating the error bounds for each dataset using the following command:
```bash
uv run python -m climatebenchpress.compressor.scripts.create_error_bounds \
    --data-loader-basepath=path/to/data-loader
```
This step creates three error bounds for each variable in the datasets and stores the information in the `datasets-error-bounds` directory.

## Compress Input Datasets

Next, compress all the input datasets by running:
```bash
uv run python -m climatebenchpress.compressor.scripts.compress \
    --data-loader-basepath=path/to/data-loader
```
This command will populate the `compressed-datasets` directory with the following structure:
```
compressed-datasets/
    dataset1/
        {var_name}-{err_bound_type}={low_err_bound_val}_{var_name2}-{err_bound_type2}={low_err_bound_val2}
            compressor1/
                decompressed.zarr
                measurements.json
            compressor2/
                ...
        {var_name}-{err_bound_type}={mid_err_bound_val}_{var_name2}-{err_bound_type2}={mid_err_bound_val2}/
            ...
        {var_name}-{err_bound_type}={high_err_bound_val}_{var_name2}-{err_bound_type2}={high_err_bound_val2}/
            ...
    dataset2/
        ...
    ...
```
For each dataset, the results for the three different error bounds are stored in different directories. The `var_name` indicates the variable(s) in the dataset that are being compressed, while `err_bound_type` will be either `abs_error` or `rel_error`.

You can use additional arguments to control which compressors and datasets are processed: `--exclude-compressor` and `--exclude-dataset` to avoid using certain compressors and datasets, or `--include-compressor` and `--include-dataset` to only use selected compressors on selected datasets.
For example, the command
```bash
uv run python -m climatebenchpress.compressor.scripts.compress \
    --data-loader-basepath=path/to/data-loader \
    --include-compressor sz3 jpeg2000 \
    --include-dataset era5
```
compresses the era5 data with the compressors SZ3 and JPEG2000.
These arguments are particularly useful if you wish to parallelize the benchmark evaluation using tools such as `xargs`.

## Compute Metrics

After compression, evaluate compression metrics on the compressed datasets using:
```bash
uv run python -m climatebenchpress.compressor.scripts.compute_metrics \
    --data-loader-basepath=path/to/data-loader
```
You can apply the same filtering options with `--exclude-compressor`, `--exclude-dataset`, `--include-compressor` and `--include-dataset` arguments as used in the compression step.

Once the metrics are computed, combine all the metrics into a single CSV file by running:
```bash
uv run python -m climatebenchpress.compressor.scripts.concatenate_metrics
```
This will create the `metrics/all_results.csv` file which contains all the results.

## Optional: Create Plots

Finally, generate visualization plots with the following command:
```bash
uv run python -m climatebenchpress.compressor.plotting.plot_metrics \
    --data-loader-basepath=path/to/data-loader
```
This will create plots in the `plots` directory. By default, this assumes access to a LaTeX compiler. If you do not have one on your system, you can add the `--avoid-latex` flag to this command.

Note that the full plotting process can take quite a lot of time because it generates individual plots for each error bound-compressor-dataset combination. If you want to avoid generating individual plots for certain datasets, you can do so with the `--exclude-dataset` command line option.
