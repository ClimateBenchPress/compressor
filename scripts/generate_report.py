from pathlib import Path

import pandas as pd

REPO = Path(__file__).parent.parent


def main():
    plots_path = REPO / "plots"
    report_path = REPO / "plots" / "report.md"

    report_string = "# ClimateBenchPress Report\n\n"
    report_string += "## Overview\n\n"

    for plot in plots_path.glob("*.png"):
        relative_path = plot.relative_to(report_path.parent)
        report_string += f"![{plot.stem}]({relative_path})\n\n"

    all_results = pd.read_csv(REPO / "metrics" / "all_results.csv")

    report_string += "## Metrics Per Datasets\n\n"
    for dataset in all_results["Dataset"].unique():
        report_string += f"### {dataset}\n"
        report_string += (
            all_results[all_results["Dataset"] == dataset]
            .drop(columns=["Dataset"])
            .to_markdown(index=False)
        ) + "\n\n"

        rd_curve = plots_path / dataset / "compression_ratio_vs_psnr.png"
        if rd_curve.exists():
            relative_path = rd_curve.relative_to(report_path.parent)
            report_string += f"![RD Curve]({relative_path})\n\n"

    report_string += "## Reconstructions\n\n"
    for dataset in all_results["Dataset"].unique():
        report_string += f"### {dataset}\n"
        df = all_results[all_results["Dataset"] == dataset]
        for comp in df["Compressor"].unique():
            report_string += f"#### {comp}\n"
            for var in df["Variable"].unique():
                plot = plots_path / dataset / f"{var}_{comp}.png"
                if plot.exists():
                    relative_path = plot.relative_to(report_path.parent)
                    report_string += f"![{var} {comp}]({relative_path})\n\n"

    with report_path.open("w") as f:
        f.write(report_string)


if __name__ == "__main__":
    main()
