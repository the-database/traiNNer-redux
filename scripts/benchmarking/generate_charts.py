import csv

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def process_table(df: pd.DataFrame) -> pd.DataFrame:
    df["psnr_df2k"] = pd.to_numeric(df["psnr_df2k"], errors="coerce")
    return df[["name", "fps", "psnr_df2k", "vram"]]


def plot_scatter(df: pd.DataFrame, scale: int, size: str) -> None:
    plt.figure(figsize=(10, 6), dpi=300)

    # Scale VRAM directly between its min and max for circle sizes
    vram_min, vram_max = df["vram"].min(), df["vram"].max()
    vram_scaled = np.interp(
        df["vram"], (vram_min, vram_max), (10, 10000)
    )  # Circle size range (10, 1000)

    # Use a colormap for dot colors
    colors = plt.cm.tab10(np.linspace(0, 1, len(df)))  # type: ignore

    for i, row in df.iterrows():
        # Plot the shaded circle (background)
        plt.scatter(
            row["fps"],
            row["psnr_df2k"],
            s=vram_scaled[i],  # type: ignore
            color=colors[i],
            alpha=0.2,
            label=row["name"],
        )

        # Plot the main dot
        plt.scatter(
            row["fps"], row["psnr_df2k"], color=colors[i], edgecolors="black", alpha=0.9
        )

        # Annotate each point with the model name
        plt.annotate(
            row["name"],
            (row["fps"], row["psnr_df2k"]),
            textcoords="offset points",
            xytext=(5, 5),
            ha="center",
            fontsize=5,
        )

    type = "Restoration" if scale == 1 else "Upscale"

    plt.title(
        f"{scale}x {type} {size} DF2K Urban100 PSNR vs FPS on 640x480 input with RTX 4090",
        fontsize=16,
    )
    plt.xlabel("FPS", fontsize=12)
    plt.ylabel("DF2K Urban100 PSNR", fontsize=12)
    plt.grid(True)
    plt.savefig(f"docs/source/resources/benchmark{scale}x_{size.lower()}.png")


def plot_fps(df: pd.DataFrame) -> None:
    df = df.sort_values(by="fps", ascending=False)
    plt.figure(figsize=(10, 6))
    plt.bar(df["name"], df["fps"], color="green", edgecolor="black")
    plt.title("FPS for each model", fontsize=16)
    plt.xlabel("Model", fontsize=12)
    plt.ylabel("FPS", fontsize=12)
    plt.xticks(rotation=45, ha="right")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_psnr(df: pd.DataFrame) -> None:
    plt.figure(figsize=(10, 6))
    plt.bar(df["name"], df["psnr_df2k"], color="purple", edgecolor="black")
    plt.title("DF2K PSNR for each model", fontsize=16)
    plt.xlabel("Model", fontsize=12)
    plt.ylabel("PSNR", fontsize=12)
    plt.xticks(rotation=45, ha="right")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def main() -> None:
    for scale in [4, 3, 2]:
        file_path = f"docs/source/resources/benchmark{scale}x.csv"

        with open(file_path, newline="") as f:
            reader = csv.DictReader(f)
            d = [row for row in reader if row["psnr_df2k"] != ""]
            for row in d:
                row["fps"] = float(row["fps"])
                row["name"] = f"{row["name"]} {row["variant"]}".strip()

            threshold1 = 2
            threshold2 = 24

            dsmall = [row for row in d if threshold2 < float(row["fps"])]
            dmed = [row for row in d if threshold1 <= float(row["fps"]) <= threshold2]
            dlarge = [row for row in d if float(row["fps"]) < threshold1]

            dfsmall = process_table(pd.DataFrame(dsmall))
            dfmed = process_table(pd.DataFrame(dmed))
            dflarge = process_table(pd.DataFrame(dlarge))
            df = process_table(pd.DataFrame(d))

            plot_scatter(dfsmall, scale, "Small")
            plot_scatter(dfmed, scale, "Medium")
            plot_scatter(dflarge, scale, "Large")
            plot_scatter(df, scale, "All")


if __name__ == "__main__":
    main()
