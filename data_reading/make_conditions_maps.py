import os
import yaml
import numpy as np
import pandas as pd
from pathlib import Path

def make_simulation_map(output_file="simulation_map.yaml"):
    alphas  = np.linspace(0.019,0.023,11)
    lambdas = np.linspace(850,1850,11)

    zpos_dic = {1: "5.0", 2: "15.0", 3: "25.0", 4: "35.0", 5: "46.5"}
    
    sim_dict = {}
    for a,alpha in enumerate(alphas):
        for l,Lambda in enumerate(lambdas):
            for z in range(1,6):
                # non capisco perche' del numero random della digi iniziale, ma tant'e'. Tocca fidarsi del nome del file, non dello step
                for step in range(1,6):
                    filename = f"data/sim/test_recosim/digi_{a}-{l}/iron_step{step}/reco_run0000{z}_3D.root"
                    if Path(filename).exists():
                        # remove the files that have only a tiny number of clusters reco (this is a sign of sim/digi/reco problem, not physics)
                        size_bytes = os.path.getsize(filename)
                        size_kb = size_bytes / 1024 
                        if size_kb>100:
                            key = f"{zpos_dic[z]},{alpha:.4f},{Lambda:.0f}"
                            sim_dict[key] = filename

    with open(output_file, "w") as f:
        yaml.safe_dump(sim_dict, f, sort_keys=True, default_flow_style=False)


def make_data_map(input_csv="calibration.csv",output_file="data_map.yaml"):
    df = pd.read_csv(input_csv, sep=";")

    nz = nP = nT = 5

    print(f"Now will choose a set of {nz} points in z, {nP} points in P and {nT} points in T sampling uniformly from their min/max values")
    
    z_bins = np.linspace(df["z"].min(), df["z"].max(), nz + 1)
    P_bins = np.linspace(df["P"].min(), df["P"].max(), nP + 1)
    T_bins = np.linspace(df["T"].min(), df["T"].max(), nT + 1)

    df = df.copy()

    df["z_bin"] = pd.cut(df["z"], z_bins, include_lowest=True)
    df["P_bin"] = pd.cut(df["P"], P_bins, include_lowest=True)
    df["T_bin"] = pd.cut(df["T"], T_bins, include_lowest=True)

    sampled = (
    df
    .dropna(subset=["z_bin", "P_bin", "T_bin"])
    .groupby(["z_bin", "P_bin", "T_bin"], observed=True)
    .sample(n=1, random_state=42)
    .reset_index(drop=True)
    )

    print(f"In principle I could select {nz*nP*nT} points, but some bins can be empty, so I selected instead {len(sampled)} runs:\n\n")
    # print(sampled[["z", "P", "T", "run"]])

    data_dict = { f'{row["z"]:.1f},{row["P"]:.4f},{row["T"]:.1f}': f"reco_run{int(row.run)}_3D.root" for _, row in sampled.iterrows() }
    
    with open(output_file, "w") as f:
        yaml.safe_dump(data_dict, f, sort_keys=True, default_flow_style=False)
    
if __name__ == "__main__":

    make_simulation_map()
    make_data_map()
    
