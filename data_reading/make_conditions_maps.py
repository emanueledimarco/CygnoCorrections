import os
import yaml
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict

def make_simulation_map(base_dir="data/sim/recosim_2k",output_file="simulation_map.yaml"):
    alphas  = np.linspace(0.019,0.023,11)
    lambdas = np.linspace(850,1850,11)

    zpos_dic = {1: "5.0", 2: "15.0", 3: "25.0", 4: "35.0", 5: "46.5"}

    #sim_dict = {}
    sim_dict = defaultdict(list)
    for a,alpha in enumerate(alphas):
        for l,Lambda in enumerate(lambdas):
            for z in range(1,6):
                # il numero del run e' fittizio (ordine del job di sim). Quindi si deve usare il numero dello step
                for step in range(1,6):
                    for d in Path(base_dir).iterdir():
                        if d.is_dir():
                            filename = f"{d}/digi_{a}-{l}/iron_step{step}/reco_run0000{z}_3D.root"
                            if Path(filename).exists():
                                # remove the files that have only a tiny number of clusters reco (this is a sign of sim/digi/reco problem, not physics)
                                size_bytes = os.path.getsize(filename)
                                size_kb = size_bytes / 1024 
                                if size_kb>100:
                                    key = f"{zpos_dic[step]},{alpha:.4f},{Lambda:.0f}"
                                    sim_dict[key].append(filename)

    with open(output_file, "w") as f:
        yaml.safe_dump(dict(sim_dict), f, sort_keys=True, default_flow_style=False)
    print (f"===> Output sim map written into {output_file}")


def make_data_map(input_csv="data/runs/recodata_run4/calibration.csv",output_file="data_map.yaml"):
    df = pd.read_csv(input_csv, sep=",")

    nP = nT = nH = 5
    
    z_vals = df["z"].unique()
    P_bins = np.linspace(df["P"].min(), df["P"].max(), nP + 1)
    T_bins = np.linspace(df["T"].min(), df["T"].max(), nT + 1)
    H_bins = np.linspace(df["H"].min(), df["H"].max(), nT + 1)

    nz = len(z_vals)
    print(f"Now will choose a set of {nz} points in z, {nP} points in P, {nT} points in T and {nH} points in H, sampling uniformly from their min/max values")
    
    df = df.copy()

    
    df["z_bin"] = (df[df["z"].isin(z_vals)])["z"]
    df["P_bin"] = pd.cut(df["P"], P_bins, include_lowest=True)
    df["T_bin"] = pd.cut(df["T"], T_bins, include_lowest=True)
    df["H_bin"] = pd.cut(df["H"], H_bins, include_lowest=True)

    grouped = (
        df
        .dropna(subset=["z_bin", "P_bin", "T_bin", "H_bin"])
        .groupby(["z_bin", "P_bin", "T_bin", "H_bin"], observed=True)["run"]
        .apply(list)
    )

    print(f"In principle I could select {nz*nP*nT*nH} points, but some bins can be empty, so I selected instead {len(grouped)} combinations of z, P, T, H.")

    data_dict = {
        (float(z), float(round(P.mid,3)), float(round(T.mid,1)), float(round(H.mid,1))): [f"data/runs/recodata_run4/reco_run{int(r)}_3D.root" for r in runs]
        for (z, P, T, H), runs in grouped.items()
    }

    data_dict_simplekeys = {", ".join(map(str, k)): v for k, v in data_dict.items()}
    
    with open(output_file, "w") as f:
        yaml.safe_dump(data_dict_simplekeys, f, sort_keys=True, default_flow_style=False)
    print (f"===> Output data map written into {output_file}")

    
if __name__ == "__main__":

    make_simulation_map()
    make_data_map()
    
