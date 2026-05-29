# Loads and treat the reconstruction tree data and simulation
# test basic plotting with python -m data_reading.read_data 


import uproot
import awkward as ak
import pandas as pd
import numpy as np

from data_reading.read_data import perform_cluster_selection

def build_cluster_dataframe(
    data_files,
    branches_scalar,
    isdata=False
):
    """
    Costruisce un dataframe:
    una riga = un cluster

    Parameters
    ----------
    data_files : list[str]
        Lista file ROOT

    branches_scalar : list[str]
        Variabili scalari per cluster
        es:
        [
            "sc_integral",
            "sc_xmean",
            "sc_ymean",
            "sc_rms"
        ]

    Returns
    -------
    pd.DataFrame
    """

    # branches necessarie per i pixel
    pixel_branches = [
        "nSc",
        "sc_redpixIdx",
        "redpix_ix",
        "redpix_iy",
        "redpix_iz",
    ]

    branches = branches_scalar + pixel_branches

    rows = []

    for data_file in data_files:

        print(f"Reading {data_file}")

        with uproot.open(data_file) as f:

            tree = f["Events"]

            arrays = tree.arrays(
                branches,
                library="ak"
            )

            # tua selezione cluster
            arrays_sel = perform_cluster_selection(
                arrays,
                isdata
            )

            n_events = len(arrays_sel["nSc"])

            for iev in range(n_events):

                nsc = arrays_sel["nSc"][iev]

                if nsc == 0:
                    continue

                redpix_idx = arrays_sel[
                    "sc_redpixIdx"
                ][iev]

                redpix_ix = arrays_sel[
                    "redpix_ix"
                ][iev]

                redpix_iy = arrays_sel[
                    "redpix_iy"
                ][iev]

                redpix_iz = arrays_sel[
                    "redpix_iz"
                ][iev]

                # loop sui cluster evento
                for isc in range(nsc):

                    start = int(redpix_idx[isc])

                    if isc < nsc - 1:
                        stop = int(
                            redpix_idx[isc + 1]
                        )
                    else:
                        stop = len(redpix_ix)

                    pix_x = np.asarray(
                        redpix_ix[start:stop]
                    )

                    pix_y = np.asarray(
                        redpix_iy[start:stop]
                    )

                    pix_z = np.asarray(
                        redpix_iz[start:stop]
                    )

                    row = {

                        # metadata
                        "event_idx": iev,
                        "cluster_idx": isc,

                        # sparse cluster
                        "pix_x": pix_x,
                        "pix_y": pix_y,
                        "pix_z": pix_z,
                    }

                    # aggiungi scalari
                    for var in branches_scalar:

                        val = arrays_sel[var][iev]

                        # cluster-wise variable
                        if isinstance(
                            val,
                            (ak.Array, list)
                        ):

                            row[var] = val[isc]

                        else:
                            # event-wise variable
                            row[var] = val

                    rows.append(row)

    df = pd.DataFrame(rows)

    return df
