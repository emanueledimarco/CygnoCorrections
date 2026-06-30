import numpy as np
import uproot
import torch

class Cluster:
    def __init__(self, pix, scalars_dict, cond, meta=None):

        self.pix = pix.astype(np.float32)
        self.cond = np.array(cond, dtype=np.float32)
        self.meta = meta or {}

        # dinamically assign scalars as attributes (fallback, use 
        for k, v in scalars_dict.items():
            setattr(self, k.replace("sc_",""), float(v))

    def n_pixels(self):
        return len(self.pix)

    def to_image(self, image_size=64):

        img = np.zeros(
            (image_size, image_size),
            dtype=np.float32
        )
     
        center = image_size // 2
     
        x = np.round(self.pix[:, 0]).astype(int)
        y = np.round(self.pix[:, 1]).astype(int)
        q = self.pix[:, 2]
     
        x += center
        y += center
     
        valid = (
            (x >= 0)
            & (x < image_size)
            & (y >= 0)
            & (y < image_size)
        )
     
        x = x[valid]
        y = y[valid]
        q = q[valid]
     
        img[y, x] += q
     
        return img


def select_cluster(c, selection_cfg):

    #print (f"Going to make selection on cluster with int = {c.integral}, xmean = {c.xmean}, ymean = {c.ymean}, npix = {c.n_pixels()}, nhits = {c.nhits}")
    return (
        selection_cfg["integral_min"] < c.integral < selection_cfg["integral_max"]
        and selection_cfg["x_min"] < c.xmean < selection_cfg["x_max"]
        and selection_cfg["y_min"] < c.ymean < selection_cfg["y_max"]
        and c.n_pixels() > selection_cfg["min_npix"]
        and c.nhits > selection_cfg["n_hits"]
    )

def build_clusters_from_event(
    arrays,
    iev,
    scalar_vars,
    conditions,
    isdata=False,
    selection_cfg=None
):

    clusters = []

    nSc = arrays["nSc"][iev]

    for isc in range(nSc):

        start = int(arrays["sc_redpixIdx"][iev][isc])

        end = (
            int(arrays["sc_redpixIdx"][iev][isc + 1])
            if isc < nSc - 1
            else len(arrays["redpix_ix"][iev])
        )

        ix = arrays["redpix_ix"][iev][start:end]
        iy = arrays["redpix_iy"][iev][start:end]
        iz = arrays["redpix_iz"][iev][start:end]

        xmean = float(arrays["sc_xmean"][iev][isc])
        ymean = float(arrays["sc_ymean"][iev][isc])

        pix = np.stack([ix, iy, iz], axis=1).astype(np.float32)

        # centering (ONLY x,y)
        pix[:, 0] -= xmean
        pix[:, 1] -= ymean

        # ------------------------
        # scalars (GENERIC)
        # ------------------------
        scalars = {}
        for var in scalar_vars:
            # Estraiamo l'intero dato dell'evento corrente
            evt_data = arrays[var][iev]

            # Controlliamo se è un array/lista con un elemento per ogni supercluster
            if isinstance(evt_data, (list, np.ndarray)) and len(evt_data) == nSc:
                # Prendiamo lo scalare corrispondente esattamente a QUESTO cluster
                scalars[var] = float(evt_data[isc])
            elif isinstance(evt_data, (list, np.ndarray)) and len(evt_data) > isc:
                # Caso di fallback se l'array ha lunghezze asimmetriche
                scalars[var] = float(evt_data[isc])
            else:
                # Se è un valore singolo per tutto l'evento (es. variabili globali)
                scalars[var] = float(evt_data)

        scalars["npix"] = len(pix)
        
        # ------------------------
        # condition
        # ------------------------
        cluster = Cluster(
            pix=pix,
            scalars_dict=scalars,
            cond=conditions,
            meta={"event_idx": iev, "cluster_idx": isc}
        )

        # ------------------------
        # selection (OPTIONAL)
        # ------------------------
        if selection_cfg is not None:
            if not select_cluster(cluster, selection_cfg):
                continue

        clusters.append(cluster)

    return clusters


def extract_all_clusters(
    arrays,
    scalar_vars,
    conditions,
    isdata=False,
    selection_cfg=None
):

    all_clusters = []

    for iev in range(len(arrays["nSc"])):

        clusters_evt = build_clusters_from_event(
            arrays,
            iev,
            scalar_vars=scalar_vars,
            conditions=conditions,
            isdata=isdata,
            selection_cfg=selection_cfg
        )

        all_clusters.extend(clusters_evt)

    return all_clusters



def build_clusters_from_root_file(
    root_file,
    scalar_vars,
    conditions,
    isdata=False,
    selection_cfg=None
):

    #print (f"Will open the rootfile {root_file}")
    f = uproot.open(root_file)
    tree = f["Events"]

    arrays = tree.arrays(library="np")

    return extract_all_clusters(
        arrays,
        scalar_vars=scalar_vars,
        conditions=conditions,
        isdata=isdata,
        selection_cfg=selection_cfg
    )


def build_dataset_from_files(
    root_files,
    scalar_vars,
    conditions,
    isdata=False,
    selection_cfg=None
):

    all_clusters = []

    for f in root_files:

        clusters = build_clusters_from_root_file(
            f,
            scalar_vars=scalar_vars,
            conditions=conditions,
            isdata=isdata,
            selection_cfg=selection_cfg
        )

        all_clusters.extend(clusters)

    return all_clusters
