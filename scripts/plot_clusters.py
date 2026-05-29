#!/usr/bin/env python3

# usage:
# evento singolo:  python plot_clusters.py run.root --event 124
# range di eventi: python plot_clusters.py run.root --first-event 100 --last-event 150

import ROOT
import os
import argparse

ROOT.gStyle.SetOptStat(0)


# ------------------------------------------------
# Disegno singolo cluster
# ------------------------------------------------
def draw_cluster(cluster_id, x, y, z, event_id, outdir):

    xmin = min(x) - 1
    xmax = max(x) + 1
    ymin = min(y) - 1
    ymax = max(y) + 1

    nx = int(xmax - xmin + 1)
    ny = int(ymax - ymin + 1)

    h2 = ROOT.TH2F(
        f"h_evt{event_id}_cl{cluster_id}",
        f"Event {event_id} Cluster {cluster_id};x;y",
        nx, xmin, xmax,
        ny, ymin, ymax
    )

    for xx, yy, zz in zip(x, y, z):
        h2.Fill(xx, yy, zz)

    c = ROOT.TCanvas(
        f"c_evt{event_id}_cl{cluster_id}",
        "",
        800,
        700
    )

    h2.Draw("COLZ")

    outname = os.path.join(
        outdir,
        f"event_{event_id:06d}_cluster_{cluster_id:03d}.png"
    )

    c.SaveAs(outname)


# ------------------------------------------------
# Disegno evento completo
# ------------------------------------------------
def draw_event(event_id, clusters, outdir):

    c = ROOT.TCanvas(
        f"c_evt_{event_id}",
        f"Event {event_id}",
        1000,
        900
    )

    mg = ROOT.TMultiGraph()
    legend = ROOT.TLegend(0.82, 0.1, 0.98, 0.9)

    colors = [
        ROOT.kRed,
        ROOT.kBlue,
        ROOT.kGreen + 2,
        ROOT.kMagenta,
        ROOT.kOrange + 7,
        ROOT.kCyan + 1,
        ROOT.kBlack,
        ROOT.kViolet,
        ROOT.kAzure + 2,
        ROOT.kPink + 7
    ]

    graphs = []

    for icl, cluster in enumerate(clusters):

        x = cluster["x"]
        y = cluster["y"]

        n = len(x)

        g = ROOT.TGraph(n)

        for i in range(n):
            g.SetPoint(i, x[i], y[i])

        color = colors[icl % len(colors)]

        g.SetMarkerStyle(20)
        g.SetMarkerSize(0.8)
        g.SetMarkerColor(color)

        mg.Add(g, "P")
        legend.AddEntry(g, f"cl {icl}", "p")

        graphs.append(g)

    mg.Draw("A")

    mg.SetTitle(
        f"Event {event_id} - reconstructed clusters;x;y"
    )

    legend.Draw()

    c.Update()

    outname = os.path.join(
        outdir,
        f"event_{event_id:06d}_all_clusters.png"
    )

    c.SaveAs(outname)


# ------------------------------------------------
# Ricostruzione cluster
# ------------------------------------------------
def reconstruct_clusters(tree):

    nSc = int(tree.nSc)

    if nSc <= 0:
        return []

    idx = tree.sc_redpixIdx
    ix = tree.redpix_ix
    iy = tree.redpix_iy
    iz = tree.redpix_iz

    n_redpix = len(ix)

    clusters = []

    for icl in range(nSc):

        start = int(idx[icl])

        if icl == nSc - 1:
            stop = int(n_redpix)
        else:
            stop = int(idx[icl + 1])

        x = []
        y = []
        z = []

        for ipix in range(start, stop):
            x.append(ix[ipix])
            y.append(iy[ipix])
            z.append(iz[ipix])

        clusters.append({
            "x": x,
            "y": y,
            "z": z
        })

    return clusters


# ------------------------------------------------
# Main
# ------------------------------------------------
def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "input_file",
        help="ROOT file"
    )

    parser.add_argument(
        "--tree",
        default="Events"
    )

    parser.add_argument(
        "--event",
        type=int,
        default=None,
        help="plotta un singolo evento"
    )

    parser.add_argument(
        "--first-event",
        type=int,
        default=0
    )

    parser.add_argument(
        "--last-event",
        type=int,
        default=None
    )

    parser.add_argument(
        "--outdir",
        default="cluster_plots"
    )

    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    f = ROOT.TFile.Open(args.input_file)

    if not f or f.IsZombie():
        raise RuntimeError("Errore apertura file ROOT")

    tree = f.Get(args.tree)

    if not tree:
        raise RuntimeError(
            f"TTree {args.tree} non trovata"
        )

    n_entries = tree.GetEntries()

    # selezione eventi
    if args.event is not None:
        events = [args.event]
    else:
        first = args.first_event
        last = (
            args.last_event
            if args.last_event is not None
            else n_entries - 1
        )

        events = range(first, last + 1)

    for iev in events:

        if iev >= n_entries:
            continue

        print(f"Processing event {iev}")

        tree.GetEntry(iev)

        clusters = reconstruct_clusters(tree)

        if len(clusters) == 0:
            print("  no clusters")
            continue

        # evento completo
        draw_event(
            iev,
            clusters,
            args.outdir
        )

        # cluster individuali
        for icl, cl in enumerate(clusters):

            draw_cluster(
                icl,
                cl["x"],
                cl["y"],
                cl["z"],
                iev,
                args.outdir
            )

    print("Done")


if __name__ == "__main__":
    main()
