import numpy as np
import torch
import torch.nn as nn
import ROOT

# Conversione tensor -> numpy
def to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return x

# fill TH2 from an array
def make_th2_from_array(name, title, arr, 
                        nx, xmin, xmax,
                        ny, ymin, ymax):
    h = ROOT.TH2D(
        name, title,
        nx, xmin, xmax,
        ny, ymin, ymax
    )
    for x, y in arr:
        h.Fill(x, y)
    return h

# just look at the 1D projection shape
def normalize(h):
    if h.Integral() > 0:
        h.Scale(1.0 / h.Integral())

class TH2FlowValidation():
    def __init__(self, A_sim, A_corr, A_data,xmin,xmax,ymin,ymax):
        self.A_sim = A_sim
        self.A_corr = A_corr
        self.A_data = A_data
        self.get_th2s(xmin,xmax,ymin,ymax)
        self.h_pull_2d = None
        self.h_pull_1d = None
    
    def get_th2s(self,xmin,xmax,ymin,ymax):
        A_sim_np  = to_numpy(self.A_sim)
        A_corr_np = to_numpy(self.A_corr)
        A_data_np = to_numpy(self.A_data)
     
        self.h_sim = make_th2_from_array(
            "h_sim", "Simulation",
            A_sim_np,
            50, xmin, xmax,
            50, ymin, ymax
        )
     
        self.h_corr = make_th2_from_array(
            "h_corr", "Corrected simulation",
            A_corr_np,
            50, xmin, xmax,
            50, ymin, ymax
        )
     
        self.h_data = make_th2_from_array(
            "h_data", "Data",
            A_data_np,
            50, xmin, xmax,
            50, ymin, ymax
        )


    def make_pull_2d(self):
        if self.h_pull_2d:
            return self.h_pull_2d
        
        h_pull = self.h_corr.Clone("h_pull")
        h_pull.Reset()
        h_pull.SetTitle("Pull: (corr - data) / sqrt(data)")
        
        for ix in range(1, self.h_corr.GetNbinsX()+1):
            for iy in range(1, self.h_corr.GetNbinsY()+1):
                n_corr = self.h_corr.GetBinContent(ix, iy)
                n_data = self.h_data.GetBinContent(ix, iy)
     
                if n_data > 0:
                    pull = (n_corr - n_data) / (n_data ** 0.5)
                    h_pull.SetBinContent(ix, iy, pull)                

        self.h_pull_2d = h_pull
        return h_pull

    def make_pull_1d(self):

        if self.h_pull_1d:
            return self.h_pull_1d
        
        h_pull_1d = ROOT.TH1D(
            "h_pull_1d",
            "Pull distribution",
            100, -5, 5
        )

        h_pull_2d = self.make_pull_2d()
        
        for ix in range(1, h_pull_2d.GetNbinsX()+1):
            for iy in range(1, h_pull_2d.GetNbinsY()+1):
                val = h_pull_2d.GetBinContent(ix, iy)
                if val != 0:
                    h_pull_1d.Fill(val)

        self.h_pull_1d = h_pull_1d
        return h_pull_1d

    def plot_validation(self,filename):

        self.make_pull_2d()
        self.make_pull_1d()
        
        hx_sim  = self.h_sim.ProjectionX("hx_sim")
        hx_corr = self.h_corr.ProjectionX("hx_corr")
        hx_data = self.h_data.ProjectionX("hx_data")
        
        hy_sim  = self.h_sim.ProjectionY("hy_sim")
        hy_corr = self.h_corr.ProjectionY("hy_corr")
        hy_data = self.h_data.ProjectionY("hy_data")
        
        for h in [hx_sim, hx_corr, hx_data, hy_sim, hy_corr, hy_data]:
            normalize(h)

        c = ROOT.TCanvas("c", "", 1200, 1000)
        c.Print("validation.pdf[")

        # --- TH2 ---
        for h, name in [
                (self.h_sim,  "Simulation"),
                (self.h_corr, "Corrected"),
                (self.h_data, "Data"),
                (self.h_pull_2d, "Pull")
        ]:
            print(f"plotting {name}") 
            h.Draw("COLZ")
            c.SetTitle(name)
            c.Print("validation.pdf")

        # --- Proiezioni X ---
        hx_data.SetLineColor(ROOT.kBlack)
        hx_data.SetMarkerColor(ROOT.kBlack)
        hx_corr.SetLineColor(ROOT.kRed)
        hx_sim.SetLineColor(ROOT.kBlue)
        
        hx_data.Draw("PE")
        hx_corr.Draw("HIST SAME")
        hx_sim.Draw("HIST SAME")

        c.Print("validation.pdf")

        # --- Proiezioni Y ---
        hy_data.SetLineColor(ROOT.kBlack)
        hy_data.SetMarkerColor(ROOT.kBlack)
        hy_corr.SetLineColor(ROOT.kRed)
        hy_sim.SetLineColor(ROOT.kBlue)

        hy_data.Draw("PE")
        hy_corr.Draw("HIST SAME")
        hy_sim.Draw("HIST SAME")
        
        c.Print("validation.pdf")
        
        # --- Pull 1D ---
        self.h_pull_1d.Draw("HIST")
        c.Print("validation.pdf")
        
        c.Print("validation.pdf]")

        f = ROOT.TFile("validation.root", "RECREATE")
        for h in [
                self.h_sim, self.h_corr, self.h_data,
                self.h_pull_2d, self.h_pull_1d,
                hx_sim, hx_corr, hx_data,
                hy_sim, hy_corr, hy_data
        ]:
            h.Write()
        f.Close()
