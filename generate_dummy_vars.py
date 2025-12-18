import ROOT
import numpy as np

if __name__ == "__main__":

    N=100000
    
    f = ROOT.TFile("dummy_simulation.root", "RECREATE")
    h2 = ROOT.TH2D("h2","dummy 2D vars", 100, -3, 7, 100, -5, 5)

    Npairs = 10
    # variabili indipendenti
    x = np.random.uniform(-1, 1, Npairs)
    y = np.random.uniform(-1, 1, Npairs)

    histos = []

    for ip in range(Npairs):
        histos.append(h2.Clone(f"h2_{ip}"))

        # -------------------------
        # Medie (polinomi in x,y)
        # -------------------------
        mu1 =  2.0 + 0.5*x[ip] - 0.3*y[ip] + 0.1*x[ip]*y[ip]
        mu2 = -1.0 + 0.4*x[ip] + 0.6*y[ip]

        # -------------------------
        # RMS (polinomi in x,y)
        # -------------------------
        sigma1 = 1.0 + 0.2*x[ip]**2 + 0.1*y[ip]**2
        sigma2 = 0.8 + 0.1*x[ip]**2 + 0.1*y[ip]**2

        # -------------------------
        # Correlazione
        # -------------------------
        rho = 0.6

        # Gaussiane standard indipendenti
        print(f"Generating two Gaussians with {N} events for pair ({x[ip]:.2f},{y[ip]:.2f})...")
        z1 = np.random.normal(size=N)
        z2 = np.random.normal(size=N)

        # -------------------------
        # Loop Monte Carlo + Fill
        # -------------------------
        print(f"Monte Carlo generation of 2D correlated distribution...")
        for i in range(N):
            if i%100 == 0:
                print(f"\tSampling the Gaussians for event {i}")
            g1 = mu1 + sigma1 * z1[i]
            g2 = mu2 + sigma2 * (rho * z1[i] + np.sqrt(1 - rho**2) * z2[i])
            histos[ip].Fill(g1, g2)

        f.cd()
        histos[ip].Write()

    f.Close()

    print("Istogramma TH2D scritto in output_h2.root")
