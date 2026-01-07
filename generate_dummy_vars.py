import ROOT
import numpy as np

# linear transformation to pass from (alpha,beta)==true params -> (x,y)==observation params
# should be small, so the output distributuion is similar in sim and data, to recycle the toy MC generation of both data and sim
def transform_xy(x,y):
    xp = 1.10*x + 0.05
    yp = 1.06*y + 0.08
    return (xp,yp)

def generate_toys(Npoints, output_file, isdata=False, N=int(1e4)):

    f = ROOT.TFile(output_file, "RECREATE")
    h2 = ROOT.TH2D("h2","dummy 2D vars", 100, -3, 7, 100, -5, 5)

    # variabili indipendenti
    xarr = np.linspace(-1, 1, Npoints)
    yarr = np.linspace(-1, 1, Npoints)

    histos = []

    for ix,x in enumerate(xarr):
        for iy,y in enumerate(yarr):
            if isdata:
                x,y = transform_xy(x,y)

            histos.append(h2.Clone(f"h2_{ix}_{iy}"))
            histos[-1].SetTitle(f"(x,y) = ({x:.2f},{y:.2f})")

            
            # -------------------------
            # Medie (polinomi in x,y)
            # -------------------------
            mu1 =  2.0 + 0.5*x - 0.3*y + 0.1*x*y
            mu2 = -1.0 + 0.4*x + 0.6*y
            
            # -------------------------
            # RMS (polinomi in x,y)
            # -------------------------
            sigma1 = 1.0 + 0.2*x**2 + 0.1*y**2
            sigma2 = 0.8 + 0.1*x**2 + 0.1*y**2
            
            # -------------------------
            # Correlazione
            # -------------------------
            rho = 0.6
            
            # Gaussiane standard indipendenti
            print(f"Generating two Gaussians with {N} events for pair ({x:.2f},{y:.2f})...")
            z1 = np.random.normal(size=N)
            z2 = np.random.normal(size=N)
            
            # -------------------------
            # Loop Monte Carlo + Fill
            # -------------------------
            print(f"Monte Carlo generation of 2D correlated distribution...")
            for i in range(N):
                if i%5000 == 0:
                    print(f"\tSampling the Gaussians for event {i}")
                g1 = mu1 + sigma1 * z1[i]
                g2 = mu2 + sigma2 * (rho * z1[i] + np.sqrt(1 - rho**2) * z2[i])
                histos[-1].Fill(g1, g2)

            f.cd()
            histos[-1].Write()

    f.Close()

    print(f"Istogramma TH2D scritto in {output_file}")

if __name__ == "__main__":

    toy_sim = "toy_simulation.root"
    print(f"Simulate the 'simulation' and store in the file {toy_sim}")
    generate_toys(11,toy_sim,false)

    toy_data = "toy_data.root"
    print(f"Simulate the 'data' and store in the file {toy_data}")
    generate_toys(5,toy_data,True)

    print("DONE")
    
