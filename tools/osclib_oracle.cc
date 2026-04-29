// One-shot oracle: print P(numu->nue) and P(numu->numu) vs energy for a grid
// of NSI scenarios at NOvA geometry, using OscCalcPMNS_NSI.
//
// Build (inside a CVMFS/UPS environment with OscLib available):
//   g++ -std=c++17 -o osclib_oracle osclib_oracle.cc \
//       $(pkg-config --cflags --libs osclib) $(root-config --cflags --libs)
//
// Redirect stdout to  ../tests/test_vs_osclib.csv  and commit it.

#include "OscLib/OscCalcPMNS_NSI.h"
#include <cmath>
#include <iostream>
#include <vector>
#include <iomanip>

int main() {
    const double L     = 810.0;   // km  (NOvA)
    const double rho   = 2.79;    // g/cm³
    const double dm21  = 7.42e-5;
    const double dm31  = 2.515e-3;
    const double th12  = std::asin(std::sqrt(0.307));     // PDG-2024 sin²θ₁₂
    const double th13  = std::asin(std::sqrt(0.0220));    // sin²θ₁₃ = 0.0220
    const double th23  = std::asin(std::sqrt(0.545));     // sin²θ₂₃
    const double dcp   = -1.601;  // radians

    // NSI scenarios: {eps_emu_mag, eps_emu_phase, eps_etau_mag, eps_etau_phase,
    //                 eps_mutau_mag, eps_mutau_phase, label}
    struct Scenario {
        double eps_emu_mag, delta_emu;
        double eps_etau_mag, delta_etau;
        double eps_mutau_mag, delta_mutau;
        const char* label;
    };
    std::vector<Scenario> scenarios = {
        {0.0, 0.0,  0.0, 0.0,  0.0, 0.0,     "standard"},
        {0.1, 0.0,  0.0, 0.0,  0.0, 0.0,     "eps_emu_0.1_phase_0"},
        {0.1, -M_PI_2, 0.0, 0.0, 0.0, 0.0,   "eps_emu_0.1_phase_-pi2"},
        {0.0, 0.0,  0.1, 0.0,  0.0, 0.0,     "eps_etau_0.1_phase_0"},
        {0.0, 0.0,  0.0, 0.0,  0.1, 0.0,     "eps_mutau_0.1_phase_0"},
    };

    std::vector<double> energies = {0.5, 1.0, 1.5, 1.9, 2.5, 3.0, 4.0, 5.0};

    std::cout << std::fixed << std::setprecision(10);
    std::cout << "scenario,E_GeV,Pme,Pmm\n";

    for (const auto& sc : scenarios) {
        osc::OscCalcPMNS_NSI calc;
        calc.SetL(L);
        calc.SetRho(rho);
        calc.SetDmsq21(dm21);
        calc.SetDmsq32(dm31 - dm21);   // OscLib uses Δm²₃₂
        calc.SetTh12(th12);
        calc.SetTh13(th13);
        calc.SetTh23(th23);
        calc.SetdCP(dcp);
        calc.SetEps_emu  (sc.eps_emu_mag);
        calc.SetDelta_emu(sc.delta_emu);
        calc.SetEps_etau  (sc.eps_etau_mag);
        calc.SetDelta_etau(sc.delta_etau);
        calc.SetEps_mutau  (sc.eps_mutau_mag);
        calc.SetDelta_mutau(sc.delta_mutau);

        for (double E : energies) {
            double Pme = calc.P(14, 12, E);   // numu -> nue
            double Pmm = calc.P(14, 14, E);   // numu -> numu
            std::cout << sc.label << "," << E << "," << Pme << "," << Pmm << "\n";
        }
    }
    return 0;
}
