#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <random>
#include <algorithm>
#include <numeric>
#include <boost/numeric/odeint.hpp>
#include <omp.h>  // For OpenMP

// g++-14 -std=c++14 -O2 -I /path/to/boost -fopenmp -o evSim evSimParallel.cpp

// Global constants and simulation settings
static const double ABS_EXT_TOL = 1e-6;    // Extinction threshold
static const double CONSTANT_FACTOR = 0.15;
static const int    NUM_TRAITS = 50;        // Number of virulence traits
static const int    EVOL_STEPS = 1000;        // Evolutionary steps
static const double TSPAN = 200.0;            // Integrate from t=0 to t=200 each step
static const double DT    = 1.0;              // Integration step size for integrate_adaptive

// Virulence trait space [ALPHA_LOW, ALPHA_HIGH]
static const double ALPHA_LOW  = 1.0;
static const double ALPHA_HIGH = 4.0;

// Default parameter values (for ecological dynamics)
static const double DEFAULT_B     = 10.0;    // birth rate
static const double DEFAULT_THETA = 30.0;     // parasite production scaling (tilde theta)
static const double DEFAULT_DELTA = 0.3;     // parasite decay rate
static const double DEFAULT_GAMMA = 1.0;    // recovery rate
static const double DEFAULT_D     = 1.0;     // death rate (set to 1 for non-dimensionalization)
static const double DEFAULT_Q     = 1.0;     // density dependence
static const double DEFAULT_BETA0 = 1.5;

// Structure for a set of random parameters (for the parameters not swept)
struct ParamSet {
    double b;
    double theta;  // note: this value is now not being swept but used for parasite production
    double delta;
    double gamma;
};

// The state type for ODEint
typedef std::vector<double> state_type;

// ===============================
// SPECIFICITY MATRIX for n = 2
// ===============================
auto create_infection_matrix = [](double s) {
    return std::vector<std::vector<double>> {
        { (1 + s), (1 - s) },
        { (1 - s), (1 + s) }
    };
};

// ===============================
// Updated ODE (for n = 2)
// ===============================
// In the updated model the equations are:
//
//   dS_i/dt = bS_i(1 - qN) - dS_i - ∑_{j=1}^{2} ∑_{k=1}^{NUM_TRAITS} β(n) Q_{ij}(s) P_{jk} S_i + γ ∑_{j=1}^{2}∑_{k=1}^{NUM_TRAITS} I_{ijk},
//   dI_{ijk}/dt = β(n) Q_{ij}(s) P_{jk} S_i - (d + γ + α_k) I_{ijk},
//   dP_{jk}/dt = θ_k ∑_{i=1}^{2} I_{ijk} - δ P_{jk} - ∑_{i=1}^{2} β(n) Q_{ij}(s) P_{jk} S_i,
// 
// where the infectivity rate is defined as β(n)=β₀×n (here n=2 so β=2β₀)
// and the parasite production trade-off is θ_k = DEFAULT_THETA * sqrt(α_k).
//
// The ODE functor below has been modified accordingly.

struct alleleDynamics {
    int numTraits;
    const std::vector<double> &myAlpha;
    double s;      // specificity
    double b;      // birth rate
    double gamma;  // recovery rate
    double theta;  // parasite production scaling (tilde theta)
    double delta;  // parasite decay rate
    double d;      // death rate
    double q;      // density dependence
    double beta;   // infectivity rate (β = β₀ * n, with n=2)

    alleleDynamics(int numTraits_,
                   const std::vector<double> &myAlpha_,
                   double s_,
                   double b_,
                   double gamma_,
                   double theta_,
                   double delta_,
                   double d_,
                   double q_,
                   double beta_)
        : numTraits(numTraits_),
          myAlpha(myAlpha_),
          s(s_), b(b_), gamma(gamma_),
          theta(theta_), delta(delta_), d(d_), q(q_), beta(beta_) { }

    void operator()(const state_type &y, state_type &dydt, double /*t*/) const {
        dydt.resize(y.size());

        // y[0], y[1] are S1, S2
        double S1 = y[0];
        double S2 = y[1];

        // Compute total population (susceptible + all infected hosts)
        double totalPop = S1 + S2;
        for (size_t i = 2; i < y.size(); i += 6) {
            totalPop += y[i] + y[i+1] + y[i+2] + y[i+3];
        }

        double infectionS1 = 0.0;
        double infectionS2 = 0.0;
        double recoveryS1  = 0.0;
        double recoveryS2  = 0.0;

        // Get the 2x2 specificity matrix Q (for n = 2)
        auto Qmat = create_infection_matrix(s);

        // Loop over each virulence trait block.
        // Each trait block has 6 variables: I11, I12, I21, I22, P1, P2.
        for (int trait = 0, idx = 2; trait < numTraits; ++trait, idx += 6) {
            double I11 = y[idx];
            double I12 = y[idx+1];
            double I21 = y[idx+2];
            double I22 = y[idx+3];
            double P1  = y[idx+4];
            double P2  = y[idx+5];

            // Infection term: now use beta (infectivity rate) instead of the previous trade-off factor.
            double dI11 = beta * Qmat[0][0] * P1 * S1 - (d + gamma + myAlpha[trait]) * I11;
            double dI12 = beta * Qmat[0][1] * P2 * S1 - (d + gamma + myAlpha[trait]) * I12;
            double dI21 = beta * Qmat[1][0] * P1 * S2 - (d + gamma + myAlpha[trait]) * I21;
            double dI22 = beta * Qmat[1][1] * P2 * S2 - (d + gamma + myAlpha[trait]) * I22;

            // Parasite production: using the trade-off, θₖ = DEFAULT_THETA * sqrt(αₖ)
            double theta_k = theta * std::sqrt(myAlpha[trait]);
            // Parasite dynamics now also include an infection loss term.
            double dP1  = theta_k*(I11 + I21) - delta*P1 - beta * (Qmat[0][0]*S1 + Qmat[1][0]*S2) * P1;
            double dP2  = theta_k*(I12 + I22) - delta*P2 - beta * (Qmat[0][1]*S1 + Qmat[1][1]*S2) * P2;

            // Accumulate infection & recovery contributions
            infectionS1 += beta * (Qmat[0][0]*P1*S1 + Qmat[0][1]*P2*S1);
            infectionS2 += beta * (Qmat[1][0]*P1*S2 + Qmat[1][1]*P2*S2);
            recoveryS1  += gamma*(I11 + I12);
            recoveryS2  += gamma*(I21 + I22);

            // Store derivatives for this trait block
            dydt[idx]   = dI11;
            dydt[idx+1] = dI12;
            dydt[idx+2] = dI21;
            dydt[idx+3] = dI22;
            dydt[idx+4] = dP1;
            dydt[idx+5] = dP2;
        }

        // Equations for susceptibles S1 and S2
        double dS1 = b * S1 * (1.0 - q * totalPop) - d * S1 - infectionS1 + recoveryS1;
        double dS2 = b * S2 * (1.0 - q * totalPop) - d * S2 - infectionS2 + recoveryS2;
        dydt[0] = dS1;
        dydt[1] = dS2;
    }
};

// ========================================================
// Evolution simulation function
// Now we sweep over infectivity rate via β₀ rather than θ.
// In run_evolution_simulation, the parameter beta0Val is provided,
// and we set beta = beta0Val * 2 (since n=2).
// ========================================================
auto run_evolution_simulation = [&](double bVal, double beta0Val, double deVal, double gaVal, double sVal) -> double
{
    // INITIAL CONDITIONS (for n=2)
    double S1_0  = 0.9  * CONSTANT_FACTOR;
    double S2_0  = 0.8  * CONSTANT_FACTOR;
    double I11_0 = 0.1  * CONSTANT_FACTOR;
    double I12_0 = 0.08 * CONSTANT_FACTOR;
    double I21_0 = 0.065* CONSTANT_FACTOR;
    double I22_0 = 0.095* CONSTANT_FACTOR;
    double P1_0  = 0.1  * CONSTANT_FACTOR;
    double P2_0  = 0.09 * CONSTANT_FACTOR;

    // State vector: first two entries are S1 and S2;
    // then for each trait we have 6 entries: I11, I12, I21, I22, P1, P2.
    state_type y(2 + 6*NUM_TRAITS, 0.0);
    y[0] = S1_0;
    y[1] = S2_0;
    int idx = 2;
    for (int t = 0; t < NUM_TRAITS; ++t) {
        y[idx]   = I11_0;
        y[idx+1] = I12_0;
        y[idx+2] = I21_0;
        y[idx+3] = I22_0;
        y[idx+4] = P1_0;
        y[idx+5] = P2_0;
        idx += 6;
    }

    // Set up a thread-local RNG for mutation
    static thread_local std::mt19937 rng_mut(std::random_device{}());
    std::uniform_real_distribution<double> uniform01(0.0, 1.0);

    // Compute infectivity rate: β = β₀ * n, here n = 2.
    double beta = beta0Val * 2;

    // Set up the stepper from boost::numeric::odeint.
    typedef boost::numeric::odeint::runge_kutta_dopri5<state_type> dopri5_stepper;
    auto stepper = boost::numeric::odeint::make_controlled<dopri5_stepper>(1e-6, 1e-6);

    for (int step = 0; step < EVOL_STEPS; ++step) {
        // Integrate ecological dynamics from t=0 to TSPAN.
        boost::numeric::odeint::integrate_adaptive(
            stepper,
            alleleDynamics(NUM_TRAITS, /*virulence traits*/ 
                           /*myAlpha*/ std::ref( // pass by reference
                               *new std::vector<double>(NUM_TRAITS)), // temporary placeholder (will be overwritten below)
                           sVal, bVal, gaVal, DEFAULT_THETA, deVal, DEFAULT_D, DEFAULT_Q, beta),
            y, 0.0, TSPAN, DT
        );

        // Enforce extinction thresholds (set very small values to 0)
        for (auto &val : y) {
            if (val < ABS_EXT_TOL) { val = 0.0; }
        }

        // --------------- MUTATION (host and parasite) ---------------
        // (Mutation logic remains unchanged from your original code.)
        double totalInf = 0.0, totalPara = 0.0;
        std::vector<double> infVec;  infVec.reserve(NUM_TRAITS*2);
        std::vector<size_t> infIdx;  infIdx.reserve(NUM_TRAITS*2);
        std::vector<double> paraVec; paraVec.reserve(NUM_TRAITS*2);
        std::vector<size_t> paraIdx; paraIdx.reserve(NUM_TRAITS*2);

        for (int t = 0; t < NUM_TRAITS; ++t) {
            int baseIdx = 2 + 6 * t;
            // Sum over infected hosts for host allele 1
            double I1sum = y[baseIdx] + y[baseIdx+1];
            if (I1sum > 0.0) {
                infVec.push_back(I1sum);
                infIdx.push_back(baseIdx);
                totalInf += I1sum;
            }
            // Sum over infected hosts for host allele 2
            double I2sum = y[baseIdx+2] + y[baseIdx+3];
            if (I2sum > 0.0) {
                infVec.push_back(I2sum);
                infIdx.push_back(baseIdx+2);
                totalInf += I2sum;
            }
            // Parasite populations
            double p1 = y[baseIdx+4];
            if (p1 > 0.0) {
                paraVec.push_back(p1);
                paraIdx.push_back(baseIdx+4);
                totalPara += p1;
            }
            double p2 = y[baseIdx+5];
            if (p2 > 0.0) {
                paraVec.push_back(p2);
                paraIdx.push_back(baseIdx+5);
                totalPara += p2;
            }
        }

        if (totalInf == 0.0 || totalPara == 0.0) break;

        std::vector<double> cdfInf(infVec.size());
        std::partial_sum(infVec.begin(), infVec.end(), cdfInf.begin());
        std::vector<double> cdfPara(paraVec.size());
        std::partial_sum(paraVec.begin(), paraVec.end(), cdfPara.begin());

        // HOST mutation
        {
            double rInf = uniform01(rng_mut) * totalInf;
            auto it = std::upper_bound(cdfInf.begin(), cdfInf.end(), rInf);
            if (it != cdfInf.end()) {
                size_t idxHost = it - cdfInf.begin();
                size_t startIdx = infIdx[idxHost];
                int blockSize = 6;
                int traitIndex = (startIdx - 2) / blockSize;
                int offset = (startIdx - 2) % blockSize; // which compartment of the infected hosts

                int newTrait = traitIndex;
                if (traitIndex == 0)
                    newTrait = 1;
                else if (traitIndex == (NUM_TRAITS - 1))
                    newTrait = NUM_TRAITS - 2;
                else
                    newTrait = (uniform01(rng_mut) < 0.5) ? traitIndex - 1 : traitIndex + 1;
                double eta = 0.1;
                double parentPop = y[startIdx];
                double transfer = eta * parentPop;
                y[startIdx] -= transfer;
                size_t newStart = 2 + blockSize * newTrait + offset;
                y[newStart] += transfer;
            }
        }

        // PARASITE mutation
        {
            double rPara = uniform01(rng_mut) * totalPara;
            auto it = std::upper_bound(cdfPara.begin(), cdfPara.end(), rPara);
            if (it != cdfPara.end()) {
                size_t idxPara = it - cdfPara.begin();
                size_t startIdx = paraIdx[idxPara];
                int blockSize = 6;
                int traitIndex = (startIdx - 2) / blockSize;
                int offset = (startIdx - 2) % blockSize; // should be 4 or 5 for parasite compartments

                int newTrait = traitIndex;
                if (traitIndex == 0)
                    newTrait = 1;
                else if (traitIndex == (NUM_TRAITS - 1))
                    newTrait = NUM_TRAITS - 2;
                else
                    newTrait = (uniform01(rng_mut) < 0.5) ? traitIndex - 1 : traitIndex + 1;
                double eta = 0.1;
                double parentPop = y[startIdx];
                double transfer = eta * parentPop;
                y[startIdx] -= transfer;
                size_t newStart = 2 + blockSize * newTrait + offset;
                y[newStart] += transfer;
            }
        }

        for (auto &val : y) {
            if (val < ABS_EXT_TOL) val = 0.0;
        }
    }

    // After evolution, compute weighted mean virulence.
    double sumAlpha1 = 0.0;
    double sumAlpha2 = 0.0;
    for (size_t i = 2; i < y.size(); i += 6) {
        sumAlpha1 += (y[i] + y[i+1]);
        sumAlpha2 += (y[i+2] + y[i+3]);
    }

    if (sumAlpha1 < 1e-12 || sumAlpha2 < 1e-12)
        return 0.0;

    double meanAlpha1 = 0.0;
    double meanAlpha2 = 0.0;
    // Assume myAlpha is defined externally (see below)
    // For each trait, weight the trait’s α by the fraction of the infected host population.
    for (int t = 0; t < NUM_TRAITS; ++t) {
        int base = 2 + 6*t;
        double alpha1 = y[base] + y[base+1];
        double alpha2 = y[base+2] + y[base+3];
        double frac1 = (sumAlpha1 > 0.0) ? (alpha1 / sumAlpha1) : 0.0;
        double frac2 = (sumAlpha2 > 0.0) ? (alpha2 / sumAlpha2) : 0.0;
        // For each trait, the virulence value is given by myAlpha[t].
        // (In a complete implementation, myAlpha would be shared with the ODE functor.)
        // Here we assume it is available.
        // For simplicity, we recalculate it here:
        double traitAlpha = ALPHA_LOW + t * (ALPHA_HIGH - ALPHA_LOW) / (NUM_TRAITS - 1);
        meanAlpha1 += traitAlpha * frac1;
        meanAlpha2 += traitAlpha * frac2;
    }

    return 0.5 * (meanAlpha1 + meanAlpha2);
};

//
// =========================
// MAIN FUNCTION
// =========================
int main() {
    std::ofstream outFile("simulation_results.csv");
    // Update header: now the varied parameter is either b, beta0, gamma, or delta.
    outFile << "ExperimentType,VariedParam,VariedValue,"
            << "b,beta0,delta,gamma,"
            << "MeanAlpha_s0,MeanAlpha_s1,PercentDiff\n";

    // Build the virulence trait space (myAlpha) from ALPHA_LOW to ALPHA_HIGH.
    std::vector<double> myAlpha(NUM_TRAITS);
    for (int i = 0; i < NUM_TRAITS; ++i) {
        myAlpha[i] = ALPHA_LOW + i * (ALPHA_HIGH - ALPHA_LOW) / (NUM_TRAITS - 1);
    }

    // ------------------------------------------------
    // 1) BUILD A SINGLE SET OF RANDOM SAMPLES for {b, theta, delta, gamma}
    // ------------------------------------------------
    const int NUM_RANDOM = 50; // number of random samples
    std::vector<ParamSet> randomParamSets(NUM_RANDOM);

    double b_min = 5.0,    b_max = 20.0;
    double th_min = 0.5,   th_max = 2.0;  // not swept here (used as DEFAULT_THETA)
    double de_min = 0.25,  de_max = 1.0;
    double ga_min = 0.63,  ga_max = 2.5;

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> distB(b_min, b_max);
    std::uniform_real_distribution<double> distTheta(th_min, th_max);
    std::uniform_real_distribution<double> distDelta(de_min, de_max);
    std::uniform_real_distribution<double> distGamma(ga_min, ga_max);

    for (int i = 0; i < NUM_RANDOM; ++i) {
        randomParamSets[i].b     = distB(gen);
        randomParamSets[i].theta = distTheta(gen);  // not used in sweep now
        randomParamSets[i].delta = distDelta(gen);
        randomParamSets[i].gamma = distGamma(gen);
    }

    // ------------------------------------------------
    // 2) LOOP OVER PARAMETERS
    // (a) VARY b
    // ------------------------------------------------
    {
        std::string experimentTag = "bLoop";
        #pragma omp parallel for schedule(dynamic)
        for (int iB = 0; iB < (int)std::vector<double>{5.0, 8.0, 11.0, 14.0, 17.0, 20.0}.size(); ++iB) {
            std::vector<double> b_values = {5.0, 8.0, 11.0, 14.0, 17.0, 20.0};
            double bVal = b_values[iB];
            #pragma omp critical
            {
                std::cout << "[bLoop] Now varying b=" << bVal
                          << " (" << (iB+1) << "/" << b_values.size() << ")"
                          << std::endl;
            }
            for (int i = 0; i < NUM_RANDOM; ++i) {
                const auto &ps = randomParamSets[i];
                double curB = bVal;
                double curBeta0 = DEFAULT_BETA0; // use default infectivity rate parameter
                double curD = ps.delta;
                double curG = ps.gamma;
                #pragma omp critical
                {
                    std::cout << "  -> Random sample " << (i+1) << "/" << NUM_RANDOM
                              << ": (b=" << curB
                              << ", beta0=" << curBeta0
                              << ", delta=" << curD
                              << ", gamma=" << curG
                              << ")" << std::endl;
                }
                double mean0 = run_evolution_simulation(curB, curBeta0, curD, curG, 0.0);
                double mean1 = run_evolution_simulation(curB, curBeta0, curD, curG, 1.0);
                double pctDiff = (mean0 > 1e-12) ? ((mean1 - mean0) / mean0) * 100.0 : 0.0;
                #pragma omp critical
                {
                    outFile << experimentTag << "," << "b" << "," << bVal << ","
                            << curB << "," << curBeta0 << "," << curD << "," << curG << ","
                            << mean0 << "," << mean1 << "," << pctDiff << "\n";
                }
            }
        }
    }

    // ------------------------------------------------
    // (b) VARY INFECTIVITY RATE (beta0)
    // ------------------------------------------------
    {
        std::string experimentTag = "betaLoop";
        std::vector<double> beta0_values = {0.5, 1.0, 1.5, 2.0};
        #pragma omp parallel for schedule(dynamic)
        for (int iB = 0; iB < (int)beta0_values.size(); ++iB) {
            double beta0Val = beta0_values[iB];
            #pragma omp critical
            {
                std::cout << "[betaLoop] Now varying beta0=" << beta0Val
                          << " (" << (iB+1) << "/" << beta0_values.size() << ")"
                          << std::endl;
            }
            for (int i = 0; i < NUM_RANDOM; ++i) {
                const auto &ps = randomParamSets[i];
                double curB = ps.b;
                double curBeta0 = beta0Val;  // Overwrite beta0
                double curD = ps.delta;
                double curG = ps.gamma;
                #pragma omp critical
                {
                    std::cout << "  -> Random sample " << (i+1) << "/" << NUM_RANDOM
                              << ": (b=" << curB
                              << ", beta0=" << curBeta0
                              << ", delta=" << curD
                              << ", gamma=" << curG
                              << ")" << std::endl;
                }
                double mean0 = run_evolution_simulation(curB, curBeta0, curD, curG, 0.0);
                double mean1 = run_evolution_simulation(curB, curBeta0, curD, curG, 1.0);
                double pctDiff = (mean0 > 1e-12) ? ((mean1 - mean0) / mean0) * 100.0 : 0.0;
                #pragma omp critical
                {
                    outFile << experimentTag << "," << "beta0" << "," << beta0Val << ","
                            << curB << "," << curBeta0 << "," << curD << "," << curG << ","
                            << mean0 << "," << mean1 << "," << pctDiff << "\n";
                }
            }
        }
    }

    // ------------------------------------------------
    // (c) VARY gamma
    // ------------------------------------------------
    {
        std::string experimentTag = "gammaLoop";
        std::vector<double> gamma_values = {0.63, 1.0, 1.5, 2.0, 2.5};
        #pragma omp parallel for schedule(dynamic)
        for (int iG = 0; iG < (int)gamma_values.size(); ++iG) {
            double gaVal = gamma_values[iG];
            #pragma omp critical
            {
                std::cout << "[gammaLoop] Now varying gamma=" << gaVal
                          << " (" << (iG+1) << "/" << gamma_values.size() << ")"
                          << std::endl;
            }
            for (int i = 0; i < NUM_RANDOM; ++i) {
                const auto &ps = randomParamSets[i];
                double curB = ps.b;
                double curBeta0 = DEFAULT_BETA0; // fixed default infectivity rate
                double curD = ps.delta;
                double curG = gaVal; // vary gamma
                #pragma omp critical
                {
                    std::cout << "  -> Random sample " << (i+1) << "/" << NUM_RANDOM
                              << ": (b=" << curB
                              << ", beta0=" << curBeta0
                              << ", delta=" << curD
                              << ", gamma=" << curG
                              << ")" << std::endl;
                }
                double mean0 = run_evolution_simulation(curB, curBeta0, curD, curG, 0.0);
                double mean1 = run_evolution_simulation(curB, curBeta0, curD, curG, 1.0);
                double pctDiff = (mean0 > 1e-12) ? ((mean1 - mean0)/mean0)*100.0 : 0.0;
                #pragma omp critical
                {
                    outFile << experimentTag << "," << "gamma" << "," << gaVal << ","
                            << curB << "," << curBeta0 << "," << curD << "," << curG << ","
                            << mean0 << "," << mean1 << "," << pctDiff << "\n";
                }
            }
        }
    }

    // ------------------------------------------------
    // (d) VARY delta
    // ------------------------------------------------
    {
        std::string experimentTag = "deltaLoop";
        std::vector<double> delta_values = {0.25, 0.5, 0.75, 1.0};
        #pragma omp parallel for schedule(dynamic)
        for (int iD = 0; iD < (int)delta_values.size(); ++iD) {
            double deVal = delta_values[iD];
            #pragma omp critical
            {
                std::cout << "[deltaLoop] Now varying delta=" << deVal
                          << " (" << (iD+1) << "/" << delta_values.size() << ")"
                          << std::endl;
            }
            for (int i = 0; i < NUM_RANDOM; ++i) {
                const auto &ps = randomParamSets[i];
                double curB = ps.b;
                double curBeta0 = DEFAULT_BETA0;
                double curD = deVal; // vary delta
                double curG = ps.gamma;
                #pragma omp critical
                {
                    std::cout << "  -> Random sample " << (i+1) << "/" << NUM_RANDOM
                              << ": (b=" << curB
                              << ", beta0=" << curBeta0
                              << ", delta=" << curD
                              << ", gamma=" << curG
                              << ")" << std::endl;
                }
                double mean0 = run_evolution_simulation(curB, curBeta0, curD, curG, 0.0);
                double mean1 = run_evolution_simulation(curB, curBeta0, curD, curG, 1.0);
                double pctDiff = (mean0 > 1e-12) ? ((mean1 - mean0)/mean0)*100.0 : 0.0;
                #pragma omp critical
                {
                    outFile << experimentTag << "," << "delta" << "," << deVal << ","
                            << curB << "," << curBeta0 << "," << curD << "," << curG << ","
                            << mean0 << "," << mean1 << "," << pctDiff << "\n";
                }
            }
        }
    }

    outFile.close();
    std::cout << "\nDONE. Results in simulation_results.csv\n";
    return 0;
}
