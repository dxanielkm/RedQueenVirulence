#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <random>
#include <algorithm>
#include <numeric>
#include <boost/numeric/odeint.hpp>
#include <omp.h>  // For OpenMP

// Compile with (example):
// g++-14 -std=c++14 -O2 -I /path/to/boost -fopenmp -o evSim evSimParallel.cpp
// On macOS, make sure you're using a compiler that supports OpenMP.

static const double ABS_EXT_TOL = 1e-6;    // Extinction threshold
static const double CONSTANT_FACTOR = 0.15;
static const int    NUM_TRAITS = 50;       // Number of virulence traits
static const int    EVOL_STEPS = 1000;     // Evolutionary steps
static const double TSPAN = 200.0;         // Integrate from t=0 to t=200 each step
static const double DT    = 1.0;           // Integration step size for integrate_adaptive

// Virulence trait space [ALPHA_LOW, ALPHA_HIGH]
static const double ALPHA_LOW  = 1.0;
static const double ALPHA_HIGH = 4.0;

// Default parameter values
static const double DEFAULT_B     = 10.0;
static const double DEFAULT_THETA = 1.0;
static const double DEFAULT_DELTA = 0.5;
static const double DEFAULT_GAMMA = 1.25;
static const double DEFAULT_D     = 1.0;  // death rate
static const double DEFAULT_Q     = 1.0;  // density dependence

struct ParamSet {
    double b;
    double theta;
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
// ODE, n = 2
// ===============================
struct alleleDynamics {
    int numTraits;
    const std::vector<double> &myAlpha;
    double s;      // specificity
    double b;      // birth
    double gamma;  // recovery
    double theta;  // parasite production
    double delta;  // parasite decay
    double d;      // death rate
    double q;      // density dependence

    alleleDynamics(int numTraits_,
                   const std::vector<double> &myAlpha_,
                   double s_,
                   double b_,
                   double gamma_,
                   double theta_,
                   double delta_,
                   double d_,
                   double q_)
        : numTraits(numTraits_),
          myAlpha(myAlpha_),
          s(s_), b(b_), gamma(gamma_),
          theta(theta_), delta(delta_), d(d_), q(q_) { }

    void operator()(const state_type &y, state_type &dydt, double /*t*/) const {
        dydt.resize(y.size());

        // y[0], y[1] are S1, S2
        double S1 = y[0];
        double S2 = y[1];

        // Total population
        double totalPop = S1 + S2;
        for (size_t i = 2; i < y.size(); i += 6) {
            // I11, I12, I21, I22
            totalPop += y[i] + y[i+1] + y[i+2] + y[i+3];
        }

        double infectionS1 = 0.0;
        double infectionS2 = 0.0;
        double recoveryS1  = 0.0;
        double recoveryS2  = 0.0;

        // 2x2 specificity matrix
        auto Qmat = create_infection_matrix(s);

        // Fill derivatives for infected compartments
        for(int trait = 0, idx = 2; trait < numTraits; ++trait, idx += 6) {
            double I11 = y[idx];
            double I12 = y[idx+1];
            double I21 = y[idx+2];
            double I22 = y[idx+3];
            double P1  = y[idx+4];
            double P2  = y[idx+5];

            // Example trade-off: beta_k = 20 sqrt(alpha_k / 10)
            double alpha_sqrt = 20.0 * std::sqrt(myAlpha[trait]/10.0);

            // ODE for I11, I12, I21, I22
            double dI11 = Qmat[0][0]*alpha_sqrt*P1*S1 - (d + gamma + myAlpha[trait])*I11;
            double dI12 = Qmat[0][1]*alpha_sqrt*P2*S1 - (d + gamma + myAlpha[trait])*I12;
            double dI21 = Qmat[1][0]*alpha_sqrt*P1*S2 - (d + gamma + myAlpha[trait])*I21;
            double dI22 = Qmat[1][1]*alpha_sqrt*P2*S2 - (d + gamma + myAlpha[trait])*I22;

            double dP1  = theta*(I11 + I21) - delta*P1;
            double dP2  = theta*(I12 + I22) - delta*P2;

            // Accumulate infection & recovery
            infectionS1 += alpha_sqrt*(Qmat[0][0]*P1*S1 + Qmat[0][1]*P2*S1);
            infectionS2 += alpha_sqrt*(Qmat[1][0]*P1*S2 + Qmat[1][1]*P2*S2);
            recoveryS1  += gamma*(I11 + I12);
            recoveryS2  += gamma*(I21 + I22);

            // Store in dydt
            dydt[idx]   = dI11;
            dydt[idx+1] = dI12;
            dydt[idx+2] = dI21;
            dydt[idx+3] = dI22;
            dydt[idx+4] = dP1;
            dydt[idx+5] = dP2;
        }

        // ODE for S1, S2
        double dS1 = b*S1*(1.0 - q*totalPop) - d*S1 - infectionS1 + recoveryS1;
        double dS2 = b*S2*(1.0 - q*totalPop) - d*S2 - infectionS2 + recoveryS2;

        dydt[0] = dS1;
        dydt[1] = dS2;
    }
};

int main() {

    std::ofstream outFile("simulation_results.csv");
    outFile << "ExperimentType,VariedParam,VariedValue,"
            << "b,theta,delta,gamma,"
            << "MeanAlpha_s0,MeanAlpha_s1,PercentDiff\n";

    // Build virulence traits from ALPHA_LOW to ALPHA_HIGH
    std::vector<double> myAlpha(NUM_TRAITS);
    for(int i=0; i<NUM_TRAITS; ++i) {
        myAlpha[i] = ALPHA_LOW + i*(ALPHA_HIGH - ALPHA_LOW)/(NUM_TRAITS-1);
    }

    // ------------------------------------------------
    // 1) BUILD A SINGLE SET OF RANDOM SAMPLES
    //    for {b, theta, delta, gamma}
    // ------------------------------------------------
    const int NUM_RANDOM = 50; // number of random samples
    std::vector<ParamSet> randomParamSets(NUM_RANDOM);

    // Bounds for each parameter when randomly sampled
    double b_min = 5.0,    b_max = 20.0;
    double th_min = 0.5,   th_max = 2.0;
    double de_min = 0.25,  de_max = 1.0;
    double ga_min = 0.63,  ga_max = 2.5;

    // Random number generator
    std::random_device rd;
    std::mt19937 gen(rd());

    // Uniform distributions
    std::uniform_real_distribution<double> distB(b_min, b_max);
    std::uniform_real_distribution<double> distTheta(th_min, th_max);
    std::uniform_real_distribution<double> distDelta(de_min, de_max);
    std::uniform_real_distribution<double> distGamma(ga_min, ga_max);

    // Fill randomParamSets
    for(int i=0; i<NUM_RANDOM; ++i) {
        randomParamSets[i].b      = distB(gen);
        randomParamSets[i].theta  = distTheta(gen);
        randomParamSets[i].delta  = distDelta(gen);
        randomParamSets[i].gamma  = distGamma(gen);
    }

    // ------------------------------------------------
    // 2) FUNCTION FOR RUNNING SIMULATION
    // ------------------------------------------------
    auto run_evolution_simulation = [&](double bVal, double thVal, double deVal, double gaVal,
                                        double sVal) -> double
    {
        // INITIAL CONDITIONS
        double S1_0  = 0.9  * CONSTANT_FACTOR;
        double S2_0  = 0.8  * CONSTANT_FACTOR;
        double I11_0 = 0.1  * CONSTANT_FACTOR;
        double I12_0 = 0.08 * CONSTANT_FACTOR;
        double I21_0 = 0.065* CONSTANT_FACTOR;
        double I22_0 = 0.095* CONSTANT_FACTOR;
        double P1_0  = 0.1  * CONSTANT_FACTOR;
        double P2_0  = 0.09 * CONSTANT_FACTOR;

        // State vector
        state_type y(2 + 6*NUM_TRAITS, 0.0);
        // Fill initial
        y[0] = S1_0;
        y[1] = S2_0;
        int idx = 2;
        for(int t=0; t<NUM_TRAITS; ++t) {
            y[idx]   = I11_0;
            y[idx+1] = I12_0;
            y[idx+2] = I21_0;
            y[idx+3] = I22_0;
            y[idx+4] = P1_0;
            y[idx+5] = P2_0;
            idx += 6;
        }

        // Use a static thread_local RNG so each thread has its own generator
        static thread_local std::mt19937 rng_mut(rd());
        std::uniform_real_distribution<double> uniform01(0.0, 1.0);

        // Stepper
        typedef boost::numeric::odeint::runge_kutta_dopri5<state_type> dopri5_stepper;
        auto stepper = boost::numeric::odeint::make_controlled<dopri5_stepper>(1e-6, 1e-6);

        for(int step=0; step<EVOL_STEPS; ++step) {
            // Integrate
            boost::numeric::odeint::integrate_adaptive(
                stepper,
                alleleDynamics(NUM_TRAITS, myAlpha, sVal,
                               bVal, gaVal, thVal, deVal,
                               DEFAULT_D, DEFAULT_Q),
                y, 0.0, TSPAN, DT
            );

            // Extinction threshold
            for(auto &val : y) {
                if(val < ABS_EXT_TOL) val = 0.0;
            }

            // ---------------- MUTATION ----------------
            // 1) Sum infected & parasite
            double totalInf = 0.0, totalPara = 0.0;
            std::vector<double> infVec;  infVec.reserve(NUM_TRAITS*2);
            std::vector<size_t> infIdx;  infIdx.reserve(NUM_TRAITS*2);
            std::vector<double> paraVec; paraVec.reserve(NUM_TRAITS*2);
            std::vector<size_t> paraIdx; paraIdx.reserve(NUM_TRAITS*2);

            for(int t=0; t<NUM_TRAITS; ++t) {
                int baseIdx = 2 + 6*t;
                // Host allele 1
                double I1sum = y[baseIdx] + y[baseIdx+1];
                if(I1sum > 0.0) {
                    infVec.push_back(I1sum);
                    infIdx.push_back(baseIdx);
                    totalInf += I1sum;
                }
                // Host allele 2
                double I2sum = y[baseIdx+2] + y[baseIdx+3];
                if(I2sum > 0.0) {
                    infVec.push_back(I2sum);
                    infIdx.push_back(baseIdx+2);
                    totalInf += I2sum;
                }
                // Parasite allele 1
                double p1 = y[baseIdx+4];
                if(p1 > 0.0) {
                    paraVec.push_back(p1);
                    paraIdx.push_back(baseIdx+4);
                    totalPara += p1;
                }
                // Parasite allele 2
                double p2 = y[baseIdx+5];
                if(p2 > 0.0) {
                    paraVec.push_back(p2);
                    paraIdx.push_back(baseIdx+5);
                    totalPara += p2;
                }
            }

            // If extinct => stop
            if(totalInf == 0.0 || totalPara == 0.0) {
                break;
            }

            // Build CDF
            std::vector<double> cdfInf(infVec.size());
            std::partial_sum(infVec.begin(), infVec.end(), cdfInf.begin());
            std::vector<double> cdfPara(paraVec.size());
            std::partial_sum(paraVec.begin(), paraVec.end(), cdfPara.begin());

            // HOST mutation
            {
                double rInf = uniform01(rng_mut)*totalInf;
                auto it = std::upper_bound(cdfInf.begin(), cdfInf.end(), rInf);
                if(it != cdfInf.end()) {
                    size_t idxHost = it - cdfInf.begin();
                    size_t startIdx = infIdx[idxHost];

                    int blockSize  = 6;
                    int traitIndex = (startIdx - 2)/blockSize;
                    int offset     = (startIdx - 2)%blockSize; // which of I11, I12, I21, I22

                    // mutate trait up or down
                    int newTrait = traitIndex;
                    if(traitIndex == 0) {
                        newTrait = 1;
                    } else if(traitIndex == (NUM_TRAITS-1)) {
                        newTrait = NUM_TRAITS-2;
                    } else {
                        if(uniform01(rng_mut) < 0.5) newTrait = traitIndex-1;
                        else                         newTrait = traitIndex+1;
                    }
                    // Transfer fraction
                    double eta = 0.1;
                    double parentPop = y[startIdx];
                    double transfer  = eta * parentPop;
                    y[startIdx] -= transfer;

                    // add to new trait
                    size_t newStart = 2 + blockSize*newTrait + offset;
                    y[newStart] += transfer;
                }
            }

            // PARASITE mutation
            {
                double rPara = uniform01(rng_mut)*totalPara;
                auto it = std::upper_bound(cdfPara.begin(), cdfPara.end(), rPara);
                if(it != cdfPara.end()) {
                    size_t idxPara = it - cdfPara.begin();
                    size_t startIdx = paraIdx[idxPara];

                    int blockSize  = 6;
                    int traitIndex = (startIdx - 2)/blockSize;
                    int offset     = (startIdx - 2)%blockSize; // 4 or 5 for parasite?

                    int newTrait = traitIndex;
                    if(traitIndex == 0) {
                        newTrait = 1;
                    } else if(traitIndex == (NUM_TRAITS-1)) {
                        newTrait = NUM_TRAITS-2;
                    } else {
                        if(uniform01(rng_mut) < 0.5) newTrait = traitIndex-1;
                        else                         newTrait = traitIndex+1;
                    }
                    double eta = 0.1;
                    double parentPop = y[startIdx];
                    double transfer  = eta * parentPop;
                    y[startIdx] -= transfer;

                    size_t newStart = 2 + blockSize*newTrait + offset;
                    y[newStart] += transfer;
                }
            }

            // final check
            for(auto &val : y) {
                if(val < ABS_EXT_TOL) val=0.0;
            }
        }

        // After evolution, compute weighted mean alpha
        double sumAlpha1 = 0.0;
        double sumAlpha2 = 0.0;
        for(size_t i=2; i<y.size(); i+=6) {
            sumAlpha1 += (y[i] + y[i+1]);
            sumAlpha2 += (y[i+2] + y[i+3]);
        }

        if(sumAlpha1 < 1e-12 || sumAlpha2 < 1e-12) {
            // Everything extinct => mean alpha = 0
            return 0.0;
        }

        double meanAlpha1 = 0.0;
        double meanAlpha2 = 0.0;
        for(int t=0; t<NUM_TRAITS; ++t) {
            int base = 2 + 6*t;

            double alpha1 = y[base]   + y[base+1];
            double alpha2 = y[base+2] + y[base+3];

            double frac1 = (sumAlpha1 > 0.0) ? (alpha1 / sumAlpha1) : 0.0;
            double frac2 = (sumAlpha2 > 0.0) ? (alpha2 / sumAlpha2) : 0.0;

            meanAlpha1 += myAlpha[t]*frac1;
            meanAlpha2 += myAlpha[t]*frac2;
        }

        return (meanAlpha1 + meanAlpha2)*0.5; // average of meanAlpha1 & meanAlpha2
    };

    // ------------------------------------------------
    // 3) LOOPS OVER DISCRETE VALUES:
    //    a) b-values
    //    b) theta-values
    //    c) gamma-values
    //    d) delta-values
    // ------------------------------------------------

    std::vector<double> b_values     = {5.0, 8.0, 11.0, 14.0, 17.0, 20.0};
    std::vector<double> theta_values = {0.5, 1.0, 1.5, 2.0};
    std::vector<double> gamma_values = {0.63, 1.0, 1.5, 2.0, 2.5};
    std::vector<double> delta_values = {0.25, 0.5, 0.75, 1.0};

    // ==========================
    // (a) VARY B
    // ==========================
    {
        std::string experimentTag = "bLoop";
        // Parallelize the outer loop with OpenMP
        #pragma omp parallel for schedule(dynamic)
        for (int iB = 0; iB < (int)b_values.size(); ++iB) {
            double bVal = b_values[iB];

            // Print status in a critical section (to avoid jumbled console output).
            #pragma omp critical
            {
                std::cout << "[bLoop] Now varying b=" << bVal
                          << " (" << (iB+1) << "/" << b_values.size() << ")"
                          << std::endl;
            }

            for(int i=0; i<NUM_RANDOM; ++i) {
                const auto &ps = randomParamSets[i];
                double curB = bVal;   // Overwrite only b
                double curT = ps.theta;
                double curD = ps.delta;
                double curG = ps.gamma;

                // Show random sample status
                #pragma omp critical
                {
                    std::cout << "  -> Random sample " << (i+1) << "/" << NUM_RANDOM
                              << ": (b=" << curB
                              << ", theta=" << curT
                              << ", delta=" << curD
                              << ", gamma=" << curG
                              << ")" << std::endl;
                }

                // Run s=0 and s=1
                double mean0 = run_evolution_simulation(curB, curT, curD, curG, 0.0);
                double mean1 = run_evolution_simulation(curB, curT, curD, curG, 1.0);

                double pctDiff = 0.0;
                if(mean0 > 1e-12) {
                    pctDiff = ((mean1 - mean0)/mean0)*100.0;
                }

                // Write to file (also critical)
                #pragma omp critical
                {
                    outFile << experimentTag << "," << "b" << "," << bVal << ","
                            << curB << "," << curT << "," << curD << "," << curG << ","
                            << mean0 << "," << mean1 << "," << pctDiff << "\n";
                }
            }
        }
    }

    // ==========================
    // (b) VARY THETA
    // ==========================
    {
        std::string experimentTag = "thetaLoop";
        #pragma omp parallel for schedule(dynamic)
        for (int iT = 0; iT < (int)theta_values.size(); ++iT) {
            double thVal = theta_values[iT];

            #pragma omp critical
            {
                std::cout << "[thetaLoop] Now varying theta=" << thVal
                          << " (" << (iT+1) << "/" << theta_values.size() << ")"
                          << std::endl;
            }

            for(int i=0; i<NUM_RANDOM; ++i) {
                const auto &ps = randomParamSets[i];
                double curB = ps.b;
                double curT = thVal;  // Overwrite only theta
                double curD = ps.delta;
                double curG = ps.gamma;

                #pragma omp critical
                {
                    std::cout << "  -> Random sample " << (i+1) << "/" << NUM_RANDOM
                              << ": (b=" << curB
                              << ", theta=" << curT
                              << ", delta=" << curD
                              << ", gamma=" << curG
                              << ")" << std::endl;
                }

                double mean0 = run_evolution_simulation(curB, curT, curD, curG, 0.0);
                double mean1 = run_evolution_simulation(curB, curT, curD, curG, 1.0);

                double pctDiff = 0.0;
                if(mean0 > 1e-12) {
                    pctDiff = ((mean1 - mean0)/mean0)*100.0;
                }

                #pragma omp critical
                {
                    outFile << experimentTag << "," << "theta" << "," << thVal << ","
                            << curB << "," << curT << "," << curD << "," << curG << ","
                            << mean0 << "," << mean1 << "," << pctDiff << "\n";
                }
            }
        }
    }

    // ==========================
    // (c) VARY GAMMA
    // ==========================
    {
        std::string experimentTag = "gammaLoop";
        #pragma omp parallel for schedule(dynamic)
        for (int iG = 0; iG < (int)gamma_values.size(); ++iG) {
            double gaVal = gamma_values[iG];

            #pragma omp critical
            {
                std::cout << "[gammaLoop] Now varying gamma=" << gaVal
                          << " (" << (iG+1) << "/" << gamma_values.size() << ")"
                          << std::endl;
            }

            for(int i=0; i<NUM_RANDOM; ++i) {
                const auto &ps = randomParamSets[i];
                double curB = ps.b;
                double curT = ps.theta;
                double curD = ps.delta;
                double curG = gaVal; // Overwrite only gamma

                #pragma omp critical
                {
                    std::cout << "  -> Random sample " << (i+1) << "/" << NUM_RANDOM
                              << ": (b=" << curB
                              << ", theta=" << curT
                              << ", delta=" << curD
                              << ", gamma=" << curG
                              << ")" << std::endl;
                }

                double mean0 = run_evolution_simulation(curB, curT, curD, curG, 0.0);
                double mean1 = run_evolution_simulation(curB, curT, curD, curG, 1.0);

                double pctDiff = 0.0;
                if(mean0 > 1e-12) {
                    pctDiff = ((mean1 - mean0)/mean0)*100.0;
                }

                #pragma omp critical
                {
                    outFile << experimentTag << "," << "gamma" << "," << gaVal << ","
                            << curB << "," << curT << "," << curD << "," << curG << ","
                            << mean0 << "," << mean1 << "," << pctDiff << "\n";
                }
            }
        }
    }

    // ==========================
    // (d) VARY DELTA
    // ==========================
    {
        std::string experimentTag = "deltaLoop";
        #pragma omp parallel for schedule(dynamic)
        for (int iD = 0; iD < (int)delta_values.size(); ++iD) {
            double deVal = delta_values[iD];

            #pragma omp critical
            {
                std::cout << "[deltaLoop] Now varying delta=" << deVal
                          << " (" << (iD+1) << "/" << delta_values.size() << ")"
                          << std::endl;
            }

            for(int i=0; i<NUM_RANDOM; ++i) {
                const auto &ps = randomParamSets[i];
                double curB = ps.b;
                double curT = ps.theta;
                double curD = deVal; // Overwrite only delta
                double curG = ps.gamma;

                #pragma omp critical
                {
                    std::cout << "  -> Random sample " << (i+1) << "/" << NUM_RANDOM
                              << ": (b=" << curB
                              << ", theta=" << curT
                              << ", delta=" << curD
                              << ", gamma=" << curG
                              << ")" << std::endl;
                }

                double mean0 = run_evolution_simulation(curB, curT, curD, curG, 0.0);
                double mean1 = run_evolution_simulation(curB, curT, curD, curG, 1.0);

                double pctDiff = 0.0;
                if(mean0 > 1e-12) {
                    pctDiff = ((mean1 - mean0)/mean0)*100.0;
                }

                #pragma omp critical
                {
                    outFile << experimentTag << "," << "delta" << "," << deVal << ","
                            << curB << "," << curT << "," << curD << "," << curG << ","
                            << mean0 << "," << mean1 << "," << pctDiff << "\n";
                }
            }
        }
    }

    outFile.close();
    std::cout << "\nDONE. Results in simulation_results.csv\n";
    return 0;
}
