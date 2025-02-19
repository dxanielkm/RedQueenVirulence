/*
Compile example (no OpenMP for clarity):
    g++-14 -std=c++14 -O2 -I /opt/homebrew/include -fopenmp -o evSim ev_sim.cpp

Run example:
    ./evSim 2 0.5 10.0

Usage:
    ./evSim <N> <s> <thetaTilde>
where
    N           = number of host/parasite alleles
    s           = specificity parameter [0..1]
    thetaTilde  = scaling factor for parasite production
*/

#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <random>
#include <algorithm>
#include <numeric>
#include <string>
#include <sstream>
#include <boost/numeric/odeint.hpp>

// ====================== GLOBAL SETTINGS ======================
static const double ABS_EXT_TOL = 1e-6;     // Extinction threshold
static const double CONSTANT_FACTOR = 10.0; // Scale for host populations
static const int    NUM_TRAITS = 50;        // Number of discrete virulence trait classes
static const int    EVOL_STEPS = 2500;      // Number of evolutionary time steps

// For the ODE integration each step
static const double TSPAN = 200.0;          // Integrate from t=0 to t=200
static const double DT    = 1.0;            // Step size for integrate_adaptive

// Virulence trait space [ALPHA_LOW, ALPHA_HIGH]
static const double ALPHA_LOW  = 1.50;
static const double ALPHA_HIGH = 4.00;

// Default ecological parameters
static const double DEFAULT_B     = 10.0;  // birth rate
static const double DEFAULT_DELTA = 0.3;   // parasite decay
static const double DEFAULT_GAMMA = 1.0;   // recovery rate
static const double DEFAULT_D     = 1.0;   // natural mortality
static const double DEFAULT_Q     = 1.0;   // density dependence
static const double DEFAULT_BETA0 = 1.5;   // baseline infection rate (beta = beta0 * N)

// ============== HELPER FUNCTIONS ==============
typedef std::vector<double> state_type;

// Compute block size for each virulence trait: (N*N infected) + (N parasites).
inline int blockSize(int N) {
    return N*N + N;
}

// Create an N x N specificity matrix Q. Q[i][j] = (1 + (N-1)*s)/N if i == j, else (1 - s)/N
std::vector<std::vector<double>> create_infection_matrix(int N, double s) {
    std::vector<std::vector<double>> Q(N, std::vector<double>(N, 0.0));
    for (int i = 0; i < N; i++){
        for (int j = 0; j < N; j++){
            if (i == j)
                Q[i][j] = (1 + (N - 1)*s)/double(N);
            else
                Q[i][j] = (1 - s)/double(N);
        }
    }
    return Q;
}

// ============== ODE FUNCTOR ==============
struct alleleDynamics {
    int N;
    int numTraits;
    const std::vector<double> &myAlpha; // virulence trait values
    double s;       // specificity
    double b;       // birth rate
    double gamma;   // recovery
    double theta;   // parasite production scaling
    double delta;   // parasite decay
    double d;       // natural mortality
    double q;       // density dependence
    double beta;    // infection rate = beta0 * N

    alleleDynamics(int N_, int nt_, const std::vector<double> &alpha_, double s_,
                   double b_, double gamma_, double theta_, double delta_,
                   double d_, double q_, double beta_)
        : N(N_), numTraits(nt_), myAlpha(alpha_), s(s_),
          b(b_), gamma(gamma_), theta(theta_), delta(delta_),
          d(d_), q(q_), beta(beta_) {}

    void operator()(const state_type &y, state_type &dydt, double /*t*/) const {
        dydt.resize(y.size());

        // First N entries = Susceptible hosts S[i]
        std::vector<double> S(N);
        for (int i = 0; i < N; i++){
            S[i] = y[i];
        }

        // Compute total population
        double totalPop = 0.0;
        for (int i = 0; i < N; i++){
            totalPop += S[i];
        }
        int bs = blockSize(N);
        for (int k = 0; k < numTraits; k++){
            int offset = N + k*bs;
            // infected compartments: I_ij for i=0..N-1, j=0..N-1
            for (int idx = 0; idx < N*N; idx++){
                totalPop += y[offset + idx];
            }
        }

        dydt.assign(dydt.size(), 0.0);
        std::vector<double> infLoss(N, 0.0);
        std::vector<double> recGain(N, 0.0);

        // Specificity matrix Q
        auto Q = create_infection_matrix(N, s);

        // Loop over virulence trait blocks
        for (int trait = 0; trait < numTraits; trait++){
            int offset = N + trait*bs;
            // production trade-off: theta_k = theta * sqrt(alpha)
            double theta_k = theta * std::sqrt(myAlpha[trait]);
            // infected hosts
            for (int i = 0; i < N; i++){
                for (int j = 0; j < N; j++){
                    int idxI = offset + i*N + j;
                    double I_ij = y[idxI];
                    int idxP = offset + N*N + j; // parasite compartment
                    double P_j = y[idxP];
                    // ODE for I_ij
                    double dI = beta * Q[i][j]*P_j*S[i] - (d + gamma + myAlpha[trait])*I_ij;
                    dydt[idxI] = dI;
                    infLoss[i] += beta * Q[i][j]*P_j*S[i];
                    recGain[i] += gamma*I_ij;
                }
            }
            // parasite compartments
            for (int j = 0; j < N; j++){
                int idxP = offset + N*N + j;
                double P_j = y[idxP];
                // sum over infected hosts with parasite j
                double sumI = 0.0;
                for (int i = 0; i < N; i++){
                    sumI += y[offset + i*N + j];
                }
                // infection loss
                double inf_loss = 0.0;
                for (int i = 0; i < N; i++){
                    inf_loss += beta * Q[i][j]*S[i];
                }
                double dP = theta_k*sumI - delta*P_j - inf_loss*P_j;
                dydt[idxP] = dP;
            }
        }
        // Susceptible dynamics
        for (int i = 0; i < N; i++){
            double dS = b*S[i]*(1 - q*totalPop) - d*S[i] - infLoss[i] + recGain[i];
            dydt[i] = dS;
        }
    }
};

// ============== MAIN ==============
int main(int argc, char* argv[]) {
    // 1) Parse command line or set defaults
    int N = 5;            // default N
    double s = 1.0;       // default specificity
    double thetaTilde = 20.0; // default parasite production scaling

    if (argc >= 2) { N = std::stoi(argv[1]); }
    if (argc >= 3) { s = std::stod(argv[2]); }
    if (argc >= 4) { thetaTilde = std::stod(argv[3]); }

    std::cout << "Running with N=" << N
              << ", s=" << s
              << ", thetaTilde=" << thetaTilde << std::endl;

    // 2) Prepare virulence trait space
    std::vector<double> myAlpha(NUM_TRAITS);
    for (int i = 0; i < NUM_TRAITS; i++){
        myAlpha[i] = ALPHA_LOW + i*(ALPHA_HIGH - ALPHA_LOW)/(NUM_TRAITS-1);
    }

    // 3) Prepare initial conditions
    int bs = blockSize(N);
    int totalSize = N + NUM_TRAITS*bs;
    state_type y0(totalSize, 0.0);

    // Set up susceptible
    for (int i = 0; i < N; i++){
        y0[i] = 0.85*CONSTANT_FACTOR;
    }
    // Infected & parasites
    for (int trait = 0; trait < NUM_TRAITS; trait++){
        int offset = N + trait*bs;
        if (trait == 0) {
            // seed small infected & parasite populations
            for (int i = 0; i < N; i++){
                for (int j = 0; j < N; j++){
                    int idxI = offset + i*N + j;
                    y0[idxI] = 0.07*CONSTANT_FACTOR;
                }
            }
            for (int j = 0; j < N; j++){
                int idxP = offset + N*N + j;
                y0[idxP] = 0.1*CONSTANT_FACTOR;
            }
        } else {
            // zero
            for (int k = 0; k < bs; k++){
                y0[offset + k] = 0.0;
            }
        }
    }

    // 4) Set up ODE parameters
    double beta  = DEFAULT_BETA0 * N;  // infection rate
    double b     = DEFAULT_B;
    double gamma = DEFAULT_GAMMA;
    double delta = DEFAULT_DELTA;
    double d     = DEFAULT_D;
    double q     = DEFAULT_Q;

    // 5) Prepare output CSV for the virulence distribution
    //    We'll record after each evolutionary step the population distribution across all alpha_k
    std::ofstream outFile("virulence_distribution.csv");
    outFile << "Step,Alpha,Population\n";

    // 6) Evolutionary simulation
    //    Each step: integrate ODE, apply extinction threshold, mutation, record distribution
    state_type y = y0; // working state
    // For random mutation
    static thread_local std::mt19937 rng(std::random_device{}());
    std::uniform_real_distribution<double> uniform01(0.0, 1.0);

    typedef boost::numeric::odeint::runge_kutta_dopri5<state_type> dopri5_stepper;
    auto stepper = boost::numeric::odeint::make_controlled<dopri5_stepper>(1e-6, 1e-6);

    for (int step = 0; step <= EVOL_STEPS; step++){
        // 6a) If step > 0, integrate ODE for TSPAN
        if (step > 0) {
            boost::numeric::odeint::integrate_adaptive(
                stepper,
                alleleDynamics(N, NUM_TRAITS, myAlpha, s, b, gamma, thetaTilde, delta, d, q, beta),
                y,
                0.0,
                TSPAN,
                DT
            );
            // extinction threshold
            for (auto &val : y){
                if (val < ABS_EXT_TOL) val = 0.0;
                if (val < 0.0) val = 0.0;
            }

            // 6b) Mutation
            // Summation for infected (host allele & trait) and parasites (allele & trait).
            // Then pick random parent in proportion to pop. Move fraction to adjacent trait.
            // For simplicity, we mutate host or parasite once each step.

            //  i) Summation for infected hosts
            double totalInf = 0.0;
            std::vector<double> infVec;
            std::vector<int> infTraitIdx; // which trait block
            infVec.reserve(NUM_TRAITS*N); // roughly
            infTraitIdx.reserve(NUM_TRAITS*N);

            //  ii) Summation for parasites
            double totalPara = 0.0;
            std::vector<double> paraVec;
            std::vector<int> paraTraitIdx;
            paraVec.reserve(NUM_TRAITS*N);
            paraTraitIdx.reserve(NUM_TRAITS*N);

            for (int trait=0; trait<NUM_TRAITS; trait++){
                int offset = N + trait*bs;
                // infected
                double sumI = 0.0;
                for (int i = 0; i < N*N; i++){
                    sumI += y[offset + i];
                }
                if (sumI > 0.0) {
                    infVec.push_back(sumI);
                    infTraitIdx.push_back(trait);
                    totalInf += sumI;
                }
                // parasites
                double sumP = 0.0;
                for (int j = 0; j < N; j++){
                    sumP += y[offset + N*N + j];
                }
                if (sumP > 0.0) {
                    paraVec.push_back(sumP);
                    paraTraitIdx.push_back(trait);
                    totalPara += sumP;
                }
            }

            if (totalInf > 0.0) {
                std::vector<double> cdfInf(infVec.size());
                std::partial_sum(infVec.begin(), infVec.end(), cdfInf.begin());
                double rInf = uniform01(rng)*totalInf;
                auto itInf = std::upper_bound(cdfInf.begin(), cdfInf.end(), rInf);
                if (itInf != cdfInf.end()){
                    size_t idxInf = std::distance(cdfInf.begin(), itInf);
                    int traitChosen = infTraitIdx[idxInf];
                    // mutate trait up or down if possible
                    int newTrait = traitChosen;
                    if (traitChosen == 0) {
                        newTrait = 1;
                    } else if (traitChosen == NUM_TRAITS-1) {
                        newTrait = NUM_TRAITS-2;
                    } else {
                        newTrait = (uniform01(rng)<0.5)?(traitChosen-1):(traitChosen+1);
                    }
                    // Move fraction of infected from traitChosen to newTrait
                    double eta = 0.1;
                    int offsetChosen = N + traitChosen*bs;
                    int offsetNew    = N + newTrait*bs;
                    for (int i=0; i<N; i++){
                        for (int j=0; j<N; j++){
                            int idxC = offsetChosen + i*N + j;
                            int idxN = offsetNew    + i*N + j;
                            double xfer = y[idxC]*eta;
                            y[idxN] += xfer;
                            y[idxC] -= xfer;
                        }
                    }
                }
            }

            if (totalPara > 0.0) {
                std::vector<double> cdfPara(paraVec.size());
                std::partial_sum(paraVec.begin(), paraVec.end(), cdfPara.begin());
                double rPara = uniform01(rng)*totalPara;
                auto itPara = std::upper_bound(cdfPara.begin(), cdfPara.end(), rPara);
                if (itPara != cdfPara.end()){
                    size_t idxPara = std::distance(cdfPara.begin(), itPara);
                    int traitChosen = paraTraitIdx[idxPara];
                    int newTrait = traitChosen;
                    if (traitChosen == 0) {
                        newTrait = 1;
                    } else if (traitChosen == NUM_TRAITS-1) {
                        newTrait = NUM_TRAITS-2;
                    } else {
                        newTrait = (uniform01(rng)<0.5)?(traitChosen-1):(traitChosen+1);
                    }
                    double eta = 0.1;
                    int offsetChosen = N + traitChosen*bs;
                    int offsetNew    = N + newTrait*bs;
                    for (int j=0; j<N; j++){
                        int idxC = offsetChosen + N*N + j;
                        int idxN = offsetNew    + N*N + j;
                        double xfer = y[idxC]*eta;
                        y[idxN] += xfer;
                        y[idxC] -= xfer;
                    }
                }
            }

            // final check
            for (auto &val : y){
                if (val < ABS_EXT_TOL) val = 0.0;
                if (val < 0.0) val = 0.0;
            }
        } // end if step > 0

        // 6c) Record distribution of virulence
        // Sum total population for each trait block (infected + parasites).
        double totalAll = 0.0;
        std::vector<double> traitPop(NUM_TRAITS, 0.0);
        for (int t=0; t<NUM_TRAITS; t++){
            int offset = N + t*bs;
            double sumTrait = 0.0;
            // infected
            for (int i=0; i<N*N; i++){
                sumTrait += y[offset + i];
            }
            // parasites
            for (int j=0; j<N; j++){
                sumTrait += y[offset + N*N + j];
            }
            traitPop[t] = sumTrait;
            totalAll += sumTrait;
        }
        // Write lines: step, alpha, population
        // (You can also store the fraction if you prefer.)
        for (int t=0; t<NUM_TRAITS; t++){
            double alphaVal = myAlpha[t];
            outFile << step << "," << alphaVal << "," << traitPop[t] << "\n";
        }
    }

    outFile.close();
    std::cout << "Simulation finished. Results in virulence_distribution.csv\n";
    return 0;
}