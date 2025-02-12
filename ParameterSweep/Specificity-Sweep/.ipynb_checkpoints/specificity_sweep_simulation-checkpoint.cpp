#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <random>
#include <algorithm>
#include <numeric>
#include <limits>
#include <boost/numeric/odeint.hpp>
#include <omp.h>  // For parallelization

// Compile with:
// g++-14 -std=c++14 -O2 -I /opt/homebrew/include -fopenmp -o specificitySim specificity_sweep_simulation.cpp

// Extinction threshold and evolution settings
double abs_ext_tol = 1e-6;  
double constantFactor = 10.0;  // Scale for host populations
// Set the number of virulence traits for evolution (here, 50 discrete traits)
int numTraits = 50;  
int evolSteps = 1000;

// Define the state type
typedef std::vector<double> state_type;

// Create the 2x2 specificity matrix Q with normalization:
// Q[i][j] = (1+s)/2 if i==j, and (1-s)/2 if i != j.
auto create_infection_matrix = [](double s) {
    return std::vector<std::vector<double>>{
        { (1 + s) / 2.0, (1 - s) / 2.0 },
        { (1 - s) / 2.0, (1 + s) / 2.0 }
    };
};

// ODE system for host-parasite dynamics with virulence evolution.
// The state vector is organized as follows:
// [S1, S2, then for each trait: I11, I12, I21, I22, P1, P2]
struct alleleDynamics {
    int numTraits;
    const std::vector<double>& myAlpha;  // virulence values for each trait
    double s;       // Specificity parameter
    double b;       // Intrinsic birth rate
    double gamma;   // Recovery rate
    double theta;   // This is interpreted as theta_tilde, used in the trade-off: theta_k = theta * sqrt(alpha)
    double delta;   // Parasite decay rate
    double d;       // Natural mortality
    double q;       // Density-dependence coefficient
    double beta;    // Infection rate (beta = beta0 * n)

    alleleDynamics(int numTraits, const std::vector<double>& myAlpha, double s,
                   double b, double gamma, double theta, double delta, double d, double q, double beta)
        : numTraits(numTraits), myAlpha(myAlpha), s(s),
          b(b), gamma(gamma), theta(theta), delta(delta), d(d), q(q), beta(beta) {}

    void operator()(const state_type& y, state_type& dydt, const double /* t */) const {
        // Extract susceptibles
        double S1 = y[0];
        double S2 = y[1];

        // Compute total host population (susceptible plus all infected hosts)
        double totalPopulation = S1 + S2;
        for (size_t i = 2; i < y.size(); i += 6) {
            totalPopulation += y[i] + y[i + 1] + y[i + 2] + y[i + 3];
        }

        dydt.resize(y.size());
        double infectionS1 = 0.0;
        double infectionS2 = 0.0;
        double recoveryS1  = 0.0;
        double recoveryS2  = 0.0;

        // Get specificity matrix Q for current s
        std::vector<std::vector<double>> Q = create_infection_matrix(s);

        // Loop over virulence traits
        for (int i = 0, idx = 2; i < numTraits; ++i, idx += 6) {
            double I11 = y[idx];
            double I12 = y[idx + 1];
            double I21 = y[idx + 2];
            double I22 = y[idx + 3];
            double P1  = y[idx + 4];
            double P2  = y[idx + 5];

            // Trade-off: parasite production now depends on the trait value.
            // Compute theta_k = theta_tilde * sqrt(myAlpha[i])
            double theta_k = theta * std::sqrt(myAlpha[i]);

            // Infected host equations:
            double dI11 = beta * Q[0][0] * P1 * S1 - (d + gamma + myAlpha[i]) * I11;
            double dI12 = beta * Q[0][1] * P2 * S1 - (d + gamma + myAlpha[i]) * I12;
            double dI21 = beta * Q[1][0] * P1 * S2 - (d + gamma + myAlpha[i]) * I21;
            double dI22 = beta * Q[1][1] * P2 * S2 - (d + gamma + myAlpha[i]) * I22;

            // Parasite equations with trait-dependent production:
            double dP1 = theta_k * (I11 + I21) - delta * P1 - beta * (Q[0][0] * S1 + Q[1][0] * S2) * P1;
            double dP2 = theta_k * (I12 + I22) - delta * P2 - beta * (Q[0][1] * S1 + Q[1][1] * S2) * P2;

            infectionS1 += beta * (Q[0][0] * P1 + Q[0][1] * P2) * S1;
            infectionS2 += beta * (Q[1][0] * P1 + Q[1][1] * P2) * S2;
            recoveryS1  += gamma * (I11 + I12);
            recoveryS2  += gamma * (I21 + I22);

            // Store derivatives for infected host and parasite compartments
            dydt[idx]     = dI11;
            dydt[idx + 1] = dI12;
            dydt[idx + 2] = dI21;
            dydt[idx + 3] = dI22;
            dydt[idx + 4] = dP1;
            dydt[idx + 5] = dP2;
        }

        // Susceptible host equations:
        double dS1 = b * S1 * (1 - q * totalPopulation) - d * S1 - infectionS1 + recoveryS1;
        double dS2 = b * S2 * (1 - q * totalPopulation) - d * S2 - infectionS2 + recoveryS2;
        dydt[0] = dS1;
        dydt[1] = dS2;
    }
};

int main() {
    // --- Set parameter values to match the Python simulation ---
    double default_b     = 10.0;   // Intrinsic birth rate
    // Here, default_theta is interpreted as theta_tilde. We use 10.0.
    double default_theta = 30.0;  
    double default_delta = 0.3;    // Parasite decay rate
    double default_gamma = 1.0;    // Recovery rate
    // Set beta = beta0 * n, with beta0 = 1.5 and n = 2:
    double default_beta  = 1.5 * 2;  // = 3.0
    double default_d     = 1.0;      // Natural mortality
    double default_q     = 1.0;      // Density-dependent competition

    // --- Prepare output file ---
    std::ofstream outputFile("specificity_sweep_results.csv");
    outputFile << "s,MeanAlpha\n";

    // --- Define the trait space for virulence evolution ---
    // For example, let alpha vary from 0.1 to 2.0 across numTraits (50) discrete values.
    double alphaLow  = 1.0;
    double alphaHigh = 4.0;
    std::vector<double> myAlpha(numTraits);
    for (int i = 0; i < numTraits; ++i) {
        myAlpha[i] = alphaLow + i * (alphaHigh - alphaLow) / (numTraits - 1);
    }

    // --- Initial conditions ---  
    // Order: S1, S2, then for each trait: I11, I12, I21, I22, P1, P2
    // We initialize the host populations as before and seed only trait 0 with nonzero infected hosts and parasites.
    double S1_0   = 0.9 * constantFactor;    // e.g., 9.0
    double S2_0   = 0.8 * constantFactor;    // e.g., 8.0
    // For trait 0, assign small positive values:
    double I11_0  = 0.07 * constantFactor;     // 0.7
    double I12_0  = 0.07 * constantFactor;     // 0.7
    double I21_0  = 0.07 * constantFactor;     // 0.7
    double I22_0  = 0.07 * constantFactor;     // 0.7
    double P1_0   = 0.1  * constantFactor;      // 1.0
    double P2_0   = 0.1  * constantFactor;      // 1.0

    // Create the state vector y0 of size: 2 + 6 * numTraits.
    state_type y0(2 + 6 * numTraits, 0.0);
    y0[0] = S1_0;
    y0[1] = S2_0;
    int idx_y = 2;
    for (int i = 0; i < numTraits; ++i) {
        if (i == 0) {
            // For the initial virulence trait (lowest value, alpha = alphaLow)
            y0[idx_y]     = I11_0;
            y0[idx_y + 1] = I12_0;
            y0[idx_y + 2] = I21_0;
            y0[idx_y + 3] = I22_0;
            y0[idx_y + 4] = P1_0;
            y0[idx_y + 5] = P2_0;
        } else {
            // For all other traits, start with zero infected hosts and parasites.
            y0[idx_y]     = 0.0;
            y0[idx_y + 1] = 0.0;
            y0[idx_y + 2] = 0.0;
            y0[idx_y + 3] = 0.0;
            y0[idx_y + 4] = 0.0;
            y0[idx_y + 5] = 0.0;
        }
        idx_y += 6;
    }

    // --- Time span for integration ---
    double t_start = 0.0;
    double t_end   = 2000.0;
    // We do not store the full time series here.

    // --- Define the range and number of s-values ---
    double s_min = 0.0;       // Minimum s value
    double s_max = 1.0;       // Maximum s value
    int num_s_values = 21;    // Total number of s values (adjustable)

    // Generate the vector of s-values using linear spacing.
    std::vector<double> s_values;
    s_values.reserve(num_s_values);
    for (int i = 0; i < num_s_values; ++i) {
        double s_val = s_min + i * ((s_max - s_min) / (num_s_values - 1));
        s_values.push_back(s_val);
    }

    // --- Parallel evolutionary simulation using OpenMP ---
    #pragma omp parallel for schedule(dynamic)
    for (size_t idx = 0; idx < s_values.size(); ++idx) {
        double s_value = s_values[idx];

        // Use the default parameters for this simulation:
        double b     = default_b;
        double theta = default_theta;  // This is theta_tilde.
        double delta = default_delta;
        double gamma = default_gamma;
        double beta  = default_beta;
        double d     = default_d;
        double q     = default_q;

        // Initialize y for this simulation from y0.
        state_type y = y0;

        // Evolution simulation loop
        for (int step = 1; step <= evolSteps; ++step) {
            // Integrate over a time span of [0, 200]
            double t0 = 0.0;
            double t1 = 200.0;
            double dt = 1.0;

            typedef boost::numeric::odeint::runge_kutta_dopri5<state_type> dopri5_stepper;
            auto stepper = boost::numeric::odeint::make_controlled<dopri5_stepper>(1e-6, 1e-6);

            boost::numeric::odeint::integrate_adaptive(
                stepper,
                alleleDynamics(numTraits, myAlpha, s_value, b, gamma, theta, delta, d, q, beta),
                y,
                t0,
                t1,
                dt
            );

            // Enforce extinction thresholds and non-negativity
            for (auto& val : y) {
                if (val < abs_ext_tol) { val = 0.0; }
                if (val < 0.0) { val = 0.0; }
            }

            // ---------- Mutation Logic ----------
            // Compute totals for infected hosts and parasites for mutation selection.
            std::vector<double> Iik_total;
            std::vector<std::pair<int,int>> h_idxs; // (host allele, trait index)
            double total_infected_hosts = 0.0;
            std::vector<double> Pjk_total;
            std::vector<std::pair<int,int>> p_idxs; // (parasite allele, trait index)
            double total_parasites = 0.0;

            for (int k = 0; k < numTraits; ++k) {
                int idx2 = 2 + 6 * k;
                double I1k_total = y[idx2] + y[idx2 + 1];  // Host allele 1 infections
                if (I1k_total > 0.0) {
                    Iik_total.push_back(I1k_total);
                    h_idxs.emplace_back(0, k);
                    total_infected_hosts += I1k_total;
                }
                double I2k_total = y[idx2 + 2] + y[idx2 + 3];  // Host allele 2 infections
                if (I2k_total > 0.0) {
                    Iik_total.push_back(I2k_total);
                    h_idxs.emplace_back(1, k);
                    total_infected_hosts += I2k_total;
                }
                double P1k = y[idx2 + 4];
                if (P1k > 0.0) {
                    Pjk_total.push_back(P1k);
                    p_idxs.emplace_back(0, k);
                    total_parasites += P1k;
                }
                double P2k = y[idx2 + 5];
                if (P2k > 0.0) {
                    Pjk_total.push_back(P2k);
                    p_idxs.emplace_back(1, k);
                    total_parasites += P2k;
                }
            }

            if (total_infected_hosts == 0.0 || total_parasites == 0.0) {
                break;
            }

            // Build cumulative distribution functions (CDFs)
            std::vector<double> host_cdf(Iik_total.size());
            std::partial_sum(Iik_total.begin(), Iik_total.end(), host_cdf.begin());
            std::vector<double> parasite_cdf(Pjk_total.size());
            std::partial_sum(Pjk_total.begin(), Pjk_total.end(), parasite_cdf.begin());

            static thread_local std::mt19937 gen_thread(std::random_device{}());
            static thread_local std::uniform_real_distribution<> dis_thread(0.0, 1.0);

            // ---------- Host Mutation ----------
            double r_host = dis_thread(gen_thread) * total_infected_hosts;
            auto host_it = std::upper_bound(host_cdf.begin(), host_cdf.end(), r_host);
            if (host_it != host_cdf.end()) {
                size_t host_idx   = std::distance(host_cdf.begin(), host_it);
                int i_p           = h_idxs[host_idx].first;   // host allele (0 or 1)
                int k_p           = h_idxs[host_idx].second;  // current trait index
                int k_m = k_p;
                if (k_p == 0) {
                    k_m = 1;
                } else if (k_p == numTraits - 1) {
                    k_m = k_p - 1;
                } else {
                    k_m = (dis_thread(gen_thread) < 0.5) ? (k_p - 1) : (k_p + 1);
                }
                double eta   = 0.1;
                int idx_p    = 2 + 6 * k_p;
                int idx_m    = 2 + 6 * k_m;
                if (i_p == 0) {  // Host allele 1
                    double transfer_I11 = y[idx_p];
                    double transfer_I12 = y[idx_p + 1];
                    double dI11 = transfer_I11 * eta;
                    double dI12 = transfer_I12 * eta;
                    y[idx_m]     += dI11;
                    y[idx_m + 1] += dI12;
                    y[idx_p]     -= dI11;
                    y[idx_p + 1] -= dI12;
                } else {         // Host allele 2
                    double transfer_I21 = y[idx_p + 2];
                    double transfer_I22 = y[idx_p + 3];
                    double dI21 = transfer_I21 * eta;
                    double dI22 = transfer_I22 * eta;
                    y[idx_m + 2] += dI21;
                    y[idx_m + 3] += dI22;
                    y[idx_p + 2] -= dI21;
                    y[idx_p + 3] -= dI22;
                }
            }

            // ---------- Parasite Mutation ----------
            double r_parasite = dis_thread(gen_thread) * total_parasites;
            auto parasite_it = std::upper_bound(parasite_cdf.begin(), parasite_cdf.end(), r_parasite);
            if (parasite_it != parasite_cdf.end()) {
                size_t parasite_idx = std::distance(parasite_cdf.begin(), parasite_it);
                int j_p             = p_idxs[parasite_idx].first;  // parasite allele (0 or 1)
                int k_p             = p_idxs[parasite_idx].second; // current trait index
                int k_m = k_p;
                if (k_p == 0) {
                    k_m = 1;
                } else if (k_p == numTraits - 1) {
                    k_m = k_p - 1;
                } else {
                    k_m = (dis_thread(gen_thread) < 0.5) ? (k_p - 1) : (k_p + 1);
                }
                double eta = 0.1;
                int idx_p  = 2 + 6 * k_p;
                int idx_m  = 2 + 6 * k_m;
                if (j_p == 0) {
                    double transfer_P1 = y[idx_p + 4] * eta;
                    y[idx_m + 4] += transfer_P1;
                    y[idx_p + 4] -= transfer_P1;
                } else {
                    double transfer_P2 = y[idx_p + 5] * eta;
                    y[idx_m + 5] += transfer_P2;
                    y[idx_p + 5] -= transfer_P2;
                }
            }

            for (auto& val : y) {
                if (val < abs_ext_tol) { val = 0.0; }
                if (val < 0.0) { val = 0.0; }
            }
        } // End of evolution steps

        // ---------- After simulation, calculate the weighted mean virulence ----------
        double sumInfectionParasite_Total = 0.0;
        for (size_t j = 2; j < y.size(); j += 6) {
            sumInfectionParasite_Total += y[j] + y[j + 1] + y[j + 2] + y[j + 3] + y[j + 4] + y[j + 5];
        }
        double weighted_mean_alpha = 0.0;
        for (int j = 0; j < numTraits; ++j) {
            int idx2 = 2 + 6 * j;
            double trait_pop = y[idx2] + y[idx2 + 1] + y[idx2 + 2] + y[idx2 + 3] 
                               + y[idx2 + 4] + y[idx2 + 5];
            if (trait_pop > 0 && sumInfectionParasite_Total > 0) {
                double proportion = trait_pop / sumInfectionParasite_Total;
                weighted_mean_alpha += myAlpha[j] * proportion;
            }
        }

        #pragma omp critical
        {
            outputFile << s_values[idx] << "," << weighted_mean_alpha << "\n";
        }
        std::cout << "Simulation completed for s=" << s_values[idx]
                  << ", mean_alpha=" << weighted_mean_alpha << std::endl;
    } // End s_values loop

    outputFile.close();
    return 0;
}
