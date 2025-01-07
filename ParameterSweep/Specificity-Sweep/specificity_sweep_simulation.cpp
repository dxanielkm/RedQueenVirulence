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
// g++-14 -std=c++14 -O2 -I /opt/homebrew/include -fopenmp -o specificitySim specificitySim.cpp

// Define parameters
double abs_ext_tol = 1e-6;  // Absolute extinction threshold
double constantFactor = 0.15;  // Adjusted to match previous code
int numTraits = 50;  // Adjust as necessary for performance
int evolSteps = 1000;

// Define the state type
typedef std::vector<double> state_type;

// Define the infection matrix Q as a function of specificity parameter s
auto create_infection_matrix = [](double s) {
    return std::vector<std::vector<double>>{
        { (1 + s), (1 - s) },
        { (1 - s), (1 + s) }
    };
};

// Define the system of differential equations
struct alleleDynamics {
    int numTraits;
    const std::vector<double>& myAlpha;
    double s;       // Specificity parameter
    double b;       // Birth rate
    double gamma;   // Recovery rate
    double theta;   // Parasite production rate
    double delta;   // Parasite mortality rate
    double d;       // Death rate
    double q;       // Density dependence

    alleleDynamics(int numTraits, const std::vector<double>& myAlpha, double s,
                   double b, double gamma, double theta, double delta, double d, double q)
        : numTraits(numTraits), myAlpha(myAlpha), s(s),
          b(b), gamma(gamma), theta(theta), delta(delta), d(d), q(q) {}

    void operator()(const state_type& y, state_type& dydt, const double /* t */) const {
        // First two elements are S1, S2
        double S1 = y[0];
        double S2 = y[1];

        // Calculate total population
        double totalPopulation = S1 + S2;
        for (size_t i = 2; i < y.size(); i += 6) {
            totalPopulation += y[i] + y[i + 1] + y[i + 2] + y[i + 3]; // Add infected populations
        }

        dydt.resize(y.size());

        // Different terms in equation for dS1/dt, dS2/dt
        double infectionS1 = 0.0;
        double infectionS2 = 0.0;
        double recoveryS1 = 0.0;
        double recoveryS2 = 0.0;

        // Get the Q matrix for the current specificity level
        std::vector<std::vector<double>> Q = create_infection_matrix(s);

        // Each iteration represents a new trait
        for (int i = 0, idx = 2; i < numTraits; ++i, idx += 6) {
            double I11 = y[idx];
            double I12 = y[idx + 1];
            double I21 = y[idx + 2];
            double I22 = y[idx + 3];
            double P1  = y[idx + 4];
            double P2  = y[idx + 5];

            // Trade-off between virulence and infectivity
            double alpha_sqrt = 20 * sqrt(myAlpha[i] / 10);

            // Compute derivatives using Q
            double dI11 = Q[0][0] * alpha_sqrt * P1 * S1 - (d + gamma + myAlpha[i]) * I11;
            double dI12 = Q[0][1] * alpha_sqrt * P2 * S1 - (d + gamma + myAlpha[i]) * I12;
            double dI21 = Q[1][0] * alpha_sqrt * P1 * S2 - (d + gamma + myAlpha[i]) * I21;
            double dI22 = Q[1][1] * alpha_sqrt * P2 * S2 - (d + gamma + myAlpha[i]) * I22;

            double dP1 = theta * (I11 + I21) - delta * P1;
            double dP2 = theta * (I12 + I22) - delta * P2;

            // Accumulate infection and recovery terms
            infectionS1 += alpha_sqrt * (Q[0][0] * P1 * S1 + Q[0][1] * P2 * S1);
            infectionS2 += alpha_sqrt * (Q[1][0] * P1 * S2 + Q[1][1] * P2 * S2);
            recoveryS1  += gamma * (I11 + I12);
            recoveryS2  += gamma * (I21 + I22);

            // Store derivatives
            dydt[idx]     = dI11;
            dydt[idx + 1] = dI12;
            dydt[idx + 2] = dI21;
            dydt[idx + 3] = dI22;
            dydt[idx + 4] = dP1;
            dydt[idx + 5] = dP2;
        }

        // Calculate dS1/dt, dS2/dt
        double dS1 = b * S1 * (1 - q * totalPopulation) - d * S1 - infectionS1 + recoveryS1;
        double dS2 = b * S2 * (1 - q * totalPopulation) - d * S2 - infectionS2 + recoveryS2;

        dydt[0] = dS1;
        dydt[1] = dS2;
    }
};

int main() {
    // Default parameter values
    double default_b     = 10.0;
    double default_theta = 10.0;
    double default_delta = 1.0;
    double default_gamma = 0.6;

    // Prepare to store the results
    std::ofstream outputFile("specificity_results.csv");

    // Output file header
    outputFile << "s,MeanAlpha\n";

    // Trait space
    double alphaLow  = 1.0;
    double alphaHigh = 5.0;
    std::vector<double> myAlpha(numTraits);
    for (int i = 0; i < numTraits; ++i) {
        myAlpha[i] = alphaLow + i * (alphaHigh - alphaLow) / (numTraits - 1);
    }

    // Initial values
    double S1_0   = 0.9  * constantFactor;
    double S2_0   = 0.8  * constantFactor;
    double I11_0  = 0.1  * constantFactor;
    double I12_0  = 0.08 * constantFactor;
    double I21_0  = 0.065* constantFactor;
    double I22_0  = 0.095* constantFactor;
    double P1_0   = 0.1  * constantFactor;
    double P2_0   = 0.09 * constantFactor;

    // Generate a list of s-values from 0, 0.01, 0.02, ..., 1.00
    std::vector<double> s_values;

    // This loop gives 101 points total from 0 to 1 in increments of 0.01
    for (int i = 0; i <= 100; ++i) {
        double s_val = i * 0.01;
        s_values.push_back(s_val);
    }

    // Use OpenMP for parallelization
    #pragma omp parallel for schedule(dynamic)
    for (size_t idx = 0; idx < s_values.size(); ++idx) {
        double s_value = s_values[idx];

        // Set parameter values
        double b     = default_b;
        double theta = default_theta;
        double delta = default_delta;
        double gamma = default_gamma;

        // Initialize y0 for each simulation
        state_type y0(2 + 6 * numTraits, 0.0);
        y0[0] = S1_0;
        y0[1] = S2_0;

        int idx_y = 2;
        for (int i = 0; i < numTraits; ++i) {
            y0[idx_y]     = I11_0;   // I11
            y0[idx_y + 1] = I12_0;   // I12
            y0[idx_y + 2] = I21_0;   // I21
            y0[idx_y + 3] = I22_0;   // I22
            y0[idx_y + 4] = P1_0;    // P1
            y0[idx_y + 5] = P2_0;    // P2
            idx_y += 6;
        }

        state_type y = y0;

        // Evolution simulation
        for (int step = 1; step <= evolSteps; ++step) {
            // Integrate over tspan = [0.0, 200.0]
            double t0 = 0.0;
            double t1 = 200.0;
            double dt = 1.0;

            // Use the runge_kutta_dopri5 stepper
            typedef boost::numeric::odeint::runge_kutta_dopri5<state_type> dopri5_stepper;

            // Create a controlled stepper using make_controlled
            auto stepper = boost::numeric::odeint::make_controlled<dopri5_stepper>(
                1e-6,   // Absolute error tolerance
                1e-6    // Relative error tolerance
            );

            // Integrate the ODE system
            boost::numeric::odeint::integrate_adaptive(
                stepper,
                alleleDynamics(numTraits, myAlpha, s_value, b, gamma, theta, delta, 1.0, 1.0),
                y,
                t0,
                t1,
                dt
            );

            // Apply extinction thresholds
            for (auto& val : y) {
                if (val < abs_ext_tol) {
                    val = 0.0;
                }
            }

            // Ensure populations remain non-negative
            for (auto& val : y) {
                if (val < 0.0) {
                    val = 0.0;
                }
            }

            // ----------------------- Mutation logic -----------------------
            // Calculate total infected host populations I_{ik}^{total} and total parasite populations P_{jk}
            std::vector<double> Iik_total;          // Stores I_{ik}^{total}
            std::vector<std::pair<int,int>> h_idxs; // (host allele i, trait k)
            double total_infected_hosts = 0.0;

            std::vector<double> Pjk_total;          // Stores P_{jk}
            std::vector<std::pair<int,int>> p_idxs; // (parasite allele j, trait k)
            double total_parasites = 0.0;

            // Build up infected host and parasite totals
            for (int k = 0; k < numTraits; ++k) {
                int idx2 = 2 + 6 * k;

                // Host allele 1
                double I1k_total = y[idx2] + y[idx2 + 1];  // I11 + I12
                if (I1k_total > 0.0) {
                    Iik_total.push_back(I1k_total);
                    h_idxs.emplace_back(0, k);
                    total_infected_hosts += I1k_total;
                }

                // Host allele 2
                double I2k_total = y[idx2 + 2] + y[idx2 + 3];  // I21 + I22
                if (I2k_total > 0.0) {
                    Iik_total.push_back(I2k_total);
                    h_idxs.emplace_back(1, k);
                    total_infected_hosts += I2k_total;
                }

                // Parasite allele 1
                double P1k = y[idx2 + 4];
                if (P1k > 0.0) {
                    Pjk_total.push_back(P1k);
                    p_idxs.emplace_back(0, k);
                    total_parasites += P1k;
                }

                // Parasite allele 2
                double P2k = y[idx2 + 5];
                if (P2k > 0.0) {
                    Pjk_total.push_back(P2k);
                    p_idxs.emplace_back(1, k);
                    total_parasites += P2k;
                }
            }

            // If hosts or parasites are extinct, end early
            if (total_infected_hosts == 0.0 || total_parasites == 0.0) {
                break;
            }

            // Construct CDFs for host and parasite
            std::vector<double> host_cdf(Iik_total.size());
            std::partial_sum(Iik_total.begin(), Iik_total.end(), host_cdf.begin());

            std::vector<double> parasite_cdf(Pjk_total.size());
            std::partial_sum(Pjk_total.begin(), Pjk_total.end(), parasite_cdf.begin());

            // Random generator for this thread (mutation picks)
            static thread_local std::mt19937 gen_thread(std::random_device{}());
            static thread_local std::uniform_real_distribution<> dis_thread(0.0, 1.0);

            // --------------- Host mutation ---------------
            double r_host = dis_thread(gen_thread) * total_infected_hosts;
            auto host_it = std::upper_bound(host_cdf.begin(), host_cdf.end(), r_host);
            if (host_it != host_cdf.end()) {
                size_t host_idx   = std::distance(host_cdf.begin(), host_it);
                int i_p           = h_idxs[host_idx].first;   // Allele (0 or 1)
                int k_p           = h_idxs[host_idx].second;  // Trait index (0..numTraits-1)

                // Determine mutant trait
                int k_m = k_p;
                if (k_p == 0) {
                    k_m = 1;
                } else if (k_p == numTraits - 1) {
                    k_m = k_p - 1;
                } else {
                    k_m = (dis_thread(gen_thread) < 0.5) ? (k_p - 1) : (k_p + 1);
                }

                // Transfer a fraction eta of the parent's infected population
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

            // --------------- Parasite mutation ---------------
            double r_parasite = dis_thread(gen_thread) * total_parasites;
            auto parasite_it = std::upper_bound(parasite_cdf.begin(), parasite_cdf.end(), r_parasite);
            if (parasite_it != parasite_cdf.end()) {
                size_t parasite_idx = std::distance(parasite_cdf.begin(), parasite_it);
                int j_p             = p_idxs[parasite_idx].first;  // Parasite allele (0 or 1)
                int k_p             = p_idxs[parasite_idx].second; // Trait index

                // Determine mutant trait
                int k_m = k_p;
                if (k_p == 0) {
                    k_m = 1;
                } else if (k_p == numTraits - 1) {
                    k_m = k_p - 1;
                } else {
                    k_m = (dis_thread(gen_thread) < 0.5) ? (k_p - 1) : (k_p + 1);
                }

                // Transfer a fraction eta of the parent's parasite population
                double eta = 0.1;
                int idx_p  = 2 + 6 * k_p;
                int idx_m  = 2 + 6 * k_m;

                if (j_p == 0) {  // Parasite allele 1
                    double transfer_P1 = y[idx_p + 4] * eta;
                    y[idx_m + 4] += transfer_P1;
                    y[idx_p + 4] -= transfer_P1;
                } else {         // Parasite allele 2
                    double transfer_P2 = y[idx_p + 5] * eta;
                    y[idx_m + 5] += transfer_P2;
                    y[idx_p + 5] -= transfer_P2;
                }
            }

            // Check for extinctions
            for (auto& val : y) {
                if (val < abs_ext_tol) {
                    val = 0.0;
                }
            }
        } // End of evolSteps loop

        // ----------------- After simulation, calculate mean alpha -----------------
        double sumInfectionParasite_Total = 0.0;
        for (size_t j = 2; j < y.size(); j += 6) {
            sumInfectionParasite_Total += y[j] + y[j + 1] + y[j + 2] + y[j + 3] + y[j + 4] + y[j + 5];
        }

        double weighted_mean_alpha = 0.0;
        // Calculate weighted mean
        for (int j = 0; j < numTraits; ++j) {
            int idx2 = 2 + 6 * j;
            double trait_pop = y[idx2] + y[idx2 + 1] + y[idx2 + 2] + y[idx2 + 3] 
                               + y[idx2 + 4] + y[idx2 + 5];
            if (trait_pop > 0 && sumInfectionParasite_Total > 0) {
                double proportion = trait_pop / sumInfectionParasite_Total;
                weighted_mean_alpha += myAlpha[j] * proportion;
            }
        }

        // Output results
        #pragma omp critical
        {
            outputFile << s_value << "," << weighted_mean_alpha << "\n";
        }

        std::cout << "Simulation completed for s=" << s_value
                  << ", mean_alpha=" << weighted_mean_alpha << std::endl;
    } // End of s_values loop

    outputFile.close();
    return 0;
}
