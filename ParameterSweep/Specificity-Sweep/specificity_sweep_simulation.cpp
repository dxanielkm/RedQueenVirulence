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
// g++ -std=c++14 -O2 -I /path/to/boost -fopenmp -o specificitySim specificity_sweep_simulation.cpp

// Global settings
double abs_ext_tol = 1e-6;  
double constantFactor = 10.0;  // Scale for host populations
int numTraits = 50;            // Number of discrete virulence trait classes
int evolSteps = 1000;          // Number of evolutionary time steps

int nAlleles = 5;  // For example, set to 3 (you can change it to 2, 3, 4, etc.)

// Each trait block has a size: (nAlleles*nAlleles infected compartments) + (nAlleles parasite compartments)
int blockSize() {
    return nAlleles * nAlleles + nAlleles;
}

typedef std::vector<double> state_type;

// Create an nAlleles x nAlleles infection specificity matrix Q.
// Q[i][j] = (1 + (nAlleles - 1)*s) / nAlleles if i == j,
//           (1 - s) / nAlleles if i != j.
std::vector<std::vector<double>> create_infection_matrix(int nAlleles, double s) {
    std::vector<std::vector<double>> Q(nAlleles, std::vector<double>(nAlleles, 0.0));
    for (int i = 0; i < nAlleles; i++){
        for (int j = 0; j < nAlleles; j++){
            if (i == j)
                Q[i][j] = (1 + (nAlleles - 1) * s) / double(nAlleles);
            else
                Q[i][j] = (1 - s) / double(nAlleles);
        }
    }
    return Q;
}

// ODE system for host-parasite dynamics with virulence evolution.
// The state vector is organized as follows:
// [S[0], S[1], ..., S[nAlleles-1],
//  then for each trait k = 0,..., numTraits-1:
//      Infected hosts: I_{ij} for i=0,..., nAlleles-1 and j=0,..., nAlleles-1,
//      Free parasites: P_j for j=0,..., nAlleles-1 ]
struct alleleDynamics {
    int nAlleles;
    int numTraits;
    const std::vector<double>& myAlpha;  // Virulence trait values for each trait class.
    double s;       // Specificity parameter.
    double b;       // Intrinsic birth rate.
    double gamma;   // Recovery rate.
    double theta;   // theta_tilde (used in the trade-off: theta_k = theta * sqrt(alpha)).
    double delta;   // Parasite decay rate.
    double d;       // Natural mortality.
    double q;       // Density-dependence coefficient.
    double beta;    // Infection rate (beta = beta0 * nAlleles).

    alleleDynamics(int nAlleles, int numTraits, const std::vector<double>& myAlpha, double s,
                   double b, double gamma, double theta, double delta, double d, double q, double beta)
        : nAlleles(nAlleles), numTraits(numTraits), myAlpha(myAlpha), s(s),
          b(b), gamma(gamma), theta(theta), delta(delta), d(d), q(q), beta(beta) {}

    void operator()(const state_type& y, state_type& dydt, const double /* t */) const {
        // Extract susceptibles (first nAlleles entries)
        std::vector<double> S(nAlleles);
        for (int i = 0; i < nAlleles; i++){
            S[i] = y[i];
        }
        
        // Compute total host population: sum of susceptibles + all infected hosts over all trait blocks.
        double totalPopulation = 0.0;
        for (int i = 0; i < nAlleles; i++){
            totalPopulation += S[i];
        }
        int bs = blockSize();
        for (int k = 0; k < numTraits; k++){
            int offset = nAlleles + k * bs;
            // Infected host compartments: indices offset to offset + (nAlleles*nAlleles - 1)
            for (int idx = 0; idx < nAlleles * nAlleles; idx++){
                totalPopulation += y[offset + idx];
            }
        }
        
        dydt.resize(y.size());
        std::vector<double> infLoss(nAlleles, 0.0);
        std::vector<double> recGain(nAlleles, 0.0);
        
        // Get the specificity matrix Q (nAlleles x nAlleles)
        std::vector<std::vector<double>> Q = create_infection_matrix(nAlleles, s);
        
        // Loop over each virulence trait block.
        for (int k = 0; k < numTraits; k++){
            int offset = nAlleles + k * bs;
            // Compute trait-dependent production: theta_k = theta * sqrt(myAlpha[k])
            double theta_k = theta * std::sqrt(myAlpha[k]);
            // Loop over host alleles i and parasite alleles j for infected hosts.
            for (int i = 0; i < nAlleles; i++){
                for (int j = 0; j < nAlleles; j++){
                    int index = offset + i * nAlleles + j; // I_{ij} for trait k.
                    double I_ij = y[index];
                    // Parasite compartment for allele j in trait k:
                    int p_index = offset + nAlleles * nAlleles + j;
                    double P_j = y[p_index];
                    
                    // Infected host dynamics:
                    // dI_{ij} = beta * Q[i][j] * P_j * S[i] - (d + gamma + myAlpha[k]) * I_{ij}
                    double dI = beta * Q[i][j] * P_j * S[i] - (d + gamma + myAlpha[k]) * I_ij;
                    dydt[index] = dI;
                    
                    // Accumulate loss and recovery contributions for susceptibles.
                    infLoss[i] += beta * Q[i][j] * P_j * S[i];
                    recGain[i]  += gamma * I_ij;
                }
            }
            // Compute free parasite dynamics for trait k.
            for (int j = 0; j < nAlleles; j++){
                int p_index = offset + nAlleles * nAlleles + j;
                double P_j = y[p_index];
                double sum_I = 0.0;
                for (int i = 0; i < nAlleles; i++){
                    int index = offset + i * nAlleles + j;
                    sum_I += y[index];
                }
                // dP_j = theta_k * (sum_{i} I_{ij}) - delta * P_j - [sum_{i} beta * Q[i][j] * S[i]] * P_j.
                double inf_loss = 0.0;
                for (int i = 0; i < nAlleles; i++){
                    inf_loss += beta * Q[i][j] * S[i];
                }
                double dP = theta_k * sum_I - delta * P_j - inf_loss * P_j;
                dydt[p_index] = dP;
            }
        }
        
        // Susceptible host dynamics for each allele i.
        for (int i = 0; i < nAlleles; i++){
            double dS_val = b * S[i] * (1 - q * totalPopulation) - d * S[i] - infLoss[i] + recGain[i];
            dydt[i] = dS_val;
        }
    }
};

////////////////////////
// Main function
////////////////////////
int main() {
    // --- Set parameter values ---
    double default_b     = 10.0;    // Intrinsic birth rate.
    double default_theta = 30.0;    // theta_tilde (scaling factor for production).
    double default_delta = 0.3;     // Parasite decay rate.
    double default_gamma = 1.0;     // Recovery rate.
    // Set beta = beta0 * nAlleles. For example, if beta0 = 1.5 then:
    double default_beta  = 1.5 * nAlleles;  
    double default_d     = 1.0;       // Natural mortality.
    double default_q     = 1.0;       // Density-dependent competition.
    
    // --- Prepare output file ---
    std::ofstream outputFile("specificity_sweep_results_n=5.csv");
    outputFile << "s,MeanAlpha\n";
    
    // --- Define the trait space for virulence evolution ---
    // Let virulence (alpha) vary from alphaLow to alphaHigh.
    double alphaLow  = 1.0;
    double alphaHigh = 4.0;
    std::vector<double> myAlpha(numTraits);
    for (int i = 0; i < numTraits; i++){
        myAlpha[i] = alphaLow + i * (alphaHigh - alphaLow) / (numTraits - 1);
    }
    
    // --- Initial conditions ---
    // Total state vector size = nAlleles (susceptibles) + numTraits * blockSize().
    int totalSize = nAlleles + numTraits * blockSize();
    state_type y0(totalSize, 0.0);
    
    // Initialize susceptibles: for example, set each S[i] = 0.85 * constantFactor.
    for (int i = 0; i < nAlleles; i++){
        y0[i] = 0.85 * constantFactor;
    }
    
    // For each trait block, seed only the first trait (k = 0) with nonzero infected hosts and parasites.
    int bs = blockSize();
    for (int k = 0; k < numTraits; k++){
        int offset = nAlleles + k * bs;
        if (k == 0) {
            // For each combination of host allele i and parasite allele j, set I_{ij} = 0.07 * constantFactor.
            for (int i = 0; i < nAlleles; i++){
                for (int j = 0; j < nAlleles; j++){
                    int index = offset + i * nAlleles + j;
                    y0[index] = 0.07 * constantFactor;
                }
            }
            // For each parasite allele j, set free parasite P_j = 0.1 * constantFactor.
            for (int j = 0; j < nAlleles; j++){
                int p_index = offset + nAlleles * nAlleles + j;
                y0[p_index] = 0.1 * constantFactor;
            }
        } else {
            // For all other traits, set the block to 0.
            for (int idx = offset; idx < offset + bs; idx++){
                y0[idx] = 0.0;
            }
        }
    }
    
    double t_start = 0.0;
    double t_end   = 2000.0;
    
    // --- Define the range and number of s-values ---
    double s_min = 0.0;
    double s_max = 1.0;
    int num_s_values = 21;  // Adjust as desired.
    std::vector<double> s_values;
    s_values.reserve(num_s_values);
    for (int i = 0; i < num_s_values; i++){
        double s_val = s_min + i * ((s_max - s_min) / (num_s_values - 1));
        s_values.push_back(s_val);
    }
    
    // --- Parallel evolutionary simulation using OpenMP ---
    #pragma omp parallel for schedule(dynamic)
    for (size_t idx = 0; idx < s_values.size(); idx++){
        double s_value = s_values[idx];
        double b = default_b;
        double theta = default_theta;
        double delta = default_delta;
        double gamma = default_gamma;
        double beta = default_beta;
        double d = default_d;
        double q = default_q;
        
        // Initialize y for this simulation from y0.
        state_type y = y0;
        
        // Evolution simulation loop.
        for (int step = 1; step <= evolSteps; step++){
            double t0 = 0.0;
            double t1 = 200.0;
            double dt = 1.0;
            
            typedef boost::numeric::odeint::runge_kutta_dopri5<state_type> dopri5_stepper;
            auto stepper = boost::numeric::odeint::make_controlled<dopri5_stepper>(1e-6, 1e-6);
            
            boost::numeric::odeint::integrate_adaptive(
                stepper,
                alleleDynamics(nAlleles, numTraits, myAlpha, s_value, b, gamma, theta, delta, d, q, beta),
                y,
                t0,
                t1,
                dt
            );
            
            // Enforce extinction thresholds and non-negativity.
            for (auto &val : y){
                if (val < abs_ext_tol) { val = 0.0; }
                if (val < 0.0) { val = 0.0; }
            }
            
            // ---------- Mutation Logic ----------
            std::vector<double> Iik_total;
            std::vector<std::pair<int,int>> h_idxs; // (host allele, trait index)
            double total_infected_hosts = 0.0;
            std::vector<double> Pjk_total;
            std::vector<std::pair<int,int>> p_idxs; // (parasite allele, trait index)
            double total_parasites = 0.0;
            
            for (int k = 0; k < numTraits; k++){
                int offset = nAlleles + k * bs;
                // For each host allele, sum infected hosts (over parasite alleles) for trait k.
                for (int i = 0; i < nAlleles; i++){
                    double sumI = 0.0;
                    for (int j = 0; j < nAlleles; j++){
                        int index = offset + i * nAlleles + j;
                        sumI += y[index];
                    }
                    if (sumI > 0.0) {
                        Iik_total.push_back(sumI);
                        h_idxs.emplace_back(i, k);
                        total_infected_hosts += sumI;
                    }
                }
                // For parasites: for each parasite allele in trait k.
                for (int j = 0; j < nAlleles; j++){
                    int p_index = offset + nAlleles * nAlleles + j;
                    double P_val = y[p_index];
                    if (P_val > 0.0) {
                        Pjk_total.push_back(P_val);
                        p_idxs.emplace_back(j, k);
                        total_parasites += P_val;
                    }
                }
            }
            
            if (total_infected_hosts == 0.0 || total_parasites == 0.0)
                break;
            
            std::vector<double> host_cdf(Iik_total.size());
            std::partial_sum(Iik_total.begin(), Iik_total.end(), host_cdf.begin());
            std::vector<double> parasite_cdf(Pjk_total.size());
            std::partial_sum(Pjk_total.begin(), Pjk_total.end(), parasite_cdf.begin());
            
            static thread_local std::mt19937 gen_thread(std::random_device{}());
            static thread_local std::uniform_real_distribution<> dis_thread(0.0, 1.0);
            
            // ---------- Host Mutation ----------
            double r_host = dis_thread(gen_thread) * total_infected_hosts;
            auto host_it = std::upper_bound(host_cdf.begin(), host_cdf.end(), r_host);
            if (host_it != host_cdf.end()){
                size_t host_idx = std::distance(host_cdf.begin(), host_it);
                int i_p = h_idxs[host_idx].first;   // host allele.
                int k_p = h_idxs[host_idx].second;    // current trait index.
                int k_m = k_p;
                if (k_p == 0)
                    k_m = 1;
                else if (k_p == numTraits - 1)
                    k_m = k_p - 1;
                else
                    k_m = (dis_thread(gen_thread) < 0.5) ? (k_p - 1) : (k_p + 1);
                double eta = 0.1;
                int idx_p = nAlleles + k_p * bs;
                int idx_m = nAlleles + k_m * bs;
                // For host allele i_p, infected compartments are at indices: idx_p + i_p*nAlleles + j.
                for (int j = 0; j < nAlleles; j++){
                    int index_p = idx_p + i_p * nAlleles + j;
                    int index_m = idx_m + i_p * nAlleles + j;
                    double transfer = y[index_p] * eta;
                    y[index_m] += transfer;
                    y[index_p] -= transfer;
                }
            }
            
            // ---------- Parasite Mutation ----------
            double r_parasite = dis_thread(gen_thread) * total_parasites;
            auto parasite_it = std::upper_bound(parasite_cdf.begin(), parasite_cdf.end(), r_parasite);
            if (parasite_it != parasite_cdf.end()){
                size_t parasite_idx = std::distance(parasite_cdf.begin(), parasite_it);
                int j_p = p_idxs[parasite_idx].first;  // parasite allele.
                int k_p = p_idxs[parasite_idx].second;   // current trait index.
                int k_m = k_p;
                if (k_p == 0)
                    k_m = 1;
                else if (k_p == numTraits - 1)
                    k_m = k_p - 1;
                else
                    k_m = (dis_thread(gen_thread) < 0.5) ? (k_p - 1) : (k_p + 1);
                double eta = 0.1;
                int idx_p = nAlleles + k_p * bs;
                int idx_m = nAlleles + k_m * bs;
                int p_index_p = idx_p + nAlleles * nAlleles + j_p;
                int p_index_m = idx_m + nAlleles * nAlleles + j_p;
                double transfer = y[p_index_p] * eta;
                y[p_index_m] += transfer;
                y[p_index_p] -= transfer;
            }
            
            for (auto &val : y) {
                if(val < abs_ext_tol) { val = 0.0; }
                if(val < 0.0) { val = 0.0; }
            }
        } // End of evolution steps
        
        // ---------- After simulation, compute the weighted mean virulence ----------
        double sumInfectionParasite_Total = 0.0;
        for (int k = 0; k < numTraits; k++){
            int offset = nAlleles + k * bs;
            // Sum all infected hosts in trait k.
            for (int i = 0; i < nAlleles; i++){
                for (int j = 0; j < nAlleles; j++){
                    int index = offset + i * nAlleles + j;
                    sumInfectionParasite_Total += y[index];
                }
            }
            // Sum free parasites in trait k.
            for (int j = 0; j < nAlleles; j++){
                int p_index = offset + nAlleles * nAlleles + j;
                sumInfectionParasite_Total += y[p_index];
            }
        }
        double weighted_mean_alpha = 0.0;
        for (int k = 0; k < numTraits; k++){
            int offset = nAlleles + k * bs;
            double trait_pop = 0.0;
            for (int i = 0; i < nAlleles; i++){
                for (int j = 0; j < nAlleles; j++){
                    int index = offset + i * nAlleles + j;
                    trait_pop += y[index];
                }
            }
            for (int j = 0; j < nAlleles; j++){
                int p_index = offset + nAlleles * nAlleles + j;
                trait_pop += y[p_index];
            }
            if (trait_pop > 0 && sumInfectionParasite_Total > 0){
                double proportion = trait_pop / sumInfectionParasite_Total;
                weighted_mean_alpha += myAlpha[k] * proportion;
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
