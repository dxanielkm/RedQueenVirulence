// Compile with:
// g++-14 -std=c++14 -O2 -I /opt/homebrew/include -fopenmp -o specificitySim specificity_sweep_simulation.cpp

#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <random>
#include <algorithm>
#include <numeric>
#include <limits>
#include <sstream>
#include <tuple>
#include <boost/numeric/odeint.hpp>
#include <omp.h>  // For parallelization

// Global settings
double abs_ext_tol = 1e-6;  
double constantFactor = 10.0;  // Scale for host populations
int numTraits = 50;            // Number of discrete virulence trait classes
int evolSteps = 1000;          // Number of evolutionary time steps

// Global trait space bounds for virulence
double alphaLow  = 1.0;
double alphaHigh = 4.0;

// Default ecological parameters (except thetã and nAlleles, which will be varied)
double default_b     = 10.0;    // Intrinsic birth rate.
double default_delta = 0.3;     // Parasite decay rate.
double default_gamma = 1.0;     // Recovery rate.
double default_d     = 1.0;     // Natural mortality.
double default_q     = 1.0;     // Density-dependent competition.
// For infection rate we use: beta = beta0 * nAlleles; here beta0 is set to 1.5.
double default_beta0 = 1.5;

// The number of replicate simulation runs for each parameter combination.
int num_replicates = 10;

// Each trait block has a size: (nAlleles*nAlleles infected compartments) + (nAlleles parasite compartments)
int blockSize(int nAlleles) {
    return nAlleles * nAlleles + nAlleles;
}

typedef std::vector<double> state_type;

// Create an nAlleles x nAlleles infection specificity matrix Q.
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
struct alleleDynamics {
    int nAlleles;
    int numTraits;
    const std::vector<double>& myAlpha;  // Virulence trait values.
    double s;       // Specificity parameter.
    double b;       // Intrinsic birth rate.
    double gamma;   // Recovery rate.
    double theta;   // thetã (scaling factor for production)
    double delta;   // Parasite decay rate.
    double d;       // Natural mortality.
    double q;       // Density-dependence coefficient.
    double beta;    // Infection rate (beta = beta0 * nAlleles).

    alleleDynamics(int nAlleles, int numTraits, const std::vector<double>& myAlpha, double s,
                   double b, double gamma, double theta, double delta, double d, double q, double beta)
        : nAlleles(nAlleles), numTraits(numTraits), myAlpha(myAlpha), s(s),
          b(b), gamma(gamma), theta(theta), delta(delta), d(d), q(q), beta(beta) {}

    void operator()(const state_type& y, state_type& dydt, const double /* t */) const {
        // Extract susceptibles.
        std::vector<double> S(nAlleles);
        for (int i = 0; i < nAlleles; i++){
            S[i] = y[i];
        }
        // Total population: susceptibles plus infected hosts.
        double totalPopulation = 0.0;
        for (int i = 0; i < nAlleles; i++){
            totalPopulation += S[i];
        }
        int bs = blockSize(nAlleles);
        for (int k = 0; k < numTraits; k++){
            int offset = nAlleles + k * bs;
            for (int idx = 0; idx < nAlleles * nAlleles; idx++){
                totalPopulation += y[offset + idx];
            }
        }
        
        dydt.resize(y.size());
        std::vector<double> infLoss(nAlleles, 0.0);
        std::vector<double> recGain(nAlleles, 0.0);
        
        std::vector<std::vector<double>> Q = create_infection_matrix(nAlleles, s);
        
        for (int k = 0; k < numTraits; k++){
            int offset = nAlleles + k * bs;
            double theta_k = theta * std::sqrt(myAlpha[k]);
            for (int i = 0; i < nAlleles; i++){
                for (int j = 0; j < nAlleles; j++){
                    int index = offset + i * nAlleles + j;
                    double I_ij = y[index];
                    int p_index = offset + nAlleles * nAlleles + j;
                    double P_j = y[p_index];
                    double dI = beta * Q[i][j] * P_j * S[i] - (d + gamma + myAlpha[k]) * I_ij;
                    dydt[index] = dI;
                    infLoss[i] += beta * Q[i][j] * P_j * S[i];
                    recGain[i]  += gamma * I_ij;
                }
            }
            for (int j = 0; j < nAlleles; j++){
                int p_index = offset + nAlleles * nAlleles + j;
                double P_j = y[p_index];
                double sum_I = 0.0;
                for (int i = 0; i < nAlleles; i++){
                    int index = offset + i * nAlleles + j;
                    sum_I += y[index];
                }
                double inf_loss = 0.0;
                for (int i = 0; i < nAlleles; i++){
                    inf_loss += beta * Q[i][j] * S[i];
                }
                double dP = theta_k * sum_I - delta * P_j - inf_loss * P_j;
                dydt[p_index] = dP;
            }
        }
        for (int i = 0; i < nAlleles; i++){
            double dS_val = b * S[i] * (1 - q * totalPopulation) - d * S[i] - infLoss[i] + recGain[i];
            dydt[i] = dS_val;
        }
    }
};

// A simple structure to store simulation results.
struct SimulationResult {
    double s;
    double meanAlpha;
    int replicate;
};

////////////////////////
// Main function
////////////////////////
int main() {
    // Define the sets of parameter values.
    std::vector<int> N_values = {5}; // Example values.
    std::vector<double> thetaTilde_values = {20};  // thetã values.
    
    // Define the s-values.
    double s_min = 1.0, s_max = 1.0;
    int num_s_values = 1;
    std::vector<double> s_values;
    s_values.push_back(1.0);
    /*
    s_values.reserve(num_s_values);
    for (int i = 0; i < num_s_values; i++){
        double s_val = s_min + i * ((s_max - s_min) / (num_s_values - 1));
        s_values.push_back(s_val);
    }
    */
    
    
    // Outer loops: for each (N, thetã) combination, we will produce one CSV file.
    for (int n : N_values) {
        for (double theta_tilde : thetaTilde_values) {
            int current_nAlleles = n;
            double current_theta = theta_tilde;
            double current_beta = default_beta0 * current_nAlleles;
            
            // Prepare the virulence trait space.
            std::vector<double> myAlpha(numTraits);
            for (int i = 0; i < numTraits; i++){
                myAlpha[i] = alphaLow + i * (alphaHigh - alphaLow) / (numTraits - 1);
            }
            
            // Prepare initial conditions y0.
            int bs = blockSize(current_nAlleles);
            int totalSize = current_nAlleles + numTraits * bs;
            state_type y0(totalSize, 0.0);
            for (int i = 0; i < current_nAlleles; i++){
                y0[i] = 0.85 * constantFactor;
            }
            for (int k = 0; k < numTraits; k++){
                int offset = current_nAlleles + k * bs;
                if (k == 0) {
                    for (int i = 0; i < current_nAlleles; i++){
                        for (int j = 0; j < current_nAlleles; j++){
                            int index = offset + i * current_nAlleles + j;
                            y0[index] = 0.07 * constantFactor;
                        }
                    }
                    for (int j = 0; j < current_nAlleles; j++){
                        int p_index = offset + current_nAlleles * current_nAlleles + j;
                        y0[p_index] = 0.1 * constantFactor;
                    }
                } else {
                    for (int idx = offset; idx < offset + bs; idx++){
                        y0[idx] = 0.0;
                    }
                }
            }
            
            // This vector will hold all simulation results for this combination.
            std::vector<SimulationResult> results;
            
            // Parallelize over replicates and s-values (each simulation run is independent).
            #pragma omp parallel for collapse(2) schedule(dynamic)
            for (int rep = 0; rep < num_replicates; rep++){
                for (size_t s_idx = 0; s_idx < s_values.size(); s_idx++){
                    double s_value = s_values[s_idx];
                    
                    // Reinitialize state for this simulation.
                    state_type y = y0;
                    
                    double b = default_b;
                    double theta = current_theta;  // current thetã
                    double delta = default_delta;
                    double gamma = default_gamma;
                    double beta = current_beta;
                    double d = default_d;
                    double q = default_q;
                    
                    // Evolution simulation loop.
                    for (int step = 1; step <= evolSteps; step++){
                        double t0 = 0.0, t1 = 200.0, dt = 1.0;
                        typedef boost::numeric::odeint::runge_kutta_dopri5<state_type> dopri5_stepper;
                        auto stepper = boost::numeric::odeint::make_controlled<dopri5_stepper>(1e-6, 1e-6);
                        
                        boost::numeric::odeint::integrate_adaptive(
                            stepper,
                            alleleDynamics(current_nAlleles, numTraits, myAlpha, s_value,
                                           b, gamma, theta, delta, d, q, beta),
                            y, t0, t1, dt
                        );
                        
                        // Enforce extinction thresholds.
                        for (auto &val : y) {
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
                            int offset = current_nAlleles + k * bs;
                            for (int i = 0; i < current_nAlleles; i++){
                                double sumI = 0.0;
                                for (int j = 0; j < current_nAlleles; j++){
                                    int index = offset + i * current_nAlleles + j;
                                    sumI += y[index];
                                }
                                if (sumI > 0.0) {
                                    Iik_total.push_back(sumI);
                                    h_idxs.emplace_back(i, k);
                                    total_infected_hosts += sumI;
                                }
                            }
                            for (int j = 0; j < current_nAlleles; j++){
                                int p_index = offset + current_nAlleles * current_nAlleles + j;
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
                        
                        // Use a local RNG.
                        std::mt19937 gen(std::random_device{}());
                        std::uniform_real_distribution<> dis(0.0, 1.0);
                        
                        // ---------- Host Mutation ----------
                        double r_host = dis(gen) * total_infected_hosts;
                        auto host_it = std::upper_bound(host_cdf.begin(), host_cdf.end(), r_host);
                        if (host_it != host_cdf.end()){
                            size_t host_idx = std::distance(host_cdf.begin(), host_it);
                            int i_p = h_idxs[host_idx].first;
                            int k_p = h_idxs[host_idx].second;
                            int k_m = k_p;
                            if (k_p == 0)
                                k_m = 1;
                            else if (k_p == numTraits - 1)
                                k_m = k_p - 1;
                            else
                                k_m = (dis(gen) < 0.5) ? (k_p - 1) : (k_p + 1);
                            double eta = 0.1;
                            int idx_p = current_nAlleles + k_p * bs;
                            int idx_m = current_nAlleles + k_m * bs;
                            for (int j = 0; j < current_nAlleles; j++){
                                int index_p = idx_p + i_p * current_nAlleles + j;
                                int index_m = idx_m + i_p * current_nAlleles + j;
                                double transfer = y[index_p] * eta;
                                y[index_m] += transfer;
                                y[index_p] -= transfer;
                            }
                        }
                        
                        // ---------- Parasite Mutation ----------
                        double r_parasite = dis(gen) * total_parasites;
                        auto parasite_it = std::upper_bound(parasite_cdf.begin(), parasite_cdf.end(), r_parasite);
                        if (parasite_it != parasite_cdf.end()){
                            size_t parasite_idx = std::distance(parasite_cdf.begin(), parasite_it);
                            int j_p = p_idxs[parasite_idx].first;
                            int k_p = p_idxs[parasite_idx].second;
                            int k_m = k_p;
                            if (k_p == 0)
                                k_m = 1;
                            else if (k_p == numTraits - 1)
                                k_m = k_p - 1;
                            else
                                k_m = (dis(gen) < 0.5) ? (k_p - 1) : (k_p + 1);
                            double eta = 0.1;
                            int idx_p = current_nAlleles + k_p * bs;
                            int idx_m = current_nAlleles + k_m * bs;
                            int p_index_p = idx_p + current_nAlleles * current_nAlleles + j_p;
                            int p_index_m = idx_m + current_nAlleles * current_nAlleles + j_p;
                            double transfer = y[p_index_p] * eta;
                            y[p_index_m] += transfer;
                            y[p_index_p] -= transfer;
                        }
                        
                        for (auto &val : y) {
                            if(val < abs_ext_tol) { val = 0.0; }
                            if(val < 0.0) { val = 0.0; }
                        }
                    } // End evolution steps
                    
                    // Compute weighted mean virulence.
                    double sumIP_Total = 0.0;
                    for (int k = 0; k < numTraits; k++){
                        int offset = current_nAlleles + k * bs;
                        for (int i = 0; i < current_nAlleles; i++){
                            for (int j = 0; j < current_nAlleles; j++){
                                int index = offset + i * current_nAlleles + j;
                                sumIP_Total += y[index];
                            }
                        }
                        for (int j = 0; j < current_nAlleles; j++){
                            int p_index = offset + current_nAlleles * current_nAlleles + j;
                            sumIP_Total += y[p_index];
                        }
                    }
                    double weighted_mean_alpha = 0.0;
                    for (int k = 0; k < numTraits; k++){
                        int offset = current_nAlleles + k * bs;
                        double trait_pop = 0.0;
                        for (int i = 0; i < current_nAlleles; i++){
                            for (int j = 0; j < current_nAlleles; j++){
                                int index = offset + i * current_nAlleles + j;
                                trait_pop += y[index];
                            }
                        }
                        for (int j = 0; j < current_nAlleles; j++){
                            int p_index = offset + current_nAlleles * current_nAlleles + j;
                            trait_pop += y[p_index];
                        }
                        if (trait_pop > 0 && sumIP_Total > 0){
                            double proportion = trait_pop / sumIP_Total;
                            weighted_mean_alpha += myAlpha[k] * proportion;
                        }
                    }
                    
                    #pragma omp critical
                    {
                        // Store result.
                        results.push_back(SimulationResult{s_value, weighted_mean_alpha, rep});
                    }
                } // End s_values loop.
            } // End replicate loop.
            
            // Write results to file.
            std::ostringstream fname;
            fname << "2specificity_sweep_results_N=" << current_nAlleles 
                  << "_theta=" << current_theta << ".csv";
            std::ofstream outputFile(fname.str());
            outputFile << "s,MeanAlpha,replicate\n";
            for (auto &r : results) {
                outputFile << r.s << "," << r.meanAlpha << "," << r.replicate << "\n";
            }
            outputFile.close();
            std::cout << "Finished simulation for N=" << current_nAlleles 
                      << ", theta_tilde=" << current_theta << std::endl;
        } // End theta_tilde loop.
    } // End N loop.
    return 0;
}
