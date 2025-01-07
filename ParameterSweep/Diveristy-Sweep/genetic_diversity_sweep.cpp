// g++-14 -std=c++14 -O2 -fopenmp -I /opt/homebrew/include -o simulation genetic_diversity_sweep.cpp

#include <iostream>
#include <fstream>
#include <sstream>
#include <cmath>
#include <vector>
#include <random>
#include <algorithm>
#include <numeric>
#include <array>     // For std::array
#include <boost/numeric/odeint.hpp>
#include <omp.h>     // OpenMP

using namespace boost::numeric::odeint;

// Parameters
const int numTraits = 50;        // Maximum number of virulence traits (variants)
const double abs_ext_tol = 0;    // Absolute extinction threshold
const int evolSteps = 1000;      // Number of evolutionary steps

// Struct to hold parameters
struct Params {
    std::vector<double> myAlpha;           // Virulence levels for each trait
    double s;
    double b;
    double gamma;
    double theta;
    double delta;
    double d;
    double q;
    std::vector<std::vector<double>> Q;    // Infection matrix Q[N][N]
    int N;                                 // Number of alleles
};

// Function to create the infection matrix Q
void create_infection_matrix(double s, std::vector<std::vector<double>>& Q, int N) {
    Q.resize(N, std::vector<double>(N, 0.0));
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            if (i == j) {
                Q[i][j] = 1 + (N - 1)*s;
            } else {
                Q[i][j] = 1 - s;
            }
        }
    }
}

// The ODE system
class AlleleDynamics {
public:
    Params params;

    AlleleDynamics(const Params& p) : params(p) {}

    void operator()(const std::vector<double>& y, std::vector<double>& dydt, double /* t */) const {
        int N = params.N;  // Number of alleles

        // Susceptible hosts S[i]
        std::vector<double> S(N, 0.0);
        for (int i = 0; i < N; ++i) {
            S[i] = y[i];
        }

        // Initialize dydt
        dydt.assign(y.size(), 0.0);

        // Infection and recovery terms
        std::vector<double> infectionS(N, 0.0);
        std::vector<double> recoveryS(N, 0.0);

        // Calculate total population
        double totalPopulation = std::accumulate(S.begin(), S.end(), 0.0);

        // Each trait has an Iab matrix and a Pb vector
        int idx = N;  // offset for the first trait
        for (int trait = 0; trait < numTraits; ++trait) {
            int idx_trait = idx + trait * (N * N + N);
            int Iab_start = idx_trait;
            int Pb_start  = Iab_start + N * N;

            // Example trade-off function for infection rate
            // (equivalent to beta_k = 20 * sqrt(alpha_k / 10.0))
            double alpha_sqrt = 2 * 10 * std::sqrt(params.myAlpha[trait] / 10.0);

            // Pb[b] for this trait
            std::vector<double> Pb(N);
            for (int i = 0; i < N; ++i) {
                Pb[i] = y[Pb_start + i];
            }

            // Iab[a][b] for this trait
            std::vector<std::vector<double>> Iab(N, std::vector<double>(N, 0.0));
            for (int i = 0; i < N * N; ++i) {
                Iab[i / N][i % N] = y[Iab_start + i];
            }

            // Skip computation if both Iab and Pb are zero
            double total_Iab_Pb = std::accumulate(Pb.begin(), Pb.end(), 0.0);
            for (int i = 0; i < N; ++i) {
                total_Iab_Pb += std::accumulate(Iab[i].begin(), Iab[i].end(), 0.0);
            }
            if (total_Iab_Pb < abs_ext_tol) {
                continue; // No active population for this trait
            }

            // Add infected hosts to totalPopulation
            for (int i = 0; i < N; ++i) {
                totalPopulation += std::accumulate(Iab[i].begin(), Iab[i].end(), 0.0);
            }

            // Compute derivatives for Iab and Pb
            std::vector<std::vector<double>> dIab(N, std::vector<double>(N, 0.0));
            for (int a = 0; a < N; ++a) {
                for (int b_idx = 0; b_idx < N; ++b_idx) {
                    double Iab_ab = Iab[a][b_idx];
                    double P_b    = Pb[b_idx];

                    dIab[a][b_idx] = params.Q[a][b_idx] * alpha_sqrt * P_b * S[a]
                                    - (params.d + params.gamma + params.myAlpha[trait]) * Iab_ab;

                    // Accumulate infection and recovery terms
                    infectionS[a] += alpha_sqrt * params.Q[a][b_idx] * P_b * S[a];
                    recoveryS[a]  += params.gamma * Iab_ab;
                }
            }

            // Flatten dIab and assign to dydt
            for (int i = 0; i < N * N; ++i) {
                dydt[Iab_start + i] = dIab[i / N][i % N];
            }

            // Compute derivatives for Pb
            std::vector<double> dPb(N, 0.0);
            for (int b_idx = 0; b_idx < N; ++b_idx) {
                double sum_Iab = 0.0;
                for (int a = 0; a < N; ++a) {
                    sum_Iab += Iab[a][b_idx];
                }
                dPb[b_idx] = params.theta * sum_Iab - params.delta * Pb[b_idx];
            }

            for (int i = 0; i < N; ++i) {
                dydt[Pb_start + i] = dPb[i];
            }
        }

        // Calculate dS/dt
        std::vector<double> dS(N);
        for (int i = 0; i < N; ++i) {
            dS[i] = params.b * S[i] * (1.0 - params.q * totalPopulation)
                    - params.d * S[i]
                    - infectionS[i]
                    + recoveryS[i];
        }

        for (int i = 0; i < N; ++i) {
            dydt[i] = dS[i];
        }
    }
};

int main() 
{
    // Fixed parameters
    const double d_fixed = 1.0;
    const double q_fixed = 1.0;

    // Ranges for random sampling
    double b_min = 5.0,    b_max = 20.0;
    double th_min = 5.0,   th_max = 20.0;
    double de_min = 0.5,   de_max = 1.5;
    double ga_min = 0.3,   ga_max = 1.2;

    // Number of random sets to run
    const int num_random_sets = 50;

    // Create distributions for parameter sampling
    std::uniform_real_distribution<> dist_b(b_min, b_max);
    std::uniform_real_distribution<> dist_th(th_min, th_max);
    std::uniform_real_distribution<> dist_de(de_min, de_max);
    std::uniform_real_distribution<> dist_ga(ga_min, ga_max);

    // Trait space for virulence
    const double alphaLow  = 1.0;
    const double alphaHigh = 4.0;
    std::vector<double> myAlpha(numTraits);
    for (int i = 0; i < numTraits; ++i) {
        myAlpha[i] = alphaLow + i * (alphaHigh - alphaLow) / (numTraits - 1);
    }

    // Values of s and N to test for each random parameter set
    std::vector<double> s_values = {0.0, 1.0};
    std::vector<int>    N_values = {1, 2, 5, 10, 15};

    // Prepare to store CSV output: s, N, b, gamma, theta, delta, Average_Virulence
    std::ofstream csvfile("average_virulence_results.csv");
    csvfile << "s,N,b,gamma,theta,delta,Average_Virulence\n";
    csvfile.close();  // We'll append inside the parallel region.

    // ---------------------------
    // Parallelize over random sets
    // ---------------------------
#pragma omp parallel
    {
        // Thread-local random device
        std::random_device rd_local;

        // Combine seeds into 32-bit integers
        std::uint32_t seed1 = static_cast<std::uint32_t>(rd_local());
        std::uint32_t seed2 = static_cast<std::uint32_t>(omp_get_thread_num());

        // Store in an array for std::seed_seq
        std::array<std::uint32_t, 2> seeds = { seed1, seed2 };
        std::seed_seq sseq(seeds.begin(), seeds.end());

        // Create a thread-local Mersenne Twister
        std::mt19937 gen_local(sseq);

        // Create a local distribution [0, 1)
        std::uniform_real_distribution<double> local_dis(0.0, 1.0);

        // Use a local string buffer to collect CSV lines
        std::stringstream local_csv_buffer;

#pragma omp for schedule(dynamic)
        for (int r = 0; r < num_random_sets; ++r)
        {
            // 1) Sample random parameters
            double b_rand  = dist_b(gen_local);
            double th_rand = dist_th(gen_local);
            double de_rand = dist_de(gen_local);
            double ga_rand = dist_ga(gen_local);

            // 2) Loop over N and s
            for (int N_val : N_values) {
                for (double s_val : s_values) 
                {
                    // Build Params
                    Params params;
                    params.d       = d_fixed;
                    params.q       = q_fixed;
                    params.b       = b_rand;
                    params.gamma   = ga_rand;
                    params.theta   = th_rand;
                    params.delta   = de_rand;
                    params.s       = s_val;
                    params.myAlpha = myAlpha;
                    params.N       = N_val;

                    // Create infection matrix
                    create_infection_matrix(params.s, params.Q, N_val);

                    // Initial Conditions
                    double constantFactor            = 1.0;
                    double total_initial_susceptible = 0.9 * constantFactor;
                    double S0_value                 = total_initial_susceptible / N_val;
                    std::vector<double> S0(N_val, S0_value);

                    double total_initial_infected    = 0.1 * constantFactor;
                    double I0_value                 = total_initial_infected / (N_val * N_val);
                    std::vector<double> I0(N_val * N_val, I0_value);

                    double total_initial_parasites   = 0.095 * constantFactor;
                    double P0_value                 = total_initial_parasites / N_val;
                    std::vector<double> P0(N_val, P0_value);

                    // State vector y0
                    const int y0_size = N_val + numTraits * (N_val * N_val + N_val);
                    std::vector<double> y0(y0_size, 0.0);

                    // Fill in susceptible hosts
                    for (int i = 0; i < N_val; ++i) {
                        y0[i] = S0[i];
                    }
                    // Initialize infected & parasite for trait 0 only
                    int idx       = N_val;
                    int trait     = 0;
                    int idx_trait = idx + trait * (N_val * N_val + N_val);
                    int Iab_start = idx_trait;
                    int Pb_start  = Iab_start + N_val * N_val;

                    for (int i = 0; i < N_val * N_val; ++i) {
                        y0[Iab_start + i] = I0[i];
                    }
                    for (int i = 0; i < N_val; ++i) {
                        y0[Pb_start + i] = P0[i];
                    }

                    // Prepare to store simulation states
                    std::vector<std::vector<double>> Yout(evolSteps, std::vector<double>(y0_size));
                    Yout[0] = y0;

                    // Evolution simulation
                    for (int step = 1; step < evolSteps; ++step) {
                        // Time span for integration
                        double t0 = 0.0;
                        double t1 = 200.0;
                        double dt = 1.0;

                        // ODE system
                        AlleleDynamics system(params);

                        // Controlled stepper
                        typedef runge_kutta_cash_karp54<std::vector<double>> error_stepper_type;
                        typedef controlled_runge_kutta<error_stepper_type>   controlled_stepper_type;
                        controlled_stepper_type controlled_stepper;

                        // Integrate
                        std::vector<double> y = Yout[step - 1];
                        integrate_adaptive(controlled_stepper, system, y, t0, t1, dt);

                        // Apply extinction thresholds
                        for (auto &val : y) {
                            if (val < abs_ext_tol) {
                                val = 0.0;
                            }
                            if (val < 0.0) {
                                val = 0.0; // ensure non-negative
                            }
                        }

                        // Mutation logic
                        std::vector<double> I_total(numTraits, 0.0);
                        for (int t_index = 0; t_index < numTraits; ++t_index) {
                            int idx_t = N_val + t_index * (N_val * N_val + N_val);
                            double sum_Iab = 0.0;
                            for (int i = 0; i < N_val * N_val; ++i) {
                                sum_Iab += y[idx_t + i];
                            }
                            I_total[t_index] = sum_Iab;
                        }

                        double total_infected = std::accumulate(I_total.begin(), I_total.end(), 0.0);
                        if (total_infected == 0.0) {
                            // Extinction
                            Yout.resize(step + 1);
                            Yout[step] = y;
                            break;
                        }

                        // Weighted random pick of parent trait
                        std::vector<double> cumsum_I(numTraits, 0.0);
                        std::partial_sum(I_total.begin(), I_total.end(), cumsum_I.begin());

                        // local_dis is our uniform(0,1) distribution
                        double rnd = local_dis(gen_local) * total_infected;
                        int parent_trait = std::distance(
                            cumsum_I.begin(),
                            std::lower_bound(cumsum_I.begin(), cumsum_I.end(), rnd)
                        );

                        // Mutant trait
                        int mutant_trait = (local_dis(gen_local) < 0.5) 
                                          ? parent_trait - 1 
                                          : parent_trait + 1;
                        if (mutant_trait < 0) mutant_trait = 0;
                        if (mutant_trait >= numTraits) mutant_trait = numTraits - 1;

                        // Transfer fraction
                        double transfer_fraction = 0.1;

                        // Indices
                        int parent_idx = N_val + parent_trait * (N_val * N_val + N_val);
                        int mutant_idx = N_val + mutant_trait * (N_val * N_val + N_val);

                        // Mutate infected (Iab)
                        for (int i = 0; i < N_val * N_val; ++i) {
                            int pIab = parent_idx + i;
                            int mIab = mutant_idx + i;
                            double transfer_amount = y[pIab] * transfer_fraction;
                            y[mIab] += transfer_amount;
                            y[pIab] -= transfer_amount;
                        }

                        // Mutate parasites (Pb)
                        for (int i = 0; i < N_val; ++i) {
                            int pPb = parent_idx + N_val * N_val + i;
                            int mPb = mutant_idx + N_val * N_val + i;
                            double transfer_amount_P = y[pPb] * transfer_fraction;
                            y[mPb] += transfer_amount_P;
                            y[pPb] -= transfer_amount_P;
                        }

                        Yout[step] = y;
                    } // end evol steps

                    // Final average virulence
                    std::vector<double> final_state = Yout.back();
                    std::vector<double> I_total_final(numTraits, 0.0);
                    double total_infected_final = 0.0;

                    for (int t_index = 0; t_index < numTraits; ++t_index) {
                        int idx_t = N_val + t_index * (N_val * N_val + N_val);
                        double sum_Iab = 0.0;
                        for (int i = 0; i < N_val * N_val; ++i) {
                            sum_Iab += final_state[idx_t + i];
                        }
                        I_total_final[t_index] = sum_Iab;
                        total_infected_final  += sum_Iab;
                    }

                    double average_virulence = 0.0;
                    if (total_infected_final > 0.0) {
                        for (int t_index = 0; t_index < numTraits; ++t_index) {
                            average_virulence += 
                                (I_total_final[t_index] / total_infected_final) 
                                * params.myAlpha[t_index];
                        }
                    }

                    // Collect CSV output in local buffer
                    local_csv_buffer 
                        << s_val << ","
                        << N_val << ","
                        << b_rand << ","
                        << ga_rand << ","
                        << th_rand << ","
                        << de_rand << ","
                        << average_virulence 
                        << "\n";
                }
            }
        } // end for (r)

        // Now append local results to the global CSV in a threadsafe way
#pragma omp critical
        {
            // Reopen file in append mode
            std::ofstream csvfile_append("average_virulence_results.csv", std::ios::app);
            csvfile_append << local_csv_buffer.str();
            csvfile_append.close();
        }
    } // end parallel region

    std::cout << "\nAll simulations completed. Results are in 'average_virulence_results.csv'.\n";
    return 0;
}
