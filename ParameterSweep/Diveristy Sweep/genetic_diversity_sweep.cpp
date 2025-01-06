// g++ -std=c++14 -O2 -I /opt/homebrew/include -o simulation evSim8.cpp

/*
Average infected population for no virulence evolution 
(average disease prevalance of 10% of evolutionary time)
(no virulence evolution)
*/

#include <iostream>
#include <fstream>
#include <cmath>
#include <vector>
#include <random>
#include <algorithm>
#include <numeric>
#include <boost/numeric/odeint.hpp>

using namespace boost::numeric::odeint;

// Parameters
const int numTraits = 50;        // Maximum number of virulence traits (variants)
const double abs_ext_tol = 0;  // Absolute extinction threshold
const int evolSteps = 1000;      // Number of evolutionary steps

// Random number generator setup
std::random_device rd;
std::mt19937 gen(rd());
std::uniform_real_distribution<> dis(0.0, 1.0);

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
                Q[i][j] = 1 + (N-1)*s;
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

        int idx = N;
        for (int trait = 0; trait < numTraits; ++trait) {
            int idx_trait = idx + trait * (N * N + N);
            int Iab_start = idx_trait;
            int Pb_start = Iab_start + N * N;

            double alpha_sqrt = 2 * 10 * sqrt(params.myAlpha[trait] / 10.0);

            // Pb vector for trait
            std::vector<double> Pb(N);
            for (int i = 0; i < N; ++i) {
                Pb[i] = y[Pb_start + i];
            }

            // Iab matrix for trait
            std::vector<std::vector<double>> Iab(N, std::vector<double>(N, 0.0));
            for (int i = 0; i < N * N; ++i) {
                Iab[i / N][i % N] = y[Iab_start + i];
            }

            // Skip computation if both Iab and Pb are zero for this trait
            double total_Iab_Pb = std::accumulate(Pb.begin(), Pb.end(), 0.0);
            for (int i = 0; i < N; ++i) {
                total_Iab_Pb += std::accumulate(Iab[i].begin(), Iab[i].end(), 0.0);
            }
            if (total_Iab_Pb < abs_ext_tol) {
                continue; // Skip this trait as it has no active populations
            }

            // Update totalPopulation with infected hosts
            for (int i = 0; i < N; ++i) {
                totalPopulation += std::accumulate(Iab[i].begin(), Iab[i].end(), 0.0);
            }

            // Compute derivatives for Iab and Pb
            std::vector<std::vector<double>> dIab(N, std::vector<double>(N, 0.0));
            for (int a = 0; a < N; ++a) {
                for (int b_idx = 0; b_idx < N; ++b_idx) {
                    double Iab_ab = Iab[a][b_idx];
                    double P_b = Pb[b_idx];

                    dIab[a][b_idx] = params.Q[a][b_idx] * alpha_sqrt * P_b * S[a] - (params.d + params.gamma + params.myAlpha[trait]) * Iab_ab;

                    // Accumulate infection and recovery terms
                    infectionS[a] += alpha_sqrt * params.Q[a][b_idx] * P_b * S[a];
                    recoveryS[a] += params.gamma * Iab_ab;
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
            dS[i] = params.b * S[i] * (1.0 - params.q * totalPopulation) - params.d * S[i] - infectionS[i] + recoveryS[i];
        }

        for (int i = 0; i < N; ++i) {
            dydt[i] = dS[i];
        }
    }
};

int main() {
    // Parameters that remain constant
    const double d = 1.0;
    const double q = 1.0;
    const double gamma = 0.2;
    const double theta = 1.2;
    const double delta = 1.5;
    const double b = 12.0;

    // Trait space
    const double alphaLow = 1.0;
    const double alphaHigh = 4.0;
    std::vector<double> myAlpha(numTraits);
    for (int i = 0; i < numTraits; ++i) {
        myAlpha[i] = alphaLow + i * (alphaHigh - alphaLow) / (numTraits - 1);
    }

    // Vectors for s and N values to test
    std::vector<double> s_values = {0.0, 1.0};
    std::vector<int> N_values = {1, 2, 5, 10, 15};  // Adjust as needed

    // Open CSV file to record results
    std::ofstream csvfile("average_virulence_results.csv");
    csvfile << "s,N,Average_Virulence\n";

    // Loop over s values
    for (int N : N_values) {
        // Loop over N values
        for (double s : s_values) {
            std::cout << "Running simulation for s = " << s << ", N = " << N << std::endl;

            Params params;
            params.d = d;
            params.q = q;
            params.gamma = gamma;
            params.theta = theta;
            params.delta = delta;
            params.b = b;
            params.s = s;
            params.myAlpha = myAlpha;
            params.N = N;

            // Create infection matrix Q
            create_infection_matrix(params.s, params.Q, N);

            // Initial conditions
            double constantFactor = 1.0;
            double total_initial_susceptible = 0.9 * constantFactor;
            double S0_value = total_initial_susceptible / N;
            std::vector<double> S0(N, S0_value);

            double total_initial_infected = 0.1 * constantFactor;
            double I0_value = total_initial_infected / (N * N);
            std::vector<double> I0(N * N, I0_value);

            double total_initial_parasites = 0.095 * constantFactor;
            double P0_value = total_initial_parasites / N;
            std::vector<double> P0(N, P0_value);

            // State vector y0
            const int y0_size = N + numTraits * (N * N + N);
            std::vector<double> y0(y0_size, 0.0);

            // Initialize S[0:N]
            for (int i = 0; i < N; ++i) {
                y0[i] = S0[i];
            }

            // Initialize Iab and Pb for trait 0 only
            int idx = N;
            int trait = 0;
            int idx_trait = idx + trait * (N * N + N);
            int Iab_start = idx_trait;
            int Pb_start = Iab_start + N * N;

            // Initialize Iab for trait 0
            for (int i = 0; i < N * N; ++i) {
                y0[Iab_start + i] = I0[i];
            }

            // Initialize Pb for trait 0
            for (int i = 0; i < N; ++i) {
                y0[Pb_start + i] = P0[i];
            }

            // Prepare to store the output
            std::vector<std::vector<double>> Yout(evolSteps, std::vector<double>(y0_size));

            Yout[0] = y0;

            // Evolution simulation
            for (int step = 1; step < evolSteps; ++step) {
                // Time span for integration
                double t0 = 0.0;
                double t1 = 200.0;
                double dt = 1.0;

                // Create the ODE system
                AlleleDynamics system(params);

                // Use the controlled stepper
                typedef runge_kutta_cash_karp54<std::vector<double>> error_stepper_type;
                typedef controlled_runge_kutta<error_stepper_type> controlled_stepper_type;
                controlled_stepper_type controlled_stepper;

                // Integrate the ODE
                std::vector<double> y = Yout[step - 1];
                integrate_adaptive(controlled_stepper, system, y, t0, t1, dt);

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

                // Mutation logic
                // Calculate total infected populations for each trait
                std::vector<double> I_total(numTraits, 0.0);

                for (int trait = 0; trait < numTraits; ++trait) {
                    int idx_trait = N + trait * (N * N + N);
                    int Iab_start = idx_trait;

                    double sum_Iab = 0.0;
                    for (int i = 0; i < N * N; ++i) {
                        sum_Iab += y[Iab_start + i];
                    }
                    I_total[trait] = sum_Iab;
                }

                // Calculate total infected population
                double total_infected = std::accumulate(I_total.begin(), I_total.end(), 0.0);

                if (total_infected == 0.0) {
                    std::cout << "Extinction occurred at step " << step << ". Ending simulation." << std::endl;
                    Yout.resize(step + 1);
                    Yout[step] = y;
                    break;
                }

                // Create cumulative sum vector for infected populations
                std::vector<double> cumsum_I(numTraits, 0.0);
                std::partial_sum(I_total.begin(), I_total.end(), cumsum_I.begin());

                // Choose parent trait to mutate based on infection levels
                double random_number = dis(gen) * total_infected;
                int parent_trait = std::distance(cumsum_I.begin(),
                                                 std::lower_bound(cumsum_I.begin(), cumsum_I.end(), random_number));

                // Determine mutant trait index
                int mutant_trait = (dis(gen) < 0.5) ? parent_trait - 1 : parent_trait + 1;
                if (mutant_trait < 0) mutant_trait = 0;
                if (mutant_trait >= numTraits) mutant_trait = numTraits - 1;

                // Transfer a fraction from parent trait to mutant trait
                double transfer_fraction = 0.1;

                // Indices for parent and mutant traits
                int parent_idx = N + parent_trait * (N * N + N);
                int mutant_idx = N + mutant_trait * (N * N + N);

                // Mutate infected populations (Iab)
                for (int i = 0; i < N * N; ++i) {
                    int parent_Iab_idx = parent_idx + i;
                    int mutant_Iab_idx = mutant_idx + i;
                    double transfer_amount = y[parent_Iab_idx] * transfer_fraction;
                    y[mutant_Iab_idx] += transfer_amount;
                    y[parent_Iab_idx] -= transfer_amount;
                }

                // Mutate parasite populations (Pb)
                for (int i = 0; i < N; ++i) {
                    int parent_Pb_idx = parent_idx + N * N + i;
                    int mutant_Pb_idx = mutant_idx + N * N + i;
                    double transfer_amount_P = y[parent_Pb_idx] * transfer_fraction;
                    y[mutant_Pb_idx] += transfer_amount_P;
                    y[parent_Pb_idx] -= transfer_amount_P;
                }

                // Store the updated state
                Yout[step] = y;

                // Output progress
                if (step % 100 == 0) {
                    std::cout << "Evolutionary step: " << step << std::endl;
                }
            }

            // After the simulation, compute the average virulence
            // We can use the last time point
            std::vector<double> final_state = Yout.back();
            std::vector<double> I_total(numTraits, 0.0);
            double total_infected = 0.0;

            for (int trait = 0; trait < numTraits; ++trait) {
                int idx_trait = N + trait * (N * N + N);
                int Iab_start = idx_trait;

                double sum_Iab = 0.0;
                for (int i = 0; i < N * N; ++i) {
                    sum_Iab += final_state[Iab_start + i];
                }
                I_total[trait] = sum_Iab;
                total_infected += sum_Iab;
            }

            // Compute average virulence weighted by infected populations
            double average_virulence = 0.0;
            if (total_infected > 0) {
                for (int trait = 0; trait < numTraits; ++trait) {
                    average_virulence += (I_total[trait] / total_infected) * params.myAlpha[trait];
                }
            }

            // Record results in CSV file
            csvfile << s << "," << N << "," << average_virulence << "\n";

            std::cout << "Average virulence for s = " << s << ", N = " << N << ": " << average_virulence << std::endl;

        }  // End of s loop
    }  // End of N loop

    csvfile.close();
    std::cout << "All simulations completed. Results are saved in 'average_virulence_results.csv'." << std::endl;

    return 0;
}
