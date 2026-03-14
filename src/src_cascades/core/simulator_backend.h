#ifndef SIMULATOR_BACKEND_H
#define SIMULATOR_BACKEND_H

#include <vector>
#include <deque>
#include <random>
#include <map>
#include <string>

// A struct to hold the detailed metrics for a single cascade run.
struct CascadeMetricsResult {
    int size;
    int depth;
    int max_width;
    double total_intensity_effort;
    std::map<int, std::vector<int>> temporal_history;
};

CascadeMetricsResult run_cascade_cpp(
    double p,
    double ell, // Assuming Poisson for speed
    int initial_intensity,
    bool record_history,
    int max_steps
) {
    if (initial_intensity <= 0) {
        return {0, 0, 0, 0.0, {}};
    }

    // High-quality random number generation, seeded once per thread
    thread_local std::mt19937 rng(std::random_device{}());
    std::poisson_distribution<int> poisson_dist(ell);
    std::uniform_real_distribution<double> uniform_dist(0.0, 1.0);

    std::deque<int> current_gen_q;
    current_gen_q.push_back(initial_intensity);
    std::deque<int> next_gen_q;

    int total_size = 0;
    int depth = 0;
    int max_width = 0;
    double total_intensity_effort = 0.0;
    std::map<int, std::vector<int>> history;

    while (!current_gen_q.empty() && total_size < max_steps) {
        int gen_width = current_gen_q.size();
        if (gen_width > max_width) {
            max_width = gen_width;
        }

        if (record_history) {
            history.emplace(depth, std::vector<int>(current_gen_q.begin(), current_gen_q.end()));
        }

        for (int i = 0; i < gen_width; ++i) {
            if (total_size >= max_steps) {
                break;
            }
            int current_intensity = current_gen_q.front();
            current_gen_q.pop_front();

            total_size++;
            total_intensity_effort += current_intensity;

            int num_children = poisson_dist(rng);
            for (int j = 0; j < num_children; ++j) {
                bool is_receptive = uniform_dist(rng) < p;
                int new_intensity = is_receptive ? current_intensity + 1 : current_intensity - 1;

                if (new_intensity > 0) {
                    next_gen_q.push_back(new_intensity);
                }
            }
        }
        current_gen_q = std::move(next_gen_q);
        depth++;
    }

    return {total_size, depth, max_width, total_intensity_effort, history};
}

#endif // SIMULATOR_BACKEND_H
