#include <iostream>
#include <vector>
#include <string>
#include <chrono>
#include <thread>
#include <cstdio>
#include <cstdlib>
#include <sstream>
#include <cmath>
#include <deque>

struct NpuMetrics {
    std::chrono::system_clock::time_point ts;
    double power_w;      // T1 power (W)
    double temp_c;       // T2 temperature (C)
    double ai_core_util; // T3 AI Core usage (%)
    double mem_util;     // T6 memory usage (%)
    double mem_bw_util;  // T7 memory bandwidth usage (%)

    NpuMetrics()
        : power_w(0.0),
          temp_c(0.0),
          ai_core_util(0.0),
          mem_util(0.0),
          mem_bw_util(0.0) {}
};

// ----------- Helper: run command and get output -----------
std::string runCommand(const std::string &cmd) {
    std::string result;
    FILE *pipe = popen(cmd.c_str(), "r");
    if (!pipe) {
        return result;
    }
    char buffer[512];
    while (fgets(buffer, sizeof(buffer), pipe) != nullptr) {
        result += buffer;
    }
    pclose(pipe);
    return result;
}

double parseLastNumber(const std::string &line) {
    std::istringstream iss(line);
    std::string token;
    double value = 0.0;
    while (iss >> token) {
        try {
            value = std::stod(token);
        } catch (...) {
        }
    }
    return value;
}

bool collectNpuMetrics(int device_id, int chip_id, NpuMetrics &m) {
    m.ts = std::chrono::system_clock::now();

    //   npu-smi info -i <id> -c <chip> -s p   -> power
    //   npu-smi info -i <id> -c <chip> -s t   -> temperature
    //   npu-smi info -i <id> -c <chip> -s a   -> AI Core usage
    //   npu-smi info -i <id> -c <chip> -s m   -> memory usage
    //   npu-smi info -i <id> -c <chip> -s b   -> memory bandwidth usage

    std::string base = "npu-smi info -i " + std::to_string(device_id) +
                       " -c " + std::to_string(chip_id) + " -s ";

    std::string out_p = runCommand(base + "p");
    std::string out_t = runCommand(base + "t");
    std::string out_a = runCommand(base + "a");
    std::string out_m = runCommand(base + "m");
    std::string out_b = runCommand(base + "b");

    if (out_p.empty() || out_t.empty()) {
        std::cerr << "Failed to get npu-smi info output.\n";
        return false;
    }

    auto getFirstLine = [](const std::string &s) {
        std::istringstream iss(s);
        std::string line;
        if (std::getline(iss, line)) return line;
        return std::string();
    };

    m.power_w      = parseLastNumber(getFirstLine(out_p));
    m.temp_c       = parseLastNumber(getFirstLine(out_t));
    m.ai_core_util = parseLastNumber(getFirstLine(out_a));
    m.mem_util     = parseLastNumber(getFirstLine(out_m));
    m.mem_bw_util  = parseLastNumber(getFirstLine(out_b));

    return true;
}

// ----------- Dynamic baseline + anomaly detection -----------

struct SlidingWindowStats {
    std::deque<double> window;
    size_t max_size;
    double mean;
    double stddev;

    SlidingWindowStats(size_t w = 100)
        : max_size(w), mean(0.0), stddev(0.0) {}

    void add(double v) {
        window.push_back(v);
        if (window.size() > max_size) {
            window.pop_front();
        }
        compute();
    }

    void compute() {
        if (window.empty()) {
            mean = 0.0;
            stddev = 0.0;
            return;
        }
        double sum = 0.0;
        for (double v : window) sum += v;
        mean = sum / static_cast<double>(window.size());
        double var = 0.0;
        for (double v : window) {
            double d = v - mean;
            var += d * d;
        }
        stddev = std::sqrt(var / static_cast<double>(window.size()));
    }

    bool ready(size_t min_samples) const {
        return window.size() >= min_samples;
    }
};

struct CusumDetector {
    double s_pos;
    double s_neg;
    double delta;   // reference offset
    double h;       // decision threshold
    double baseline_mean;

    CusumDetector(double d = 0.1, double h_val = 5.0)
        : s_pos(0.0),
          s_neg(0.0),
          delta(d),
          h(h_val),
          baseline_mean(0.0) {}

    void setBaseline(double mu) {
        baseline_mean = mu;
    }

    bool update(double x) {
        double k = delta;
        double y = x - baseline_mean;

        s_pos = std::max(0.0, s_pos + (y - k));
        s_neg = std::max(0.0, s_neg - (y + k));

        if (s_pos > h || s_neg > h) {
            s_pos = 0.0;
            s_neg = 0.0;
            return true;
        }
        return false;
    }
};

struct ThresholdState {
    double lower;
    double upper;
    int consec_limit;
    int consec_count;

    ThresholdState(double lo, double up, int k)
        : lower(lo), upper(up), consec_limit(k), consec_count(0) {}

    bool update(double x) {
        if (x < lower || x > upper) {
            consec_count++;
        } else {
            consec_count = 0;
        }
        return consec_count >= consec_limit;
    }
};

int main(int argc, char *argv[]) {
    int device_id = 0;
    int chip_id   = 0;
    double freq_hz = 1.0;

    if (argc > 1) {
        freq_hz = std::stod(argv[1]);
    }
    if (freq_hz <= 0.0) freq_hz = 1.0;

    std::cout << "NPU monitor started, frequency = "
              << freq_hz << " Hz\n";

    // Dynamic baselines for temperature and power
    SlidingWindowStats temp_stats(100);
    SlidingWindowStats power_stats(100);

    // CUSUM detectors
    CusumDetector temp_cusum(0.2, 5.0);
    CusumDetector power_cusum(1.0, 5.0);

    // Static thresholds
    ThresholdState temp_thr(0.0, 85.0, 3);
    ThresholdState power_thr(0.0, 300.0, 3);

    const size_t MIN_BASELINE_SAMPLES = 20;
    const double K_SIGMA = 3.0;

    auto interval = std::chrono::milliseconds(
        static_cast<int>(1000.0 / freq_hz)
    );

    while (true) {
        NpuMetrics m;
        if (!collectNpuMetrics(device_id, chip_id, m)) {
            std::this_thread::sleep_for(interval);
            continue;
        }

        // Update sliding windows
        temp_stats.add(m.temp_c);
        power_stats.add(m.power_w);

        // Update CUSUM baseline when enough data
        if (temp_stats.ready(MIN_BASELINE_SAMPLES)) {
            temp_cusum.setBaseline(temp_stats.mean);
        }
        if (power_stats.ready(MIN_BASELINE_SAMPLES)) {
            power_cusum.setBaseline(power_stats.mean);
        }

        // Print current metrics
        auto t = std::chrono::system_clock::to_time_t(m.ts);
        std::cout << "----------------------------------------\n";
        std::cout << "Time: " << std::ctime(&t); // has newline
        std::cout << "Power(W): " << m.power_w
                  << ", Temp(C): " << m.temp_c
                  << ", AI Core(%): " << m.ai_core_util
                  << ", Mem(%): " << m.mem_util
                  << ", MemBW(%): " << m.mem_bw_util << "\n";

        // ---- Static threshold anomalies (R1-like) ----
        bool temp_thr_anom  = temp_thr.update(m.temp_c);
        bool power_thr_anom = power_thr.update(m.power_w);

        if (temp_thr_anom) {
            std::cout << "[ALERT][R1] Temperature static threshold exceeded. "
                      << "value=" << m.temp_c
                      << " (limit<=85C, 3 consecutive)\n";
        }
        if (power_thr_anom) {
            std::cout << "[ALERT][R1] Power static threshold exceeded. "
                      << "value=" << m.power_w
                      << " (limit<=300W, 3 consecutive)\n";
        }

        // ---- Dynamic sigma-rule anomalies (baseline) ----
        if (temp_stats.ready(MIN_BASELINE_SAMPLES) &&
            temp_stats.stddev > 0.0) {
            double z = std::abs(m.temp_c - temp_stats.mean) /
                       temp_stats.stddev;
            if (z > K_SIGMA) {
                std::cout << "[ALERT][DynamicSigma] Temperature out of "
                          << K_SIGMA << " sigma range. "
                          << "value=" << m.temp_c
                          << ", mean=" << temp_stats.mean
                          << ", std=" << temp_stats.stddev << "\n";
            }
        }

        if (power_stats.ready(MIN_BASELINE_SAMPLES) &&
            power_stats.stddev > 0.0) {
            double z = std::abs(m.power_w - power_stats.mean) /
                       power_stats.stddev;
            if (z > K_SIGMA) {
                std::cout << "[ALERT][DynamicSigma] Power out of "
                          << K_SIGMA << " sigma range. "
                          << "value=" << m.power_w
                          << ", mean=" << power_stats.mean
                          << ", std=" << power_stats.stddev << "\n";
            }
        }

        // ---- CUSUM trend anomalies (R2-like) ----
        if (temp_stats.ready(MIN_BASELINE_SAMPLES)) {
            if (temp_cusum.update(m.temp_c)) {
                std::cout << "[ALERT][CUSUM] Temperature trend change "
                          << "detected around baseline=" << temp_stats.mean
                          << "\n";
            }
        }
        if (power_stats.ready(MIN_BASELINE_SAMPLES)) {
            if (power_cusum.update(m.power_w)) {
                std::cout << "[ALERT][CUSUM] Power trend change "
                          << "detected around baseline=" << power_stats.mean
                          << "\n";
            }
        }

        std::this_thread::sleep_for(interval);
    }

    return 0;
}