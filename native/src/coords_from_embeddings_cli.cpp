#include <algorithm>
#include <cmath>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <iostream>\n#include <limits>\n#include <sstream>
#include <string>
#include <vector>

namespace {

struct Args {
    std::string input_root;
    std::string out_dir;
    std::string anchors_path;
    int dims = 3;
    int pca_iters = 64;
};

void usage() {
    std::cerr << "Coords From Embeddings CLI\n"
              << "Usage:\n"
              << "  coords_from_embeddings_cli --input-root <dir>"
              << " [--out-dir <dir>] [--dims 2|3]"
              << " [--anchors <jsonl>] [--pca-iters N]\n";
}

bool parse_args(int argc, char** argv, Args& out) {
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--input-root" && i + 1 < argc) {
            out.input_root = argv[++i];
        } else if (arg == "--out-dir" && i + 1 < argc) {
            out.out_dir = argv[++i];
        } else if (arg == "--dims" && i + 1 < argc) {
            out.dims = std::stoi(argv[++i]);
        } else if (arg == "--anchors" && i + 1 < argc) {
            out.anchors_path = argv[++i];
        } else if (arg == "--pca-iters" && i + 1 < argc) {
            out.pca_iters = std::stoi(argv[++i]);
        } else if (arg == "--help" || arg == "-h") {
            return false;
        } else {
            std::cerr << "Unknown argument: " << arg << "\n";
            return false;
        }
    }
    if (out.input_root.empty()) {
        return false;
    }
    if (out.dims != 2 && out.dims != 3) {
        std::cerr << "dims must be 2 or 3\n";
        return false;
    }
    if (out.pca_iters < 8) {
        out.pca_iters = 8;
    }
    return true;
}

std::string join_path(const std::string& a, const std::string& b) {
    if (a.empty()) return b;
    char last = a.back();
    if (last == '\\' || last == '/') {
        return a + b;
    }
    return a + "\\" + b;
}

std::string now_iso() {
    std::time_t t = std::time(nullptr);
    std::tm tm{};
#ifdef _WIN32
    localtime_s(&tm, &t);
#else
    localtime_r(&t, &tm);
#endif
    char buf[32];
    std::strftime(buf, sizeof(buf), "%Y-%m-%dT%H:%M:%S", &tm);
    return std::string(buf);
}

bool parse_embedding_line(const std::string& line, std::vector<double>& out) {
    const std::string key = "\"embedding\"";
    auto key_pos = line.find(key);
    if (key_pos == std::string::npos) {
        return false;
    }
    auto open_pos = line.find('[', key_pos);
    auto close_pos = line.find(']', open_pos);
    if (open_pos == std::string::npos || close_pos == std::string::npos || close_pos <= open_pos) {
        return false;
    }
    std::string inner = line.substr(open_pos + 1, close_pos - open_pos - 1);
    std::vector<double> values;
    values.reserve(1024);
    std::stringstream ss(inner);
    std::string token;
    while (std::getline(ss, token, ',')) {
        std::stringstream conv(token);
        double val = 0.0;
        conv >> val;
        if (!conv.fail()) {
            values.push_back(val);
        }
    }
    if (values.empty()) {
        return false;
    }
    out = std::move(values);
    return true;
}

bool parse_idx(const std::string& line, int& out) {
    const std::string key = "\"idx\"";
    auto pos = line.find(key);
    if (pos == std::string::npos) {
        return false;
    }
    auto colon = line.find(':', pos + key.size());
    if (colon == std::string::npos) {
        return false;
    }
    std::stringstream ss(line.substr(colon + 1));
    int val = 0;
    ss >> val;
    if (ss.fail()) {
        return false;
    }
    out = val;
    return true;
}

bool read_embeddings_jsonl(const std::string& path,
                           std::vector<int>& idxs,
                           std::vector<std::vector<double>>& embeddings) {
    std::ifstream in(path);
    if (!in) {
        return false;
    }
    std::string line;
    int line_idx = 0;
    while (std::getline(in, line)) {
        if (line.empty()) {
            continue;
        }
        std::vector<double> emb;
        if (!parse_embedding_line(line, emb)) {
            return false;
        }
        int idx = line_idx;
        parse_idx(line, idx);
        idxs.push_back(idx);
        embeddings.push_back(std::move(emb));
        ++line_idx;
    }
    return !embeddings.empty();
}

std::vector<double> mean_vector(const std::vector<std::vector<double>>& data) {
    std::vector<double> mean;
    if (data.empty()) {
        return mean;
    }
    const size_t dim = data.front().size();
    mean.assign(dim, 0.0);
    for (const auto& row : data) {
        if (row.size() != dim) {
            return {};
        }
        for (size_t d = 0; d < dim; ++d) {
            mean[d] += row[d];
        }
    }
    for (double& v : mean) {
        v /= static_cast<double>(data.size());
    }
    return mean;
}

void center_data(const std::vector<std::vector<double>>& data,
                 const std::vector<double>& mean,
                 std::vector<std::vector<double>>& centered) {
    centered.clear();
    centered.reserve(data.size());
    for (const auto& row : data) {
        std::vector<double> c(row.size(), 0.0);
        for (size_t d = 0; d < row.size(); ++d) {
            c[d] = row[d] - mean[d];
        }
        centered.push_back(std::move(c));
    }
}

bool normalize(std::vector<double>& v) {
    double norm = 0.0;
    for (double x : v) norm += x * x;
    norm = std::sqrt(norm);
    if (norm < 1e-12) {
        return false;
    }
    for (double& x : v) x /= norm;
    return true;
}

std::vector<double> cov_times_vec(const std::vector<std::vector<double>>& centered,
                                  const std::vector<double>& v) {
    const size_t n = centered.size();
    const size_t dim = v.size();
    std::vector<double> tmp(n, 0.0);
    for (size_t i = 0; i < n; ++i) {
        const auto& row = centered[i];
        double dot = 0.0;
        for (size_t d = 0; d < dim; ++d) {
            dot += row[d] * v[d];
        }
        tmp[i] = dot;
    }
    std::vector<double> out(dim, 0.0);
    const double denom = (n > 1) ? (1.0 / static_cast<double>(n - 1)) : 1.0;
    for (size_t i = 0; i < n; ++i) {
        const auto& row = centered[i];
        const double scale = tmp[i] * denom;
        for (size_t d = 0; d < dim; ++d) {
            out[d] += row[d] * scale;
        }
    }
    return out;
}

void orthogonalize(std::vector<double>& v, const std::vector<std::vector<double>>& basis) {
    for (const auto& b : basis) {
        double dot = 0.0;
        for (size_t i = 0; i < v.size(); ++i) {
            dot += v[i] * b[i];
        }
        for (size_t i = 0; i < v.size(); ++i) {
            v[i] -= dot * b[i];
        }
    }
}

std::vector<std::vector<double>> pca_components(const std::vector<std::vector<double>>& centered,
                                                int dims,
                                                int iters) {
    std::vector<std::vector<double>> comps;
    if (centered.empty()) {
        return comps;
    }
    const size_t dim = centered.front().size();
    for (int k = 0; k < dims; ++k) {
        std::vector<double> v(dim, 0.0);
        for (size_t i = 0; i < dim; ++i) {
            v[i] = static_cast<double>((i + 1) * (k + 1));
        }
        orthogonalize(v, comps);
        if (!normalize(v)) {
            break;
        }
        for (int it = 0; it < iters; ++it) {
            std::vector<double> w = cov_times_vec(centered, v);
            orthogonalize(w, comps);
            if (!normalize(w)) {
                break;
            }
            v.swap(w);
        }
        comps.push_back(v);
    }
    return comps;
}

std::vector<std::vector<double>> project_coords(const std::vector<std::vector<double>>& centered,
                                                const std::vector<std::vector<double>>& comps) {
    std::vector<std::vector<double>> coords;
    const size_t dims = comps.size();
    coords.reserve(centered.size());
    for (const auto& row : centered) {
        std::vector<double> c(dims, 0.0);
        for (size_t k = 0; k < dims; ++k) {
            double dot = 0.0;
            for (size_t d = 0; d < row.size(); ++d) {
                dot += row[d] * comps[k][d];
            }
            c[k] = dot;
        }
        coords.push_back(std::move(c));
    }
    return coords;
}

void minmax(const std::vector<std::vector<double>>& coords,
            std::vector<double>& minv,
            std::vector<double>& maxv) {
    if (coords.empty()) {
        return;
    }
    const size_t dims = coords.front().size();
    minv.assign(dims, std::numeric_limits<double>::max());
    maxv.assign(dims, std::numeric_limits<double>::lowest());
    for (const auto& row : coords) {
        for (size_t d = 0; d < dims; ++d) {
            minv[d] = std::min(minv[d], row[d]);
            maxv[d] = std::max(maxv[d], row[d]);
        }
    }
}

std::vector<std::vector<double>> scale_minmax(const std::vector<std::vector<double>>& coords,
                                              const std::vector<double>& minv,
                                              const std::vector<double>& maxv) {
    std::vector<std::vector<double>> out;
    out.reserve(coords.size());
    for (const auto& row : coords) {
        std::vector<double> s(row.size(), 0.0);
        for (size_t d = 0; d < row.size(); ++d) {
            double denom = maxv[d] - minv[d];
            if (std::fabs(denom) < 1e-12) {
                denom = 1.0;
            }
            s[d] = (row[d] - minv[d]) / denom;
        }
        out.push_back(std::move(s));
    }
    return out;
}

void write_csv(const std::string& path,
               const std::vector<int>& idxs,
               const std::vector<std::vector<double>>& coords) {
    std::ofstream out(path, std::ios::out | std::ios::trunc);
    if (!out) {
        return;
    }
    const size_t dims = coords.empty() ? 0 : coords.front().size();
    if (dims == 2) {
        out << "idx,x,y\n";
    } else {
        out << "idx,x,y,z\n";
    }
    out << std::fixed << std::setprecision(8);
    for (size_t i = 0; i < coords.size() && i < idxs.size(); ++i) {
        out << idxs[i];
        for (size_t d = 0; d < coords[i].size(); ++d) {
            out << "," << coords[i][d];
        }
        out << "\n";
    }
}

void write_json_matrix(const std::string& path,
                       const std::vector<std::vector<double>>& coords) {
    std::ofstream out(path, std::ios::out | std::ios::trunc);
    if (!out) {
        return;
    }
    out << std::fixed << std::setprecision(8);
    out << "[\n";
    for (size_t i = 0; i < coords.size(); ++i) {
        out << "  [";
        for (size_t d = 0; d < coords[i].size(); ++d) {
            out << coords[i][d];
            if (d + 1 < coords[i].size()) out << ", ";
        }
        out << "]";
        if (i + 1 < coords.size()) out << ",";
        out << "\n";
    }
    out << "]\n";
}

void write_json_vector(const std::string& path, const std::vector<double>& vec) {
    std::ofstream out(path, std::ios::out | std::ios::trunc);
    if (!out) {
        return;
    }
    out << std::fixed << std::setprecision(8);
    out << "[";
    for (size_t i = 0; i < vec.size(); ++i) {
        out << vec[i];
        if (i + 1 < vec.size()) out << ", ";
    }
    out << "]\n";
}

void write_meta(const std::string& path,
                const std::string& mode,
                const std::string& may_source,
                const std::string& feb_source,
                const std::string& anchor_source,
                size_t dim,
                int coords_dims) {
    std::ofstream out(path, std::ios::out | std::ios::trunc);
    if (!out) {
        return;
    }
    out << "{\n";
    out << "  \"mode\": \"" << mode << "\",\n";
    out << "  \"created\": \"" << now_iso() << "\",\n";
    out << "  \"may_source\": \"" << may_source << "\",\n";
    out << "  \"feb_source\": \"" << feb_source << "\",\n";
    out << "  \"anchor_source\": ";
    if (!anchor_source.empty()) {
        out << "\"" << anchor_source << "\"";
    } else {
        out << "null";
    }
    out << ",\n";
    out << "  \"embedding_dim\": " << dim << ",\n";
    out << "  \"coords_dims\": " << coords_dims << "\n";
    out << "}\n";
}

void write_minmax(const std::string& path,
                  const std::string& mode,
                  const std::vector<double>& minv,
                  const std::vector<double>& maxv) {
    std::ofstream out(path, std::ios::out | std::ios::trunc);
    if (!out) {
        return;
    }
    out << std::fixed << std::setprecision(8);
    out << "{\n";
    out << "  \"mode\": \"" << mode << "\",\n";
    out << "  \"min\": [";
    for (size_t i = 0; i < minv.size(); ++i) {
        out << minv[i];
        if (i + 1 < minv.size()) out << ", ";
    }
    out << "],\n";
    out << "  \"max\": [";
    for (size_t i = 0; i < maxv.size(); ++i) {
        out << maxv[i];
        if (i + 1 < maxv.size()) out << ", ";
    }
    out << "]\n";
    out << "}\n";
}

bool process_mode(const std::string& mode,
                  const std::string& may_path,
                  const std::string& feb_path,
                  const std::string& out_dir,
                  int dims,
                  int pca_iters,
                  const std::string& anchors_path) {
    std::vector<int> may_idxs;
    std::vector<int> feb_idxs;
    std::vector<std::vector<double>> may_embeddings;
    std::vector<std::vector<double>> feb_embeddings;
    if (!read_embeddings_jsonl(may_path, may_idxs, may_embeddings)) {
        std::cerr << "Failed to read May embeddings: " << may_path << "\n";
        return false;
    }
    if (!read_embeddings_jsonl(feb_path, feb_idxs, feb_embeddings)) {
        std::cerr << "Failed to read Feb embeddings: " << feb_path << "\n";
        return false;
    }
    if (may_embeddings.empty() || feb_embeddings.empty()) {
        std::cerr << "Empty embeddings for mode: " << mode << "\n";
        return false;
    }
    if (may_embeddings.front().size() != feb_embeddings.front().size()) {
        std::cerr << "Embedding dimension mismatch for " << mode << "\n";
        return false;
    }
    const size_t dim = may_embeddings.front().size();

    std::vector<double> mean = mean_vector(may_embeddings);
    if (mean.empty()) {
        std::cerr << "Failed to compute mean for " << mode << "\n";
        return false;
    }
    std::vector<std::vector<double>> may_centered;
    std::vector<std::vector<double>> feb_centered;
    center_data(may_embeddings, mean, may_centered);
    center_data(feb_embeddings, mean, feb_centered);

    std::vector<std::vector<double>> comps = pca_components(may_centered, dims, pca_iters);
    if (comps.size() != static_cast<size_t>(dims)) {
        std::cerr << "Failed to compute PCA components for " << mode << "\n";
        return false;
    }

    auto may_coords = project_coords(may_centered, comps);
    auto feb_coords = project_coords(feb_centered, comps);

    std::vector<double> minv, maxv;
    minmax(may_coords, minv, maxv);
    auto may_scaled = scale_minmax(may_coords, minv, maxv);
    auto feb_scaled = scale_minmax(feb_coords, minv, maxv);

    const std::string prefix = mode + "_pca_" + std::to_string(dims) + "d";
    write_meta(join_path(out_dir, prefix + "_meta.json"), mode, may_path, feb_path, anchors_path, dim, dims);
    write_minmax(join_path(out_dir, prefix + "_minmax.json"), mode, minv, maxv);
    write_json_vector(join_path(out_dir, prefix + "_mean.json"), mean);

    std::vector<double> flat;
    for (const auto& c : comps) {
        for (double v : c) {
            flat.push_back(v);
        }
    }
    write_json_vector(join_path(out_dir, prefix + "_components.json"), flat);

    write_csv(join_path(out_dir, "may_" + mode + "_coords_" + std::to_string(dims) + "d.csv"), may_idxs, may_scaled);
    write_csv(join_path(out_dir, "feb_" + mode + "_coords_" + std::to_string(dims) + "d.csv"), feb_idxs, feb_scaled);
    write_json_matrix(join_path(out_dir, "may_" + mode + "_coords_" + std::to_string(dims) + "d.json"), may_scaled);
    write_json_matrix(join_path(out_dir, "feb_" + mode + "_coords_" + std::to_string(dims) + "d.json"), feb_scaled);

    if (!anchors_path.empty()) {
        std::vector<int> anchor_idxs;
        std::vector<std::vector<double>> anchor_embeddings;
        if (!read_embeddings_jsonl(anchors_path, anchor_idxs, anchor_embeddings)) {
            std::cerr << "Failed to read anchors embeddings: " << anchors_path << "\n";
            return false;
        }
        std::vector<std::vector<double>> anchor_centered;
        center_data(anchor_embeddings, mean, anchor_centered);
        auto anchor_coords = project_coords(anchor_centered, comps);
        auto anchor_scaled = scale_minmax(anchor_coords, minv, maxv);
        write_csv(join_path(out_dir, "anchors_" + mode + "_coords_" + std::to_string(dims) + "d.csv"), anchor_idxs, anchor_scaled);
        write_json_matrix(join_path(out_dir, "anchors_" + mode + "_coords_" + std::to_string(dims) + "d.json"), anchor_scaled);
    }

    return true;
}

} // namespace

int main(int argc, char** argv) {
    Args args;
    if (!parse_args(argc, argv, args)) {
        usage();
        return 1;
    }

    const std::string input_root = args.input_root;
    const std::string out_dir = args.out_dir.empty()
        ? join_path(input_root, args.dims == 3 ? "coords_out_3d" : "coords_out")
        : args.out_dir;

#ifdef _WIN32
    std::string mkdir_cmd = "mkdir \"" + out_dir + "\" 2>NUL";
#else
    std::string mkdir_cmd = "mkdir -p \"" + out_dir + "\"";
#endif
    std::system(mkdir_cmd.c_str());

    const std::string may_ctxon = join_path(input_root, "MAY_CTXON_EMBEDDINGS.jsonl");
    const std::string feb_ctxon = join_path(input_root, "FEB_CTXON_EMBEDDINGS.jsonl");
    const std::string may_ctxoff = join_path(input_root, "MAY_CTXOFF_EMBEDDINGS.jsonl");
    const std::string feb_ctxoff = join_path(input_root, "FEB_CTXOFF_EMBEDDINGS.jsonl");

    if (!process_mode("ctxon", may_ctxon, feb_ctxon, out_dir, args.dims, args.pca_iters, args.anchors_path)) {
        return 2;
    }
    if (!process_mode("ctxoff", may_ctxoff, feb_ctxoff, out_dir, args.dims, args.pca_iters, args.anchors_path)) {
        return 3;
    }

    std::cout << "[OK] Wrote coords to: " << out_dir << "\n";
    return 0;
}

