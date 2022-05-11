#include <algorithm>
#include <atomic>
#include <array>
#include <concepts>
#include <cstdint>
#include <numeric>
#include <random>
#include <ranges>
#include <string>
#include <thread>
#include <valarray>

#include <blockingconcurrentqueue.h>
#include <mio/mmap.hpp>
#include <pcg_random.hpp>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include "mtgml/structs.hpp"

namespace py = pybind11;

struct AdjMtxData {
    std::uint16_t card_idx;
    std::array<float, 1 << 15> adj_mtx_row{0.f};
};

constexpr auto& get_cards(const CubeCards& cards) {
    return cards;
}

constexpr auto& get_cards(const Deck& deck) {
    return deck.main;
}

constexpr Pool empty_pool{0};
constexpr auto& get_cards(const Pick& pick) {
    if (pick.pool[1] > 0) return pick.pool;
    else return empty_pool;
}

constexpr void populate_adj_mtx(std::vector<float>& adj_mtx, const auto& range, std::size_t num_cards) {
    std::size_t i = 0;
    for (auto idx1 : range) {
        if (idx1 == 0) return;
        i += 1;
        const std::size_t col_idx = idx1 - 1;
        for (auto idx2 : range | std::ranges::views::take(i)) {
            const std::size_t col_idx2 = idx2 - 1;
            const std::size_t idx_max = std::max(col_idx, col_idx2);
            const std::size_t idx_min = std::min(col_idx, col_idx2);
            adj_mtx[idx_max * num_cards + idx_min] += 1;
        }
    }
}

constexpr float EPSILON = 1e-06;

constexpr void normalize_range(const auto& range, std::size_t id) {
    const float sum = std::reduce(std::begin(range), std::end(range));
    if (sum > 0) {
        for (auto& x : range) {
            x /= sum;
        }
    } else {
        auto iter = std::begin(range);
        std::advance(iter, id);
        *iter = 1.f;
    }
}

constexpr void fill_upper_triangle(std::vector<float>& adj_mtx, std::size_t num_cards) {
    for (std::size_t j = 0; j < num_cards; j++) {
        const std::size_t row_idx = j * num_cards;
        for (std::size_t i = j + 1; i < num_cards; i++) {
            adj_mtx[row_idx + i] = adj_mtx[i * num_cards + j];
        }
    }
}

constexpr void normalize_mtx(std::vector<float>& mtx1, std::vector<float>& mtx2, std::size_t num_cards) {
    for (std::size_t i = 0; i < num_cards; i++) {
        std::size_t row_idx = num_cards * i;
        for (std::size_t idx = row_idx; idx <= row_idx + i; idx++) {
            mtx2[idx] = std::max(mtx2[idx], 1.f);
            mtx1[idx] /= mtx2[idx];
        }
    }
}

constexpr void add_epsilon(std::vector<float>& adj_mtx, std::size_t num_cards) {
    float divisor = 1 + num_cards * EPSILON;
    for (std::size_t i = 0; i < std::size(adj_mtx); i++) {
        adj_mtx[i] = (adj_mtx[i] + EPSILON) / divisor;
    }
}

template <typename T>
struct AdjMtxGenerator {
    using Result = std::tuple<py::array_t<std::uint16_t>, py::array_t<float>>;

private:
    const std::size_t num_cards;
    std::size_t batch_size;
    std::size_t length;
    std::size_t pos{0};

    std::vector<float> adj_mtx;
    std::vector<float> reg_adj_mtx;
    std::vector<float> weights;

    std::size_t initial_seed;
    pcg32 main_rng;

    std::vector<std::uint16_t> card_indices;
    std::thread initialization_thread;
    static constexpr std::size_t load_thread_count = 32;
    std::uniform_real_distribution<float> sampler{0, 1};


    constexpr void load_rows(std::size_t num_objects, std::size_t offset, const T* file_start) {
        for (std::size_t i=offset; i < num_objects; i += load_thread_count) {
            const auto& row = get_cards(file_start[i]);
            populate_adj_mtx(adj_mtx, row, num_cards);
            if constexpr (std::same_as<T, Pick>) {
                for (std::size_t j=0; j < row.size(); j++);
                auto seen = std::ranges::join_view(file_start[i].seen_packs);
                populate_adj_mtx(reg_adj_mtx, seen, num_cards);
            }
        }
    }

    constexpr void normalize_rows(std::size_t offset) {
        for (std::size_t i=offset; i < num_cards; i += load_thread_count) {
            normalize_range(adj_mtx | std::ranges::views::drop(i * num_cards) | std::ranges::views::take(num_cards), i);
        }
    }

    void initialize(std::string filename) {
        mio::basic_mmap_source<std::byte> mmap(filename);
        std::size_t num_objects = mmap.size() / sizeof(T);
        std::cout << num_objects << " for adj_mtx " << batch_size << std::endl;
        auto file_start = reinterpret_cast<const T*>(mmap.data());
        std::array<std::thread, load_thread_count> loading_array;
        std::cout << "Created loading array." << std::endl;
        for (std::size_t i=0; i < load_thread_count; i++) loading_array[i] = std::thread([=, this](){ load_rows(num_objects, i, file_start); });
        std::cout << "Started first set of threads." << std::endl;
        for (std::size_t i=0; i < load_thread_count; i++) loading_array[i].join();
        for (std::size_t i=0; i < num_cards; i++) {
            weights[i] = adj_mtx[i * (num_cards + 1)];
            adj_mtx[i * (num_cards + 1)] = 0;
        }
        const float sum = std::reduce(std::begin(weights), std::end(weights));
        if (sum > 0) {
            for (auto& x : weights) {
                x /= sum;
            }
        } else {
            std::cout << "No diagonal" << std::endl;
        }
        add_epsilon(weights, std::size(weights));
        std::cout << "Greatest value " << std::ranges::max(weights) << std::endl;

        std::inclusive_scan(std::begin(weights), std::end(weights), std::begin(weights));
        if constexpr (std::same_as<T, Pick>) normalize_mtx(adj_mtx, reg_adj_mtx, num_cards);
        fill_upper_triangle(adj_mtx, num_cards);
        for (std::size_t i=0; i < load_thread_count; i++) loading_array[i] = std::thread([=, this](){ normalize_rows(i); });
        std::cout << "Started second set of threads." << std::endl;
        for (std::size_t i=0; i < load_thread_count; i++) loading_array[i].join();
        add_epsilon(adj_mtx, num_cards);
        std::iota(card_indices.begin(), card_indices.end(), 0);
        std::cout << "Finished loading adj_mtx " << batch_size << std::endl;
    }

    void join_thread() {
        if (initialization_thread.joinable()) {
            initialization_thread.join();
        }
    }

public:
    AdjMtxGenerator(std::string filename, std::size_t num_cards, std::size_t batch_size, std::size_t seed)
        : num_cards{num_cards}, batch_size{batch_size}, length{num_cards / batch_size},
          adj_mtx(num_cards * num_cards, 0.0), reg_adj_mtx(num_cards * num_cards, 0.0),
          weights(num_cards, 0.0), initial_seed{seed}, main_rng(initial_seed, 0),
          card_indices(num_cards, 0), initialization_thread([this, filename]() { this->initialize(filename); })
    { }

    void queue_new_epoch() & {
        std::ranges::shuffle(card_indices, main_rng);
    }
    py::array_t<float> get_adj_mtx() {
        using namespace std::chrono_literals;
        {
            py::gil_scoped_release release;
            join_thread();
        }
        std::cout << "Read adj_mtx " << batch_size << std::endl;
        return py::array_t<float>{std::array<std::size_t, 2>{num_cards, num_cards},
                                  std::array<std::size_t, 2>{sizeof(float) * num_cards, sizeof(float)},
                                  adj_mtx.data()};
    }

    Result next() & {
        using namespace std::chrono_literals;
        std::vector<AdjMtxData>* batched;
        {
            py::gil_scoped_release release;
            join_thread();
            batched = new std::vector<AdjMtxData>(batch_size);
            for (std::size_t i=0; i < batched->size(); i++) {
                float value = sampler(main_rng);
                auto iter = std::upper_bound(std::begin(weights), std::end(weights), value);
                std::size_t row = std::distance(std::begin(weights), iter);
                row = std::min(num_cards - 1, row);
                batched->at(i).card_idx = row + 1;
                for (std::size_t j=0; j < num_cards; j++) {
                    batched->at(i).adj_mtx_row[j] = adj_mtx[row * num_cards + j];
                }
            }
        }
        AdjMtxData& first_sample = batched->at(0);
        py::capsule free_when_done{batched,  [](void* ptr) { delete reinterpret_cast<std::vector<AdjMtxData>*>(ptr); }};
        return {
            py::array_t<std::uint16_t>{single_card_shape(), single_card_strides, &first_sample.card_idx, free_when_done},
            py::array_t<float>{adj_mtx_row_shape(), adj_mtx_row_strides, first_sample.adj_mtx_row.data(), free_when_done},
        };
    }

    std::size_t size() { return length; }

    Result get_item(std::size_t) { return next(); }

private:
    constexpr std::array<std::size_t, 1> single_card_shape() { return {batch_size}; }
    constexpr std::array<std::size_t, 2> adj_mtx_row_shape() { return {batch_size, num_cards}; }

    static constexpr std::array<std::size_t, 1> single_card_strides{sizeof(AdjMtxData)};
    static constexpr std::array<std::size_t, 2> adj_mtx_row_strides{sizeof(AdjMtxData), sizeof(float)};
};

using CubeAdjMtxGenerator = AdjMtxGenerator<CubeCards>;
using DeckAdjMtxGenerator = AdjMtxGenerator<Deck>;
using PickAdjMtxGenerator = AdjMtxGenerator<Pick>;

PYBIND11_MODULE(adj_mtx_generator, m) {
    using namespace pybind11::literals;
    py::class_<CubeAdjMtxGenerator>(m, "CubeAdjMtxGenerator")
        .def(py::init<std::string, std::size_t, std::size_t, std::size_t>())
        .def("__len__", &CubeAdjMtxGenerator::size)
        .def("__getitem__", &CubeAdjMtxGenerator::get_item)
        .def("get_adj_mtx", &CubeAdjMtxGenerator::get_adj_mtx)
        .def("next", &CubeAdjMtxGenerator::next)
        .def("on_epoch_end", &CubeAdjMtxGenerator::queue_new_epoch);
    py::class_<DeckAdjMtxGenerator>(m, "DeckAdjMtxGenerator")
        .def(py::init<std::string, std::size_t, std::size_t, std::size_t>())
        .def("__len__", &DeckAdjMtxGenerator::size)
        .def("__getitem__", &DeckAdjMtxGenerator::get_item)
        .def("get_adj_mtx", &DeckAdjMtxGenerator::get_adj_mtx)
        .def("next", &DeckAdjMtxGenerator::next)
        .def("on_epoch_end", &DeckAdjMtxGenerator::queue_new_epoch);
    py::class_<PickAdjMtxGenerator>(m, "PickAdjMtxGenerator")
        .def(py::init<std::string, std::size_t, std::size_t, std::size_t>())
        .def("__len__", &PickAdjMtxGenerator::size)
        .def("__getitem__", &PickAdjMtxGenerator::get_item)
        .def("get_adj_mtx", &PickAdjMtxGenerator::get_adj_mtx)
        .def("next", &PickAdjMtxGenerator::next)
        .def("on_epoch_end", &PickAdjMtxGenerator::queue_new_epoch);
}
