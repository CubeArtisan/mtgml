#include <algorithm>
#include <atomic>
#include <array>
#include <cstdint>
#include <numeric>
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

auto& get_cards(const CubeCards& cards) {
    return cards;
}

auto& get_cards(const Deck& deck) {
    return deck.main;
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

    std::size_t initial_seed;
    pcg32 main_rng;

    std::vector<std::uint16_t> card_indices;
    bool joined{false};
    std::thread initialization_thread;
    static constexpr std::size_t load_thread_count = 32;

    void load_rows(std::size_t num_objects, std::size_t offset, const T* file_start) {
        for (std::size_t i=offset; i < num_objects; i += load_thread_count) {
            const auto& row = get_cards(file_start[i]);
            for (std::size_t j=0; j < row.size(); j++) {
                std::uint16_t j_val = row[j];
                if (j_val == 0) break;
                /* if (j_val > num_cards) std::cout << i << ", " << j_val << std::endl; */
                for (std::size_t k=0; k <= j; k++) {
                    std::uint16_t k_val = row[k];
                    adj_mtx[(j_val - 1) * num_cards + (k_val - 1)] += 1;
                    adj_mtx[(k_val - 1) * num_cards + (j_val - 1)] += 1;
                }
            }
        }
    }

    void normalize_rows(std::size_t offset) {
        std::size_t empty_rows = 0;
        for (std::size_t i=offset; i < num_cards; i += load_thread_count) {
            auto begin = adj_mtx.begin();
            auto end = adj_mtx.begin();
            std::advance(begin, i * num_cards);
            std::advance(end, i * num_cards + num_cards);
            /* const float row_sum = std::accumulate(begin, end, 0); */
            const float row_sum = adj_mtx[i * (num_cards + 1)];
            if (row_sum > 0) for (std::size_t j=i * num_cards; j < (i + 1) * num_cards; j++) adj_mtx[j] /= row_sum;
            else {
                adj_mtx[i * (num_cards + 1)] = 1.0;
                /* std::cout << "empty row " << i  << " adj_mtx " << batch_size << std::endl; */
                empty_rows += 1;
            }
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
        for (std::size_t i=0; i < load_thread_count; i++) loading_array[i] = std::thread([=, this](){ normalize_rows(i); });
        std::cout << "Started second set of threads." << std::endl;
        for (std::size_t i=0; i < load_thread_count; i++) loading_array[i].join();
        std::iota(card_indices.begin(), card_indices.end(), 0);
        std::cout << "Finished loading adj_mtx " << batch_size << std::endl;
        joined = true;
    }

public:
    AdjMtxGenerator(std::string filename, std::size_t num_cards, std::size_t batch_size, std::size_t seed)
        : num_cards{num_cards}, batch_size{batch_size}, length{num_cards / batch_size},
          adj_mtx(num_cards * num_cards, 0.0), initial_seed{seed}, main_rng(initial_seed, 0),
          card_indices(num_cards, 0), initialization_thread([this, filename]() { this->initialize(filename); })
    { }

    void queue_new_epoch() & {
        std::ranges::shuffle(card_indices, main_rng);
    }
    py::array_t<float> get_adj_mtx() {
        using namespace std::chrono_literals;
        {
            py::gil_scoped_release release;
            while (!joined) {
                std::cout << "Waiting on the adjacency matrix." << std::endl;
                std::this_thread::sleep_for(1'000ms);
            }
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
            while (!joined) {
                std::cout << "Waiting on the adjacency matrix." << std::endl;
                std::this_thread::sleep_for(1'000ms);
            }
            batched = new std::vector<AdjMtxData>(batch_size);
            for (std::size_t i=0; i < batched->size(); i++) {
                std::uint16_t row = card_indices[pos++ % num_cards];
                batched->at(i).card_idx = row;
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
}
