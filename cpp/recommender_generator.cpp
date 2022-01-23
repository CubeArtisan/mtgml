#include <algorithm>
#include <array>
#include <numeric>
#include <queue>
#include <random>
#include <thread>
#include <tuple>
#include <utility>
#include <valarray>
#include <vector>

#include <blockingconcurrentqueue.h>
#include <mio/mmap.hpp>
#include <pcg_random.hpp>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

constexpr std::size_t MAX_CUBE_SIZE = 1080;
constexpr std::size_t MAX_DECK_SIZE = 80;
constexpr std::size_t MAX_SIDEBOARD_SIZE = 48;
constexpr std::size_t READ_SIZE = 64;

std::exponential_distribution<float> exp_dist(1);

std::valarray<std::size_t> sample_no_replacement(const std::size_t count, std::valarray<float> weights,
                                                 const std::vector<std::size_t>& actual_indices, pcg32& rng) {
    weights /= weights.sum();
    std::vector<float> cum_sum(weights.size());
    std::inclusive_scan(std::begin(weights), std::end(weights), std::begin(cum_sum));
    std::priority_queue<std::pair<float, std::size_t>> reservoir;
    std::valarray<std::size_t> results(count);
    for (std::size_t i = 0; i < count; i++) reservoir.emplace(exp_dist(rng) / weights[i], i);
    for (auto iter = std::begin(weights) + count; iter != std::end(weights); ++iter) {
        const float t_w = reservoir.top().first;
        const float x_w = exp_dist(rng) / t_w;
        const size_t cur_index = std::distance(std::begin(weights), iter);
        const auto cum_sum_iter = std::upper_bound(std::begin(cum_sum) + cur_index, std::end(cum_sum),
                                                   cum_sum[cur_index] + x_w);
        if (cum_sum_iter != std::end(cum_sum)) {
            iter += std::distance(std::begin(cum_sum) + cur_index, cum_sum_iter);
            const float nt_w = -t_w * *iter;
            const float e_2 = std::log(std::uniform_real_distribution<float>(std::exp(nt_w), 1)(rng));
            const float k_i = -e_2 / *iter;

            reservoir.pop();
            reservoir.emplace(k_i, std::distance(std::begin(weights), iter));
        }
    }
    for (size_t i = 0; i < count; i++) {
        results[i] = actual_indices[reservoir.top().second];
        reservoir.pop();
    }
    return results;
}

struct Deck {
    std::array<std::uint16_t, MAX_DECK_SIZE> main;
    std::array<std::uint16_t, MAX_SIDEBOARD_SIZE> side;
};

using CubeCards = std::array<std::uint16_t, MAX_CUBE_SIZE>;

struct CubeData {
    CubeCards noisy_cube{0};
    CubeCards single_card_cube{0};
    CubeCards true_cube{0};
    std::array<float, 1 << 15> adj_mtx_row{0.f};
};

struct CubeGenerator {
    using Result = std::tuple<std::tuple<py::array_t<std::uint16_t>, py::array_t<std::uint16_t>>,
                              std::tuple<py::array_t<std::uint16_t>, py::array_t<float>>>;
    using ProcessedValue = std::unique_ptr<std::vector<CubeData>>;

private:
    mio::basic_mmap_source<std::byte> cube_mmap;
    mio::basic_mmap_source<std::byte> deck_mmap;
    std::size_t num_cards;
    std::size_t num_cubes;
    std::size_t num_decks;
    std::size_t batch_size;
    std::size_t length;
    std::size_t initial_seed;

    std::valarray<float> adj_mtx;
    std::valarray<float> neg_sampler;
    std::vector<std::size_t> true_indices;
    std::valarray<float> replacing_neg_sampler;

    std::normal_distribution<float> noise_dist;
    std::uniform_real_distribution<float> neg_sampler_rand;
    pcg32 main_rng;

    std::jthread worker_thread;

    moodycamel::BlockingConcurrentQueue<std::vector<std::size_t>> task_queue;
    moodycamel::ProducerToken task_producer;
    moodycamel::BlockingConcurrentQueue<ProcessedValue> processed_queue;
    moodycamel::ConsumerToken processed_consumer;

public:
    CubeGenerator(std::string cube_filename, std::string decks_filename, std::size_t num_cards,
                  std::size_t batch_size, std::size_t seed, float noise, float noise_std)
            : cube_mmap(cube_filename), num_cubes{cube_mmap.size() / sizeof(CubeCards)},
              deck_mmap(decks_filename), num_decks{deck_mmap.size() / sizeof(Deck)},
              batch_size{batch_size}, length{(num_cubes + batch_size - 1) / batch_size},
              initial_seed{seed}, adj_mtx(0.0, num_cards * num_cards),
              neg_sampler(0.0, num_cards), true_indices(0, num_cards),
              replacing_neg_sampler(0.0, num_cards),
              noise_dist(noise, noise_std), main_rng{initial_seed, 1},
              task_producer{task_queue}, processed_consumer{processed_queue}
    {
        py::gil_scoped_release release;
        auto deck_file_start = reinterpret_cast<const std::uint16_t*>(deck_mmap.data());
        for (std::size_t i=0; i < num_decks; i++) {
            const std::uint16_t* row = deck_file_start + sizeof(Deck) * i / 2;
            for (std::size_t j=0; j < MAX_DECK_SIZE; j++) {
                std::uint16_t j_val = row[j];
                if (j_val == 0) break;
                for (std::size_t k=0; k <= j; k++) {
                    std::uint16_t k_val = row[k];
                    adj_mtx[(j_val - 1) * num_cards + (k_val - 1)] += 1;
                    adj_mtx[(k_val - 1) * num_cards + (j_val - 1)] += 1;
                }
            }
        }
        for (std::size_t i=0; i < num_cards; i++) {
            // We need to materialize the valarray here to have it read nicely since you can't sum a slice.
            const auto slice = adj_mtx[std::slice(i, num_cards, num_cards)];
            const std::valarray<float> column{slice};
            const float column_sum = column.sum();
            if (column_sum == 0) adj_mtx[i * (num_cards + 1)] = 1.0;
            else slice /= std::valarray(column_sum, num_cards);
        }
        for (std::size_t i=0; i < num_cards; i++) {
            // We need to materialize the valarray here to have it read nicely since you can't sum a slice.
            const auto slice = adj_mtx[std::slice(i * num_cards, num_cards, 1)];
            const std::valarray<float> row{slice};
            const float row_sum = row.sum();
            if (row_sum == 0) adj_mtx[i * (num_cards + 1)] = 1.0;
            else slice /= std::valarray(row_sum, num_cards);
        }
        auto cube_file_start = reinterpret_cast<const std::uint16_t*>(deck_mmap.data());
        std::vector<float> cube_counts(num_cards, 0.0);
        for (std::size_t i=0; i < num_cubes; i++) {
            const std::uint16_t* row = cube_file_start + sizeof(CubeCards) * i / 2;
            for (std::size_t j=0; j < MAX_CUBE_SIZE; j++) {
                if (row[i] == 0) break;
                cube_counts[row[i] - 1] += 1.0;
            }
        }
        std::vector<std::pair<float, std::size_t>> sortable_sampler;
        sortable_sampler.reserve(num_cards);
        for (size_t i=0; i < num_cards; i++) {
            float count = cube_counts[i];
            if (count == 0) count = 1;
            sortable_sampler.emplace_back(count, i);
        }
        std::ranges::sort(sortable_sampler, [](const auto& p1, const auto& p2) { return p1.first > p2.first; });
        std::ranges::transform(sortable_sampler, std::begin(neg_sampler), [](const auto& p) { return p.first; });
        std::ranges::transform(sortable_sampler, std::begin(true_indices), [](const auto& p) { return p.second; });
        std::inclusive_scan(std::begin(neg_sampler), std::end(neg_sampler), std::begin(replacing_neg_sampler));
        neg_sampler_rand = std::uniform_real_distribution<float>(0, replacing_neg_sampler[num_cards - 1]);
    }

    CubeGenerator& enter() & {
        py::gil_scoped_release release;
        worker_thread = std::jthread([this](std::stop_token st) { this->worker_func(st, pcg32(this->initial_seed, 0)); });
        main_rng = pcg32(initial_seed, 0);
        queue_new_epoch();
        return *this;
    }

    bool exit(py::object, py::object, py::object) {
        worker_thread.request_stop();
        worker_thread.join();
        return false;
    }

    size_t size() const {
        return length;
    }

    void queue_new_epoch() {
        std::vector<size_t> indices(num_cubes);
        std::iota(indices.begin(), indices.end(), 0);
        std::ranges::shuffle(indices, main_rng);
        std::vector<std::vector<std::size_t>> tasks;
        tasks.reserve(length);
        const size_t full_batches = num_cubes / batch_size;
        for (size_t i=0; i < full_batches; i++) {
            tasks.emplace_back(indices.begin() + i * batch_size, indices.begin() + (i + 1) * batch_size);
        }
        task_queue.enqueue_bulk(task_producer, std::make_move_iterator(tasks.begin()), length);
    }

    Result next() & {
        ProcessedValue processed_value;
        if (!processed_queue.wait_dequeue_timed(processed_consumer, processed_value, 10'000)) {
            py::gil_scoped_release gil_release;
            queue_new_epoch();
            while (!processed_queue.wait_dequeue_timed(processed_consumer, processed_value, 100'000)) {}
        }
        std::vector<CubeData>* batched = processed_value.release();
        py::capsule free_when_done{batched,  [](void* ptr) { delete reinterpret_cast<std::vector<CubeData>*>(ptr); }};
        CubeData& first_sample = batched->front();
        return {
            {
                py::array_t<std::uint16_t>{noisy_cube_shape(), noisy_cube_strides, first_sample.noisy_cube.data(), free_when_done},
                py::array_t<std::uint16_t>{single_card_cube_shape(), single_card_cube_strides, first_sample.single_card_cube.data(), free_when_done},
            },
            {
                py::array_t<std::uint16_t>{true_cube_shape(), true_cube_strides, first_sample.true_cube.data(), free_when_done},
                py::array_t<float>{adj_mtx_row_shape(), adj_mtx_row_strides, first_sample.adj_mtx_row.data(), free_when_done},
            },
        };
    }

    Result getitem(std::size_t) & {
        return next();
    }

    CubeData process_cube(const std::size_t index, pcg32& rng) {
        float noise = std::ranges::clamp(noise_dist(rng), 0.3f, 0.9f);
        std::valarray<float> x1(0.0, num_cards);
        auto row_start = reinterpret_cast<const std::uint16_t*>(cube_mmap.data()) + index * MAX_CUBE_SIZE;
        for (std::size_t i=0; i < MAX_CUBE_SIZE; i++) {
            if (row_start[i] == 0) break;
            x1[row_start[i] - 1] += 1;
        }
        std::size_t count = static_cast<std::size_t>(x1.sum());
        size_t to_flip = std::ranges::clamp(noise * count, 1.0f, count - 1.0f);

        std::valarray<float> y1 = x1;
        auto to_exclude = sample_no_replacement(to_flip, x1, true_indices, rng);
        auto to_include = sample_no_replacement(to_flip, neg_sampler * x1, true_indices, rng);
        std::valarray<float> to_exclude_sampler(0.0, num_cards);
        to_exclude_sampler[to_exclude] = neg_sampler[to_exclude];
        auto y_to_exclude = sample_no_replacement(to_flip / 5, to_exclude_sampler, true_indices, rng);

        x1[to_exclude] = 0;
        x1[to_include] = 1;
        y1[y_to_exclude] = 0;

        const std::size_t actual_index = std::uniform_int_distribution<std::size_t>(0, num_cards - 1)(rng);

        std::array<std::uint16_t, MAX_CUBE_SIZE> noisy_cube{0};
        std::array<std::uint16_t, MAX_CUBE_SIZE> true_cube{0};
        std::size_t cur_index = 0;
        for (std::uint16_t i=0; i < num_cards; i++) {
            for (std::size_t j=0; j < x1[i]; j++) {
                if (cur_index >= MAX_CUBE_SIZE) break;
                noisy_cube[cur_index++] = i + 1;
            }
        }
        for (std::uint16_t i=0; i < num_cards; i++) {
            if (cur_index >= MAX_CUBE_SIZE) break;
            if (y1[i]) true_cube[cur_index++] = i + 1;
        }

        std::array<float, 1 << 15> adj_mtx_row{0.f};
        for (std::size_t i=0; i < num_cards; i++) {
            adj_mtx_row[i] = adj_mtx[actual_index * num_cards + i];
        }

        std::array<std::uint16_t, MAX_CUBE_SIZE> single_card_cube{0};
        single_card_cube[0] = actual_index;

        return {noisy_cube, single_card_cube, true_cube, adj_mtx_row};
    }

    void worker_func(std::stop_token st, pcg32 rng) {
        moodycamel::ConsumerToken consume_token(task_queue);
        moodycamel::ProducerToken produce_token(processed_queue);
        std::vector<std::size_t> task;
        while(!st.stop_requested()) {
            // Time here is in microseconds.
            if(task_queue.wait_dequeue_timed(consume_token, task, 100'000)) {
                std::unique_ptr<std::vector<CubeData>> batch = std::make_unique<std::vector<CubeData>>();
                batch->reserve(batch_size);
                for (const std::size_t index : task) {
                    batch->push_back(process_cube(index, rng));
                }
                processed_queue.enqueue(produce_token, std::move(batch));
            }
        }
    }

private:
    constexpr std::array<std::size_t, 2> noisy_cube_shape() { return {batch_size, MAX_CUBE_SIZE}; }
    constexpr std::array<std::size_t, 2> single_card_cube_shape() { return {batch_size, MAX_CUBE_SIZE}; }
    constexpr std::array<std::size_t, 2> true_cube_shape() { return {batch_size, MAX_CUBE_SIZE}; }
    constexpr std::array<std::size_t, 2> adj_mtx_row_shape() { return {batch_size, num_cards}; }

    static constexpr std::array<std::size_t, 2> noisy_cube_strides{sizeof(CubeData), sizeof(std::uint16_t)};
    static constexpr std::array<std::size_t, 2> single_card_cube_strides{sizeof(CubeData), sizeof(std::uint16_t)};
    static constexpr std::array<std::size_t, 2> true_cube_strides{sizeof(CubeData), sizeof(std::uint16_t)};
    static constexpr std::array<std::size_t, 2> adj_mtx_row_strides{sizeof(CubeData), sizeof(float)};
};

PYBIND11_MODULE(recommender_generator, m) {
    using namespace pybind11::literals;
    py::class_<CubeGenerator>(m, "RecommenderGenerator")
        .def(py::init<std::string, std::string, std::size_t, std::size_t, std::size_t, float, float>())
        .def("__enter__", &CubeGenerator::enter)
        .def("__exit__", &CubeGenerator::exit)
        .def("__len__", &CubeGenerator::size)
        .def("__getitem__", &CubeGenerator::getitem)
        .def("next", &CubeGenerator::next)
        .def("on_epoch_end", &CubeGenerator::queue_new_epoch);
};
