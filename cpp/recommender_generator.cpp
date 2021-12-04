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
#include <pcg_random.hpp>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

std::exponential_distribution<double> exp_dist(1);

std::valarray<std::size_t> sample_no_replacement(const std::size_t count, std::valarray<double> weights,
                                                 const std::vector<std::size_t>& actual_indices, pcg32& rng) {
    weights /= weights.sum();
    std::vector<double> cum_sum(weights.size());
    std::inclusive_scan(std::begin(weights), std::end(weights), std::begin(cum_sum));
    std::priority_queue<std::pair<double, std::size_t>> reservoir;
    std::valarray<std::size_t> results(count);
    for (std::size_t i = 0; i < count; i++) reservoir.emplace(exp_dist(rng) / weights[i], i);
    for (auto iter = std::begin(weights) + count; iter != std::end(weights); ++iter) {
        const double t_w = reservoir.top().first;
        const double x_w = exp_dist(rng) / t_w;
        const size_t cur_index = std::distance(std::begin(weights), iter);
        const auto cum_sum_iter = std::upper_bound(std::begin(cum_sum) + cur_index, std::end(cum_sum),
                                                   cum_sum[cur_index] + x_w);
        if (cum_sum_iter != std::end(cum_sum)) {
            iter += std::distance(std::begin(cum_sum) + cur_index, cum_sum_iter);
            const double nt_w = -t_w * *iter;
            const double e_2 = std::log(std::uniform_real_distribution<double>(std::exp(nt_w), 1)(rng));
            const double k_i = -e_2 / *iter;

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

struct Generator {
    using result_type = std::tuple<std::tuple<py::array_t<double>, py::array_t<double>>,
                                   std::tuple<py::array_t<double>, py::array_t<double>>>;
    using queue_values = std::tuple<std::array<std::size_t, 2>,
                                    std::tuple<std::valarray<double>, std::valarray<double>>,
                                    std::tuple<std::valarray<double>, std::valarray<double>>>;
private:
    std::size_t num_cards;
    std::size_t num_cubes;
    std::size_t num_threads;
    std::size_t max_batch_size;
    std::size_t length;
    std::size_t initial_seed;

    std::valarray<double> y_mtx;
    std::valarray<double> x_mtx;
    std::valarray<double> neg_sampler;
    std::vector<std::size_t> true_indices;
    std::valarray<double> replacing_neg_sampler;

    std::normal_distribution<double> noise_dist;
    std::uniform_real_distribution<double> neg_sampler_rand;
    pcg32 main_rng;

    std::vector<std::jthread> threads;

    moodycamel::BlockingConcurrentQueue<std::vector<std::size_t>> task_queue;
    moodycamel::ProducerToken task_producer;
    moodycamel::BlockingConcurrentQueue<queue_values> result_queue;
    moodycamel::ConsumerToken result_consumer;

public:
    Generator(const py::array_t<double, py::array::c_style>& adj_mtx,
              const py::array_t<double, py::array::c_style>& cubes,
              std::size_t num_workers, std::size_t batch_size, std::size_t seed,
              double noise, double noise_std)
            : num_cards{static_cast<std::size_t>(adj_mtx.shape(0))}, num_cubes{static_cast<std::size_t>(cubes.shape(0))},
              num_threads{num_workers},
              max_batch_size{batch_size}, length{(num_cubes + batch_size - 1) / batch_size},
              initial_seed{seed},
              x_mtx({cubes.data(), static_cast<std::size_t>(cubes.size())}),
              y_mtx({adj_mtx.data(), static_cast<std::size_t>(adj_mtx.size())}),
              neg_sampler(num_cards), true_indices(num_cards),
              replacing_neg_sampler(num_cards),
              noise_dist(noise, noise_std), main_rng{initial_seed, num_threads},
              task_producer{task_queue}, result_consumer{result_queue}
    {
        py::gil_scoped_release release;
        for (size_t i=0; i < num_cards; i++) {
            const std::valarray<double> row{y_mtx[std::slice(i * num_cards, num_cards, 1)]};
            const double sum = row.sum();
            if (sum == 0) y_mtx[i * (num_cards + 1)] = 1;
            else y_mtx[std::slice(i * num_cards, num_cards, 1)] /= std::valarray(sum, num_cards);
        }
        std::vector<std::pair<double, std::size_t>> sortable_sampler;
        sortable_sampler.reserve(num_cards);
        for (size_t i=0; i < num_cards; i++) {
            double count = adj_mtx.data()[i * (num_cards + 1)];
            if (count == 0) count = 1;
            sortable_sampler.emplace_back(count, i);
        }
        std::ranges::sort(sortable_sampler, [](const auto& p1, const auto& p2) { return p1.first > p2.first; });
        std::ranges::transform(sortable_sampler, std::begin(neg_sampler), [](const auto& p) { return p.first; });
        std::ranges::transform(sortable_sampler, std::begin(true_indices), [](const auto& p) { return p.second; });
        std::inclusive_scan(std::begin(neg_sampler), std::end(neg_sampler), std::begin(replacing_neg_sampler));
        neg_sampler_rand = std::uniform_real_distribution<double>(0, replacing_neg_sampler[num_cards - 1]);
    }

    Generator& enter() {
        py::gil_scoped_release release;
        for (size_t i=0; i < num_threads; i++) {
            threads.emplace_back([i, this](std::stop_token st) { this->worker_thread(st, pcg32(this->initial_seed, i)); });
        }
        main_rng = pcg32(initial_seed, num_threads);
        queue_new_epoch();
        return *this;
    }

    bool exit(py::object, py::object, py::object) {
        for (auto& worker : threads) worker.request_stop();
        threads.clear();
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
        const size_t full_batches = num_cubes / max_batch_size;
        for (size_t i=0; i < full_batches; i++) {
            tasks.emplace_back(indices.begin() + i * max_batch_size, indices.begin() + (i + 1) * max_batch_size);
        }
        if (full_batches < length) {
            tasks.emplace_back(indices.begin() + full_batches * max_batch_size, indices.end());
        }
        task_queue.enqueue_bulk(task_producer, std::make_move_iterator(tasks.begin()), length);
    }

    result_type next() {
        queue_values result;
        if (task_queue.size_approx() < num_threads && result_queue.size_approx() < 2 * max_batch_size) queue_new_epoch();
        result_queue.wait_dequeue(result_consumer, result);
        return {
            {
                py::array_t<double>(std::get<0>(result), &std::get<0>(std::get<1>(result))[0]),
                py::array_t<double>(std::get<0>(result), &std::get<1>(std::get<1>(result))[0]),
            },
            {
                py::array_t<double>(std::get<0>(result), &std::get<0>(std::get<2>(result))[0]),
                py::array_t<double>(std::get<0>(result), &std::get<1>(std::get<2>(result))[0]),
            },
        };
    }

    result_type getitem(std::size_t) {
        return next();
    }

    std::array<std::valarray<double>, 4> process_cube(const std::size_t index, pcg32& rng) {
        double noise = std::ranges::clamp(noise_dist(rng), 0.3, 0.7);
        std::valarray<double> x1 = x_mtx[std::slice(index * num_cards, num_cards, 1)];
        std::size_t count = static_cast<std::size_t>(x1.sum());
        size_t to_flip = std::ranges::clamp(noise * count, 1.0, count - 1.0);

        std::valarray<double> y1 = x1;
        auto to_exclude = sample_no_replacement(to_flip, x1, true_indices, rng);
        auto to_include = sample_no_replacement(to_flip, neg_sampler * x1, true_indices, rng);
        std::valarray<double> to_exclude_sampler(0.0, num_cards);
        to_exclude_sampler[to_exclude] = neg_sampler[to_exclude];
        auto y_to_exclude = sample_no_replacement(to_flip / 8, to_exclude_sampler, true_indices, rng);

        x1[to_exclude] = 0;
        x1[to_include] = 1;
        y1[y_to_exclude] = 0;

        /* const double rand_value = neg_sampler_rand(rng); */
        /* auto found_iter = std::upper_bound(std::begin(replacing_neg_sampler), std::end(replacing_neg_sampler), rand_value); */
        /* const std::size_t card_index = found_iter != std::end(replacing_neg_sampler) ? std::distance(std::begin(replacing_neg_sampler), found_iter) : replacing_neg_sampler.size() - 1; */
        /* const std::size_t actual_index = true_indices[card_index]; */
        const std::size_t actual_index = std::uniform_int_distribution<std::size_t>(0, num_cards - 1)(rng);

        std::valarray<double> x2(0.0, num_cards);
        x2[actual_index] = 1;

        return {x1, x2, y1, y_mtx[std::slice(actual_index * num_cards, num_cards, 1)]};
    }

    void worker_thread(std::stop_token st, pcg32 rng) {
        moodycamel::ConsumerToken consume_token(task_queue);
        moodycamel::ProducerToken produce_token(result_queue);
        std::vector<size_t> task;
        while(!st.stop_requested()) {
            // Time here is in microseconds.
            if(task_queue.wait_dequeue_timed(consume_token, task, 100'000)) {
                std::valarray<double> result_x1(task.size() * num_cards);
                std::valarray<double> result_x2(task.size() * num_cards);
                std::valarray<double> result_y1(task.size() * num_cards);
                std::valarray<double> result_y2(task.size() * num_cards);
                size_t offset = 0;
                for (const size_t index : task) {
                    const auto [x1, x2, y1, y2] = process_cube(index, rng);
                    result_x1[std::slice(offset * num_cards, num_cards, 1)] = x1;
                    result_x2[std::slice(offset * num_cards, num_cards, 1)] = x2;
                    result_y1[std::slice(offset * num_cards, num_cards, 1)] = y1;
                    result_y2[std::slice(offset * num_cards, num_cards, 1)] = y2;
                    offset++;
                }
                const std::array<std::size_t, 2> shape{task.size(), num_cards};
                result_queue.enqueue(produce_token, std::tuple{
                    shape,
                    std::tuple{
                        result_x1,
                        result_x2,
                    },
                    std::tuple{
                        result_y1,
                        result_y2,
                    }
                });
            }
        }
    }
};

PYBIND11_MODULE(generator, m) {
    using namespace pybind11::literals;
    py::object KerasSequence = py::module_::import("tensorflow.keras.utils").attr("Sequence");
    py::class_<Generator>(m, "Generator")
        .def(py::init<py::array_t<double, py::array::c_style>,
                      py::array_t<double, py::array::c_style>,
                      std::size_t, std::size_t, std::size_t,
                      double, double>())
                      /* double, double>("adj_mtx"_a, "cubes"_a, "num_workers"_a, "batch_size"_a, */
                      /*                 "seed"_a, "noise"_a=0.3, "noise_std"_a=0.1)) */
        .def("__enter__", &Generator::enter)
        .def("__exit__", &Generator::exit)
        .def("__len__", &Generator::size)
        .def("__getitem__", &Generator::getitem)
        .def("on_epoch_end", &Generator::queue_new_epoch);
}
