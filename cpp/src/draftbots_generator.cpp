#include <array>
#include <atomic>
#include <iostream>
#include <memory>
#include <tuple>
#include <valarray>
#include <vector>

#include <blockingconcurrentqueue.h>
#include <mio/mmap.hpp>
#include <pcg_random.hpp>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include "mtgml/structs.hpp"

namespace py = pybind11;

struct DraftbotGenerator {
  using ProcessedValue = std::unique_ptr<std::vector<Draft>>;
  using Result =
      std::tuple<py::array_t<std::int16_t>, py::array_t<std::int16_t>,
                 py::array_t<std::int16_t>, py::array_t<std::int8_t>,
                 py::array_t<float>, py::array_t<std::int8_t>,
                 py::array_t<float>>;

private:
  mio::basic_mmap_source<std::byte> mmap;
  std::size_t batch_size;
  std::size_t seed;
  std::size_t num_picks;
  pcg32 main_rng;
  moodycamel::BlockingConcurrentQueue<std::size_t> chunk_queue;
  moodycamel::ProducerToken chunk_producer;
  moodycamel::BlockingConcurrentQueue<ProcessedValue> processed_queue;
  moodycamel::ConsumerToken processed_consumer;
  std::jthread worker_thread;
  std::jthread worker_thread2;

  static constexpr std::size_t SHUFFLE_BUFFER_SIZE = READ_SIZE * (1 << 6);

public:
  DraftbotGenerator(std::string filename, std::size_t batch_size,
                    std::size_t seed)
      : mmap(filename), batch_size{batch_size}, seed{seed},
        num_picks{mmap.size() / sizeof(Draft)}, main_rng{seed, 1},
        chunk_producer{chunk_queue}, processed_consumer{processed_queue} {
    // std::cout << "Sizeof Draft " << sizeof(Draft) << std::endl;
  }

  DraftbotGenerator &enter() & {
    py::gil_scoped_release gil_release;
    worker_thread = std::jthread([this](std::stop_token st) {
      this->worker_func(st, pcg32(this->seed, 0));
    });
    worker_thread2 = std::jthread([this](std::stop_token st) {
      this->worker_func(st, pcg32(this->seed, 1));
    });
    queue_new_epoch();
    return *this;
  }

  bool exit(py::object, py::object, py::object) {
    worker_thread.request_stop();
    worker_thread2.request_stop();
    worker_thread.join();
    worker_thread2.join();
    return false;
  }

  std::size_t size() const { return num_picks / batch_size; }

  void queue_new_epoch() & {
    // We may lose some picks this way but it greatly simplifies the code.
    std::vector<std::size_t> indices(num_picks / READ_SIZE);
    std::iota(indices.begin(), indices.end(), 0);
    std::ranges::shuffle(indices, main_rng);
    std::size_t read_ratio = std::max(1ul, batch_size / READ_SIZE);
    chunk_queue.enqueue_bulk(chunk_producer,
                             std::make_move_iterator(indices.begin()),
                             (indices.size() / read_ratio) * read_ratio);
  }

  Result next() & {
    std::vector<Draft> *batched;
    {
      py::gil_scoped_release gil_release;
      ProcessedValue processed_value;
      if (!processed_queue.wait_dequeue_timed(processed_consumer,
                                              processed_value, 10'000)) {
        std::cout << "Waiting on a draftbot sample." << std::endl;
        queue_new_epoch();
        while (!processed_queue.wait_dequeue_timed(processed_consumer,
                                                   processed_value, 250'000)) {
          std::cout << "Waiting on a draftbots sample." << std::endl;
          queue_new_epoch();
        }
      }
      batched = processed_value.release();
    }
    py::capsule free_when_done{
        batched,
        [](void *ptr) { delete reinterpret_cast<std::vector<Draft> *>(ptr); }};
    Draft &first_pick = batched->front();
    return {
        py::array_t<std::int16_t>{basics_shape(), basics_strides,
                                  first_pick.basics.data(), free_when_done},
        py::array_t<std::int16_t>{pool_shape(), pool_strides,
                                  first_pick.pool.data(), free_when_done},
        py::array_t<std::int16_t>{seen_packs_shape(), seen_packs_strides,
                                  first_pick.seen_packs[0].data(),
                                  free_when_done},
        py::array_t<std::int8_t>{seen_coords_shape(), seen_coords_strides,
                                 first_pick.seen_coords[0][0].data(),
                                 free_when_done},
        py::array_t<float>{
            seen_coord_weights_shape(), seen_coord_weights_strides,
            first_pick.seen_coord_weights[0].data(), free_when_done},
        py::array_t<std::int8_t>{is_trashed_shape(), is_trashed_strides,
                                 first_pick.is_trashed.data(), free_when_done},
        py::array_t<float>{riskiness_shape(), riskiness_strides,
                           first_pick.riskiness[0].data(), free_when_done},
    };
  }

  Result getitem(std::size_t) & { return next(); }

private:
  static constexpr std::array<std::size_t, 2> basics_strides{
      sizeof(Draft), sizeof(std::int16_t)};
  static constexpr std::array<std::size_t, 2> pack_strides{
      sizeof(Draft), sizeof(std::int16_t)};
  static constexpr std::array<std::size_t, 2> pool_strides{
      sizeof(Draft), sizeof(std::int16_t)};
  static constexpr std::array<std::size_t, 3> seen_packs_strides{
      sizeof(Draft), sizeof(Pack), sizeof(std::int16_t)};
  static constexpr std::array<std::size_t, 4> seen_coords_strides{
      sizeof(Draft), sizeof(Coords), sizeof(CoordPair), sizeof(std::int8_t)};
  static constexpr std::array<std::size_t, 3> seen_coord_weights_strides{
      sizeof(Draft), sizeof(CoordWeights), sizeof(float)};
  static constexpr std::array<std::size_t, 3> coords_strides{
      sizeof(Draft), sizeof(CoordPair), sizeof(std::int8_t)};
  static constexpr std::array<std::size_t, 2> coord_weights_strides{
      sizeof(Draft), sizeof(float)};
  static constexpr std::array<std::size_t, 2> is_trashed_strides{
      sizeof(Draft), sizeof(std::int8_t)};
  static constexpr std::array<std::size_t, 3> riskiness_strides{
      sizeof(Draft), sizeof(std::array<float, MAX_CARDS_IN_PACK>),
      sizeof(float)};

  constexpr std::array<std::size_t, 2> basics_shape() const noexcept {
    return {batch_size, MAX_BASICS};
  }
  constexpr std::array<std::size_t, 2> pool_shape() const noexcept {
    return {batch_size, MAX_PICKED};
  }
  constexpr std::array<std::size_t, 3> seen_packs_shape() const noexcept {
    return {batch_size, MAX_SEEN_PACKS, MAX_CARDS_IN_PACK};
  }
  constexpr std::array<std::size_t, 4> seen_coords_shape() const noexcept {
    return {batch_size, MAX_SEEN_PACKS, 4, 2};
  }
  constexpr std::array<std::size_t, 3>
  seen_coord_weights_shape() const noexcept {
    return {batch_size, MAX_SEEN_PACKS, 4};
  }
  constexpr std::array<std::size_t, 2> is_trashed_shape() const noexcept {
    return {batch_size, MAX_SEEN_PACKS};
  }
  constexpr std::array<std::size_t, 3> riskiness_shape() const noexcept {
    return {batch_size, MAX_SEEN_PACKS, MAX_CARDS_IN_PACK};
  }

  void worker_func(std::stop_token st, pcg32 rng) {
    using namespace std::chrono_literals;
    std::uniform_int_distribution<std::size_t> index_selector(
        0, SHUFFLE_BUFFER_SIZE - 1);
    auto current_batch = std::make_unique<std::vector<Draft>>(batch_size);
    std::size_t current_batch_idx = 0;
    std::vector<Draft> shuffle_buffer;
    shuffle_buffer.reserve(SHUFFLE_BUFFER_SIZE);
    moodycamel::ConsumerToken chunk_consumer{chunk_queue};
    moodycamel::ProducerToken processed_producer{processed_queue};
    std::array<std::size_t, READ_SIZE> read_indices;
    while (!st.stop_requested()) {
      auto iter = read_indices.begin();
      while (iter != read_indices.end()) {
        iter += chunk_queue.try_dequeue_bulk(
            chunk_consumer, iter, std::distance(iter, read_indices.end()));
        if (st.stop_requested())
          return;
      }
      for (std::size_t read_idx : read_indices) {
        if (shuffle_buffer.size() < SHUFFLE_BUFFER_SIZE) {
          for (std::size_t i = 0; i < READ_SIZE; i++) {
            const Draft *read_start = reinterpret_cast<const Draft *>(
                mmap.data() + sizeof(Draft) * READ_SIZE * read_idx);
            shuffle_buffer.push_back(read_start[i]);
          }
          // std::cout << "Reading into Shuffle Buffer." << std::endl;
        } else {
          if (current_batch_idx >= batch_size) {
            if (processed_queue.size_approx() >=
                16 * SHUFFLE_BUFFER_SIZE / batch_size) {
              while (processed_queue.size_approx() >=
                         8 * SHUFFLE_BUFFER_SIZE / batch_size &&
                     !st.stop_requested()) {
                // std::cout << "Waiting for queue to empty." << std::endl;
                std::this_thread::sleep_for(100ms);
              }
            }
            processed_queue.enqueue(processed_producer,
                                    std::move(current_batch));
            current_batch = std::make_unique<std::vector<Draft>>(batch_size);
            current_batch_idx = 0;
          }
          std::size_t read_amount =
              std::min(current_batch->size() - current_batch_idx, READ_SIZE);
          std::memcpy(current_batch->data() + current_batch_idx,
                      reinterpret_cast<const Draft *>(
                          mmap.data() + sizeof(Draft) * READ_SIZE * read_idx),
                      sizeof(Draft) * read_amount);
          for (std::size_t i = 0; i < read_amount; i++) {
            std::swap(current_batch->at(current_batch_idx++),
                      shuffle_buffer[index_selector(rng)]);
          }
        }
      }
    }
  }
};

PYBIND11_MODULE(draftbot_generator, m) {
  using namespace pybind11::literals;
  py::class_<DraftbotGenerator>(m, "DraftbotGenerator")
      .def(py::init<std::string, std::size_t, std::size_t>())
      .def("__enter__", &DraftbotGenerator::enter)
      .def("__exit__", &DraftbotGenerator::exit)
      .def("__len__", &DraftbotGenerator::size)
      .def("__getitem__", &DraftbotGenerator::getitem)
      .def("next", &DraftbotGenerator::next)
      .def("on_epoch_end", &DraftbotGenerator::queue_new_epoch);
}
