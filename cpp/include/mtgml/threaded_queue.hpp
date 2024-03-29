#include <atomic>
#include <chrono>
#include <functional>
#include <map>
#include <memory>
#include <shared_mutex>
#include <vector>

template <typename Data, std::size_t num_slots, std::size_t num_writers>
struct ThreadedQueue {

  Data pop() {
    /* std::shared_lock lck(data_lock, std::try_to_lock); */
    if (num_writers == 0) return {};
    Data old_value;
    do {
      old_value = data[read_index % data.size()];
    } while (old_value && old_value.compare_exchange_strong(old_value, {}));
    return old_value;
  }

  Data pop_blocking() {
    /* std::shared_lock lck(data_lock, std::try_to_lock); */
    if (num_writers == 0) return {};
    Data old_value;
    std::size_t index = read_index % data.size();
    do {
      old_value = data[index].load();
      if (!old_value) {
        index = read_index % data.size();
        data[index].wait();
      }
    } while (!old_value || data[index].compare_exchange_strong(old_value, {}));
    return old_value;
  }

  ThreadedQueue() = default;
  ~ThreadedQueue() {
    for (auto& x : data) x.store({});
  }

private:
  std::array<std::atomic<Data>, num_writers * num_slots> data;
  std::atomic<std::size_t> read_index{0};
  /* std::shared_timed_mutex data_lock; */
  std::array<std::atomic<bool>, num_writers> assigned_writers{false};

public:
  // We assume the default value of Data can be cast to bool and is false.
  struct Writer {
    bool write(const Data& data) {
      ThreadedQueue& queue = queue_ref.get();
      if(!queue.data[pos].load()) {
        queue.data[pos].store(data);
        next_pos = (next_pos + 1) % num_slots;
        return true;
      } else return false;
    }

    bool write(Data&& data) {
      /* std::shared_lock lck(queue_ref.get().data_lock, std::try_to_lock); */
      /* if (!lck) return false; */
      return write_internal(std::move(data));
    }

    bool write_blocking(Data data) {
      /* std::shared_lock lck(queue_ref.get().data_lock); */
      ThreadedQueue& queue = queue_ref.get();

      if(!queue.data[position].load()) {
        queue.data[position].store(data);
        next_pos = (next_pos + 1) % num_slots;
        return true;
      } else {
        queue.data[pos].wait();
        return write_internal(std::move(data));
      };
    }

    // These are not safe to share.
    Writer(const Writer&) = delete;
    Writer& operator=(const Writer&) = delete;
    Writer(Writer&&) = default;
    Writer& operator=(Writer&&) = delete;

  private:
    std::size_t position;
    std::reference_wrapper<ThreadedQueue> queue_ref;
    Writer(std::size_t index, std::reference_wrapper<ThreadedQueue> queue) : position{index}, queue_ref{queue} { }

    friend struct ThreadedQueue;
  };

  std::optional<Writer> get_writer() {
    for (auto& flag : 
    if (num_assigned_writers < num_writers) {
      return Writer(num_assigned_writers++, *this);
    } else {
      return std::nullopt;
    }
  }
};

template <typename Data, std::size_t num_batches>
struct BatchPool {
  std::array<std::pair<std::atomic<bool>, std::shared_ptr<std::vector<Data>>>, num_batches> batches;

  std::shared_ptr<std::vector<Data>> get_batch() {
    for (auto& batch : batches) {
      if (!batch.first.load() && batch.first.compare_exchange_strong(false, true)) {
        return batch.second;
      }
    }
    return {};
  }

  void release_batch(const std::shared_ptr<std::vector<Data>>& ptr) {
    for (auto& batch : batches) {
      if (batch.second == ptr) batch.first.store(false);
    }
  }
  void release_batch(const std::vector<Data>* ptr) {
    for (auto& batch : batches) {
      if (batch.second == ptr) batch.first.store(false);
    }
  }

  BatchPool(std::size_t batch_size) {
    for (auto& batch : batches) {
      batch.first.store(false);
      batch.second = std::make_shared<std::vector<Data>>(batch_size);
    }
  }
};
