#pragma once

#include "../../utils.h"
#include <Eigen/Dense>
#include <chrono>
#include <memory>
#include <shared_mutex>

#include <type_traits>
#include <unordered_map>

/**
 * @brief
 *
 * Sensor namespace describes Sensor hooks and operations used for the
 * Common Kalman Filtering (CKF) framework developed in this software. CKF
 * describes an trigger-based filtering approach where a task is queued into a
 * filter and is executed asynchronously from the queue time.
 *
 * By forcing the onus on the sensor suite to trigger Kalman updates, we also
 * remove the need for filters to implement a strict heartbeat.
 *
 * @see ckf.hpp
 * @authors Krish Sridhar
 * @date July 7th, 2026
 */
namespace Sensor {

// just some conveniences...
using SteadyClock = std::chrono::steady_clock;
using CorrectFn =
    std::function<void(const Eigen::MatrixXd &H, const Eigen::MatrixXd &R,
                       const Eigen::VectorXd &r)>;
/**
 * @brief
 *
 * Virtual Sensor class "interface". ISensor acts as a contract that
 * new Sensor implementations must subclass to be accepted into the Sensor
 * Table.
 *
 * Defines basic implementation requirements for a sensor to interface with
 * the framework.
 */
class ISensor {
protected:
  // common shared mutex. Allows for read/write acces on common data.
  mutable std::shared_mutex data_mtx;
  // name of the sensor
  std::string id;
  // this is constant over the entire runtime (sensor characteristic)
  Eigen::MatrixXd R;

public:
  virtual ~ISensor() = default;

  /**
   * @brief Measurement noise matrix of the sensor.
   *
   * @return Eigen::MatrixXd
   */
  virtual Eigen::MatrixXd get_R() final {
    WITH_SH_LOCK(this->data_mtx) return this->R;
  }

  /**
   * @brief Get the id, or name, of the sensor
   *
   * @return std::string
   */
  virtual std::string get_id() final { return this->id; }
};

/**
 * @brief Sensor Table stores the sensors used by the framework, and provides
 * simple data retrieval and push commands.
 */
class SensorTable {
private:
  /**
   * @brief SensorHook allows for processes to attach themselves onto
   * the SensorTable and do work when the sensor populates data into the table.
   *
   * @tparam T Datatype returned by the Sensor
   */
  template <typename T> class SensorHook final {
  public:
    /// SensorSnapshotFunction allows users to interface with the internal
    /// Measurment Noise matrix and delivered sensor by data, timestamped by
    /// when the data arrived at the sensor.
    using SensorSnapshotFunction = std::function<void(
        const T &data, const Eigen::MatrixXd &R, SteadyClock::time_point t)>;

    /// InitializeFunction is a one-time initialization protocol that is
    /// activated immediately after the Sensor's statistics engine @ref
    /// FilteredSampleProducer says the Sensor has been initialized with
    /// sufficient data.
    using InitializeFunction =
        std::function<void(const FilteredSampleProducer<T> &)>;

    /// Helper function indicating that the hooked process requires no
    /// initialization.
    inline static constexpr auto NoInitialize =
        [](const FilteredSampleProducer<T> &) {};

  private:
    // sensor hook should have the name of the process its hooked onto
    std::string id;
    // serializes each function in a common vector so that it remembers who to
    // alert
    std::vector<SensorSnapshotFunction> snap_bindings;
    // same as above, but for initialization instead of data arrivals
    std::vector<InitializeFunction> init_bindings;

  public:
    /**
     * @brief Construct a new Sensor Hook object.
     *
     * @param id_in id of the Sensor that you're hooked onto. Useful for
     * debugging.
     */
    explicit SensorHook(std::string id_in) : id(id_in) {}
    /**
     * @brief Get the id object.
     *
     * @return const std::string&
     */
    const std::string &get_id() const { return this->id; }
    /**
     * @brief Get the id object.
     *
     * @return std::string&
     */
    std::string &get_id() { return this->id; }

    /**
     * @brief Binds a process including a snapshot and initialize function to
     * trigger once data arrives or initialization of the sensor has occurred.
     *
     * @param fn SensorSnapshotFunction implementation
     * @param init_fn InitializeFunction implementation
     */
    void bind(SensorSnapshotFunction fn, InitializeFunction init_fn) {
      snap_bindings.emplace_back(fn);
      init_bindings.emplace_back(init_fn);
    }

    /**
     * @brief Triggers the initialize call to initialize all sensors and trigger
     * the set initialization functions.
     *
     * @warning Will clear the initialization function binding vector, and will
     * not work more than once!
     * @param prod @ref FilteredSampleProducer data
     */
    void initialize(const FilteredSampleProducer<T> &prod) {
      for (auto &binding : init_bindings)
        binding(prod);

      init_bindings.clear();
    }

    /**
     * @brief Emits a signal to the bound functions to trigger on data arrival.
     *
     * @param z Sensor input data.
     * @param R Measurement noise matrix.
     * @param t timestamp of data arrival.
     */
    void emit(const T &z, const Eigen::MatrixXd &R, SteadyClock::time_point t) {
      for (auto &binding : snap_bindings)
        binding(z, R, t);
    }
  };

  /// SensorContainer backs the SensorTbale object by encapsulating incoming
  /// data filtered through the @ref FilteredSampleProducer. Each container has
  /// a hook that is revealed.
  template <typename T> class SensorContainer final : public ISensor {
  private:
    // incoming data
    FilteredSampleProducer<T> prod;
    // minimum samples before initialized
    uint32_t min_samples;
    // calibration flag
    bool calibrated{false};
    // the hook.
    std::unique_ptr<SensorHook<T>> hook;

  public:
    /**
     * @brief Construct a new Sensor Container object
     *
     * @param init initial data
     * @param min_samples_in minimum samples to calibrate
     * @param id_in the name of the sensor
     */
    SensorContainer(const T &init, uint32_t min_samples_in, std::string id_in)
        : prod{init}, min_samples(min_samples_in),
          hook(std::make_unique<SensorHook<T>>(id_in)) {
      this->id = id_in;
    };

    /**
     * @brief Get the mean from the data
     *
     * @param mean average sensor reading
     * @return true data is calibrated
     * @return false data is not calibrated
     */
    bool get_mean(T &mean) {
      WITH_SH_LOCK(this->data_mtx)

      if (!this->is_ready())
        return false;

      mean = this->prod.get_mean();
      return true;
    }

    /**
     *  Binds callbacks to the container's internal sensor hook.
     *
     *  snap_fn function to call when sensor data arrives
     *  init_fn function to call when sensor calibration completes
     */
    void bind(typename SensorHook<T>::SensorSnapshotFunction snap_fn,
              typename SensorHook<T>::InitializeFunction init_fn) {
      this->hook->bind(std::move(snap_fn), std::move(init_fn));
    }

    /**
     *  Checks if the calibration is finished.
     *
     *  true Sensor is calibrated
     *  false Sensor is not calibrated
     */
    bool is_ready() { return this->prod.get_count() > min_samples; }

    /**
     * @brief Puts data into the Sensor Table.
     *
     * @param data The input data.
     * @param sample_time when the data was collected.
     */
    void put(const T &data,
             SteadyClock::time_point sample_time = SteadyClock::now()) {
      T z;
      SteadyClock::time_point t;

      {
        // put data into the producer on a unique lock.
        WITH_UQ_LOCK(this->data_mtx) this->prod.put(data, sample_time);

        // checks if calibrated
        if (!this->is_ready())
          return;

        // if constexpr checks if the type at compile time is an scalar or not
        IF_CONSTEXPR_ASSIGN(
            this->R, T,
            Eigen::MatrixXd::Constant(1, 1, this->prod.get_variance()),
            this->prod.get_variance().asDiagonal());

        // if I got this far, then initialize the data through the hook and
        // return
        if (!this->calibrated) {
          this->calibrated = true;
          this->hook->initialize(this->prod);
          return;
        }

        // get the latest data off the producer and emit under the unique hook
        this->prod.get_latest(z, t);
      }
      hook->emit(z, this->R, t);
    }

    /**
     * @brief Get the data off the producer.
     *
     * @param data data to store
     * @param t timestamp
     * @return true data has been acquired
     * @return false data has not been acquired
     */
    bool get(T &data, SteadyClock::time_point &t) {
      {
        // check with the fast read lock if I have the data and return
        WITH_SH_LOCK(this->data_mtx)

        // if not ready, return false
        if (!this->is_ready())
          return false;

        // if calibrated just quickly get the latest data and return true
        if (this->calibrated) {
          this->prod.get_latest(data, t);
          return true;
        }
      }
      {
        // guess i dont... have to do the slow unique locking and update the
        // information
        WITH_UQ_LOCK(this->data_mtx)

        // return false if not ready
        if (!this->is_ready())
          return false;

        // initial calibration
        if (!this->calibrated) {
          this->R = this->prod.get_variance().asDiagonal();
          this->calibrated = true;
        }

        // and get the data and return.
        this->prod.get_latest(data, t);
        return true;
      }
    }
  };

  /// hashtable map to map the ID to the sensor.
  inline static std::unordered_map<std::string, std::shared_ptr<ISensor>>
      hashtable;

  /// get the shared mutex on table
  inline static std::shared_mutex table_mtx;

  /// check if an ID in the table
  inline static std::shared_ptr<ISensor> find_in_table(const std::string &id) {
    WITH_SH_LOCK(SensorTable::table_mtx)
    auto it = SensorTable::hashtable.find(id);

    return (it == SensorTable::hashtable.end()) ? nullptr : it->second;
  }

public:
  SensorTable() = delete;

  /**
   * @brief Bind the snapshot function and initialize function to the internal
   * sensor hook.
   *
   * @tparam T sensor datatype
   * @param id sensor name
   * @param snap_fn SensorSnapshotFunction funciton
   * @param init_fn InitializeFunction function
   * @return true functions succesfully
   * @return false
   */
  template <typename T>
  static bool bind(std::string id,
                   typename SensorHook<T>::SensorSnapshotFunction snap_fn,
                   typename SensorHook<T>::InitializeFunction init_fn) {
    auto container = std::static_pointer_cast<SensorContainer<T>>(
        SensorTable::find_in_table(id));
    if (container == nullptr)
      return false;

    container->bind(std::move(snap_fn), std::move(init_fn));
    return true;
  }

  /**
   * @brief Get the mean of the sensor data
   *
   * @tparam T sensor datatype
   * @param id sensor name
   * @param data container for retrieved data
   * @return true data has been retrieved
   * @return false container does not exist
   */
  template <typename T> static bool get_mean(std::string id, T &data) {
    auto container = std::static_pointer_cast<SensorContainer<T>>(
        SensorTable::find_in_table(id));
    if (container == nullptr)
      return false;

    return container->get_mean(data);
  }

  /**
   * @brief Checks if the sensor is calibrated
   *
   * @tparam T sensor datatype
   * @param id name of the sensor
   * @return true sensor is ready and calibrated
   * @return false sensor is not ready OR the sensor with identification "id"
   * doesn't exist
   */
  template <typename T> static bool is_ready(std::string id) {
    auto container = std::static_pointer_cast<SensorContainer<T>>(
        SensorTable::find_in_table(id));
    if (container == nullptr)
      return false;

    return container->is_ready();
  }

  /**
   * @brief Registers a sensor into the sensor table.
   *
   * @tparam T sensor datatypea
   * @param init initial data
   * @param min_samples minimum samples before calibration
   * @param id_in sensor name
   */
  template <typename T>
  static void register_sensor(const T &init, uint32_t min_samples,
                              std::string id_in) {
    WITH_UQ_LOCK(SensorTable::table_mtx)
    SensorTable::hashtable[id_in] =
        std::make_shared<SensorContainer<T>>(init, min_samples, id_in);
  }

  /**
   * @brief Puts data into a specific sensor container.
   *
   * @tparam T sensor datatype
   * @param id sensor name
   * @param data sensor data
   * @param sample_time timestamp of the sensor data.
   */
  template <typename T>
  static void put(const std::string &id, const T &data,
                  SteadyClock::time_point sample_time = SteadyClock::now()) {

    auto container = std::static_pointer_cast<SensorContainer<T>>(
        SensorTable::find_in_table(id));
    if (container == nullptr)
      return;

    container->put(data, sample_time);
  }
};
}; // namespace Sensor
