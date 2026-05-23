#pragma once

#include <vulkan/vulkan.h>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <queue>
#include <functional>
#include <unordered_map>
#include <string>
#include <fstream>
#include <vector>
#include <atomic>
#include <iostream>
#include <cstdint>

// ---------------------------------------------------------
// MALI-G68 ASYNC PIPELINE OPTIMIZER
// ---------------------------------------------------------
class AsyncPipelineManager {
public:
    using CompileTask = std::function<VkPipeline()>;

    static AsyncPipelineManager& GetInstance() {
        static AsyncPipelineManager instance;
        return instance;
    }

    void Initialize(VkDevice device, const std::string& cacheFilePath) {
        device_ = device;
        cacheFilePath_ = cacheFilePath;
        shutdown_.store(false);

        LoadCache();  // stub, does nothing

        // Spawn background worker
        workerThread_ = std::thread(&AsyncPipelineManager::WorkerLoop, this);
    }

    // FIXED: Shutdown with proper locking
    void Shutdown() {
        if (shutdown_.exchange(true)) return;

        cv_.notify_all();
        if (workerThread_.joinable()) {
            workerThread_.join();
        }

        // Lock the I/O mutex BEFORE destroying the cache
        std::lock_guard<std::mutex> lock(cacheIOMutex_);

    }

    VkPipelineCache GetCacheHandle() const {
        return pipelineCache_;
    }

    bool QueuePipeline(uint64_t hash, CompileTask task) {
        std::lock_guard<std::mutex> lock(stateMutex_);

        if (pipelineStatus_.count(hash)) {
            return false; // Already compiled or in queue
        }

        pipelineStatus_[hash] = VK_NULL_HANDLE; // Mark as compiling

        {
            std::lock_guard<std::mutex> qLock(queueMutex_);
            taskQueue_.push({hash, std::move(task)});
        }
        cv_.notify_one();
        return true;
    }

    VkPipeline GetPipeline(uint64_t hash) {
        std::lock_guard<std::mutex> lock(stateMutex_);
        auto it = pipelineStatus_.find(hash);
        if (it != pipelineStatus_.end()) {
            return it->second;
        }
        return VK_NULL_HANDLE;
    }

    void SaveCache() {
        // stub - pipeline cache persistence handled by Vita3K core
    }

private:
    AsyncPipelineManager() = default;
    ~AsyncPipelineManager() { Shutdown(); }

    struct TaskWrapper {
        uint64_t hash;
        CompileTask task;
    };

    VkDevice device_ = VK_NULL_HANDLE;
    VkPipelineCache pipelineCache_ = VK_NULL_HANDLE;
    std::string cacheFilePath_;

    std::thread workerThread_;
    std::atomic<bool> shutdown_{true};

    std::mutex queueMutex_;
    std::condition_variable cv_;
    std::queue<TaskWrapper> taskQueue_;

    std::mutex stateMutex_;
    std::unordered_map<uint64_t, VkPipeline> pipelineStatus_;

    std::mutex cacheIOMutex_;
    uint32_t pipelinesCompiledSinceLastSave_ = 0;

    void LoadCache() {
        // stub - pipeline cache persistence handled by Vita3K core
        pipelineCache_ = VK_NULL_HANDLE;
    }

    void WorkerLoop() {
        while (!shutdown_.load()) {
            TaskWrapper currentTask;
            {
                std::unique_lock<std::mutex> lock(queueMutex_);
                cv_.wait(lock, [this] { return !taskQueue_.empty() || shutdown_.load(); });

                if (shutdown_.load() && taskQueue_.empty()) break;

                currentTask = std::move(taskQueue_.front());
                taskQueue_.pop();
            }

            // Execute the heavy Mali driver compilation
            VkPipeline compiledPipeline = currentTask.task();

            if (compiledPipeline != VK_NULL_HANDLE) {
                bool need_save = false;
                {
                    std::lock_guard<std::mutex> lock(stateMutex_);
                    pipelineStatus_[currentTask.hash] = compiledPipeline;

                    pipelinesCompiledSinceLastSave_++;
                    if (pipelinesCompiledSinceLastSave_ >= 10) {
                        pipelinesCompiledSinceLastSave_ = 0;
                        need_save = true;
                    }
                }
                // Save outside the stateMutex_ to avoid blocking pipeline lookups
                if (need_save) {
                    SaveCache();
                }
            }
        }
    }
};
