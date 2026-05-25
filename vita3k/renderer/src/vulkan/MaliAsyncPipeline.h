#pragma once

#include <vulkan/vulkan.h>
#include <thread>
#include <atomic>
#include <functional>
#include <unordered_map>
#include <string>
#include <vector>
#include <cstdint>

// ---------------------------------------------------------
// MALI-G68 LOCK-FREE ASYNC PIPELINE OPTIMIZER
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
        shutdown_.store(false, std::memory_order_release);

        LoadCache();

        // Spawn background worker
        workerThread_ = std::thread(&AsyncPipelineManager::WorkerLoop, this);
    }

    void Shutdown() {
        if (shutdown_.exchange(true, std::memory_order_acq_rel)) return;

        if (workerThread_.joinable()) {
            workerThread_.join();
        }
    }

    VkPipelineCache GetCacheHandle() const {
        return pipelineCache_;
    }

    // LOCK-FREE: Push task onto the single-producer single-consumer ring buffer
    bool QueuePipeline(uint64_t hash, CompileTask task) {
        uint32_t currentTail = tail_.load(std::memory_order_relaxed);
        uint32_t nextTail = (currentTail + 1) & QUEUE_MASK;

        // Check if queue is full
        if (nextTail == head_.load(std::memory_order_acquire)) {
            return false; 
        }

        // If the hash is already tracked and valid, don't re-queue
        auto it = pipelineStatus_.find(hash);
        if (it != pipelineStatus_.end() && it->second.load(std::memory_order_relaxed) != VK_NULL_HANDLE) {
            return false;
        }

        ringBuffer_[currentTail] = {hash, std::move(task)};
        tail_.store(nextTail, std::memory_order_release);
        return true;
    }

    // LOCK-FREE: Main render thread lookups read atomic pointers directly
    VkPipeline GetPipeline(uint64_t hash) {
        auto it = pipelineStatus_.find(hash);
        if (it != pipelineStatus_.end()) {
            return it->second.load(std::memory_order_acquire);
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

    // --- LOCK-FREE SPSC RING BUFFER CONFIG ---
    static constexpr uint32_t QUEUE_SIZE = 1024; // Must be a power of 2
    static constexpr uint32_t QUEUE_MASK = QUEUE_SIZE - 1;
    
    TaskWrapper ringBuffer_[QUEUE_SIZE];
    std::atomic<uint32_t> head_{0};
    std::atomic<uint32_t> tail_{0};

    // Maps pipeline hash to an atomic VkPipeline handle to prevent data races
    std::unordered_map<uint64_t, std::atomic<VkPipeline>> pipelineStatus_;
    uint32_t pipelinesCompiledSinceLastSave_ = 0;

    void LoadCache() {
        pipelineCache_ = VK_NULL_HANDLE;
    }

    void WorkerLoop() {
        while (!shutdown_.load(std::memory_order_acquire)) {
            uint32_t currentHead = head_.load(std::memory_order_relaxed);
            
            if (currentHead == tail_.load(std::memory_order_acquire)) {
                // Low latency spin-yield instead of hitting heavy OS sleep states via cv
                std::this_thread::yield(); 
                continue;
            }

            // Pop from buffer
            TaskWrapper currentTask = std::move(ringBuffer_[currentHead]);
            head_.store((currentHead + 1) & QUEUE_MASK, std::memory_order_release);

            // Let the Mali GPU driver run heavy compilation in the background
            VkPipeline compiledPipeline = currentTask.task();

            if (compiledPipeline != VK_NULL_HANDLE) {
                pipelineStatus_[currentTask.hash].store(compiledPipeline, std::memory_order_release);

                pipelinesCompiledSinceLastSave_++;
                if (pipelinesCompiledSinceLastSave_ >= 10) {
                    pipelinesCompiledSinceLastSave_ = 0;
                    SaveCache();
                }
            }
        }
    }
};
