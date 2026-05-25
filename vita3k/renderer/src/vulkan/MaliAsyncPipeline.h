#pragma once

#include <vulkan/vulkan.h>
#include <thread>
#include <atomic>
#include <functional>
#include <unordered_map>
#include <string>
#include <vector>
#include <mutex>
#include <cstdint>

// ---------------------------------------------------------
// MALI-G68 LOCK-FREE QUEUE + SPIN-PROTECTED STATUS
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
        workerThread_ = std::thread(&AsyncPipelineManager::WorkerLoop, this);
    }

    void Shutdown() {
        if (shutdown_.exchange(true, std::memory_order_acq_rel)) return;
        if (workerThread_.joinable()) {
            workerThread_.join();
        }
    }

    bool QueuePipeline(uint64_t hash, CompileTask task) {
        // 1. Ultra-fast state check
        {
            std::lock_guard<std::mutex> lock(statusMutex_);
            auto it = pipelineStatus_.find(hash);
            if (it != pipelineStatus_.end()) {
                return false; // Already queued or compiled
            }
            // Mark as compiling immediately so we don't queue duplicates
            pipelineStatus_[hash] = VK_NULL_HANDLE; 
        }

        // 2. Lock-free ring buffer push
        uint32_t currentTail = tail_.load(std::memory_order_relaxed);
        uint32_t nextTail = (currentTail + 1) & QUEUE_MASK;

        if (nextTail == head_.load(std::memory_order_acquire)) {
            return false; // Queue full
        }

        ringBuffer_[currentTail] = {hash, std::move(task)};
        tail_.store(nextTail, std::memory_order_release);
        return true;
    }

    VkPipeline GetPipeline(uint64_t hash) {
        std::lock_guard<std::mutex> lock(statusMutex_);
        auto it = pipelineStatus_.find(hash);
        if (it != pipelineStatus_.end()) {
            return it->second;
        }
        return VK_NULL_HANDLE;
    }

private:
    AsyncPipelineManager() = default;
    ~AsyncPipelineManager() { Shutdown(); }

    struct TaskWrapper {
        uint64_t hash;
        CompileTask task;
    };

    VkDevice device_ = VK_NULL_HANDLE;
    std::string cacheFilePath_;
    std::thread workerThread_;
    std::atomic<bool> shutdown_{true};

    // --- LOCK-FREE SPSC RING BUFFER ---
    static constexpr uint32_t QUEUE_SIZE = 1024; // Must be a power of 2
    static constexpr uint32_t QUEUE_MASK = QUEUE_SIZE - 1;
    
    TaskWrapper ringBuffer_[QUEUE_SIZE];
    std::atomic<uint32_t> head_{0};
    std::atomic<uint32_t> tail_{0};

    // --- PROTECTED STATUS MAP ---
    std::mutex statusMutex_;
    std::unordered_map<uint64_t, VkPipeline> pipelineStatus_;

    void WorkerLoop() {
        while (!shutdown_.load(std::memory_order_acquire)) {
            uint32_t currentHead = head_.load(std::memory_order_relaxed);
            
            if (currentHead == tail_.load(std::memory_order_acquire)) {
                std::this_thread::yield(); 
                continue;
            }

            // Pop task lock-free
            TaskWrapper currentTask = std::move(ringBuffer_[currentHead]);
            head_.store((currentHead + 1) & QUEUE_MASK, std::memory_order_release);

            // Execute heavy Mali compilation OUTSIDE the mutex lock
            VkPipeline compiledPipeline = currentTask.task();

            if (compiledPipeline != VK_NULL_HANDLE) {
                // Lock only for the split second it takes to write the pointer
                std::lock_guard<std::mutex> lock(statusMutex_);
                pipelineStatus_[currentTask.hash] = compiledPipeline;
            }
        }
    }
};
