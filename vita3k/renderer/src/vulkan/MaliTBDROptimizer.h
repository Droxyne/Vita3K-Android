#pragma once

#include <vector>
#include <stdexcept>
#include <vulkan/vulkan.h>   // ensure VK headers are included

class MaliTBDROptimizer {
public:
    static bool IsDepthStencilFormat(VkFormat format) {
        return format == VK_FORMAT_D16_UNORM ||
               format == VK_FORMAT_D32_SFLOAT ||
               format == VK_FORMAT_D16_UNORM_S8_UINT ||
               format == VK_FORMAT_D24_UNORM_S8_UINT ||
               format == VK_FORMAT_D32_SFLOAT_S8_UINT ||
               format == VK_FORMAT_S8_UINT;
    }

    // Helper to find the right memory type index
    static uint32_t FindMemoryType(VkPhysicalDevice physicalDevice, uint32_t typeFilter, VkMemoryPropertyFlags properties) {
        VkPhysicalDeviceMemoryProperties memProperties;
        vkGetPhysicalDeviceMemoryProperties(physicalDevice, &memProperties);
        for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++) {
            if ((typeFilter & (1 << i)) && (memProperties.memoryTypes[i].propertyFlags & properties) == properties) {
                return i;
            }
        }
        return ~0u;
    }

    // ------------------------------------------------------------------------
    // 1. INTERCEPT IMAGE CREATION
    // ------------------------------------------------------------------------
    static VkResult CreateOptimizedImage(
        VkPhysicalDevice physicalDevice,
        VkDevice device,
        VkImageCreateInfo* pCreateInfo,
        VkImage* pImage,
        VkDeviceMemory* pImageMemory,
        bool guest_requires_readback = false)
    {
        bool isDepthStencil = IsDepthStencilFormat(pCreateInfo->format);

        // If the guest doesn't explicitly need to sample this texture later, trap it in Tile SRAM
        if (isDepthStencil && !guest_requires_readback) {
            // Strip sampled/transfer bits that desktop emulators lazily leave on
            pCreateInfo->usage &= ~VK_IMAGE_USAGE_SAMPLED_BIT;
            pCreateInfo->usage &= ~VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
            pCreateInfo->usage &= ~VK_IMAGE_USAGE_TRANSFER_DST_BIT;
            // Force Transient Attachment
            pCreateInfo->usage |= VK_IMAGE_USAGE_TRANSIENT_ATTACHMENT_BIT;
        }

        VkResult result = vkCreateImage(device, pCreateInfo, nullptr, pImage);
        if (result != VK_SUCCESS) return result;

        VkMemoryRequirements memRequirements;
        vkGetImageMemoryRequirements(device, *pImage, &memRequirements);

        VkMemoryAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        allocInfo.allocationSize = memRequirements.size;

        // Attempt to find LAZILY_ALLOCATED memory (Mali supports this for Transient images)
        uint32_t memType = ~0u;
        if (pCreateInfo->usage & VK_IMAGE_USAGE_TRANSIENT_ATTACHMENT_BIT) {
            memType = FindMemoryType(physicalDevice, memRequirements.memoryTypeBits,
                                     VK_MEMORY_PROPERTY_LAZILY_ALLOCATED_BIT);
        }

        // Fallback to standard DEVICE_LOCAL if lazy allocation isn't supported or requested
        if (memType == ~0u) {
            memType = FindMemoryType(physicalDevice, memRequirements.memoryTypeBits,
                                     VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        }

        // ----- FIX: check if memory type was actually found -----
        if (memType == ~0u) {
            // No suitable memory type – destroy the image and return error
            vkDestroyImage(device, *pImage, nullptr);
            return VK_ERROR_OUT_OF_DEVICE_MEMORY;
        }

        allocInfo.memoryTypeIndex = memType;
        result = vkAllocateMemory(device, &allocInfo, nullptr, pImageMemory);
        if (result == VK_SUCCESS) {
            vkBindImageMemory(device, *pImage, *pImageMemory, 0);
        } else {
            vkDestroyImage(device, *pImage, nullptr);
        }
        return result;
    }

    // ------------------------------------------------------------------------
    // 2. INTERCEPT RENDER PASS CREATION
    // ------------------------------------------------------------------------
    static void OptimizeRenderPassAttachments(
        std::vector<VkAttachmentDescription>& attachments,
        bool is_guest_clear_pass,
        bool guest_requires_depth_readback)
    {
        for (auto& att : attachments) {
            bool isDepthStencil = IsDepthStencilFormat(att.format);

            if (isDepthStencil) {
                // FORCE STORE OP: DONT_CARE
                if (!guest_requires_depth_readback) {
                    att.storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
                    att.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
                }

                // FORCE LOAD OP: CLEAR or DONT_CARE
                if (att.loadOp == VK_ATTACHMENT_LOAD_OP_LOAD) {
                    if (is_guest_clear_pass) {
                        att.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
                        att.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
                    } else {
                        att.loadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
                        att.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
                    }
                }
            } else {
                // Color Attachments: optimize LoadOp
                if (att.loadOp == VK_ATTACHMENT_LOAD_OP_LOAD && is_guest_clear_pass) {
                    att.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
                }
            }
        }
    }
};
