#pragma once
#include <vulkan/vulkan.h>
#include <vector>
#include <stdexcept>

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

    // FIXED: Upgraded from single boolean to per-attachment tracking array
    static void OptimizeRenderPassAttachments(
        std::vector<VkAttachmentDescription>& attachments,
        bool is_guest_clear_pass,
        const std::vector<bool>& attachments_require_readback)
    {
        if (attachments.size() != attachments_require_readback.size()) {
            throw std::runtime_error("MaliTBDROptimizer Error: Size mismatch between attachments and readback tracking flags.");
        }

        for (size_t i = 0; i < attachments.size(); ++i) {
            auto& att = attachments[i];
            bool isDepthStencil = IsDepthStencilFormat(att.format);
            bool guestNeedsData = attachments_require_readback[i];

            if (isDepthStencil) {
                if (!guestNeedsData) {
                    // Safe to optimize out: keep it entirely inside local GPU tile memory
                    att.storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
                    att.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
                } else {
                    // CRITICAL FIX: Spill to system RAM so the guest can read it back safely
                    att.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
                    att.stencilStoreOp = VK_ATTACHMENT_STORE_OP_STORE;
                }

                if (att.loadOp == VK_ATTACHMENT_LOAD_OP_LOAD) {
                    if (is_guest_clear_pass) {
                        att.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
                        att.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
                    } else {
                        // Keep as DONT_CARE if we don't need historical data preserved on load
                        att.loadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
                        att.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
                    }
                }
            } else {
                // Color attachments
                if (att.loadOp == VK_ATTACHMENT_LOAD_OP_LOAD && is_guest_clear_pass) {
                    att.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
                }

                // Balance bandwidth on color buffers as well
                if (!guestNeedsData) {
                    att.storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
                } else {
                    att.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
                }
            }
        }
    }
};
