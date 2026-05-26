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
            bool isPresentAttachment = (att.finalLayout == VK_IMAGE_LAYOUT_PRESENT_SRC_KHR);

            if (isDepthStencil) {
                // Depth/stencil: always store for now.
                // Vita games do CPU readbacks on depth constantly (shadow maps etc).
                // Re-enable DONT_CARE here only after rendering is confirmed stable
                // and you have measured actual bandwidth savings.
                att.storeOp        = VK_ATTACHMENT_STORE_OP_STORE;
                att.stencilStoreOp = VK_ATTACHMENT_STORE_OP_STORE;

                if (att.loadOp == VK_ATTACHMENT_LOAD_OP_LOAD) {
                    if (is_guest_clear_pass) {
                        att.loadOp        = VK_ATTACHMENT_LOAD_OP_CLEAR;
                        att.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
                    } else {
                        att.loadOp        = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
                        att.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
                    }
                }
            } else {
                // Color attachments
                if (att.loadOp == VK_ATTACHMENT_LOAD_OP_LOAD && is_guest_clear_pass) {
                    att.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
                }

                // NEVER discard the final present attachment — that's the actual frame.
                // Only intermediate color buffers that nothing reads back are safe to discard.
                if (!guestNeedsData && !isPresentAttachment) {
                    att.storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
                } else {
                    att.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
                }
            }
        }
    }
};
