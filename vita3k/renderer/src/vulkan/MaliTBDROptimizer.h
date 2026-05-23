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
        bool guest_requires_depth_readback)
    {
        for (auto& att : attachments) {
            bool isDepthStencil = IsDepthStencilFormat(att.format);

            if (isDepthStencil) {
                if (!guest_requires_depth_readback) {
                    att.storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
                    att.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
                }

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
                if (att.loadOp == VK_ATTACHMENT_LOAD_OP_LOAD && is_guest_clear_pass) {
                    att.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
                }
            }
        }
    }
};
