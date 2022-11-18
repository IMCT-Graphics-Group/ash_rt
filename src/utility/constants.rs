use std::os::raw::c_char;

use crate::utility::debug::ValidationInfo;
use crate::utility::structures::*;

use ash::vk;
use winit::event::VirtualKeyCode;

// Constants
pub const WINDOW_TITLE: &'static str = "测试窗口";
// pub const TEXTURE_PATH: &'static str = "textures/texture.jpg";
pub const MODEL_PATH: &'static str = "assets/viking_room.obj";
pub const TEXTURE_PATH: &'static str = "assets/viking_room.png";
pub const WINDOW_WIDTH: u32 = 800;
pub const WINDOW_HEIGHT: u32 = 600;
pub const WINDOW_KEYCODE_EXIT: VirtualKeyCode = VirtualKeyCode::Escape;

pub const VALIDATION: ValidationInfo = ValidationInfo {
    is_enable: true,
    required_validation_layers: ["VK_LAYER_KHRONOS_validation"],
};
pub const APPLICATION_VERSION: u32 = vk::make_api_version(0, 1, 0, 0);
pub const API_VERSION: u32 = vk::make_api_version(0, 1, 3, 0);
pub const ENGINE_VERSION: u32 = vk::make_api_version(0, 1, 0, 0);
pub const DEVICE_EXTENSIONS: DeviceExtension = DeviceExtension {
    names: ["VK_KHR_swapchain"],
};

impl DeviceExtension {
    pub fn get_extensions_raw_names(&self) -> [*const c_char; 5] {
        [
            ash::extensions::khr::Swapchain::name().as_ptr(),
            ash::extensions::nv::RayTracing::name().as_ptr(),
            vk::ExtDescriptorIndexingFn::name().as_ptr(),
            vk::ExtScalarBlockLayoutFn::name().as_ptr(),
            vk::KhrGetMemoryRequirements2Fn::name().as_ptr(),
        ]
    }
}

pub const MAX_FRAMES_IN_FLIGHT: usize = 2;
