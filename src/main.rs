use std::{ffi::CString, fs::File, mem::align_of, path::Path, ptr, rc::Rc};

use ash_rt::{
    utility,
    utility::{
        constants::*,
        structures::*,
        tools::load_model,
        window::{ProgramProc, VulkanApp},
    },
};
use cgmath::{Deg, Matrix4, Point3, Vector3};

use ash::{
    extensions::nv,
    util::{read_spv, Align},
    vk,
};

#[repr(C)]
#[derive(Clone, Debug, Copy)]
struct GeometryInstance {
    transform: [f32; 12],
    instance_id_and_mask: u32,
    instance_offset_and_flags: u32,
    acceleration_handle: u64,
}

impl GeometryInstance {
    fn new(
        transform: [f32; 12],
        id: u32,
        mask: u8,
        offset: u32,
        flags: vk::GeometryInstanceFlagsNV,
        acceleration_handle: u64,
    ) -> Self {
        let mut instance = GeometryInstance {
            transform,
            instance_id_and_mask: 0,
            instance_offset_and_flags: 0,
            acceleration_handle,
        };
        instance.set_id(id);
        instance.set_mask(mask);
        instance.set_offset(offset);
        instance.set_flags(flags);
        instance
    }

    fn set_id(&mut self, id: u32) {
        let id = id & 0x00ffffff;
        self.instance_id_and_mask |= id;
    }

    fn set_mask(&mut self, mask: u8) {
        let mask = mask as u32;
        self.instance_id_and_mask |= mask << 24;
    }

    fn set_offset(&mut self, offset: u32) {
        let offset = offset & 0x00ffffff;
        self.instance_offset_and_flags |= offset;
    }

    fn set_flags(&mut self, flags: vk::GeometryInstanceFlagsNV) {
        let flags = flags.as_raw() as u32;
        self.instance_offset_and_flags |= flags << 24;
    }
}

#[derive(Clone)]
struct ImageResource {
    image: vk::Image,
    memory: vk::DeviceMemory,
    view: vk::ImageView,
    sampler: vk::Sampler,
    base: Rc<VulkanRenderer>,
}

impl ImageResource {
    fn new(base: Rc<VulkanRenderer>) -> Self {
        ImageResource {
            image: vk::Image::null(),
            memory: vk::DeviceMemory::null(),
            view: vk::ImageView::null(),
            sampler: vk::Sampler::null(),
            base,
        }
    }

    fn create_image(
        &mut self,
        image_type: vk::ImageType,
        format: vk::Format,
        extent: vk::Extent3D,
        tiling: vk::ImageTiling,
        usage: vk::ImageUsageFlags,
        memory_flags: vk::MemoryPropertyFlags,
    ) {
        unsafe {
            let create_info = vk::ImageCreateInfo::builder()
                .image_type(image_type)
                .format(format)
                .extent(extent)
                .mip_levels(1)
                .array_layers(1)
                .samples(vk::SampleCountFlags::TYPE_1)
                .tiling(tiling)
                .usage(usage)
                .sharing_mode(vk::SharingMode::EXCLUSIVE)
                .initial_layout(vk::ImageLayout::UNDEFINED)
                .build();

            self.image = self.base.device.create_image(&create_info, None).unwrap();

            let requirements = self.base.device.get_image_memory_requirements(self.image);
            let memory_index = utility::raytracing_aid::find_memorytype_index(
                &requirements,
                &self.base.memory_properties,
                memory_flags,
            )
            .expect("Unable to find suitable memory index image.");

            let allocate_info = vk::MemoryAllocateInfo {
                allocation_size: requirements.size,
                memory_type_index: memory_index,
                ..Default::default()
            };

            self.memory = self
                .base
                .device
                .allocate_memory(&allocate_info, None)
                .unwrap();

            self.base
                .device
                .bind_image_memory(self.image, self.memory, 0)
                .expect("Unable to bind image memory");
        }
    }

    fn create_view(
        &mut self,
        view_type: vk::ImageViewType,
        format: vk::Format,
        range: vk::ImageSubresourceRange,
    ) {
        let create_info = vk::ImageViewCreateInfo::builder()
            .view_type(view_type)
            .format(format)
            .subresource_range(range)
            .image(self.image)
            .components(vk::ComponentMapping {
                r: vk::ComponentSwizzle::R,
                g: vk::ComponentSwizzle::G,
                b: vk::ComponentSwizzle::B,
                a: vk::ComponentSwizzle::A,
            })
            .build();
        self.view = unsafe {
            self.base
                .device
                .create_image_view(&create_info, None)
                .unwrap()
        };
    }
}

impl Drop for ImageResource {
    fn drop(&mut self) {
        unsafe {
            self.base.device.destroy_image_view(self.view, None);
            self.base.device.free_memory(self.memory, None);
            self.base.device.destroy_image(self.image, None);
            self.base.device.destroy_sampler(self.sampler, None);
        }
    }
}

#[derive(Clone)]
struct BufferResource {
    buffer: vk::Buffer,
    memory: vk::DeviceMemory,
    size: vk::DeviceSize,
    base: Rc<VulkanRenderer>,
}

impl BufferResource {
    fn new(
        size: vk::DeviceSize,
        usage: vk::BufferUsageFlags,
        memory_properties: vk::MemoryPropertyFlags,
        base: Rc<VulkanRenderer>,
    ) -> Self {
        unsafe {
            let buffer_info = vk::BufferCreateInfo::builder()
                .size(size)
                .usage(usage)
                .sharing_mode(vk::SharingMode::EXCLUSIVE)
                .build();

            let buffer = base.device.create_buffer(&buffer_info, None).unwrap();

            let memory_req = base.device.get_buffer_memory_requirements(buffer);

            let memory_index = utility::raytracing_aid::find_memorytype_index(
                &memory_req,
                &base.memory_properties,
                memory_properties,
            )
            .unwrap();

            let allocate_info = vk::MemoryAllocateInfo {
                allocation_size: memory_req.size,
                memory_type_index: memory_index,
                ..Default::default()
            };

            let memory = base.device.allocate_memory(&allocate_info, None).unwrap();

            base.device.bind_buffer_memory(buffer, memory, 0).unwrap();

            BufferResource {
                buffer,
                memory,
                size,
                base,
            }
        }
    }

    fn store<T: Copy>(&mut self, data: &[T]) {
        unsafe {
            let size = (std::mem::size_of::<T>() * data.len()) as u64;
            let mapped_ptr = self.map(size);
            let mut mapped_slice = Align::new(mapped_ptr, align_of::<T>() as u64, size);
            mapped_slice.copy_from_slice(&data);
            self.unmap();
        }
    }

    fn map(&mut self, size: vk::DeviceSize) -> *mut std::ffi::c_void {
        unsafe {
            let data: *mut std::ffi::c_void = self
                .base
                .device
                .map_memory(self.memory, 0, size, vk::MemoryMapFlags::empty())
                .unwrap();
            data
        }
    }

    fn unmap(&mut self) {
        unsafe {
            self.base.device.unmap_memory(self.memory);
        }
    }
}

impl Drop for BufferResource {
    fn drop(&mut self) {
        unsafe {
            self.base.device.destroy_buffer(self.buffer, None);
            self.base.device.free_memory(self.memory, None);
        }
    }
}
struct VulkanRenderer {
    window: winit::window::Window,

    _entry: ash::Entry,
    instance: ash::Instance,
    surface_loader: ash::extensions::khr::Surface,
    surface_format: vk::SurfaceFormatKHR,
    surface: vk::SurfaceKHR,
    debug_utils_loader: ash::extensions::ext::DebugUtils,
    debug_messenger: vk::DebugUtilsMessengerEXT,

    physical_device: vk::PhysicalDevice,
    memory_properties: vk::PhysicalDeviceMemoryProperties,
    device: ash::Device,

    queue_family: QueueFamilyIndices,
    graphics_queue: vk::Queue,
    present_queue: vk::Queue,

    swapchain_loader: ash::extensions::khr::Swapchain,
    swapchain: vk::SwapchainKHR,
    swapchain_images: Vec<vk::Image>,
    swapchain_format: vk::Format,
    swapchain_extent: vk::Extent2D,
    swapchain_imageviews: Vec<vk::ImageView>,
    swapchain_framebuffers: Vec<vk::Framebuffer>,

    render_pass: vk::RenderPass,
    ubo_layout: vk::DescriptorSetLayout,
    pipeline_layout: vk::PipelineLayout,
    graphics_pipeline: vk::Pipeline,

    color_image: vk::Image,
    color_image_view: vk::ImageView,
    color_image_memory: vk::DeviceMemory,

    depth_image: vk::Image,
    depth_image_view: vk::ImageView,
    depth_image_memory: vk::DeviceMemory,

    msaa_samples: vk::SampleCountFlags,

    _mip_levels: u32,
    texture_image: vk::Image,
    texture_image_view: vk::ImageView,
    texture_sampler: vk::Sampler,
    texture_image_memory: vk::DeviceMemory,

    _vertices: Vec<Vertex>,
    indices: Vec<u32>,

    vertex_buffer: vk::Buffer,
    vertex_buffer_memory: vk::DeviceMemory,
    index_buffer: vk::Buffer,
    index_buffer_memory: vk::DeviceMemory,

    uniform_transform: UniformBufferObject,
    uniform_buffers: Vec<vk::Buffer>,
    uniform_buffers_memory: Vec<vk::DeviceMemory>,

    descriptor_pool: vk::DescriptorPool,
    descriptor_sets: Vec<vk::DescriptorSet>,

    command_pool: vk::CommandPool,
    command_buffers: Vec<vk::CommandBuffer>,

    image_available_semaphores: Vec<vk::Semaphore>,
    render_finished_semaphores: Vec<vk::Semaphore>,
    in_flight_fences: Vec<vk::Fence>,
    current_frame: usize,

    is_framebuffer_resized: bool,
}

impl VulkanRenderer {
    pub fn new(event_loop: &winit::event_loop::EventLoop<()>) -> VulkanRenderer {
        let window =
            utility::window::init_window(event_loop, WINDOW_TITLE, WINDOW_WIDTH, WINDOW_HEIGHT);

        let entry = ash::Entry::linked();
        let instance = utility::general::create_instance(
            &entry,
            WINDOW_TITLE,
            VALIDATION.is_enable,
            &VALIDATION.required_validation_layers.to_vec(),
        );
        let surface_stuff = utility::general::create_surface(
            &entry,
            &instance,
            &window,
            WINDOW_WIDTH,
            WINDOW_HEIGHT,
        );
        let (debug_utils_loader, debug_messenger) =
            utility::debug::setup_debug_utils(VALIDATION.is_enable, &entry, &instance);

        let physical_device =
            utility::general::pick_physcial_device(&instance, &surface_stuff, &DEVICE_EXTENSIONS);
        let msaa_samples =
            utility::general::get_max_usable_sample_count(&instance, physical_device);
        let physical_device_memory_properties =
            unsafe { instance.get_physical_device_memory_properties(physical_device) };
        let (device, queue_family) = utility::general::create_logical_device(
            &instance,
            physical_device,
            &VALIDATION,
            &DEVICE_EXTENSIONS,
            &surface_stuff,
        );
        let surface_format =
            utility::general::create_surface_format(physical_device, &surface_stuff);

        let graphics_queue =
            unsafe { device.get_device_queue(queue_family.graphics_family.unwrap(), 0) };
        let present_queue =
            unsafe { device.get_device_queue(queue_family.present_family.unwrap(), 0) };

        let swapchain_stuff = utility::general::create_swapchain(
            &instance,
            &device,
            physical_device,
            &window,
            &surface_stuff,
            &queue_family,
        );
        let swapchain_imageviews = utility::general::create_image_views(
            &device,
            swapchain_stuff.swapchain_format,
            &swapchain_stuff.swapchain_images,
        );
        let render_pass = utility::general::create_render_pass(
            &instance,
            &device,
            physical_device,
            swapchain_stuff.swapchain_format,
            msaa_samples,
        );
        let ubo_layout = utility::general::create_descriptor_set_layout(&device);
        let (graphics_pipeline, pipeline_layout) = utility::general::create_graphics_pipeline(
            &device,
            render_pass,
            swapchain_stuff.swapchain_extent,
            ubo_layout,
            msaa_samples,
        );
        let command_pool = utility::general::create_command_pool(&device, &queue_family);
        let (color_image, color_image_view, color_image_memory) =
            utility::general::create_color_resources(
                &device,
                swapchain_stuff.swapchain_format,
                swapchain_stuff.swapchain_extent,
                &physical_device_memory_properties,
                msaa_samples,
            );
        let (depth_image, depth_image_view, depth_image_memory) =
            utility::general::create_depth_resources(
                &instance,
                &device,
                physical_device,
                command_pool,
                graphics_queue,
                swapchain_stuff.swapchain_extent,
                &physical_device_memory_properties,
                msaa_samples,
            );
        let swapchain_framebuffers = utility::general::create_framebuffers(
            &device,
            render_pass,
            &swapchain_imageviews,
            depth_image_view,
            color_image_view,
            swapchain_stuff.swapchain_extent,
        );
        let (vertices, indices) = load_model(&Path::new(MODEL_PATH));
        utility::general::check_mipmap_support(
            &instance,
            physical_device,
            vk::Format::R8G8B8A8_SRGB,
        );
        let (texture_image, texture_image_memory, mip_levels) =
            utility::general::create_texture_image(
                &device,
                command_pool,
                graphics_queue,
                &physical_device_memory_properties,
                &Path::new(TEXTURE_PATH),
            );
        let texture_image_view =
            utility::general::create_texture_image_view(&device, texture_image, mip_levels);
        let texture_sampler = utility::general::create_texture_sampler(&device, mip_levels);
        let (vertex_buffer, vertex_buffer_memory) = utility::general::create_vertex_buffer(
            &device,
            &physical_device_memory_properties,
            command_pool,
            graphics_queue,
            &vertices,
        );
        let (index_buffer, index_buffer_memory) = utility::general::create_index_buffer(
            &device,
            &physical_device_memory_properties,
            command_pool,
            graphics_queue,
            &indices,
        );
        let (uniform_buffers, uniform_buffers_memory) = utility::general::create_uniform_buffers(
            &device,
            &physical_device_memory_properties,
            swapchain_stuff.swapchain_images.len(),
        );
        let descriptor_pool = utility::general::create_descriptor_pool(
            &device,
            swapchain_stuff.swapchain_images.len(),
        );
        let descriptor_sets = utility::general::create_descriptor_sets(
            &device,
            descriptor_pool,
            ubo_layout,
            &uniform_buffers,
            texture_image_view,
            texture_sampler,
            swapchain_stuff.swapchain_images.len(),
        );
        let command_buffers = utility::general::create_command_buffers(
            &device,
            command_pool,
            graphics_pipeline,
            &swapchain_framebuffers,
            render_pass,
            swapchain_stuff.swapchain_extent,
            vertex_buffer,
            index_buffer,
            pipeline_layout,
            &descriptor_sets,
            indices.len() as u32,
        );
        let sync_objects = utility::general::create_sync_objects(&device, MAX_FRAMES_IN_FLIGHT);

        VulkanRenderer {
            window,

            _entry: entry,
            instance,
            surface: surface_stuff.surface,
            surface_loader: surface_stuff.surface_loader,
            surface_format,
            debug_utils_loader,
            debug_messenger,

            physical_device,
            memory_properties: physical_device_memory_properties,
            device,

            queue_family,
            graphics_queue,
            present_queue,

            swapchain_loader: swapchain_stuff.swapchain_loader,
            swapchain: swapchain_stuff.swapchain,
            swapchain_format: swapchain_stuff.swapchain_format,
            swapchain_images: swapchain_stuff.swapchain_images,
            swapchain_extent: swapchain_stuff.swapchain_extent,
            swapchain_imageviews,
            swapchain_framebuffers,

            pipeline_layout,
            ubo_layout,
            render_pass,
            graphics_pipeline,

            color_image,
            color_image_view,
            color_image_memory,

            depth_image,
            depth_image_view,
            depth_image_memory,

            msaa_samples,

            _mip_levels: mip_levels,
            texture_image,
            texture_image_view,
            texture_sampler,
            texture_image_memory,

            _vertices: vertices,
            indices,

            vertex_buffer,
            vertex_buffer_memory,
            index_buffer,
            index_buffer_memory,

            uniform_transform: UniformBufferObject {
                model: Matrix4::from_angle_z(Deg(90.0)),
                view: Matrix4::look_at_rh(
                    Point3::new(2.0, 2.0, 2.0),
                    Point3::new(0.0, 0.0, 0.0),
                    Vector3::new(0.0, 0.0, 1.0),
                ),
                proj: {
                    let mut proj = cgmath::perspective(
                        Deg(45.0),
                        swapchain_stuff.swapchain_extent.width as f32
                            / swapchain_stuff.swapchain_extent.height as f32,
                        0.1,
                        10.0,
                    );
                    proj[1][1] = proj[1][1] * -1.0;
                    proj
                },
            },
            uniform_buffers,
            uniform_buffers_memory,

            descriptor_pool,
            descriptor_sets,

            command_pool,
            command_buffers,

            image_available_semaphores: sync_objects.image_available_semaphores,
            render_finished_semaphores: sync_objects.render_finished_semaphores,
            in_flight_fences: sync_objects.inflight_fences,
            current_frame: 0,

            is_framebuffer_resized: false,
        }
    }
}

impl VulkanRenderer {
    fn update_uniform_buffer(&mut self, current_image: usize, delta_time: f32) {
        self.uniform_transform.model =
            Matrix4::from_axis_angle(Vector3::new(0.0, 0.0, 1.0), Deg(90.0) * delta_time)
                * self.uniform_transform.model;

        let ubos = [self.uniform_transform.clone()];

        let buffer_size = (std::mem::size_of::<UniformBufferObject>() * ubos.len()) as u64;

        unsafe {
            let data_ptr =
                self.device
                    .map_memory(
                        self.uniform_buffers_memory[current_image],
                        0,
                        buffer_size,
                        vk::MemoryMapFlags::empty(),
                    )
                    .expect("Failed to Map Memory") as *mut UniformBufferObject;

            data_ptr.copy_from_nonoverlapping(ubos.as_ptr(), ubos.len());

            self.device
                .unmap_memory(self.uniform_buffers_memory[current_image]);
        }
    }
}

impl Drop for VulkanRenderer {
    fn drop(&mut self) {
        unsafe {
            for i in 0..MAX_FRAMES_IN_FLIGHT {
                self.device
                    .destroy_semaphore(self.image_available_semaphores[i], None);
                self.device
                    .destroy_semaphore(self.render_finished_semaphores[i], None);
                self.device.destroy_fence(self.in_flight_fences[i], None);
            }

            self.cleanup_swapchain();

            self.device
                .destroy_descriptor_pool(self.descriptor_pool, None);

            for i in 0..self.uniform_buffers.len() {
                self.device.destroy_buffer(self.uniform_buffers[i], None);
                self.device
                    .free_memory(self.uniform_buffers_memory[i], None);
            }

            self.device.destroy_buffer(self.index_buffer, None);
            self.device.free_memory(self.index_buffer_memory, None);

            self.device.destroy_buffer(self.vertex_buffer, None);
            self.device.free_memory(self.vertex_buffer_memory, None);

            self.device.destroy_sampler(self.texture_sampler, None);
            self.device
                .destroy_image_view(self.texture_image_view, None);

            self.device.destroy_image(self.texture_image, None);
            self.device.free_memory(self.texture_image_memory, None);

            self.device
                .destroy_descriptor_set_layout(self.ubo_layout, None);

            self.device.destroy_command_pool(self.command_pool, None);

            self.device.destroy_device(None);
            self.surface_loader.destroy_surface(self.surface, None);

            if VALIDATION.is_enable {
                self.debug_utils_loader
                    .destroy_debug_utils_messenger(self.debug_messenger, None);
            }
            self.instance.destroy_instance(None);
        }
    }
}

impl VulkanApp for VulkanRenderer {
    fn draw_frame(&mut self, delta_time: f32) {
        let wait_fences = [self.in_flight_fences[self.current_frame]];

        unsafe {
            self.device
                .wait_for_fences(&wait_fences, true, std::u64::MAX)
                .expect("Failed to wait for Fence!");
        }

        let (image_index, _is_sub_optimal) = unsafe {
            let result = self.swapchain_loader.acquire_next_image(
                self.swapchain,
                std::u64::MAX,
                self.image_available_semaphores[self.current_frame],
                vk::Fence::null(),
            );
            match result {
                Ok(image_index) => image_index,
                Err(vk_result) => match vk_result {
                    vk::Result::ERROR_OUT_OF_DATE_KHR => {
                        self.recreate_swapchain();
                        return;
                    }
                    _ => panic!("Failed to acquire Swap Chain Image!"),
                },
            }
        };

        self.update_uniform_buffer(image_index as usize, delta_time);

        let wait_semaphores = [self.image_available_semaphores[self.current_frame]];
        let wait_stages = [vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT];
        let signal_semaphores = [self.render_finished_semaphores[self.current_frame]];

        let submit_infos = [vk::SubmitInfo {
            s_type: vk::StructureType::SUBMIT_INFO,
            p_next: ptr::null(),
            wait_semaphore_count: wait_semaphores.len() as u32,
            p_wait_semaphores: wait_semaphores.as_ptr(),
            p_wait_dst_stage_mask: wait_stages.as_ptr(),
            command_buffer_count: 1,
            p_command_buffers: &self.command_buffers[image_index as usize],
            signal_semaphore_count: signal_semaphores.len() as u32,
            p_signal_semaphores: signal_semaphores.as_ptr(),
        }];

        unsafe {
            self.device
                .reset_fences(&wait_fences)
                .expect("Failed to reset Fence!");

            self.device
                .queue_submit(
                    self.graphics_queue,
                    &submit_infos,
                    self.in_flight_fences[self.current_frame],
                )
                .expect("Failed to execute queue submit.");
        }

        let swapchains = [self.swapchain];

        let present_info = vk::PresentInfoKHR {
            s_type: vk::StructureType::PRESENT_INFO_KHR,
            p_next: ptr::null(),
            wait_semaphore_count: 1,
            p_wait_semaphores: signal_semaphores.as_ptr(),
            swapchain_count: 1,
            p_swapchains: swapchains.as_ptr(),
            p_image_indices: &image_index,
            p_results: ptr::null_mut(),
        };

        let result = unsafe {
            self.swapchain_loader
                .queue_present(self.present_queue, &present_info)
        };

        let is_resized = match result {
            Ok(_) => self.is_framebuffer_resized,
            Err(vk_result) => match vk_result {
                vk::Result::ERROR_OUT_OF_DATE_KHR | vk::Result::SUBOPTIMAL_KHR => true,
                _ => panic!("Failed to execute queue present."),
            },
        };
        if is_resized {
            self.is_framebuffer_resized = false;
            self.recreate_swapchain();
        }

        self.current_frame = (self.current_frame + 1) % MAX_FRAMES_IN_FLIGHT;
    }

    fn recreate_swapchain(&mut self) {
        let surface_stuff = SurfaceStuff {
            surface_loader: self.surface_loader.clone(),
            surface: self.surface,
            screen_width: WINDOW_WIDTH,
            screen_height: WINDOW_HEIGHT,
        };

        self.wait_device_idle();

        self.cleanup_swapchain();

        let swapchain_stuff = utility::general::create_swapchain(
            &self.instance,
            &self.device,
            self.physical_device,
            &self.window,
            &surface_stuff,
            &self.queue_family,
        );
        self.swapchain_loader = swapchain_stuff.swapchain_loader;
        self.swapchain = swapchain_stuff.swapchain;
        self.swapchain_images = swapchain_stuff.swapchain_images;
        self.swapchain_format = swapchain_stuff.swapchain_format;
        self.swapchain_extent = swapchain_stuff.swapchain_extent;

        self.swapchain_imageviews = utility::general::create_image_views(
            &self.device,
            self.swapchain_format,
            &self.swapchain_images,
        );
        self.render_pass = utility::general::create_render_pass(
            &self.instance,
            &self.device,
            self.physical_device,
            self.swapchain_format,
            self.msaa_samples,
        );
        let (graphics_pipeline, pipeline_layout) = utility::general::create_graphics_pipeline(
            &self.device,
            self.render_pass,
            swapchain_stuff.swapchain_extent,
            self.ubo_layout,
            self.msaa_samples,
        );
        self.graphics_pipeline = graphics_pipeline;
        self.pipeline_layout = pipeline_layout;

        let color_resources = utility::general::create_color_resources(
            &self.device,
            self.swapchain_format,
            self.swapchain_extent,
            &self.memory_properties,
            self.msaa_samples,
        );
        self.color_image = color_resources.0;
        self.color_image_view = color_resources.1;
        self.color_image_memory = color_resources.2;

        let depth_resources = utility::general::create_depth_resources(
            &self.instance,
            &self.device,
            self.physical_device,
            self.command_pool,
            self.graphics_queue,
            self.swapchain_extent,
            &self.memory_properties,
            self.msaa_samples,
        );
        self.depth_image = depth_resources.0;
        self.depth_image_view = depth_resources.1;
        self.depth_image_memory = depth_resources.2;

        self.swapchain_framebuffers = utility::general::create_framebuffers(
            &self.device,
            self.render_pass,
            &self.swapchain_imageviews,
            self.depth_image_view,
            self.color_image_view,
            self.swapchain_extent,
        );
        self.command_buffers = utility::general::create_command_buffers(
            &self.device,
            self.command_pool,
            self.graphics_pipeline,
            &self.swapchain_framebuffers,
            self.render_pass,
            self.swapchain_extent,
            self.vertex_buffer,
            self.index_buffer,
            self.pipeline_layout,
            &self.descriptor_sets,
            self.indices.len() as u32,
        );
    }

    fn cleanup_swapchain(&self) {
        unsafe {
            self.device.destroy_image(self.depth_image, None);
            self.device.destroy_image_view(self.depth_image_view, None);
            self.device.free_memory(self.depth_image_memory, None);

            self.device.destroy_image(self.color_image, None);
            self.device.destroy_image_view(self.color_image_view, None);
            self.device.free_memory(self.color_image_memory, None);

            self.device
                .free_command_buffers(self.command_pool, &self.command_buffers);
            for &framebuffer in self.swapchain_framebuffers.iter() {
                self.device.destroy_framebuffer(framebuffer, None);
            }
            self.device.destroy_pipeline(self.graphics_pipeline, None);
            self.device
                .destroy_pipeline_layout(self.pipeline_layout, None);
            self.device.destroy_render_pass(self.render_pass, None);
            for &image_view in self.swapchain_imageviews.iter() {
                self.device.destroy_image_view(image_view, None);
            }
            self.swapchain_loader
                .destroy_swapchain(self.swapchain, None);
        }
    }

    fn wait_device_idle(&self) {
        unsafe {
            self.device
                .device_wait_idle()
                .expect("Failed to wait device idle!")
        };
    }

    fn resize_framebuffer(&mut self) {
        self.is_framebuffer_resized = true;
    }

    fn window_ref(&self) -> &winit::window::Window {
        &self.window
    }
}

#[derive(Clone)]
struct RayTracingApp {
    base: Rc<VulkanRenderer>,
    ray_tracing: Rc<nv::RayTracing>,
    properties: vk::PhysicalDeviceRayTracingPropertiesNV,
    top_as_memory: vk::DeviceMemory,
    top_as: vk::AccelerationStructureNV,
    bottom_as_memory: vk::DeviceMemory,
    bottom_as: vk::AccelerationStructureNV,
    descriptor_set_layout: vk::DescriptorSetLayout,
    pipeline_layout: vk::PipelineLayout,
    pipeline: vk::Pipeline,
    shader_binding_table: Option<BufferResource>,
    color0_buffer: Option<BufferResource>,
    color1_buffer: Option<BufferResource>,
    color2_buffer: Option<BufferResource>,
    descriptor_pool: vk::DescriptorPool,
    descriptor_set: vk::DescriptorSet,
    offscreen_target: ImageResource,
    rgen_shader_module: vk::ShaderModule,
    chit_shader_module: vk::ShaderModule,
    miss_shader_module: vk::ShaderModule,
    lib_shader_module: vk::ShaderModule,
}
impl RayTracingApp {
    fn new(
        base: Rc<VulkanRenderer>,
        ray_tracing: Rc<nv::RayTracing>,
        properties: vk::PhysicalDeviceRayTracingPropertiesNV,
    ) -> Self {
        RayTracingApp {
            base: base.clone(),
            ray_tracing,
            properties,
            top_as_memory: vk::DeviceMemory::null(),
            top_as: vk::AccelerationStructureNV::null(),
            bottom_as_memory: vk::DeviceMemory::null(),
            bottom_as: vk::AccelerationStructureNV::null(),
            descriptor_set_layout: vk::DescriptorSetLayout::null(),
            pipeline_layout: vk::PipelineLayout::null(),
            pipeline: vk::Pipeline::null(),
            shader_binding_table: None,
            color0_buffer: None,
            color1_buffer: None,
            color2_buffer: None,
            descriptor_pool: vk::DescriptorPool::null(),
            descriptor_set: vk::DescriptorSet::null(),
            offscreen_target: ImageResource::new(base),
            rgen_shader_module: vk::ShaderModule::null(),
            chit_shader_module: vk::ShaderModule::null(),
            miss_shader_module: vk::ShaderModule::null(),
            lib_shader_module: vk::ShaderModule::null(),
        }
    }

    fn initialize(&mut self) {
        self.create_offscreen_target();
        self.create_acceleration_structures();
        self.create_bindless_uniform_buffers();
        self.create_pipeline();
        self.create_shader_binding_table();
        self.create_descriptor_set();
    }

    fn create_offscreen_target(&mut self) {
        self.offscreen_target.create_image(
            vk::ImageType::TYPE_2D,
            self.base.surface_format.format,
            vk::Extent3D::builder()
                .width(self.base.swapchain_extent.width)
                .height(self.base.swapchain_extent.height)
                .depth(1)
                .build(),
            vk::ImageTiling::OPTIMAL,
            vk::ImageUsageFlags::STORAGE | vk::ImageUsageFlags::TRANSFER_SRC,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        );

        self.offscreen_target.create_view(
            vk::ImageViewType::TYPE_2D,
            self.base.surface_format.format,
            vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                base_mip_level: 0,
                level_count: 1,
                base_array_layer: 0,
                layer_count: 1,
            },
        );
    }

    fn create_acceleration_structures(&mut self) {
        unsafe {
            // Create geometry

            let vertices = [
                VertexRt {
                    pos: [-0.5, -0.5, 0.0],
                },
                VertexRt {
                    pos: [0.0, 0.5, 0.0],
                },
                VertexRt {
                    pos: [0.5, -0.5, 0.0],
                },
            ];

            let vertex_count = vertices.len();
            let vertex_stride = std::mem::size_of::<VertexRt>();

            let vertex_buffer_size = vertex_stride * vertex_count;
            let mut vertex_buffer = BufferResource::new(
                vertex_buffer_size as u64,
                vk::BufferUsageFlags::VERTEX_BUFFER,
                vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
                self.base.clone(),
            );
            vertex_buffer.store(&vertices);

            let indices = [0u16, 1, 2];
            let index_count = indices.len();
            let index_buffer_size = std::mem::size_of::<u16>() * index_count;
            let mut index_buffer = BufferResource::new(
                index_buffer_size as u64,
                vk::BufferUsageFlags::INDEX_BUFFER,
                vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
                self.base.clone(),
            );
            index_buffer.store(&indices);

            let geometry = vec![vk::GeometryNV::builder()
                .geometry_type(vk::GeometryTypeNV::TRIANGLES)
                .geometry(
                    vk::GeometryDataNV::builder()
                        .triangles(
                            vk::GeometryTrianglesNV::builder()
                                .vertex_data(vertex_buffer.buffer)
                                .vertex_offset(0)
                                .vertex_count(vertex_count as u32)
                                .vertex_stride(vertex_stride as u64)
                                .vertex_format(vk::Format::R32G32B32_SFLOAT)
                                .index_data(index_buffer.buffer)
                                .index_offset(0)
                                .index_count(index_count as u32)
                                .index_type(vk::IndexType::UINT16)
                                .build(),
                        )
                        .build(),
                )
                .flags(vk::GeometryFlagsNV::OPAQUE)
                .build()];

            println!("Geometry: {:?}", geometry.len());
            // Create bottom-level acceleration structure

            let accel_info = vk::AccelerationStructureCreateInfoNV::builder()
                .compacted_size(0)
                .info(
                    vk::AccelerationStructureInfoNV::builder()
                        .ty(vk::AccelerationStructureTypeNV::BOTTOM_LEVEL)
                        .geometries(&geometry)
                        .flags(vk::BuildAccelerationStructureFlagsNV::PREFER_FAST_TRACE)
                        .build(),
                )
                .build();

            self.bottom_as = self
                .ray_tracing
                .create_acceleration_structure(&accel_info, None)
                .unwrap();

            let memory_requirements = self
                .ray_tracing
                .get_acceleration_structure_memory_requirements(
                    &vk::AccelerationStructureMemoryRequirementsInfoNV::builder()
                        .acceleration_structure(self.bottom_as)
                        .ty(vk::AccelerationStructureMemoryRequirementsTypeNV::OBJECT)
                        .build(),
                );

            self.bottom_as_memory = self
                .base
                .device
                .allocate_memory(
                    &vk::MemoryAllocateInfo::builder()
                        .allocation_size(memory_requirements.memory_requirements.size)
                        .memory_type_index(
                            utility::general::find_memorytype_index(
                                &memory_requirements.memory_requirements,
                                &self.base.memory_properties,
                                vk::MemoryPropertyFlags::DEVICE_LOCAL,
                            )
                            .unwrap(),
                        )
                        .build(),
                    None,
                )
                .unwrap();

            self.ray_tracing
                .bind_acceleration_structure_memory(&[
                    vk::BindAccelerationStructureMemoryInfoNV::builder()
                        .acceleration_structure(self.bottom_as)
                        .memory(self.bottom_as_memory)
                        .build(),
                ])
                .unwrap();

            // let bottom_as_info = vk::AccelerationStructureInfoNV {
            //     s_type: vk::StructureType::ACCELERATION_STRUCTURE_INFO_NV,
            //     p_next: ptr::null(),
            //     ty: vk::AccelerationStructureTypeNV::BOTTOM_LEVEL,
            //     geometry_count: geometry.len() as u32,
            //     p_geometries: geometry.as_ptr(),
            //     flags: vk::BuildAccelerationStructureFlagsNV::PREFER_FAST_TRACE,
            //     ..Default::default()
            // };

            // let bottom_as_create_info = vk::AccelerationStructureCreateInfoNV {
            //     s_type: vk::StructureType::ACCELERATION_STRUCTURE_CREATE_INFO_NV,
            //     p_next: ptr::null(),
            //     compacted_size: 0,
            //     info: bottom_as_info,
            // };

            // self.bottom_as = self
            //     .ray_tracing
            //     .create_acceleration_structure(&bottom_as_create_info, None)
            //     .expect("Failed to create bottom AS.");

            // let bottom_as_memory_requirements_info =
            //     vk::AccelerationStructureMemoryRequirementsInfoNV {
            //         s_type: vk::StructureType::ACCELERATION_STRUCTURE_MEMORY_REQUIREMENTS_INFO_NV,
            //         p_next: ptr::null(),
            //         acceleration_structure: self.bottom_as,
            //         ty: vk::AccelerationStructureMemoryRequirementsTypeNV::OBJECT,
            //     };

            // let bottom_as_memory_requirements = self
            //     .ray_tracing
            //     .get_acceleration_structure_memory_requirements(
            //         &bottom_as_memory_requirements_info,
            //     );

            // let bottom_as_memory_allocate_info = vk::MemoryAllocateInfo {
            //     s_type: vk::StructureType::MEMORY_ALLOCATE_INFO,
            //     p_next: ptr::null(),
            //     allocation_size: bottom_as_memory_requirements.memory_requirements.size,
            //     memory_type_index: utility::general::find_memorytype_index(
            //         &bottom_as_memory_requirements.memory_requirements,
            //         &self.base.memory_properties,
            //         vk::MemoryPropertyFlags::DEVICE_LOCAL,
            //     )
            //     .expect("Failed to find suitable AS memory type."),
            // };

            // self.bottom_as_memory = self
            //     .base
            //     .device
            //     .allocate_memory(&bottom_as_memory_allocate_info, None)
            //     .expect("Failed to allocate AS memory.");

            // let bind_bottom_as_memory_infos = [vk::BindAccelerationStructureMemoryInfoNV {
            //     s_type: vk::StructureType::BIND_ACCELERATION_STRUCTURE_MEMORY_INFO_NV,
            //     p_next: ptr::null(),
            //     acceleration_structure: self.bottom_as,
            //     memory: self.bottom_as_memory,
            //     ..Default::default()
            // }];

            // self.ray_tracing
            //     .bind_acceleration_structure_memory(&bind_bottom_as_memory_infos)
            //     .expect("Failed to bind AS memory.");

            // Create instance buffer

            let bottom_as_handle = self
                .ray_tracing
                .get_acceleration_structure_handle(self.bottom_as)
                .expect("Failed to get AS handle.");

            let transform_0: [f32; 12] =
                [1.0, 0.0, 0.0, -1.5, 0.0, 1.0, 0.0, 1.1, 0.0, 0.0, 1.0, 0.0];

            let transform_1: [f32; 12] =
                [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, -1.1, 0.0, 0.0, 1.0, 0.0];

            let transform_2: [f32; 12] =
                [1.0, 0.0, 0.0, 1.5, 0.0, 1.0, 0.0, 1.1, 0.0, 0.0, 1.0, 0.0];

            let instances = vec![
                GeometryInstance::new(
                    transform_0,
                    0,
                    0xff,
                    0,
                    vk::GeometryInstanceFlagsNV::TRIANGLE_CULL_DISABLE_NV,
                    bottom_as_handle,
                ),
                GeometryInstance::new(
                    transform_1,
                    1,
                    0xff,
                    0,
                    vk::GeometryInstanceFlagsNV::TRIANGLE_CULL_DISABLE_NV,
                    bottom_as_handle,
                ),
                GeometryInstance::new(
                    transform_2,
                    2,
                    0xff,
                    0,
                    vk::GeometryInstanceFlagsNV::TRIANGLE_CULL_DISABLE_NV,
                    bottom_as_handle,
                ),
            ];

            let instance_buffer_size = std::mem::size_of::<GeometryInstance>() * instances.len();
            let mut instance_buffer = BufferResource::new(
                instance_buffer_size as u64,
                vk::BufferUsageFlags::RAY_TRACING_NV,
                vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
                self.base.clone(),
            );
            instance_buffer.store(&instances);

            // Create top-level acceleration structure

            let accel_info = vk::AccelerationStructureCreateInfoNV::builder()
                .compacted_size(0)
                .info(
                    vk::AccelerationStructureInfoNV::builder()
                        .ty(vk::AccelerationStructureTypeNV::TOP_LEVEL)
                        .instance_count(instances.len() as u32)
                        .build(),
                )
                .build();

            self.top_as = self
                .ray_tracing
                .create_acceleration_structure(&accel_info, None)
                .unwrap();

            let memory_requirements = self
                .ray_tracing
                .get_acceleration_structure_memory_requirements(
                    &vk::AccelerationStructureMemoryRequirementsInfoNV::builder()
                        .acceleration_structure(self.top_as)
                        .ty(vk::AccelerationStructureMemoryRequirementsTypeNV::OBJECT)
                        .build(),
                );

            self.top_as_memory = self
                .base
                .device
                .allocate_memory(
                    &vk::MemoryAllocateInfo::builder()
                        .allocation_size(memory_requirements.memory_requirements.size)
                        .memory_type_index(
                            utility::general::find_memorytype_index(
                                &memory_requirements.memory_requirements,
                                &self.base.memory_properties,
                                vk::MemoryPropertyFlags::DEVICE_LOCAL,
                            )
                            .unwrap(),
                        )
                        .build(),
                    None,
                )
                .unwrap();

            self.ray_tracing
                .bind_acceleration_structure_memory(&[
                    vk::BindAccelerationStructureMemoryInfoNV::builder()
                        .acceleration_structure(self.top_as)
                        .memory(self.top_as_memory)
                        .build(),
                ])
                .unwrap();

            // let top_as_create_info = vk::AccelerationStructureCreateInfoNV {
            //     s_type: vk::StructureType::ACCELERATION_STRUCTURE_CREATE_INFO_NV,
            //     p_next: ptr::null(),
            //     compacted_size: 0,
            //     info: vk::AccelerationStructureInfoNV::builder()
            //         .ty(vk::AccelerationStructureTypeNV::TOP_LEVEL)
            //         .instance_count(instances.len() as u32)
            //         .build(),
            // };

            // self.top_as = self
            //     .ray_tracing
            //     .create_acceleration_structure(&top_as_create_info, None)
            //     .expect("Failed to create top AS.");

            // let top_as_memory_requirements_info =
            //     vk::AccelerationStructureMemoryRequirementsInfoNV {
            //         s_type: vk::StructureType::ACCELERATION_STRUCTURE_MEMORY_REQUIREMENTS_INFO_NV,
            //         p_next: ptr::null(),
            //         ty: vk::AccelerationStructureMemoryRequirementsTypeNV::OBJECT,
            //         acceleration_structure: self.top_as,
            //     };

            // let top_as_memory_requirements = self
            //     .ray_tracing
            //     .get_acceleration_structure_memory_requirements(&top_as_memory_requirements_info);

            // let top_as_memory_allocate_info = vk::MemoryAllocateInfo {
            //     s_type: vk::StructureType::MEMORY_ALLOCATE_INFO,
            //     p_next: ptr::null(),
            //     allocation_size: top_as_memory_requirements.memory_requirements.size,
            //     memory_type_index: utility::general::find_memorytype_index(
            //         &top_as_memory_requirements.memory_requirements,
            //         &self.base.memory_properties,
            //         vk::MemoryPropertyFlags::DEVICE_LOCAL,
            //     )
            //     .expect("Failed to find suitable AS memory type."),
            // };

            // self.top_as_memory = self
            //     .base
            //     .device
            //     .allocate_memory(&top_as_memory_allocate_info, None)
            //     .expect("Failed to allocate AS memory");

            // let bind_top_as_memory_infos = [vk::BindAccelerationStructureMemoryInfoNV {
            //     s_type: vk::StructureType::BIND_ACCELERATION_STRUCTURE_MEMORY_INFO_NV,
            //     acceleration_structure: self.top_as,
            //     memory: self.top_as_memory,
            //     ..Default::default()
            // }];

            // self.ray_tracing
            //     .bind_acceleration_structure_memory(&bind_top_as_memory_infos)
            //     .expect("Failed to bind AS memory");

            // Build accleration structures

            let bottom_as_size = {
                let requirements = self
                    .ray_tracing
                    .get_acceleration_structure_memory_requirements(
                        &vk::AccelerationStructureMemoryRequirementsInfoNV::builder()
                            .acceleration_structure(self.bottom_as)
                            .ty(vk::AccelerationStructureMemoryRequirementsTypeNV::BUILD_SCRATCH)
                            .build(),
                    );
                requirements.memory_requirements.size
            };

            let top_as_size = {
                let requirements = self
                    .ray_tracing
                    .get_acceleration_structure_memory_requirements(
                        &vk::AccelerationStructureMemoryRequirementsInfoNV::builder()
                            .acceleration_structure(self.top_as)
                            .ty(vk::AccelerationStructureMemoryRequirementsTypeNV::BUILD_SCRATCH)
                            .build(),
                    );
                requirements.memory_requirements.size
            };

            let scratch_buffer_size = std::cmp::max(bottom_as_size, top_as_size);
            let scratch_buffer = BufferResource::new(
                scratch_buffer_size,
                vk::BufferUsageFlags::RAY_TRACING_NV,
                vk::MemoryPropertyFlags::DEVICE_LOCAL,
                self.base.clone(),
            );

            let allocate_info = vk::CommandBufferAllocateInfo::builder()
                .command_buffer_count(1)
                .command_pool(self.base.command_pool)
                .level(vk::CommandBufferLevel::PRIMARY)
                .build();

            let command_buffers = self
                .base
                .device
                .allocate_command_buffers(&allocate_info)
                .unwrap();
            let build_command_buffer = command_buffers[0];

            self.base
                .device
                .begin_command_buffer(
                    build_command_buffer,
                    &vk::CommandBufferBeginInfo::builder()
                        .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT)
                        .build(),
                )
                .unwrap();

            let memory_barrier = vk::MemoryBarrier::builder()
                .src_access_mask(
                    vk::AccessFlags::ACCELERATION_STRUCTURE_WRITE_NV
                        | vk::AccessFlags::ACCELERATION_STRUCTURE_READ_NV,
                )
                .dst_access_mask(
                    vk::AccessFlags::ACCELERATION_STRUCTURE_WRITE_NV
                        | vk::AccessFlags::ACCELERATION_STRUCTURE_READ_NV,
                )
                .build();

            self.ray_tracing.cmd_build_acceleration_structure(
                build_command_buffer,
                &vk::AccelerationStructureInfoNV::builder()
                    .ty(vk::AccelerationStructureTypeNV::BOTTOM_LEVEL)
                    .geometries(&geometry)
                    .build(),
                vk::Buffer::null(),
                0,
                false,
                self.bottom_as,
                vk::AccelerationStructureNV::null(),
                scratch_buffer.buffer,
                0,
            );

            self.base.device.cmd_pipeline_barrier(
                build_command_buffer,
                vk::PipelineStageFlags::ACCELERATION_STRUCTURE_BUILD_NV,
                vk::PipelineStageFlags::ACCELERATION_STRUCTURE_BUILD_NV,
                vk::DependencyFlags::empty(),
                &[memory_barrier],
                &[],
                &[],
            );

            self.ray_tracing.cmd_build_acceleration_structure(
                build_command_buffer,
                &vk::AccelerationStructureInfoNV::builder()
                    .ty(vk::AccelerationStructureTypeNV::TOP_LEVEL)
                    .instance_count(instances.len() as u32)
                    .build(),
                instance_buffer.buffer,
                0,
                false,
                self.top_as,
                vk::AccelerationStructureNV::null(),
                scratch_buffer.buffer,
                0,
            );

            self.base.device.cmd_pipeline_barrier(
                build_command_buffer,
                vk::PipelineStageFlags::ACCELERATION_STRUCTURE_BUILD_NV,
                vk::PipelineStageFlags::ACCELERATION_STRUCTURE_BUILD_NV,
                vk::DependencyFlags::empty(),
                &[memory_barrier],
                &[],
                &[],
            );

            self.base
                .device
                .end_command_buffer(build_command_buffer)
                .unwrap();

            self.base
                .device
                .queue_submit(
                    self.base.present_queue,
                    &[vk::SubmitInfo::builder()
                        .command_buffers(&[build_command_buffer])
                        .build()],
                    vk::Fence::null(),
                )
                .expect("queue submit failed.");

            match self.base.device.queue_wait_idle(self.base.present_queue) {
                Ok(_) => println!("Successfully built acceleration structures"),
                Err(err) => {
                    println!("Failed to build acceleration structures: {:?}", err);
                    panic!("GPU ERROR");
                }
            }

            // let bottom_as_size = bottom_as_memory_requirements.memory_requirements.size;

            // let top_as_size = top_as_memory_requirements.memory_requirements.size;

            // let scratch_buffer_size = std::cmp::max(bottom_as_size, top_as_size);
            // let scratch_buffer = BufferResource::new(
            //     scratch_buffer_size,
            //     vk::BufferUsageFlags::RAY_TRACING_NV,
            //     vk::MemoryPropertyFlags::DEVICE_LOCAL,
            //     self.base.clone(),
            // );

            // let allocate_info = vk::CommandBufferAllocateInfo {
            //     s_type: vk::StructureType::COMMAND_BUFFER_ALLOCATE_INFO,
            //     p_next: ptr::null(),
            //     command_buffer_count: 1,
            //     command_pool: self.base.command_pool,
            //     level: vk::CommandBufferLevel::PRIMARY,
            // };

            // let command_buffers = self
            //     .base
            //     .device
            //     .allocate_command_buffers(&allocate_info)
            //     .expect("Failed to allocate command buffer.");

            // let build_command_buffer = command_buffers[0];

            // let command_buffer_begin_info = vk::CommandBufferBeginInfo {
            //     s_type: vk::StructureType::COMMAND_BUFFER_BEGIN_INFO,
            //     p_next: ptr::null(),
            //     flags: vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT,
            //     ..Default::default()
            // };

            // self.base
            //     .device
            //     .begin_command_buffer(build_command_buffer, &command_buffer_begin_info)
            //     .expect("Failed to begin command buffer.");

            // let memory_barrier = vk::MemoryBarrier {
            //     s_type: vk::StructureType::MEMORY_BARRIER,
            //     p_next: ptr::null(),
            //     src_access_mask: vk::AccessFlags::ACCELERATION_STRUCTURE_WRITE_NV
            //         | vk::AccessFlags::ACCELERATION_STRUCTURE_READ_NV,
            //     dst_access_mask: vk::AccessFlags::ACCELERATION_STRUCTURE_WRITE_NV
            //         | vk::AccessFlags::ACCELERATION_STRUCTURE_READ_NV,
            // };

            // let bottom_as_info = vk::AccelerationStructureInfoNV {
            //     s_type: vk::StructureType::ACCELERATION_STRUCTURE_INFO_NV,
            //     p_next: ptr::null(),
            //     ty: vk::AccelerationStructureTypeNV::BOTTOM_LEVEL,
            //     geometry_count: geometry.len() as u32,
            //     p_geometries: geometry.as_ptr(),
            //     ..Default::default()
            // };

            // self.ray_tracing.cmd_build_acceleration_structure(
            //     build_command_buffer,
            //     &bottom_as_info,
            //     vk::Buffer::null(),
            //     0,
            //     false,
            //     self.bottom_as,
            //     vk::AccelerationStructureNV::null(),
            //     scratch_buffer.buffer,
            //     0,
            // );

            // self.base.device.cmd_pipeline_barrier(
            //     build_command_buffer,
            //     vk::PipelineStageFlags::ACCELERATION_STRUCTURE_BUILD_NV,
            //     vk::PipelineStageFlags::ACCELERATION_STRUCTURE_BUILD_NV,
            //     vk::DependencyFlags::empty(),
            //     &[memory_barrier],
            //     &[],
            //     &[],
            // );

            // let top_as_info = vk::AccelerationStructureInfoNV {
            //     s_type: vk::StructureType::ACCELERATION_STRUCTURE_INFO_NV,
            //     p_next: ptr::null(),
            //     ty: vk::AccelerationStructureTypeNV::TOP_LEVEL,
            //     instance_count: instances.len() as u32,
            //     ..Default::default()
            // };

            // self.ray_tracing.cmd_build_acceleration_structure(
            //     build_command_buffer,
            //     &top_as_info,
            //     instance_buffer.buffer,
            //     0,
            //     false,
            //     self.top_as,
            //     vk::AccelerationStructureNV::null(),
            //     scratch_buffer.buffer,
            //     0,
            // );

            // self.base.device.cmd_pipeline_barrier(
            //     build_command_buffer,
            //     vk::PipelineStageFlags::ACCELERATION_STRUCTURE_BUILD_NV,
            //     vk::PipelineStageFlags::ACCELERATION_STRUCTURE_BUILD_NV,
            //     vk::DependencyFlags::empty(),
            //     &[memory_barrier],
            //     &[],
            //     &[],
            // );

            // self.base
            //     .device
            //     .end_command_buffer(build_command_buffer)
            //     .expect("Failed to end command buffer.");

            // let submit_info = vk::SubmitInfo {
            //     s_type: vk::StructureType::SUBMIT_INFO,
            //     p_next: ptr::null(),
            //     command_buffer_count: 1,
            //     p_command_buffers: [build_command_buffer].as_ptr(),
            //     ..Default::default()
            // };

            // self.base
            //     .device
            //     .queue_submit(self.base.present_queue, &[submit_info], vk::Fence::null())
            //     .expect("Failed to submit queue.");

            // match self.base.device.queue_wait_idle(self.base.present_queue) {
            //     Ok(_) => println!("Successfully built acceleration structures"),
            //     Err(err) => {
            //         println!("Failed to build acceleration structures: {:?}", err);
            //         panic!("GPU ERROR");
            //     }
            // }

            self.base
                .device
                .free_command_buffers(self.base.command_pool, &[build_command_buffer]);
        }
    }

    fn create_bindless_uniform_buffers(&mut self) {
        let color0: [f32; 3] = [1.0, 0.0, 0.0];
        let color1: [f32; 3] = [0.0, 1.0, 0.0];
        let color2: [f32; 3] = [0.0, 0.0, 1.0];

        let buffer_size = (std::mem::size_of::<f32>() * 3) as vk::DeviceSize;

        let mut color0_buffer = BufferResource::new(
            buffer_size,
            vk::BufferUsageFlags::UNIFORM_BUFFER,
            vk::MemoryPropertyFlags::HOST_VISIBLE,
            self.base.clone(),
        );
        color0_buffer.store(&color0);
        self.color0_buffer = Some(color0_buffer);

        let mut color1_buffer = BufferResource::new(
            buffer_size,
            vk::BufferUsageFlags::UNIFORM_BUFFER,
            vk::MemoryPropertyFlags::HOST_VISIBLE,
            self.base.clone(),
        );
        color1_buffer.store(&color1);
        self.color1_buffer = Some(color1_buffer);

        let mut color2_buffer = BufferResource::new(
            buffer_size,
            vk::BufferUsageFlags::UNIFORM_BUFFER,
            vk::MemoryPropertyFlags::HOST_VISIBLE,
            self.base.clone(),
        );
        color2_buffer.store(&color2);
        self.color2_buffer = Some(color2_buffer);
    }

    fn create_pipeline(&mut self) {
        let binding_flags = [
            vk::DescriptorBindingFlagsEXT::empty(),
            vk::DescriptorBindingFlagsEXT::empty(),
            vk::DescriptorBindingFlagsEXT::VARIABLE_DESCRIPTOR_COUNT,
        ];

        let mut descriptor_set_layout_binding_create_info =
            vk::DescriptorSetLayoutBindingFlagsCreateInfoEXT {
                s_type: vk::StructureType::DESCRIPTOR_SET_LAYOUT_BINDING_FLAGS_CREATE_INFO_EXT,
                p_next: ptr::null(),
                p_binding_flags: binding_flags.as_ptr(),
                ..Default::default()
            };

        unsafe {
            let descriptor_set_layout_bindings = [
                vk::DescriptorSetLayoutBinding {
                    descriptor_count: 1,
                    descriptor_type: vk::DescriptorType::ACCELERATION_STRUCTURE_NV,
                    stage_flags: vk::ShaderStageFlags::RAYGEN_NV,
                    binding: 0,
                    ..Default::default()
                },
                vk::DescriptorSetLayoutBinding {
                    descriptor_count: 1,
                    descriptor_type: vk::DescriptorType::STORAGE_IMAGE,
                    stage_flags: vk::ShaderStageFlags::RAYGEN_NV,
                    binding: 1,
                    ..Default::default()
                },
                vk::DescriptorSetLayoutBinding {
                    descriptor_count: 1,
                    descriptor_type: vk::DescriptorType::UNIFORM_BUFFER,
                    stage_flags: vk::ShaderStageFlags::CLOSEST_HIT_NV,
                    binding: 0,
                    ..Default::default()
                },
            ];

            let descriptor_set_layout_create_info = vk::DescriptorSetLayoutCreateInfo::builder()
                .bindings(&descriptor_set_layout_bindings)
                .push_next(&mut descriptor_set_layout_binding_create_info)
                .build();

            self.descriptor_set_layout = self
                .base
                .device
                .create_descriptor_set_layout(&descriptor_set_layout_create_info, None)
                .expect("Failed to create descriptor set layout.");

            let use_lib = false;
            let use_hlsl = true;
            let use_bindless = true;
            if use_lib && use_hlsl {
                let lib_path = Path::new("shaders/compiled/triangle.hlsl_lib.spv");
                let mut lib_file = File::open(lib_path)
                    .expect(&format!("Could not open lib file: {:?}", lib_path));

                let lib_code = read_spv(&mut lib_file)
                    .expect(&format!("Could not load lib file: {:?}", lib_path));

                let lib_shader_info = vk::ShaderModuleCreateInfo::builder().code(&lib_code);
                self.lib_shader_module = self
                    .base
                    .device
                    .create_shader_module(&lib_shader_info, None)
                    .expect("Failed to create Library shader module.");
            } else {
                let lang = if use_hlsl { "hlsl_" } else { "glsl_" };

                let variant = if use_bindless { "bindless_" } else { "" };

                let rgen_path = format!("shaders/compiled/triangle.{}rgen.spv", lang);
                let rgen_path = Path::new(&rgen_path);

                let rchit_path = format!("shaders/compiled/triangle.{}{}rchit.spv", lang, variant);
                let rchit_path = Path::new(&rchit_path);

                let rmiss_path = format!("shaders/compiled/triangle.{}rmiss.spv", lang);
                let rmiss_path = Path::new(&rmiss_path);

                let mut rgen_file = File::open(&rgen_path)
                    .expect(&format!("Failed to open rgen file: {:?}", rgen_path));

                let mut rchit_file = File::open(&rchit_path)
                    .expect(&format!("Failed to open rchit file: {:?}", rchit_path));

                let mut rmiss_file = File::open(&rmiss_path)
                    .expect(&format!("Failed to open rmiss file: {:?}", rmiss_path));

                let rgen_code = read_spv(&mut rgen_file)
                    .expect(&format!("Failed to load rgen file: {:?}", rgen_path));

                let rgen_shader_info = vk::ShaderModuleCreateInfo::builder().code(&rgen_code);
                self.rgen_shader_module = self
                    .base
                    .device
                    .create_shader_module(&rgen_shader_info, None)
                    .expect("Failed to create rgen shader module.");

                let rchit_code = read_spv(&mut rchit_file)
                    .expect(&format!("Failed to load rchit file: {:?}", rchit_file));
                let rchit_shader_info = vk::ShaderModuleCreateInfo::builder().code(&rchit_code);
                self.chit_shader_module = self
                    .base
                    .device
                    .create_shader_module(&rchit_shader_info, None)
                    .expect("Failded to create rchit shader module");

                let rmiss_code = read_spv(&mut rmiss_file)
                    .expect(&format!("Failed to load rmiss file: {:?}", rmiss_file));
                let rmiss_shader_info = vk::ShaderModuleCreateInfo::builder().code(&rmiss_code);
                self.miss_shader_module = self
                    .base
                    .device
                    .create_shader_module(&rmiss_shader_info, None)
                    .expect("Failed to create rmiss shader module.");
            }

            let layouts = vec![self.descriptor_set_layout];
            let layout_create_info = vk::PipelineLayoutCreateInfo {
                s_type: vk::StructureType::PIPELINE_LAYOUT_CREATE_INFO,
                p_next: ptr::null(),
                set_layout_count: layouts.len() as u32,
                p_set_layouts: layouts.as_ptr(),
                ..Default::default()
            };

            self.pipeline_layout = self
                .base
                .device
                .create_pipeline_layout(&layout_create_info, None)
                .expect("Failed to create pipeline layout.");

            let shader_groups = vec![
                // group0 = [ raygen ]
                vk::RayTracingShaderGroupCreateInfoNV {
                    s_type: vk::StructureType::RAY_TRACING_SHADER_GROUP_CREATE_INFO_NV,
                    p_next: ptr::null(),
                    ty: vk::RayTracingShaderGroupTypeNV::GENERAL,
                    general_shader: 0,
                    closest_hit_shader: vk::SHADER_UNUSED_NV,
                    any_hit_shader: vk::SHADER_UNUSED_NV,
                    intersection_shader: vk::SHADER_UNUSED_NV,
                },
                // group1 = [ chit ]
                vk::RayTracingShaderGroupCreateInfoNV {
                    s_type: vk::StructureType::RAY_TRACING_SHADER_GROUP_CREATE_INFO_NV,
                    p_next: ptr::null(),
                    ty: vk::RayTracingShaderGroupTypeNV::TRIANGLES_HIT_GROUP,
                    general_shader: vk::SHADER_UNUSED_NV,
                    closest_hit_shader: 1,
                    any_hit_shader: vk::SHADER_UNUSED_NV,
                    intersection_shader: vk::SHADER_UNUSED_NV,
                },
                // group2 = [ miss ]
                vk::RayTracingShaderGroupCreateInfoNV {
                    s_type: vk::StructureType::RAY_TRACING_SHADER_GROUP_CREATE_INFO_NV,
                    p_next: ptr::null(),
                    ty: vk::RayTracingShaderGroupTypeNV::GENERAL,
                    general_shader: 2,
                    closest_hit_shader: vk::SHADER_UNUSED_NV,
                    any_hit_shader: vk::SHADER_UNUSED_NV,
                    intersection_shader: vk::SHADER_UNUSED_NV,
                },
            ];

            let rgen_name = CString::new("rgen_main").unwrap();
            let rchit_name = CString::new("rchit_main").unwrap();
            let rmiss_name = CString::new("rmiss_main").unwrap();
            let else_name = CString::new("main").unwrap();
            let shader_stages = if use_lib && use_hlsl {
                vec![
                    vk::PipelineShaderStageCreateInfo {
                        s_type: vk::StructureType::PIPELINE_SHADER_STAGE_CREATE_INFO,
                        p_next: ptr::null(),
                        stage: vk::ShaderStageFlags::RAYGEN_NV,
                        module: self.lib_shader_module,
                        p_name: rgen_name.as_ptr(),
                        ..Default::default()
                    },
                    vk::PipelineShaderStageCreateInfo {
                        s_type: vk::StructureType::PIPELINE_SHADER_STAGE_CREATE_INFO,
                        p_next: ptr::null(),
                        stage: vk::ShaderStageFlags::CLOSEST_HIT_NV,
                        module: self.lib_shader_module,
                        p_name: rchit_name.as_ptr(),
                        ..Default::default()
                    },
                    vk::PipelineShaderStageCreateInfo {
                        s_type: vk::StructureType::PIPELINE_SHADER_STAGE_CREATE_INFO,
                        p_next: ptr::null(),
                        stage: vk::ShaderStageFlags::MISS_NV,
                        module: self.lib_shader_module,
                        p_name: rmiss_name.as_ptr(),
                        ..Default::default()
                    },
                ]
            } else {
                vec![
                    vk::PipelineShaderStageCreateInfo {
                        s_type: vk::StructureType::PIPELINE_SHADER_STAGE_CREATE_INFO,
                        p_next: ptr::null(),
                        stage: vk::ShaderStageFlags::RAYGEN_NV,
                        module: self.lib_shader_module,
                        p_name: else_name.as_ptr(),
                        ..Default::default()
                    },
                    vk::PipelineShaderStageCreateInfo {
                        s_type: vk::StructureType::PIPELINE_SHADER_STAGE_CREATE_INFO,
                        p_next: ptr::null(),
                        stage: vk::ShaderStageFlags::CLOSEST_HIT_NV,
                        module: self.lib_shader_module,
                        p_name: else_name.as_ptr(),
                        ..Default::default()
                    },
                    vk::PipelineShaderStageCreateInfo {
                        s_type: vk::StructureType::PIPELINE_SHADER_STAGE_CREATE_INFO,
                        p_next: ptr::null(),
                        stage: vk::ShaderStageFlags::MISS_NV,
                        module: self.lib_shader_module,
                        p_name: else_name.as_ptr(),
                        ..Default::default()
                    },
                ]
            };

            let rt_pipeline_create_info = vk::RayTracingPipelineCreateInfoNV {
                s_type: vk::StructureType::RAY_TRACING_PIPELINE_CREATE_INFO_NV,
                p_next: ptr::null(),
                stage_count: shader_stages.len() as u32,
                p_stages: shader_stages.as_ptr(),
                group_count: shader_groups.len() as u32,
                p_groups: shader_groups.as_ptr(),
                max_recursion_depth: 1,
                layout: self.pipeline_layout,
                ..Default::default()
            };

            self.pipeline = self
                .ray_tracing
                .create_ray_tracing_pipelines(
                    vk::PipelineCache::null(),
                    &[rt_pipeline_create_info],
                    None,
                )
                .expect("Failed to create ray tracing pipeline.")[0];
        }
    }

    fn create_shader_binding_table(&mut self) {
        let group_count = 3;
        let table_size = (self.properties.shader_group_handle_size * group_count) as u64;
        let mut table_data: Vec<u8> = vec![0u8; table_size as usize];

        unsafe {
            self.ray_tracing
                .get_ray_tracing_shader_group_handles(
                    self.pipeline,
                    0,
                    group_count,
                    &mut table_data,
                )
                .expect("Failed to get ray tracing shader group handles.");
        }

        let mut shader_binding_table = BufferResource::new(
            table_size,
            vk::BufferUsageFlags::TRANSFER_SRC,
            vk::MemoryPropertyFlags::HOST_VISIBLE,
            self.base.clone(),
        );
        shader_binding_table.store(&table_data);
        self.shader_binding_table = Some(shader_binding_table);
    }

    fn create_descriptor_set(&mut self) {
        unsafe {
            let descriptor_sizes = [
                vk::DescriptorPoolSize {
                    ty: vk::DescriptorType::ACCELERATION_STRUCTURE_NV,
                    descriptor_count: 1,
                },
                vk::DescriptorPoolSize {
                    ty: vk::DescriptorType::STORAGE_IMAGE,
                    descriptor_count: 1,
                },
                vk::DescriptorPoolSize {
                    ty: vk::DescriptorType::UNIFORM_BUFFER,
                    descriptor_count: 3,
                },
            ];

            let descriptor_pool_info = vk::DescriptorPoolCreateInfo {
                s_type: vk::StructureType::DESCRIPTOR_POOL_CREATE_INFO,
                p_next: ptr::null(),
                pool_size_count: descriptor_sizes.len() as u32,
                p_pool_sizes: descriptor_sizes.as_ptr(),
                max_sets: 1,
                ..Default::default()
            };

            self.descriptor_pool = self
                .base
                .device
                .create_descriptor_pool(&descriptor_pool_info, None)
                .expect("Failed to create descriptor pool.");

            let descriptor_set_allocate_info = vk::DescriptorSetAllocateInfo {
                s_type: vk::StructureType::DESCRIPTOR_SET_ALLOCATE_INFO,
                p_next: ptr::null(),
                descriptor_pool: self.descriptor_pool,
                p_set_layouts: [self.descriptor_set_layout].as_ptr(),
                ..Default::default()
            };
            let descriptor_sets = self
                .base
                .device
                .allocate_descriptor_sets(&descriptor_set_allocate_info)
                .expect("Failed to allocate descriptor sets.");

            self.descriptor_set = descriptor_sets[0];

            let accel_structs = [self.top_as];
            let mut accel_info = vk::WriteDescriptorSetAccelerationStructureNV::builder()
                .acceleration_structures(&accel_structs)
                .build();

            let mut accel_write = vk::WriteDescriptorSet::builder()
                .dst_set(self.descriptor_set)
                .dst_binding(0)
                .dst_array_element(0)
                .descriptor_type(vk::DescriptorType::ACCELERATION_STRUCTURE_NV)
                .push_next(&mut accel_info)
                .build();

            accel_write.descriptor_count = 1;

            let image_info = [vk::DescriptorImageInfo {
                image_layout: vk::ImageLayout::GENERAL,
                image_view: self.offscreen_target.view,
                ..Default::default()
            }];

            let image_write = vk::WriteDescriptorSet::builder()
                .dst_set(self.descriptor_set)
                .dst_binding(1)
                .dst_array_element(0)
                .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                .image_info(&image_info)
                .build();

            let buffer0 = self.color0_buffer.as_ref().unwrap().buffer;
            let buffer1 = self.color1_buffer.as_ref().unwrap().buffer;
            let buffer2 = self.color2_buffer.as_ref().unwrap().buffer;

            let buffer_info = [
                vk::DescriptorBufferInfo {
                    buffer: buffer0,
                    range: vk::WHOLE_SIZE,
                    ..Default::default()
                },
                vk::DescriptorBufferInfo {
                    buffer: buffer1,
                    range: vk::WHOLE_SIZE,
                    ..Default::default()
                },
                vk::DescriptorBufferInfo {
                    buffer: buffer2,
                    range: vk::WHOLE_SIZE,
                    ..Default::default()
                },
            ];

            let buffer_write = vk::WriteDescriptorSet::builder()
                .dst_set(self.descriptor_set)
                .dst_binding(2)
                .dst_array_element(0)
                .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                .buffer_info(&buffer_info)
                .build();

            self.base
                .device
                .update_descriptor_sets(&[accel_write, image_write, buffer_write], &[]);
        }
    }

    fn release(&mut self) {
        unsafe {
            self.base.wait_device_idle();

            self.ray_tracing
                .destroy_acceleration_structure(self.top_as, None);
            self.base.device.free_memory(self.top_as_memory, None);

            self.ray_tracing
                .destroy_acceleration_structure(self.bottom_as, None);
            self.base.device.free_memory(self.bottom_as_memory, None);

            self.base
                .device
                .destroy_descriptor_pool(self.descriptor_pool, None);

            self.shader_binding_table = None;

            self.color0_buffer = None;
            self.color1_buffer = None;
            self.color2_buffer = None;

            self.base.device.destroy_pipeline(self.pipeline, None);
            self.base
                .device
                .destroy_pipeline_layout(self.pipeline_layout, None);
            self.base
                .device
                .destroy_descriptor_set_layout(self.descriptor_set_layout, None);

            self.base
                .device
                .destroy_shader_module(self.rgen_shader_module, None);
            self.base
                .device
                .destroy_shader_module(self.chit_shader_module, None);
            self.base
                .device
                .destroy_shader_module(self.miss_shader_module, None);
            self.base
                .device
                .destroy_shader_module(self.lib_shader_module, None);
        }
    }
}

fn main() {
    let program_proc = ProgramProc::new();
    let vulkan_renderer = Rc::new(VulkanRenderer::new(&program_proc.event_loop));

    unsafe {
        let props_rt = nv::RayTracing::get_properties(
            &vulkan_renderer.instance,
            vulkan_renderer.physical_device,
        );
        let ray_tracing = Rc::new(nv::RayTracing::new(
            &vulkan_renderer.instance,
            &vulkan_renderer.device,
        ));
        let mut app = RayTracingApp::new(vulkan_renderer.clone(), ray_tracing, props_rt);

        app.initialize();

        println!("NV Ray Tracing Properties:");
        println!(
            " shader_group_handle_size: {}",
            props_rt.shader_group_handle_size
        );
        println!(" max_recursion_depth: {}", props_rt.max_recursion_depth);
        println!(
            " max_shader_group_stride: {}",
            props_rt.max_shader_group_stride
        );
        println!(
            " shader_group_base_alignment: {}",
            props_rt.shader_group_base_alignment
        );
        println!(" max_geometry_count: {}", props_rt.max_geometry_count);
        println!(" max_instance_count: {}", props_rt.max_instance_count);
        println!(" max_triangle_count: {}", props_rt.max_triangle_count);
        println!(
            " max_descriptor_set_acceleration_structures: {}",
            props_rt.max_descriptor_set_acceleration_structures
        );

        vulkan_renderer.wait_device_idle();
        app.release();
    }
    // program_proc.main_loop(vulkan_renderer);
}
