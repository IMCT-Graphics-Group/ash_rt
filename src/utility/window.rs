use crate::utility::{constants::*, fps_limiter};
use winit::{
    event::{ElementState, Event, KeyboardInput, WindowEvent},
    event_loop::{ControlFlow, EventLoop},
};

const IS_PAINT_FPS_COUNTER: bool = true;

pub fn init_window(
    event_loop: &winit::event_loop::EventLoop<()>,
    title: &str,
    width: u32,
    height: u32,
) -> winit::window::Window {
    winit::window::WindowBuilder::new()
        .with_title(title)
        .with_inner_size(winit::dpi::LogicalSize::new(width, height))
        .build(event_loop)
        .expect("Failed to create window.")
}

pub trait VulkanApp {
    fn draw_frame(&mut self, delta_time: f32);
    fn recreate_swapchain(&mut self);
    fn cleanup_swapchain(&self);
    fn wait_device_idle(&self);
    fn resize_framebuffer(&mut self);
    fn window_ref(&self) -> &winit::window::Window;
}

pub struct ProgramProc {
    pub event_loop: EventLoop<()>,
}

impl ProgramProc {
    pub fn new() -> ProgramProc {
        let event_loop = EventLoop::new();
        ProgramProc { event_loop }
    }

    pub fn main_loop<A: 'static + VulkanApp>(self, mut vulkan_app: A) {
        let mut tick_counter = fps_limiter::FPSLimiter::new();

        self.event_loop
            .run(move |event, _, control_flow| match event {
                Event::WindowEvent { event, .. } => match event {
                    WindowEvent::CloseRequested => {
                        vulkan_app.wait_device_idle();
                        *control_flow = ControlFlow::Exit
                    }
                    WindowEvent::KeyboardInput { input, .. } => match input {
                        KeyboardInput {
                            virtual_keycode,
                            state,
                            ..
                        } => match (virtual_keycode, state) {
                            (Some(WINDOW_KEYCODE_EXIT), ElementState::Pressed) => {
                                vulkan_app.wait_device_idle();
                                *control_flow = ControlFlow::Exit
                            }
                            _ => {}
                        },
                    },
                    WindowEvent::Resized(_new_size) => {
                        vulkan_app.wait_device_idle();
                        vulkan_app.resize_framebuffer();
                    }
                    _ => {}
                },
                Event::MainEventsCleared => {
                    vulkan_app.window_ref().request_redraw();
                }
                Event::RedrawRequested(_window_id) => {
                    let delta_time = tick_counter.delta_time();
                    vulkan_app.draw_frame(delta_time);

                    if IS_PAINT_FPS_COUNTER {
                        print!("FPS: {}\r", tick_counter.fps());
                    }

                    tick_counter.tick_frame();
                }
                Event::LoopDestroyed => {
                    vulkan_app.wait_device_idle();
                }
                _ => (),
            })
    }
}
