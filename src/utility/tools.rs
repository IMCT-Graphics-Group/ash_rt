use std::{ffi::CStr, os::raw::c_char, path::Path};

use super::structures::Vertex;

pub fn vk_to_string(raw_string_array: &[c_char]) -> String {
    let raw_string = unsafe {
        let pointer = raw_string_array.as_ptr();
        CStr::from_ptr(pointer)
    };

    raw_string
        .to_str()
        .expect("Failed to convert vulkan raw string.")
        .to_owned()
}

pub fn read_shader_code(shader_path: &Path) -> Vec<u8> {
    use std::fs::File;
    use std::io::Read;

    let spv_file =
        File::open(shader_path).expect(&format!("Failed to find spv file at {:?}", shader_path));
    let bytes_code: Vec<u8> = spv_file.bytes().filter_map(|byte| byte.ok()).collect();

    bytes_code
}

pub fn load_model(model_path: &Path) -> (Vec<Vertex>, Vec<u32>) {
    let load_options = tobj::LoadOptions {
        single_index: true,
        ..Default::default()
    };
    let model_obj =
        tobj::load_obj(model_path, &load_options).expect("Failed to load model object!");

    let mut vertices = vec![];
    let mut indices = vec![];

    let (models, _) = model_obj;
    for m in models.iter() {
        let mesh = &m.mesh;

        if mesh.texcoords.len() == 0 {
            panic!("Missing texture coordinate for the model.")
        }

        let total_vertices_count = mesh.positions.len() / 3;
        for i in 0..total_vertices_count {
            let vertex = Vertex {
                pos: [
                    mesh.positions[i * 3],
                    mesh.positions[i * 3 + 1],
                    mesh.positions[i * 3 + 2],
                    1.0,
                ],
                color: [1.0, 1.0, 1.0, 1.0],
                tex_coord: [mesh.texcoords[i * 2], mesh.texcoords[i * 2 + 1]],
            };
            vertices.push(vertex);
        }

        indices = mesh.indices.clone();
    }
    println!("Num of vertices: {}", vertices.len());
    (vertices, indices)
}
