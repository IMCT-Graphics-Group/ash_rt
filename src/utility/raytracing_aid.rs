use ash::vk;

pub fn find_memorytype_index(
    memory_req: &vk::MemoryRequirements,
    memory_prop: &vk::PhysicalDeviceMemoryProperties,
    flags: vk::MemoryPropertyFlags,
) -> Option<u32> {
    // Try to find an exactly matching memory flag
    let best_suitable_index =
        find_memorytype_index_f(memory_req, memory_prop, flags, |property_flags, flags| {
            property_flags == flags
        });
    if best_suitable_index.is_some() {
        return best_suitable_index;
    }
    // Otherwise find a memory flag that works
    find_memorytype_index_f(memory_req, memory_prop, flags, |property_flags, flags| {
        property_flags & flags == flags
    })
}

pub fn find_memorytype_index_f<F: Fn(vk::MemoryPropertyFlags, vk::MemoryPropertyFlags) -> bool>(
    memory_req: &vk::MemoryRequirements,
    memory_prop: &vk::PhysicalDeviceMemoryProperties,
    flags: vk::MemoryPropertyFlags,
    f: F,
) -> Option<u32> {
    let mut memory_type_bits = memory_req.memory_type_bits;
    for (index, ref memory_type) in memory_prop.memory_types.iter().enumerate() {
        if memory_type_bits & 1 == 1 {
            if f(memory_type.property_flags, flags) {
                return Some(index as u32);
            }
        }
        memory_type_bits = memory_type_bits >> 1;
    }
    None
}
