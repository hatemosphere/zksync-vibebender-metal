#[cfg(test)]
mod scan_tests {
    use field::Field;
    use itertools::Itertools;
    use rand::distr::Uniform;
    use rand::{rng, Rng};

    use crate::field::BaseField;
    use crate::metal_runtime::{
        init_shader_library, system_default_device, MetalBuffer, MetalCommandQueue,
    };
    use crate::ops_cub::device_scan::{self, ScanOperation};

    type BF = BaseField;

    #[test]
    fn test_scan_inclusive_add_u32() {
        const NUM_ITEMS: usize = 1 << 16;
        const RANGE_MAX: u32 = 1 << 16;

        let device = system_default_device().unwrap();
        init_shader_library(device).unwrap();
        let queue = MetalCommandQueue::new(device).unwrap();

        let h_in: Vec<u32> = rng()
            .sample_iter(Uniform::new(0, RANGE_MAX).unwrap())
            .take(NUM_ITEMS)
            .collect_vec();

        let d_in = MetalBuffer::from_slice(device, &h_in).unwrap();
        let d_out = MetalBuffer::<u32>::alloc(device, NUM_ITEMS).unwrap();
        let temp_elems = device_scan::get_scan_temp_storage_elems(NUM_ITEMS as u32) as usize;
        let d_temp = MetalBuffer::<u32>::alloc(device, temp_elems).unwrap();

        let cmd_buf = queue.new_command_buffer().unwrap();
        device_scan::scan_u32(
            device,
            &cmd_buf,
            true, // inclusive
            &d_in,
            &d_out,
            &d_temp,
            NUM_ITEMS as u32,
        )
        .unwrap();
        cmd_buf.commit_and_wait();
        cmd_buf.status().unwrap();

        let mut h_out = vec![0u32; NUM_ITEMS];
        unsafe {
            d_out.copy_to_slice(&mut h_out);
        }

        // CPU reference: inclusive prefix sum
        let mut expected = Vec::with_capacity(NUM_ITEMS);
        let mut acc: u32 = 0;
        for &v in &h_in {
            acc = acc.wrapping_add(v);
            expected.push(acc);
        }
        assert_eq!(expected, h_out);
    }

    #[test]
    fn test_scan_exclusive_add_u32() {
        const NUM_ITEMS: usize = 1 << 16;
        const RANGE_MAX: u32 = 1 << 16;

        let device = system_default_device().unwrap();
        init_shader_library(device).unwrap();
        let queue = MetalCommandQueue::new(device).unwrap();

        let h_in: Vec<u32> = rng()
            .sample_iter(Uniform::new(0, RANGE_MAX).unwrap())
            .take(NUM_ITEMS)
            .collect_vec();

        let d_in = MetalBuffer::from_slice(device, &h_in).unwrap();
        let d_out = MetalBuffer::<u32>::alloc(device, NUM_ITEMS).unwrap();
        let temp_elems = device_scan::get_scan_temp_storage_elems(NUM_ITEMS as u32) as usize;
        let d_temp = MetalBuffer::<u32>::alloc(device, temp_elems).unwrap();

        let cmd_buf = queue.new_command_buffer().unwrap();
        device_scan::scan_u32(
            device,
            &cmd_buf,
            false, // exclusive
            &d_in,
            &d_out,
            &d_temp,
            NUM_ITEMS as u32,
        )
        .unwrap();
        cmd_buf.commit_and_wait();
        cmd_buf.status().unwrap();

        let mut h_out = vec![0u32; NUM_ITEMS];
        unsafe {
            d_out.copy_to_slice(&mut h_out);
        }

        // CPU reference: exclusive prefix sum
        let mut expected = Vec::with_capacity(NUM_ITEMS);
        let mut acc: u32 = 0;
        for &v in &h_in {
            expected.push(acc);
            acc = acc.wrapping_add(v);
        }
        assert_eq!(expected, h_out);
    }

    #[test]
    fn test_scan_inclusive_add_bf() {
        const NUM_ITEMS: usize = 1 << 16;
        const RANGE_MAX: u32 = 1 << 16;

        let device = system_default_device().unwrap();
        init_shader_library(device).unwrap();
        let queue = MetalCommandQueue::new(device).unwrap();

        let h_in: Vec<BF> = rng()
            .sample_iter(Uniform::new(0u32, RANGE_MAX).unwrap())
            .map(BF::from_nonreduced_u32)
            .take(NUM_ITEMS)
            .collect_vec();

        let d_in = MetalBuffer::from_slice(device, &h_in).unwrap();
        let d_out = MetalBuffer::<BF>::alloc(device, NUM_ITEMS).unwrap();
        let temp_elems = device_scan::get_scan_temp_storage_elems(NUM_ITEMS as u32) as usize;
        let d_temp = MetalBuffer::<BF>::alloc(device, temp_elems).unwrap();

        let cmd_buf = queue.new_command_buffer().unwrap();
        device_scan::scan_bf(
            device,
            &cmd_buf,
            ScanOperation::Sum,
            true,
            &d_in,
            &d_out,
            &d_temp,
            NUM_ITEMS as u32,
        )
        .unwrap();
        cmd_buf.commit_and_wait();
        cmd_buf.status().unwrap();

        let mut h_out = vec![BF::ZERO; NUM_ITEMS];
        unsafe {
            d_out.copy_to_slice(&mut h_out);
        }

        // CPU reference
        let mut acc = BF::ZERO;
        for (i, &v) in h_in.iter().enumerate() {
            acc.add_assign(&v);
            assert_eq!(acc, h_out[i], "mismatch at index {i}");
        }
    }

    #[test]
    fn test_scan_exclusive_add_bf() {
        const NUM_ITEMS: usize = 1 << 16;
        const RANGE_MAX: u32 = 1 << 16;

        let device = system_default_device().unwrap();
        init_shader_library(device).unwrap();
        let queue = MetalCommandQueue::new(device).unwrap();

        let h_in: Vec<BF> = rng()
            .sample_iter(Uniform::new(0u32, RANGE_MAX).unwrap())
            .map(BF::from_nonreduced_u32)
            .take(NUM_ITEMS)
            .collect_vec();

        let d_in = MetalBuffer::from_slice(device, &h_in).unwrap();
        let d_out = MetalBuffer::<BF>::alloc(device, NUM_ITEMS).unwrap();
        let temp_elems = device_scan::get_scan_temp_storage_elems(NUM_ITEMS as u32) as usize;
        let d_temp = MetalBuffer::<BF>::alloc(device, temp_elems).unwrap();

        let cmd_buf = queue.new_command_buffer().unwrap();
        device_scan::scan_bf(
            device,
            &cmd_buf,
            ScanOperation::Sum,
            false,
            &d_in,
            &d_out,
            &d_temp,
            NUM_ITEMS as u32,
        )
        .unwrap();
        cmd_buf.commit_and_wait();
        cmd_buf.status().unwrap();

        let mut h_out = vec![BF::ZERO; NUM_ITEMS];
        unsafe {
            d_out.copy_to_slice(&mut h_out);
        }

        // CPU reference: exclusive prefix sum
        let mut acc = BF::ZERO;
        for (i, &v) in h_in.iter().enumerate() {
            assert_eq!(acc, h_out[i], "mismatch at index {i}");
            acc.add_assign(&v);
        }
    }

    #[test]
    fn test_scan_inclusive_mul_bf() {
        const NUM_ITEMS: usize = 1 << 16;
        const RANGE_MAX: u32 = 1 << 16;

        let device = system_default_device().unwrap();
        init_shader_library(device).unwrap();
        let queue = MetalCommandQueue::new(device).unwrap();

        let h_in: Vec<BF> = rng()
            .sample_iter(Uniform::new(0u32, RANGE_MAX).unwrap())
            .map(BF::from_nonreduced_u32)
            .take(NUM_ITEMS)
            .collect_vec();

        let d_in = MetalBuffer::from_slice(device, &h_in).unwrap();
        let d_out = MetalBuffer::<BF>::alloc(device, NUM_ITEMS).unwrap();
        let temp_elems = device_scan::get_scan_temp_storage_elems(NUM_ITEMS as u32) as usize;
        let d_temp = MetalBuffer::<BF>::alloc(device, temp_elems).unwrap();

        let cmd_buf = queue.new_command_buffer().unwrap();
        device_scan::scan_bf(
            device,
            &cmd_buf,
            ScanOperation::Product,
            true,
            &d_in,
            &d_out,
            &d_temp,
            NUM_ITEMS as u32,
        )
        .unwrap();
        cmd_buf.commit_and_wait();
        cmd_buf.status().unwrap();

        let mut h_out = vec![BF::ZERO; NUM_ITEMS];
        unsafe {
            d_out.copy_to_slice(&mut h_out);
        }

        // CPU reference: inclusive prefix product
        let mut acc = BF::ONE;
        for (i, &v) in h_in.iter().enumerate() {
            acc.mul_assign(&v);
            assert_eq!(acc, h_out[i], "mismatch at index {i}");
        }
    }

    #[test]
    fn test_scan_exclusive_mul_bf() {
        const NUM_ITEMS: usize = 1 << 16;
        const RANGE_MAX: u32 = 1 << 16;

        let device = system_default_device().unwrap();
        init_shader_library(device).unwrap();
        let queue = MetalCommandQueue::new(device).unwrap();

        let h_in: Vec<BF> = rng()
            .sample_iter(Uniform::new(0u32, RANGE_MAX).unwrap())
            .map(BF::from_nonreduced_u32)
            .take(NUM_ITEMS)
            .collect_vec();

        let d_in = MetalBuffer::from_slice(device, &h_in).unwrap();
        let d_out = MetalBuffer::<BF>::alloc(device, NUM_ITEMS).unwrap();
        let temp_elems = device_scan::get_scan_temp_storage_elems(NUM_ITEMS as u32) as usize;
        let d_temp = MetalBuffer::<BF>::alloc(device, temp_elems).unwrap();

        let cmd_buf = queue.new_command_buffer().unwrap();
        device_scan::scan_bf(
            device,
            &cmd_buf,
            ScanOperation::Product,
            false,
            &d_in,
            &d_out,
            &d_temp,
            NUM_ITEMS as u32,
        )
        .unwrap();
        cmd_buf.commit_and_wait();
        cmd_buf.status().unwrap();

        let mut h_out = vec![BF::ZERO; NUM_ITEMS];
        unsafe {
            d_out.copy_to_slice(&mut h_out);
        }

        // CPU reference: exclusive prefix product
        let mut acc = BF::ONE;
        for (i, &v) in h_in.iter().enumerate() {
            assert_eq!(acc, h_out[i], "mismatch at index {i}");
            acc.mul_assign(&v);
        }
    }
}

#[cfg(test)]
mod reduce_tests {
    use field::{Field, Rand};
    use rand::rng;

    use crate::field::BaseField;
    use crate::metal_runtime::{
        init_shader_library, system_default_device, MetalBuffer, MetalCommandQueue,
    };
    use crate::ops_cub::device_reduce::{self, ReduceOperation};

    type BF = BaseField;

    #[test]
    fn test_reduce_sum_bf() {
        const NUM_ITEMS: usize = 1 << 16;

        let device = system_default_device().unwrap();
        init_shader_library(device).unwrap();
        let queue = MetalCommandQueue::new(device).unwrap();

        let mut rng = rng();
        let h_in: Vec<BF> = (0..NUM_ITEMS)
            .map(|_| BF::random_element(&mut rng))
            .collect();

        let d_in = MetalBuffer::from_slice(device, &h_in).unwrap();
        let d_out = MetalBuffer::<BF>::alloc(device, 1).unwrap();
        let temp_elems = device_reduce::get_reduce_temp_storage_elems(NUM_ITEMS as u32) as usize;
        let d_partials = MetalBuffer::<BF>::alloc(device, temp_elems).unwrap();

        let cmd_buf = queue.new_command_buffer().unwrap();
        device_reduce::reduce_bf(
            device,
            &cmd_buf,
            ReduceOperation::Sum,
            &d_in,
            &d_out,
            &d_partials,
            NUM_ITEMS as u32,
        )
        .unwrap();
        cmd_buf.commit_and_wait();
        cmd_buf.status().unwrap();

        let mut h_out = [BF::ZERO];
        unsafe {
            d_out.copy_to_slice(&mut h_out);
        }

        let expected = h_in.iter().fold(BF::ZERO, |mut acc, x| {
            acc.add_assign(x);
            acc
        });
        assert_eq!(expected, h_out[0]);
    }

    #[test]
    fn test_reduce_product_bf() {
        const NUM_ITEMS: usize = 1 << 16;

        let device = system_default_device().unwrap();
        init_shader_library(device).unwrap();
        let queue = MetalCommandQueue::new(device).unwrap();

        let mut rng = rng();
        let h_in: Vec<BF> = (0..NUM_ITEMS)
            .map(|_| BF::random_element(&mut rng))
            .collect();

        let d_in = MetalBuffer::from_slice(device, &h_in).unwrap();
        let d_out = MetalBuffer::<BF>::alloc(device, 1).unwrap();
        let temp_elems = device_reduce::get_reduce_temp_storage_elems(NUM_ITEMS as u32) as usize;
        let d_partials = MetalBuffer::<BF>::alloc(device, temp_elems).unwrap();

        let cmd_buf = queue.new_command_buffer().unwrap();
        device_reduce::reduce_bf(
            device,
            &cmd_buf,
            ReduceOperation::Product,
            &d_in,
            &d_out,
            &d_partials,
            NUM_ITEMS as u32,
        )
        .unwrap();
        cmd_buf.commit_and_wait();
        cmd_buf.status().unwrap();

        let mut h_out = [BF::ZERO];
        unsafe {
            d_out.copy_to_slice(&mut h_out);
        }

        let expected = h_in.iter().fold(BF::ONE, |mut acc, x| {
            acc.mul_assign(x);
            acc
        });
        assert_eq!(expected, h_out[0]);
    }

    #[test]
    fn test_reduce_sum_e2() {
        use crate::field::Ext2Field;
        type E2 = Ext2Field;

        const NUM_ITEMS: usize = 1 << 16;

        let device = system_default_device().unwrap();
        init_shader_library(device).unwrap();
        let queue = MetalCommandQueue::new(device).unwrap();

        let mut rng = rng();
        let h_in: Vec<E2> = (0..NUM_ITEMS)
            .map(|_| E2::random_element(&mut rng))
            .collect();

        let d_in = MetalBuffer::from_slice(device, &h_in).unwrap();
        let d_out = MetalBuffer::<E2>::alloc(device, 1).unwrap();
        let temp_elems = device_reduce::get_reduce_temp_storage_elems(NUM_ITEMS as u32) as usize;
        let d_partials = MetalBuffer::<E2>::alloc(device, temp_elems).unwrap();

        let cmd_buf = queue.new_command_buffer().unwrap();
        device_reduce::reduce_e2(
            device,
            &cmd_buf,
            ReduceOperation::Sum,
            &d_in,
            &d_out,
            &d_partials,
            NUM_ITEMS as u32,
        )
        .unwrap();
        cmd_buf.commit_and_wait();
        cmd_buf.status().unwrap();

        let mut h_out = [E2::ZERO];
        unsafe {
            d_out.copy_to_slice(&mut h_out);
        }

        let expected = h_in.iter().fold(E2::ZERO, |mut acc, x| {
            acc.add_assign(x);
            acc
        });
        assert_eq!(expected, h_out[0]);
    }

    #[test]
    fn test_reduce_sum_e4() {
        use crate::field::Ext4Field;
        type E4 = Ext4Field;

        const NUM_ITEMS: usize = 1 << 16;

        let device = system_default_device().unwrap();
        init_shader_library(device).unwrap();
        let queue = MetalCommandQueue::new(device).unwrap();

        let mut rng = rng();
        let h_in: Vec<E4> = (0..NUM_ITEMS)
            .map(|_| E4::random_element(&mut rng))
            .collect();

        let d_in = MetalBuffer::from_slice(device, &h_in).unwrap();
        let d_out = MetalBuffer::<E4>::alloc(device, 1).unwrap();
        let temp_elems = device_reduce::get_reduce_temp_storage_elems(NUM_ITEMS as u32) as usize;
        let d_partials = MetalBuffer::<E4>::alloc(device, temp_elems).unwrap();

        let cmd_buf = queue.new_command_buffer().unwrap();
        device_reduce::reduce_e4(
            device,
            &cmd_buf,
            ReduceOperation::Sum,
            &d_in,
            &d_out,
            &d_partials,
            NUM_ITEMS as u32,
        )
        .unwrap();
        cmd_buf.commit_and_wait();
        cmd_buf.status().unwrap();

        let mut h_out = [E4::ZERO];
        unsafe {
            d_out.copy_to_slice(&mut h_out);
        }

        let expected = h_in.iter().fold(E4::ZERO, |mut acc, x| {
            acc.add_assign(x);
            acc
        });
        assert_eq!(expected, h_out[0]);
    }

    #[test]
    fn test_reduce_product_e4() {
        use crate::field::Ext4Field;
        type E4 = Ext4Field;

        const NUM_ITEMS: usize = 1 << 16;

        let device = system_default_device().unwrap();
        init_shader_library(device).unwrap();
        let queue = MetalCommandQueue::new(device).unwrap();

        let mut rng = rng();
        let h_in: Vec<E4> = (0..NUM_ITEMS)
            .map(|_| E4::random_element(&mut rng))
            .collect();

        let d_in = MetalBuffer::from_slice(device, &h_in).unwrap();
        let d_out = MetalBuffer::<E4>::alloc(device, 1).unwrap();
        let temp_elems = device_reduce::get_reduce_temp_storage_elems(NUM_ITEMS as u32) as usize;
        let d_partials = MetalBuffer::<E4>::alloc(device, temp_elems).unwrap();

        let cmd_buf = queue.new_command_buffer().unwrap();
        device_reduce::reduce_e4(
            device,
            &cmd_buf,
            ReduceOperation::Product,
            &d_in,
            &d_out,
            &d_partials,
            NUM_ITEMS as u32,
        )
        .unwrap();
        cmd_buf.commit_and_wait();
        cmd_buf.status().unwrap();

        let mut h_out = [E4::ZERO];
        unsafe {
            d_out.copy_to_slice(&mut h_out);
        }

        let expected = h_in.iter().fold(E4::ONE, |mut acc, x| {
            acc.mul_assign(x);
            acc
        });
        assert_eq!(expected, h_out[0]);
    }
}

#[cfg(test)]
mod sort_tests {
    use itertools::Itertools;
    use rand::random;

    use crate::metal_runtime::{
        init_shader_library, system_default_device, MetalBuffer, MetalCommandQueue,
    };
    use crate::ops_cub::device_radix_sort::{self};

    #[test]
    fn test_sort_keys_ascending_u32() {
        const NUM_ITEMS: usize = 1 << 16;

        let device = system_default_device().unwrap();
        init_shader_library(device).unwrap();
        let queue = MetalCommandQueue::new(device).unwrap();

        let mut h_keys_in: Vec<u32> = (0..NUM_ITEMS).map(|_| random()).collect_vec();

        let d_keys_in = MetalBuffer::from_slice(device, &h_keys_in).unwrap();
        let d_keys_out = MetalBuffer::<u32>::alloc(device, NUM_ITEMS).unwrap();
        let temp_elems = device_radix_sort::get_sort_temp_storage_elems(NUM_ITEMS as u32) as usize;
        let d_histograms = MetalBuffer::<u32>::alloc(device, temp_elems).unwrap();

        let cmd_buf = queue.new_command_buffer().unwrap();
        device_radix_sort::sort_keys(
            device,
            &cmd_buf,
            false,
            &d_keys_in,
            &d_keys_out,
            &d_histograms,
            NUM_ITEMS as u32,
            0,
            32,
        )
        .unwrap();
        cmd_buf.commit_and_wait();
        cmd_buf.status().unwrap();

        let mut h_keys_out = vec![0u32; NUM_ITEMS];
        unsafe {
            d_keys_out.copy_to_slice(&mut h_keys_out);
        }

        h_keys_in.sort();
        assert_eq!(h_keys_in, h_keys_out);
    }
}

#[test]

fn test_scan_inclusive_mul_e4() {
    use crate::metal_runtime::*;
    use crate::field::Ext4Field as E4;
    use field::{Field, FieldExtension, Mersenne31Field as BF};

    let device = device::system_default_device().unwrap();
    pipeline::init_shader_library(device).unwrap();
    let queue = command_queue::MetalCommandQueue::new(device).unwrap();

    let n = 1024usize;
    let mut input = vec![E4::ONE; n];
    // Fill with non-trivial values
    for i in 0..n {
        let val = BF::from_nonreduced_u32((i as u32 + 2) % 2147483647);
        input[i] = E4::from_coeffs_in_base(&[val, BF::ZERO, BF::ZERO, BF::ZERO]);
    }

    // CPU reference: inclusive prefix product
    let mut cpu_result = vec![E4::ONE; n];
    let mut running = E4::ONE;
    for i in 0..n {
        running.mul_assign(&input[i]);
        cpu_result[i] = running;
    }

    // GPU scan
    let d_in = buffer::MetalBuffer::<E4>::from_slice(device, &input).unwrap();
    let d_out = buffer::MetalBuffer::<E4>::alloc(device, n).unwrap();
    let temp_elems = super::device_scan::get_scan_temp_storage_elems(n as u32);
    let d_temp = buffer::MetalBuffer::<E4>::alloc(device, temp_elems as usize).unwrap();

    let cmd_buf = queue.new_command_buffer().unwrap();
    super::device_scan::scan_e4(
        device, &cmd_buf,
        super::device_scan::ScanOperation::Product,
        true, // inclusive
        &d_in, &d_out, &d_temp, n as u32,
    ).unwrap();
    cmd_buf.commit_and_wait();

    let gpu_result = unsafe { d_out.as_slice() };
    let mut mismatches = 0;
    for i in 0..n {
        if gpu_result[i] != cpu_result[i] {
            mismatches += 1;
            if mismatches <= 3 {
                println!("MISMATCH at {}: GPU={:?} CPU={:?}", i,
                    gpu_result[i].into_coeffs_in_base().map(|e: BF| e.to_reduced_u32()),
                    cpu_result[i].into_coeffs_in_base().map(|e: BF| e.to_reduced_u32()));
            }
        }
    }
    println!("E4 inclusive mul scan: {}/{} mismatches", mismatches, n);
    assert_eq!(mismatches, 0);
}
