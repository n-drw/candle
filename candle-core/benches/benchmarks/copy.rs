use crate::benchmarks::{BenchDevice, BenchDeviceHandler};
use candle_core::{DType, Device, Tensor};
use criterion::{black_box, criterion_group, Criterion, Throughput};
use std::time::Instant;

fn run(a: &Tensor) {
    a.copy().unwrap();
}

fn run_copy_benchmark(c: &mut Criterion, device: &Device, dtype: DType, name: &str) {
    let b = 1;
    let m = 1024;
    let k = 1024;

    let tensor = Tensor::arange(0.0f32, (b * m * k) as f32, device)
        .unwrap()
        .to_dtype(dtype)
        .unwrap()
        .reshape((b, m, k))
        .unwrap();

    let flops = b * m * k * dtype.size_in_bytes();

    let mut group = c.benchmark_group(device.bench_name(name));
    group.throughput(Throughput::Bytes(flops as u64));
    group.bench_function("iter", move |b| {
        b.iter_custom(|iters| {
            let start = Instant::now();
            for _i in 0..iters {
                run(black_box(&tensor));
            }
            device.sync().unwrap();
            start.elapsed()
        })
    });
    group.finish();
}

fn criterion_benchmark(c: &mut Criterion) {
    let handler = BenchDeviceHandler::new().unwrap();
    for device in handler.devices {
        for dtype in [DType::F32, DType::BF16, DType::F16] {
            let name = format!("copy_{:?}", dtype);
            if device.is_dtype_available(dtype) {
                run_copy_benchmark(c, &device, dtype, &name);
            }
        }
    }
}

criterion_group!(benches, criterion_benchmark);
use crate::benchmarks::{BenchDevice, BenchDeviceHandler};
use candle_core::{Device, Tensor, WithDType};
use criterion::{black_box, criterion_group, Criterion, Throughput};
use std::time::Instant;

fn run_copy_mask_benchmark<D: WithDType>(c: &mut Criterion, device: &Device, name: &str) {
    let batch_size = 128;
    let in_seq_len = 1;
    let kv_seq_len = 1024;

    let attn_mask = vec![vec![vec![D::zero(); kv_seq_len]; in_seq_len]; batch_size];
    let size_in_bytes = batch_size * in_seq_len * kv_seq_len * D::DTYPE.size_in_bytes();

    let mut group = c.benchmark_group(device.bench_name(name));
    group.throughput(Throughput::Bytes(size_in_bytes as u64));
    group.bench_function("iter", move |b| {
        b.iter_custom(|iters| {
            let attn_masks = vec![attn_mask.clone(); iters as usize];
            let start = Instant::now();
            for attn_mask in attn_masks.into_iter() {
                let tensor = Tensor::new(black_box(attn_mask), device).unwrap();
                black_box(tensor);
            }
            device.sync().unwrap();
            start.elapsed()
        })
    });
    group.finish();
}

fn criterion_benchmark(c: &mut Criterion) {
    let handler = BenchDeviceHandler::new().unwrap();
    for device in handler.devices {
        run_copy_mask_benchmark::<f32>(c, &device, "copy_mask");
    }
}

criterion_group!(benches, criterion_benchmark);
