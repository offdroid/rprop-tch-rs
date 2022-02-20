# Rprop for tch-rs

[Resilient Propagation](http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.21.1417) for [tch-rs](https://github.com/LaurentMazare/tch-rs) ported from [PyTorch](https://github.com/pytorch/pytorch), specifically [torch.optim.Rprop](https://github.com/pytorch/pytorch/blob/bf03d934965d0fae47a66756dd70304ad718b125/torch/optim/rprop.py).

> :warning: Currently only tested with simple models!

Licensed under the same terms as PyTorch, see [LICENSE](./LICENSE)

## Usage

Add to `Cargo.toml`
```toml
rprop-tch = { git = "https://github.com/offdroid/rprop-tch-rs.git" }
```

Usage matches `tch::nn::Optimizer`
```rust
let vs = tch::nn::VarStore::new(tch::Device::Cpu);
// Init model with `vs`
let net: &dyn tch::nn::Module = todo!();
// Build Rprop optimizer, here with default paramters
let mut opt = rprop_tch::Rprop::build_default(&vs, Some(0.01));
// Training loop
for epoch in 1..10 {
    let (x, y) = todo!();
    let loss: tch::Tensor = net.forward(&x).mse_loss(&y);
    // Use it like tch::nn::Optimizer
    opt.zero_grad();
    loss.backward();
    opt.step();
}
```

## Example

Check [examples](./examples) and/or run
```bash
cargo run --example basic
```
