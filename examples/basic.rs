use std::f64::consts::PI;

use tch::kind;
use tch::nn;
use tch::nn::Module;
use tch::Device;
use tch::TchError;
use tch::Tensor;

fn net(vs: &nn::Path) -> impl Module {
    nn::seq()
        .add(nn::linear(vs / "layer1", 1, 5, Default::default()))
        .add_fn(|xs| xs.relu())
        .add(nn::linear(vs, 5, 1, Default::default()))
}

pub fn main() -> Result<(), TchError> {
    // Fit noisy sine in [0, π]. Inspired by Bishop2006PRML 1.1
    let x = Tensor::linspace(0.0, 2.0 * PI, 10, kind::FLOAT_CPU);
    let y = Tensor::normal(&x.sin(), &x.zeros_like(), 0.2);
    let test_x = Tensor::linspace(0.0, 2.0 * PI, 10, kind::FLOAT_CPU);

    let vs = nn::VarStore::new(Device::Cpu);
    let net = net(&vs.root());

    let mut opt = rprop_tch::Rprop::build_default(&vs, None);
    // Alternatively try with Adam
    // let mut opt = Adam::default().build(&vs, 0.01)?;

    for epoch in 1..200 {
        let loss = net
            .forward(&x.unsqueeze(1))
            .mse_loss(&y.unsqueeze(1), tch::Reduction::Mean);

        opt.zero_grad();
        loss.backward();
        opt.step();

        tch::no_grad(|| {
            let test_loss = net
                .forward(&test_x.unsqueeze(1))
                .mse_loss(&test_x.sin().unsqueeze(1), tch::Reduction::Mean);
            println!(
                "epoch: {:3} train loss: {:.6}; test loss: {:.6}",
                epoch,
                f64::from(&loss),
                f64::from(&test_loss),
            );
        });
    }
    Ok(())
}
