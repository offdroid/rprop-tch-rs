use std::ops::Mul;
use std::sync::Arc;
use std::sync::Mutex;
use tch::nn::VarStore;
use tch::nn::Variables;
use tch::no_grad;
use tch::Device;
use tch::Kind;
use tch::Tensor;

struct Buffer {
    prev: Tensor,
    step_size: Tensor,
    kind: Kind,
}

impl Buffer {
    pub fn new(size: &[i64], kind: Kind, lr: f64) -> Buffer {
        Buffer {
            prev: Tensor::zeros(size, (Kind::Double, Device::Cpu)),
            step_size: Tensor::full(size, lr, (Kind::Double, Device::Cpu)),
            kind,
        }
    }
}

/// [Rprop optimizer](http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.21.1417)
pub struct Rprop {
    etas: (f64, f64),
    step_sizes: (f64, f64),

    vars: Arc<Mutex<Variables>>,
    buffers: Vec<Buffer>,
}

impl Rprop {
    pub fn build(vs: &VarStore, lr: f64, etas: (f64, f64), step_sizes: (f64, f64)) -> Rprop {
        let vars = vs.variables_.clone();
        // Create buffers for every trainable variable
        let buffers = vars
            .lock()
            .unwrap()
            .trainable_variables
            .iter()
            .map(|x| Buffer::new(&x.tensor.size(), x.tensor.kind(), lr))
            .collect();

        Rprop {
            etas,
            step_sizes,
            vars,
            buffers,
        }
    }

    pub fn build_default(vs: &VarStore, lr: Option<f64>) -> Rprop {
        Self::build(vs, lr.unwrap_or(1e-2), (0.5, 1.2), (1e-6, 50.0))
    }

    /// Ensure that the gradient update is not part of the autograd routine
    pub fn step(&mut self) {
        no_grad(|| self._step());
    }

    pub fn _step(&mut self) {
        let mut vars = self.vars.lock().unwrap();

        let etaminus = self.etas.0;
        let etaplus = self.etas.1;
        let step_size_min = self.step_sizes.0;
        let step_size_max = self.step_sizes.1;

        // Iterate through all trainable variables
        for (var, buffer) in vars.trainable_variables.iter_mut().zip(&mut self.buffers) {
            // Convert to double as tch-rs only really supports f64 atm
            let grad = var.tensor.grad().to_dtype(Kind::Double, false, false);
            if grad.is_sparse() {
                unimplemented!("Rprop does not support sparse gradients");
            }

            let sign = grad.shallow_clone().mul(buffer.prev.shallow_clone()).sign();
            let sign = sign
                .where_scalarother(&sign.lt(0.0), etaplus)
                .where_scalarother(&sign.gt(0.0), etaminus)
                .where_scalarother(&sign.ne(0.0), 1.0);

            // Update stepsizes with step size updates
            let _ = buffer
                .step_size
                .g_mul_(&sign)
                .clamp_(step_size_min, step_size_max);

            // For dir<0, dfdx=0
            // For dir>=0 dfdx=dfdx
            let grad = grad.where_scalarother(&sign.ne(etaminus), 0.0);
            // Update parameters
            let _ = var.tensor.addcmul_(
                &grad.sign().mul(-1.0).to_dtype(buffer.kind, false, false),
                &buffer.step_size.to_dtype(buffer.kind, false, false),
            );
            buffer.prev.copy_(&grad);
        }
    }

    /// Zero the gradient of all trainable variables
    pub fn zero_grad(&mut self) {
        let mut vars = self.vars.lock().unwrap();
        for var in &mut vars.trainable_variables {
            var.tensor.zero_grad();
        }
    }

    /// Clips gradient value at some specified maximum value.
    pub fn clip_grad_value(&self, max: f64) {
        let v = self.vars.lock().unwrap();
        for var in v.trainable_variables.iter() {
            let _t = var.tensor.grad().clamp_(-max, max);
        }
    }

    /// Clips gradient L2 norm over all trainable parameters.
    ///
    /// The norm is computed over all gradients together, as if they were
    /// concatenated into a single vector.
    pub fn clip_grad_norm(&self, max: f64) {
        crate::no_grad(|| {
            let v = self.vars.lock().unwrap();
            let mut norms = vec![];
            for var in v.trainable_variables.iter() {
                norms.push(var.tensor.grad().norm());
            }
            let total_norm = f64::from(Tensor::stack(&norms, 0).norm());
            let clip_coef = max / (total_norm + 1e-6);
            if clip_coef < 1.0 {
                for var in v.trainable_variables.iter() {
                    let _t = var.tensor.grad().g_mul_scalar_(clip_coef);
                }
            }
        })
    }

    /// Applies a backward step pass, update the gradients, and performs an optimization step.
    pub fn backward_step(&mut self, loss: &Tensor) {
        self.zero_grad();
        loss.backward();
        self.step();
    }

    /// Applies a backward step pass, update the gradients, and performs an optimization step.
    ///
    /// The gradients are clipped based on `max` before being applied.
    pub fn backward_step_clip(&mut self, loss: &Tensor, max: f64) {
        self.zero_grad();
        loss.backward();
        self.clip_grad_value(max);
        self.step();
    }

    /// Applies a backward step pass, update the gradients, and performs an optimization step.
    ///
    /// The gradients L2 norm is clipped based on `max`.
    pub fn backward_step_clip_norm(&mut self, loss: &Tensor, max: f64) {
        self.zero_grad();
        loss.backward();
        self.clip_grad_norm(max);
        self.step();
    }
}
