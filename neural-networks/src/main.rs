use anyhow::Result;
use candle_core::{DType, Device, Tensor, D};
use candle_nn::{loss, ops, Linear, Module, Optimizer, VarBuilder, VarMap};
use clap::Parser;
use std::rc::Rc;

const IMAGE_DIM: usize = 28 * 28;
const LABELS: usize = 10;

struct Dataset {
    pub training_data: Tensor,
    pub training_labels: Tensor,
    pub test_data: Tensor,
    pub test_labels: Tensor,
}

fn load_tensors(csv: &str, device: &Device) -> Result<(Tensor, Tensor)> {
    let mut data = Vec::new();
    let mut labels = Vec::new();

    let mut rdr = csv::Reader::from_path(csv)?;
    for result in rdr.records() {
        let record = result?;
        let label = record.get(0).unwrap().parse::<u32>()?;
        let mut features = Vec::new();
        for i in 1..record.len() {
            features.push(record.get(i).unwrap().parse::<f32>()? / 255.0);
        }
        labels.push(label);
        data.push(features);
    }

    let data = data.into_iter().flatten().collect::<Vec<f32>>();
    let data = Tensor::from_slice(&data, (labels.len(), IMAGE_DIM), device)?;
    let labels = Tensor::from_slice(&labels, (labels.len(),), device)?;

    Ok((data, labels))
}

fn load_dataset(train_csv: &str, test_csv: &str, device: &Device) -> Result<Dataset> {
    let (training_data, training_labels) = load_tensors(train_csv, device)?;
    let (test_data, test_labels) = load_tensors(test_csv, device)?;

    Ok(Dataset {
        training_data,
        training_labels,
        test_data,
        test_labels,
    })
}
#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    #[arg(long)]
    train_csv: String,

    #[arg(long)]
    test_csv: String,

    // Print the Cost and Loss at each epoch
    #[arg(long, default_value_t = false)]
    progress: bool,

    // The learning rate
    #[arg(long, default_value = "0.01")]
    learning_rate: f64,

    // The regularization parameter
    #[arg(long, default_value = "0.01")]
    regularization: f32,

    // The number of epochs
    #[arg(long, default_value = "5000")]
    epochs: i32,
}

struct Mlp {
    ln1: Linear,
    ln2: Linear,
}

impl Mlp {
    fn new(vs: VarBuilder) -> Result<Self> {
        let ln1 = candle_nn::linear(IMAGE_DIM, 100, vs.pp("ln1"))?;
        let ln2 = candle_nn::linear(100, LABELS, vs.pp("ln2"))?;
        Ok(Self { ln1, ln2 })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let xs = self.ln1.forward(xs)?;
        let xs = xs.relu()?;
        Ok(self.ln2.forward(&xs)?)
    }
}

fn main() -> Result<()> {
    let args = Args::parse();
    let device = Rc::new(Device::cuda_if_available(0)?);
    let dataset = load_dataset(&args.train_csv, &args.test_csv, &device)?;

    let varmap = VarMap::new();
    let vs = VarBuilder::from_varmap(&varmap, DType::F32, &device);
    let model = Mlp::new(vs)?;
    let mut sgd = candle_nn::SGD::new(varmap.all_vars(), args.learning_rate)?;

    let test_images = dataset.test_data.to_device(&device)?;
    let test_labels = dataset
        .test_labels
        .to_dtype(DType::U32)?
        .to_device(&device)?;
    for epoch in 1..args.epochs {
        let logits = model.forward(&dataset.training_data)?;
        let log_sm = ops::log_softmax(&logits, D::Minus1)?;
        let loss = loss::nll(&log_sm, &dataset.training_labels)?;
        sgd.backward_step(&loss)?;

        let test_logits = model.forward(&test_images)?;
        let sum_ok = test_logits
            .argmax(D::Minus1)?
            .eq(&test_labels)?
            .to_dtype(DType::F32)?
            .sum_all()?
            .to_scalar::<f32>()?;
        let test_accuracy = sum_ok / test_labels.dims1()? as f32;
        if args.progress && epoch % 100 == 0 {
            println!(
                "{epoch:4} train loss: {:8.5} test acc: {:5.2}%",
                loss.to_scalar::<f32>()?,
                100. * test_accuracy
            );
        }
    }

    Ok(())
}
