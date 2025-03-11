use std::{thread, time};

mod neuron;
use neuron::Neuron;

/// A simplified neural network with two layers:
/// - The MRF layer represents inhibitory neurons.
/// - The reticular nucleus of the thalamus receives sensory (excitatory) inputs and inhibitory inputs from the MRF.
#[derive(Debug)]
struct NeuralNetwork {
    mrf_layer: Vec<Neuron>,
    rn_layer: Vec<Neuron>,
    timestep: usize,
}

impl NeuralNetwork {
    /// Create a new network.
    ///
    /// * `num_mrf_inputs` – Number of excitatory inputs for MRF neurons.
    /// * `num_sensory_inputs` – Number of sensory inputs for thalamic reticular nucleus neurons.
    /// * `num_mrf` – Number of MRF (inhibitory) neurons.
    /// * `num_rn` – Number of thalamic reticular nucleus neurons.
    fn new(
        num_mrf_inputs: usize,
        num_sensory_inputs: usize,
        num_mrf: usize,
        num_rn: usize,
    ) -> Self {
        // Build MRF layer (inhibitory neurons). These neurons do not use any baseline bias.
        let mut mrf_layer = (0..num_mrf)
            .map(|_| Neuron::new(num_mrf_inputs, 0))
            .collect::<Vec<_>>();
        // Set bias = 0 for all MRF neurons.
        for neuron in mrf_layer.iter_mut() {
            neuron.bias = 0.0;
        }

        // Build thalamus layer (sensory-driven neurons).
        let mut rn_layer = (0..num_rn)
            .map(|_| Neuron::new(num_sensory_inputs, num_mrf))
            .collect::<Vec<_>>();
        // Give thalamic neurons a positive baseline bias and reduce their sensitivity to inhibition.
        for neuron in rn_layer.iter_mut() {
            neuron.bias = 5.0;
            neuron.inhibitory_multiplier = 5.0;
        }

        NeuralNetwork {
            mrf_layer,
            rn_layer,
            timestep: 0,
        }
    }

    /// Process one simulation timestep.
    ///
    /// * `sensory_inputs` – Excitatory inputs for thalamic neurons.
    /// * `mrf_inputs` – Excitatory inputs for MRF neurons.
    ///
    /// Returns outputs of both layers.
    fn feedforward(&mut self, sensory_inputs: &[f64], mrf_inputs: &[f64]) -> (Vec<f64>, Vec<f64>) {
        self.timestep += 1;

        // Add a small amount of noise to all neurons.
        for neuron in self.mrf_layer.iter_mut() {
            neuron.add_noise(0.05);
        }
        for neuron in self.rn_layer.iter_mut() {
            neuron.add_noise(0.05);
        }

        // Process the MRF layer (inhibitory neurons) using only excitatory inputs.
        for neuron in self.mrf_layer.iter_mut() {
            neuron.process_inputs(mrf_inputs, &[]);
        }

        // Use the MRF outputs as inhibitory inputs for thalamic neurons.
        let mrf_outputs: Vec<f64> = self.mrf_layer.iter().map(|n| n.output).collect();
        for neuron in self.rn_layer.iter_mut() {
            neuron.process_inputs(sensory_inputs, &mrf_outputs);
        }

        let rn_outputs = self.rn_layer.iter().map(|n| n.output).collect();
        (mrf_outputs, rn_outputs)
    }

    /// Log network activity including the ratio (MRF activity / RN activity).
    fn print_activity(&self, mrf_outputs: &[f64], rn_outputs: &[f64]) {
        let mrf_avg: f64 = mrf_outputs.iter().sum::<f64>() / mrf_outputs.len() as f64;
        let rn_avg: f64 = rn_outputs.iter().sum::<f64>() / rn_outputs.len() as f64;
        let ratio = if rn_avg.abs() > f64::EPSILON {
            mrf_avg / rn_avg
        } else {
            0.0
        };
        println!(
            "Timestep: {} | MRF Activity: {:.4} | RN Activity: {:.4} | Ratio (MRF/RN): {:.4}",
            self.timestep, mrf_avg, rn_avg, ratio
        );
    }

    /// Increase the activity multiplier for all MRF neurons.
    ///
    /// This directly scales the output of MRF neurons. For example, a multiplier of 2 means the
    /// inhibitory (MRF) activity is doubled.
    fn increase_mrf_activity(&mut self, increment: f64) {
        for neuron in self.mrf_layer.iter_mut() {
            neuron.activity_multiplier += increment;
        }
        println!(
            "MRF Activity Multiplier now: {:.4}",
            self.mrf_layer[0].activity_multiplier
        );
    }
}

fn main() {
    // Create a neural network:
    // - MRF layer: 4 inhibitory neurons with 3 excitatory inputs each.
    // - RN layer: 6 neurons with 3 sensory inputs and 4 inhibitory inputs.
    let mut nn = NeuralNetwork::new(3, 3, 4, 6);

    // Define constant input signals.
    // - Strong sensory inputs drive thalamic neurons.
    // - Moderate excitatory inputs drive MRF neurons.
    let sensory_inputs = vec![0.8, 0.8, 0.8];
    let mrf_inputs = vec![0.3, 0.3, 0.3];

    // Set the interval for boosting MRF activity.
    let mrf_increase_interval = time::Duration::from_secs(3);
    let mut last_mrf_increase_time = time::Instant::now();

    let mut toggle_incr = false;

    // Run the simulation indefinitely.
    loop {
        let (mrf_outputs, rn_outputs) = nn.feedforward(&sensory_inputs, &mrf_inputs);
        nn.print_activity(&mrf_outputs, &rn_outputs);

        if time::Instant::now().duration_since(last_mrf_increase_time) >= mrf_increase_interval {
            if toggle_incr {
                // Less MRF, more RN
                println!("\n--- DECREASING MRF ACTIVITY MULTIPLIER ---\n");
                nn.increase_mrf_activity(-0.5);

                // If activity of RN is very high
                if rn_outputs.iter().sum::<f64>() > 0.9 {
                    toggle_incr = false;
                }
            } else {
                // More MRF, less RN
                println!("\n+++ INCREASING MRF ACTIVITY MULTIPLIER +++\n");
                nn.increase_mrf_activity(0.5);

                // If activity of RN is very low
                if rn_outputs.iter().sum::<f64>() < 0.001 {
                    toggle_incr = true;
                }
            }

            last_mrf_increase_time = time::Instant::now();
        }
        thread::sleep(time::Duration::from_millis(100));
    }
}
