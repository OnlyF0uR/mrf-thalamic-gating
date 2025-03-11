use rand::Rng;

/// Represents a simplified neuron with biologically inspired properties.
#[derive(Debug)]
pub struct Neuron {
    // Synaptic weights for excitatory and inhibitory inputs.
    pub excitatory_weights: Vec<f64>,
    pub inhibitory_weights: Vec<f64>,
    /// Bias modulates baseline excitability.
    pub bias: f64,
    /// Output signal (spike value or graded response).
    pub output: f64,
    /// Refractory period information.
    pub refractory_period: usize,
    pub refractory_countdown: usize,
    /// Membrane potential properties (in mV).
    pub membrane_potential: f64,
    pub threshold: f64,
    pub time_constant: f64,
    pub resting_potential: f64,
    /// Multipliers for scaling input contributions.
    pub excitatory_multiplier: f64,
    pub inhibitory_multiplier: f64,
    /// For MRF neurons only: scales the final output.
    pub activity_multiplier: f64,
}

impl Neuron {
    /// Create a new neuron.
    ///
    /// * `num_excitatory_inputs` – Number of excitatory synapses.
    /// * `num_inhibitory_inputs` – Number of inhibitory synapses.
    pub fn new(num_excitatory_inputs: usize, num_inhibitory_inputs: usize) -> Self {
        let mut rng = rand::rng();

        // Initialize excitatory weights in the positive range.
        let excitatory_weights = (0..num_excitatory_inputs)
            .map(|_| rng.random_range(0.0..0.5))
            .collect::<Vec<f64>>();

        // Initialize inhibitory weights in the negative range.
        let inhibitory_weights = (0..num_inhibitory_inputs)
            .map(|_| -rng.random_range(0.3..0.8))
            .collect::<Vec<f64>>();

        let bias = rng.random_range(-0.2..0.2);

        Neuron {
            excitatory_weights,
            inhibitory_weights,
            bias,
            output: 0.0,
            refractory_period: 3,
            refractory_countdown: 0,
            membrane_potential: -70.0, // Resting potential (mV).
            threshold: -55.0,          // Spike threshold (mV).
            time_constant: 0.8,        // Decay factor per timestep.
            resting_potential: -70.0,
            excitatory_multiplier: 15.0,
            inhibitory_multiplier: 20.0, // Default; may be adjusted externally.
            activity_multiplier: 1.0,    // For MRF neurons (multiplied onto output).
        }
    }

    /// Process incoming signals and update the neuron's state.
    ///
    /// * `excitatory_inputs` – List of excitatory input signals.
    /// * `inhibitory_inputs` – List of inhibitory input signals.
    pub fn process_inputs(&mut self, excitatory_inputs: &[f64], inhibitory_inputs: &[f64]) {
        // If the neuron is in its refractory period, dampen its output.
        if self.refractory_countdown > 0 {
            self.refractory_countdown -= 1;
            self.output *= 0.5;
            return;
        }

        // Calculate contributions from excitatory and inhibitory inputs.
        let excitatory_effect: f64 = excitatory_inputs
            .iter()
            .zip(self.excitatory_weights.iter())
            .map(|(i, w)| i * w)
            .sum();
        let inhibitory_effect: f64 = inhibitory_inputs
            .iter()
            .zip(self.inhibitory_weights.iter())
            .map(|(i, w)| i * w)
            .sum();

        // Compute the net input, including any bias.
        let net_input = (excitatory_effect * self.excitatory_multiplier)
            + (inhibitory_effect * self.inhibitory_multiplier)
            + self.bias;
        self.membrane_potential += net_input;

        // Determine whether to fire.
        if self.membrane_potential >= self.threshold {
            self.output = 1.0;
            self.refractory_countdown = self.refractory_period;
            // Hyperpolarize after a spike.
            self.membrane_potential = self.resting_potential - 10.0;
        } else {
            // Compute a graded response using a sigmoid function.
            let normalized = (self.membrane_potential - self.resting_potential)
                / (self.threshold - self.resting_potential);
            self.output = self.sigmoid(normalized * 4.0);
        }

        // Apply exponential decay of the membrane potential toward the resting potential.
        self.membrane_potential = self.membrane_potential * self.time_constant
            + self.resting_potential * (1.0 - self.time_constant);

        // For MRF neurons (no inhibitory inputs), scale the output by the activity multiplier.
        if self.inhibitory_weights.is_empty() {
            self.output *= self.activity_multiplier;
        }
    }

    /// Sigmoid activation function for a smooth graded response.
    pub fn sigmoid(&self, x: f64) -> f64 {
        1.0 / (1.0 + (-x).exp())
    }

    /// Add noise to the membrane potential (scaled to mV).
    pub fn add_noise(&mut self, noise_level: f64) {
        let mut rng = rand::rng();
        self.membrane_potential += rng.random_range(-noise_level..noise_level) * 10.0;
    }
}
