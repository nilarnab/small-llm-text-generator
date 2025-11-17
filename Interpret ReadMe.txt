

This folder contains interpretability experiments for both the Transformer and LSTM models I trained on the QA dataset.
The goal was to understand what the models are doing internally following ideas from the AlphaFold attention maps and Anthropic’s monosemanticity analyses.


Install the required packages:

pip install torch matplotlib tiktoken


Run everything using:

python interpretability.py


This script supports four interpretability modes:
- attention → Transformer attention heads (AlphaFold-style)
- hidden → LSTM hidden activations
- monosemantic → Neuron-level analysis for both models
- attention_grid → same as attention, for completeness

All generated figures will appear inside:

interpretability_outputs/


Commands to Run

Each command below corresponds to a different interpretability task.
All checkpoint paths and prompts will need to be adjusted as per your prompt and checkpoint path.

---

Transformer — Attention Visualization (AlphaFold style), add your path and prompts!!


python interpretability.py --mode attention --model_type transformer --checkpoint checkpoints/transformer_20251111_211051/step_4212_LOSS_0.3353.pt --device cpu --prompt "Question: Who won Super Bowl XX? Answer:"
```

Output files:

interpretability_outputs/block_0_grid.png
interpretability_outputs/block_1_grid.png


Description:
Each image shows a grid of attention maps for one Transformer block.
Each small square is an attention head, showing which input tokens it focuses on.
Bright areas = higher attention weight.
Some heads strongly focus on “Question” or “Answer,” while others spread over content tokens like “Super” and “Bowl.”

This matches the “attention head” visualizations shown in the AlphaFold paper appendices.


Transformer — Monosemantic Neuron Analysis


python interpretability.py --mode monosemantic --model_type transformer --checkpoint checkpoints/transformer_20251111_211051/step_4212_LOSS_0.3353.pt --device cpu --prompt "Question: Who won Super Bowl XX? Answer:"


Output file:

interpretability_outputs/monosemantic_top_neurons.png


Description:
This bar chart shows the top 10 neurons with the highest activation variance.
These are “monosemantic” neurons — units that respond strongly to specific input features.
In my results, neurons were most active for question-related tokens and named entities, consistent with feature-selective behavior described by Anthropic.

NOTE:

1- Use the exact same embed_size, hidden_size, and num_layers as during training. These should match those you used for trainings to those present in this file.
(For example: 256, 512, 2)

2- Double-check the correct checkpoint path — make sure you’re using an LSTM checkpoint, not a Transformer one.

3- If you modify the model code, don’t remove or rename layers, or old checkpoints won’t match.

 LSTM — Hidden State Activations

python interpretability.py --mode hidden --model_type lstm --checkpoint checkpoints/lstm_seq_20251111_175054/step_40056_LOSS_1.4254.pt --device cpu --prompt "Question: Who won Super Bowl XX? Answer:"


Output file:

interpretability_outputs/lstm_hidden_states.png


Description:
This heatmap visualizes hidden activations across time steps.
Each horizontal stripe corresponds to one LSTM neuron.
The x-axis shows token positions (e.g., “Question”, “Who”, “won”, etc.), and color intensity shows how strongly that neuron is activated.
Bright yellow = strong activation; dark purple = low or negative activation.
Distinct activation waves can be seen for important words like “Super” and “Bowl,” showing how LSTM memory evolves over time.



 LSTM — Monosemantic Neuron Analysis


python interpretability.py --mode monosemantic --model_type lstm --checkpoint checkpoints/lstm_seq_20251111_175054/step_40056_LOSS_1.4254.pt --device cpu --prompt "Question: Who won Super Bowl XX? Answer:"


Output file:

interpretability_outputs/monosemantic_top_neurons.png

K-gram

 python interpretability.py --mode kgram --model_type kgram --checkpoint checkpoints/kgram_mlp_seq_20251116_035832/step_11001_LOSS_1.7580.pt --device cpu --prompt "Question: In which year was CNN founded? Answer:"


Description:
A bar chart showing the 10 most active neurons (highest activation variance) in the LSTM.
These are roughly the “feature neurons” that respond most consistently to particular tokens.
The pattern is similar to the transformer’s analysis but less modular — activations are more distributed across neurons.



Interpretation Summary

- Transformer:
  The multi-head attention visualizations show distinct focus patterns — some heads lock onto “Question” or “Answer,” while others distribute across relevant content tokens.
  This behavior aligns with the hierarchical specialization seen in attention-based models like AlphaFold and GPT.

- LSTM:
  The LSTM’s hidden state heatmap shows temporal dependency — neurons activate in waves as the sequence is read.
  Monosemantic analysis highlights a few dominant neurons that respond selectively to input structure, though the overall pattern is more entangled than in the Transformer.



References

- AlphaFold Paper (Appendix Figures) — for attention visualization style
- Anthropic’s Interpretability Series:
  - Scaling Monosemanticity https://transformer-circuits.pub/2024/scaling-monosemanticity/
  - Monosemantic Features https://transformer-circuits.pub/2023/monosemantic-features/index.html
