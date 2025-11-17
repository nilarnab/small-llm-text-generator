import torch

import matplotlib.pyplot as plt
import argparse
import os
import tiktoken
import math


# manual import since pico-llm.py has a dash in its name (can't do normal import)
import importlib.util, sys
spec = importlib.util.spec_from_file_location("pico_llm", os.path.join(os.getcwd(), "pico-llm.py"))
pico_llm = importlib.util.module_from_spec(spec)
sys.modules["pico_llm"] = pico_llm
spec.loader.exec_module(pico_llm)

TransformerModel = pico_llm.TransformerModel
LSTMSeqModel = pico_llm.LSTMSeqModel
KGramMLPSeqModel = pico_llm.KGramMLPSeqModel


def visualize_attention(model, enc, prompt, device):

    os.makedirs("interpretability_outputs", exist_ok=True)

    print("Encoding prompt...")
    token_ids = enc.encode(prompt)
    tokens = torch.tensor(token_ids, dtype=torch.long, device=device).unsqueeze(1)
    print("Token shape:", tokens.shape)

    model.eval()
    attention_per_block = []

    # Run the prompt through the transformer but manually extract attn weights
    print("Running forward pass through the transformer blocks...")
    with torch.no_grad():
        x = model.embedding(tokens)
        seq_len = x.size(0)

        # Add positional embeddings like usual
        pos = torch.arange(seq_len, device=device).unsqueeze(1)
        x = x + model.position_embedding(pos)

        # Go block by block and grab attention from each head
        for idx, block in enumerate(model.blocks):
            x_norm = block.norm_1(x)

            attn_out, attn_weights = block.attn1(
                x_norm, x_norm, x_norm,
                need_weights=True,
                average_attn_weights=False
            )

            # attn_weights shape: (1, num_heads, seq_len, seq_len)
            attention_per_block.append(attn_weights[0].cpu())

            # Normal transformer updates
            x = x + attn_out
            x = x + block.mlp_g_1(block.norm_2(x))

    # Convert token IDs to readable text so the heatmaps have labels
    token_labels = [enc.decode([t]) for t in token_ids]

    print("\nSaving attention visualizations...")

    # Go through each block and save the two figures
    for block_i, attn_heads in enumerate(attention_per_block):
        num_heads = attn_heads.size(0)
        grid_side = math.ceil(math.sqrt(num_heads))


        # (A) Usual multi-head attention grid

        fig, axes = plt.subplots(grid_side, grid_side, figsize=(14, 14))
        fig.suptitle(f"Transformer Block {block_i+1} – All Heads", fontsize=18, y=1.02)

        for h in range(num_heads):
            r, c = divmod(h, grid_side)
            ax = axes[r][c] if grid_side > 1 else axes
            ax.imshow(attn_heads[h], cmap="viridis", aspect="equal")
            ax.set_title(f"Head {h}", fontsize=9)

            ax.set_xticks(range(len(token_labels)))
            ax.set_yticks(range(len(token_labels)))
            ax.set_xticklabels(token_labels, rotation=90, fontsize=5)
            ax.set_yticklabels(token_labels, fontsize=5)

        # Remove empty slots in the grid if head count isn't a perfect square
        for h in range(num_heads, grid_side * grid_side):
            r, c = divmod(h, grid_side)
            fig.delaxes(axes[r][c])

        # Add a small explanation under the figure
        explanation = (
            "Legend:\n"
            "• Rows = tokens doing the attending\n"
            "• Columns = tokens being attended to\n"
            "• Color = attention weight (softmax probability)"
        )

        fig.text(
            0.5, -0.04, explanation,
            ha="center", va="top",
            fontsize=12,
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.7)
        )

        plt.tight_layout()
        save_path = f"interpretability_outputs/block_{block_i}_grid.png"
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        plt.close()
        print("Saved:", save_path)


        # Square “giant” 12-head grid in a 3x4 layout

        heads_per_row = 4
        num_rows = math.ceil(num_heads / heads_per_row)

        rows_combined = []
        for r in range(num_rows):
            row_maps = []
            for c in range(heads_per_row):
                head_idx = r * heads_per_row + c
                if head_idx < num_heads:
                    row_maps.append(attn_heads[head_idx])
                else:
                    # If the number of heads doesn't fill the grid evenly
                    row_maps.append(torch.zeros_like(attn_heads[0]))
            rows_combined.append(torch.cat(row_maps, dim=1))

        square_map = torch.cat(rows_combined, dim=0)

        plt.figure(figsize=(10, 10))
        plt.imshow(square_map, cmap="viridis", aspect="equal")
        plt.title(f"Transformer Block {block_i+1} – All 12 Heads (Square Grid)", fontsize=16)
        plt.colorbar(label="Attention Strength")

        plt.text(
            0.5, -0.08,
            "This figure arranges all heads in a 3×4 layout.\n"
            "Each small square is a full head attention matrix.\n"
            "Brighter = higher attention weight.",
            ha="center", va="top",
            fontsize=10,
            transform=plt.gca().transAxes,
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.7)
        )

        save_path = f"interpretability_outputs/block_{block_i}_square_giant.png"
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        plt.close()
        print("Saved:", save_path)

    print("\nDone.\n")





def visualize_lstm(model, enc, prompt, device="cpu"):
    model.eval()
    with torch.no_grad():
        tokens = torch.tensor(enc.encode(prompt), dtype=torch.long, device=device).unsqueeze(0)
        emb = model.embedding(tokens)
        out, _ = model.lstm(emb)

        activations = out.squeeze(0).detach().cpu()

        plt.figure(figsize=(10, 6))
        im = plt.imshow(activations.T, aspect="auto", cmap="plasma")
        cbar = plt.colorbar(im)
        cbar.set_label("Activation Value", fontsize=10)

        plt.xlabel("Token Index")
        plt.ylabel("Neuron Index")
        plt.title("LSTM Hidden Layer Activations")

        os.makedirs("interpretability_outputs", exist_ok=True)
        plt.savefig("interpretability_outputs/lstm_hidden_states.png", dpi=150)
        plt.close()

        print("Saved interpretability_outputs/lstm_hidden_states.png")


# monosemantic analysis (Anthropic style, works for both models)
def visualize_monosemantic(model, enc, prompt, device="cpu"):
    print("\nRunning simple monosemantic analysis...")
    os.makedirs("interpretability_outputs", exist_ok=True)
    tokens = torch.tensor(enc.encode(prompt), dtype=torch.long, device=device).unsqueeze(0)

    model.eval()
    with torch.no_grad():
        if hasattr(model, "embedding"):
            emb = model.embedding(tokens)
        else:
            print("No embedding layer found.")
            return

        # if LSTM model
        if hasattr(model, "lstm"):
            out, _ = model.lstm(emb)
            activations = out.squeeze(0).cpu()
        # if Transformer model
        elif hasattr(model, "blocks"):
            x = emb
            seq_len = x.size(1)
            positions = torch.arange(seq_len, device=device).unsqueeze(0)
            x = x + model.position_embedding(positions)
            for block in model.blocks:
                x = block.norm_1(x)
                attn_out, _ = block.attn1(x, x, x, need_weights=False)
                x = x + attn_out
                x = x + block.mlp_g_1(block.norm_2(x))
            activations = x.squeeze(0).cpu()
        else:
            print("Unknown model type, skipping.")
            return

        variances = activations.var(dim=0)
        top_neurons = torch.topk(variances, 10)
        plt.figure(figsize=(8, 4))
        plt.bar(range(10), top_neurons.values.numpy())
        plt.title("Top 10 Most Active (Monosemantic) Neurons")
        plt.xlabel("Neuron Index (sorted by variance)")
        plt.ylabel("Activation Variance")
        plt.grid(axis="y", linestyle="--", alpha=0.4)
        plt.savefig("interpretability_outputs/monosemantic_top_neurons.png", dpi=150)
        plt.close()

def visualize_kgram(model, enc, prompt, device):

    os.makedirs("interpretability_outputs", exist_ok=True)
    token_ids = enc.encode(prompt)
    tokens = torch.tensor(token_ids, dtype=torch.long, device=device).unsqueeze(1)

    seq_len = len(token_ids)
    k = model.k

    model.eval()

    with torch.no_grad():
        # run it once so weights/embeddings are loaded
        _ = model(tokens)

        # pad so sliding windows at the beginning don't break
        pad = torch.zeros((k - 1, 1), dtype=torch.long, device=device)
        padded = torch.cat([pad, tokens], dim=0)

        # collect window slices used for each prediction
        window_list = []
        for i in range(k):
            window_list.append(padded[i:i + seq_len])
        windows = torch.stack(window_list, dim=2)     # shape: (seq_len, 1, k)

        # embed each window position
        emb = model.embedding(windows)                # (seq_len, 1, k, embed_dim)
        emb = emb.squeeze(1)                          # (seq_len, k, embed_dim)

        # L2 norm as a rough measure of how active each window slot is
        contrib = emb.norm(dim=2).cpu()

    plt.figure(figsize=(8, 6))
    plt.imshow(contrib, cmap="viridis", aspect="auto")
    plt.colorbar(label="Embedding Contribution (L2 Norm)")

    plt.title("K-gram MLP – Window Position Influence")
    plt.xlabel("Window Position")     # 0 is oldest, k-1 is the newest
    plt.ylabel("Tokens in Prompt")

    # x-axis labels
    x_labels = [f"t-{(k - 1) - i}" for i in range(k)]
    plt.xticks(range(k), x_labels)

    # y-axis labels (actual tokens)
    decoded_tokens = [enc.decode([tid]) for tid in token_ids]
    plt.yticks(range(seq_len), decoded_tokens, fontsize=6)

    out_path = "interpretability_outputs/kgram_contribution_heatmap.png"
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()

    print("Saved:", out_path)





def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="attention")
    parser.add_argument("--model_type", type=str, default="transformer")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--prompt", type=str,
                        default="Question: Who won Super Bowl XX? Answer:")
    args = parser.parse_args()

    enc = tiktoken.get_encoding("gpt2")

    print(f"Loading {args.model_type} model...")
    if args.model_type == "transformer":
        model = TransformerModel(d_model=768, n_heads=12, n_blocks=12)

    elif args.model_type == "lstm":
        model = LSTMSeqModel(vocab_size=50257, embed_size=256, hidden_size=256, num_layers=1)

    elif args.model_type == "kgram":
        model = KGramMLPSeqModel(
            vocab_size=50257,
            k=3,
            embed_size=256,
            num_inner_layers=1,
            chunk_size=1
        )

    else:
        print("Unknown model type")
        return

    ckpt = torch.load(args.checkpoint, map_location=args.device)
    model.load_state_dict(ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt)
    print(f"Loaded checkpoint: {args.checkpoint}")

    # handle all interpretability modes
    if args.mode == "attention":
        visualize_attention(model, enc, args.prompt, args.device)
    elif args.mode == "attention_grid":
        visualize_attention(model, enc, args.prompt, args.device)
    elif args.mode == "hidden":
        visualize_lstm(model, enc, args.prompt, args.device)
    elif args.mode == "monosemantic":
        visualize_monosemantic(model, enc, args.prompt, args.device)
    elif args.mode == "kgram":
        visualize_kgram(model, enc, args.prompt, args.device)
    else:
        print("Unknown mode")


if __name__ == "__main__":
    main()