import torch

import matplotlib.pyplot as plt
import argparse
import os
import math
import tiktoken


# manual import since pico-llm.py has a dash in its name (can't do normal import)
import importlib.util, sys
spec = importlib.util.spec_from_file_location("pico_llm", os.path.join(os.getcwd(), "pico-llm.py"))
pico_llm = importlib.util.module_from_spec(spec)
sys.modules["pico_llm"] = pico_llm
spec.loader.exec_module(pico_llm)

TransformerModel = pico_llm.TransformerModel
LSTMSeqModel = pico_llm.LSTMSeqModel


def visualize_attention(model, enc, prompt, device):
    import matplotlib.pyplot as plt
    import math
    import torch
    import os

    os.makedirs("interpretability_outputs", exist_ok=True)

    print("Encoding prompt...")
    tokens = torch.tensor(enc.encode(prompt), dtype=torch.long, device=device).unsqueeze(1)
    print(f"Prompt tokenized to shape: {tokens.shape}")

    model.eval()
    attn_maps = []

    print("Running through transformer blocks...")
    with torch.no_grad():
        x = model.embedding(tokens)
        seq_len = x.size(0)

        positions = torch.arange(seq_len, device=device).unsqueeze(1)
        x = x + model.position_embedding(positions)

        for i, block in enumerate(model.blocks):
            x_norm = block.norm_1(x)

            attn_out, attn_weights = block.attn1(
                x_norm, x_norm, x_norm,
                need_weights=True,
                average_attn_weights=False
            )

            # attn_weights: (1, num_heads, tgt_len, src_len)
            attn_maps.append(attn_weights[0].cpu())

            x = x + attn_out
            x = x + block.mlp_g_1(block.norm_2(x))

    # Decode token strings
    token_texts = [enc.decode([tok]) for tok in tokens.squeeze().tolist()]

    print("\nSaving attention grids...")


    for block_i, attn in enumerate(attn_maps):

        num_heads = attn.size(0)
        grid_size = math.ceil(math.sqrt(num_heads))
        fig, axes = plt.subplots(grid_size, grid_size, figsize=(14, 14))
        fig.suptitle(f"Transformer Block {block_i+1} – All Heads", fontsize=18, y=1.02)

        for h in range(num_heads):
            row, col = divmod(h, grid_size)
            ax = axes[row][col] if grid_size > 1 else axes
            ax.imshow(attn[h], cmap='viridis', aspect='equal')
            ax.set_title(f"Head {h}", fontsize=9)

            ax.set_xticks(range(len(token_texts)))
            ax.set_yticks(range(len(token_texts)))
            ax.set_xticklabels(token_texts, rotation=90, fontsize=5)
            ax.set_yticklabels(token_texts, fontsize=5)

        # Remove unused tiles
        for h in range(num_heads, grid_size * grid_size):
            row, col = divmod(h, grid_size)
            fig.delaxes(axes[row][col])

        # Legend Box
        legend_text = (
            "Legend:\n"
            "• Rows = Query tokens (who is attending)\n"
            "• Columns = Key tokens (what they attend to)\n"
            "• Color = Attention strength (softmax probability)"
        )

        fig.text(
            0.5, -0.04,
            legend_text,
            ha='center', va='top',
            fontsize=12,
            bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.7)
        )

        plt.tight_layout()
        out_path = f"interpretability_outputs/block_{block_i}_grid.png"
        plt.savefig(out_path, dpi=200, bbox_inches='tight')
        plt.close()
        print(f"Saved {out_path}")


        heads_per_row = 4
        rows = math.ceil(num_heads / heads_per_row)

        grid_rows = []
        for r in range(rows):
            row_heads = []
            for c in range(heads_per_row):
                idx = r * heads_per_row + c
                if idx < num_heads:
                    row_heads.append(attn[idx])
                else:
                    row_heads.append(torch.zeros_like(attn[0]))
            grid_rows.append(torch.cat(row_heads, dim=1))

        square_map = torch.cat(grid_rows, dim=0)

        plt.figure(figsize=(10, 10))
        plt.imshow(square_map, cmap='viridis', aspect='equal')
        plt.title(f"Transformer Block {block_i+1} – All 12 Heads (Square Grid)", fontsize=16)
        plt.colorbar(label="Attention Strength")

        # Add legend box
        plt.text(
            0.5, -0.08,
            "Square Giant Map Legend:\n"
            "Each cell = one attention head arranged in a 3×4 grid.\n"
            "Rows and columns reflect token-to-token attention.\n"
            "Color represents attention probability.",
            ha='center', va='top',
            fontsize=10,
            transform=plt.gca().transAxes,
            bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.7)
        )

        out_path = f"interpretability_outputs/block_{block_i}_square_giant.png"
        plt.savefig(out_path, dpi=200, bbox_inches='tight')
        plt.close()
        print(f"Saved {out_path}")

    print("Done.\n")




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

    import torch
    import matplotlib.pyplot as plt
    import os

    os.makedirs("interpretability_outputs", exist_ok=True)

    print("Encoding prompt...")
    tokens = enc.encode(prompt)
    token_tensor = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(1)  # (seq_len, 1)
    seq_len = len(tokens)
    k = model.k

    model.eval()

    with torch.no_grad():
        _ = model(token_tensor)

        padded = torch.zeros((k - 1, 1), dtype=torch.long, device=device)
        padded = torch.cat([padded, token_tensor], dim=0)  # (seq_len + k - 1)

        windows = []
        for i in range(k):
            windows.append(padded[i:i + seq_len])
        windows = torch.stack(windows, dim=2)                # (seq_len, 1, k)

        emb_context = model.embedding(windows)               # (seq_len, 1, k, embed)
        emb_context = emb_context.squeeze(1)                 # (seq_len, k, embed)

        contrib = emb_context.norm(dim=2).cpu()              # (seq_len, k)

    plt.figure(figsize=(8, 6))
    plt.imshow(contrib, cmap="viridis", aspect="auto")
    plt.colorbar(label="Embedding Contribution (L2 Norm)")
    plt.title("K-gram MLP – Contribution of Each Window Position")
    plt.xlabel("Window Position (0=oldest → k-1=latest)")
    plt.ylabel("Token")

    xticks = [f"t-{(k-1)-i}" for i in range(k)]
    plt.xticks(range(k), xticks)

    token_texts = [enc.decode([tok]) for tok in tokens]
    plt.yticks(range(seq_len), token_texts, fontsize=6)

    out_path = "interpretability_outputs/kgram_contribution_heatmap.png"
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close()

    print(f"Saved {out_path}")




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
        from pico_llm import KGramMLPSeqModel
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