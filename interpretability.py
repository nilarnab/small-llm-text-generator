import torch

import matplotlib.pyplot as plt
import argparse
import os
import tiktoken
import math


# manual import since pico_llm.py has a dash in its name (can't do normal import)
import importlib.util, sys
spec = importlib.util.spec_from_file_location("pico_llm", os.path.join(os.getcwd(), "pico_llm.py"))
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

    print("Running forward pass through the transformer blocks...")
    with torch.no_grad():
        x = model.embedding(tokens)
        seq_len = x.size(0)

        pos = torch.arange(seq_len, device=device).unsqueeze(1)
        x = x + model.position_embedding(pos)

        for idx, block in enumerate(model.blocks):
            x_norm = block.norm_1(x)

            attn_out, attn_weights = block.attn1(
                x_norm, x_norm, x_norm,
                need_weights=True,
                average_attn_weights=False
            )

            attention_per_block.append(attn_weights[0].cpu())

            x = x + attn_out
            x = x + block.mlp_g_1(block.norm_2(x))

    token_labels = [enc.decode([t]) for t in token_ids]

    print("Now, time to save hte visualisations")

    for block_i, attn_heads in enumerate(attention_per_block):
        num_heads = attn_heads.size(0)
        grid_side = math.ceil(math.sqrt(num_heads))

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

        for h in range(num_heads, grid_side * grid_side):
            r, c = divmod(h, grid_side)
            fig.delaxes(axes[r][c])

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



def visualize_kgram(model, enc, prompt, device):

    os.makedirs("interpretability_outputs", exist_ok=True)
    token_ids = enc.encode(prompt)
    tokens = torch.tensor(token_ids, dtype=torch.long, device=device).unsqueeze(1)

    seq_len = len(token_ids)
    k = model.k

    model.eval()

    with torch.no_grad():
        _ = model(tokens)

        pad = torch.zeros((k - 1, 1), dtype=torch.long, device=device)
        padded = torch.cat([pad, tokens], dim=0)

        window_list = []
        for i in range(k):
            window_list.append(padded[i:i + seq_len])
        windows = torch.stack(window_list, dim=2)     # shape: (seq_len, 1, k)

        emb = model.embedding(windows)                # (seq_len, 1, k, embed_dim)
        emb = emb.squeeze(1)                          # (seq_len, k, embed_dim)

        contrib = emb.norm(dim=2).cpu()

    plt.figure(figsize=(8, 6))
    plt.imshow(contrib, cmap="viridis", aspect="auto")
    plt.colorbar(label="Embedding Contribution (L2 Norm)")

    plt.title("K-gram MLP – Window Position Influence")
    plt.xlabel("Window Position")
    plt.ylabel("Tokens in Prompt")

    x_labels = [f"t-{(k - 1) - i}" for i in range(k)]
    plt.xticks(range(k), x_labels)

    decoded_tokens = [enc.decode([tid]) for tid in token_ids]
    plt.yticks(range(seq_len), decoded_tokens, fontsize=6)

    out_path = "interpretability_outputs/kgram_contribution_heatmap.png"
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()

    print("saving the file:", out_path)





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
    print(f"looaded checkpoint: {args.checkpoint}")

    if args.mode == "attention":
        visualize_attention(model, enc, args.prompt, args.device)
    elif args.mode == "kgram":
        visualize_kgram(model, enc, args.prompt, args.device)
    else:
        print("unnknown mode")


if __name__ == "__main__":
    main()