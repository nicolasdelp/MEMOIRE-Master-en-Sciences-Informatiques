import argparse
import re
import torch
from pathlib import Path


def load_state_dict(path):
    ckpt = torch.load(str(path), map_location='cpu', weights_only=False)
    if isinstance(ckpt, dict) and 'state_dict' in ckpt:
        return ckpt['state_dict']
    return ckpt

def merge_peft(sd, alpha):
    """
    For each base_layer.weight, find the corresponding LoRAs
    and merge them into a new key without the .base_layer prefix.
    """
    new_sd = {}

    # Identify all LoRA-wrapped modules (by their "root" path)
    lora_roots = set()
    for k in sd.keys():
        m = re.match(r'(.+?)\.lora_A\.default\.weight$', k)
        if m:
            lora_roots.add(m.group(1))

    print(f"Found {len(lora_roots)} LoRA-wrapped modules")

    merged_count = 0
    base_renamed_count = 0
    skipped_count = 0

    # Keep a record of keys to exclude (the original LoRA components)
    keys_to_skip = set()

    for root in lora_roots:
        a_key = f"{root}.lora_A.default.weight"
        b_key = f"{root}.lora_B.default.weight"
        base_w_key = f"{root}.base_layer.weight"
        base_b_key = f"{root}.base_layer.bias"
        target_w_key = f"{root}.weight"
        target_b_key = f"{root}.bias"

        if base_w_key not in sd:
            print(f"  [WARN] No base_layer.weight for {root}")
            skipped_count += 1
            continue

        A = sd[a_key].float()
        B = sd[b_key].float()
        W = sd[base_w_key].float()
        rank = A.shape[0]
        scaling = alpha / rank

        delta = B @ A

        if W.dim() > 2:
            delta = delta.reshape(W.shape)

        W_merged = W + delta * scaling
        W_merged = W_merged.to(sd[base_w_key].dtype)

        new_sd[target_w_key] = W_merged
        merged_count += 1

        # Copy the bias if it exists (not modified by LoRA)
        if base_b_key in sd:
            new_sd[target_b_key] = sd[base_b_key]
            base_renamed_count += 1

        # Mark the old keys as to be excluded from the final dict
        keys_to_skip.add(a_key)
        keys_to_skip.add(b_key)
        keys_to_skip.add(base_w_key)
        if base_b_key in sd:
            keys_to_skip.add(base_b_key)

    print(f"Merged {merged_count} LoRA layers")
    print(f"Renamed {base_renamed_count} biases base_layer.bias -> bias")

    # Copy all other keys that are not LoRA
    other_count = 0
    for k, v in sd.items():
        if k in keys_to_skip:
            continue
        if k in new_sd:
            continue  # déjà mergé
        new_sd[k] = v
        other_count += 1
    print(f"Copied {other_count} other keys")

    return new_sd


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True, help='Finetuned checkpoint avec LoRA (format PEFT)')
    parser.add_argument('--output', required=True, help='Merged checkpoint with LoRA weights applied (format standard)')
    parser.add_argument('--alpha', type=int, default=1, help='alpha LoRA. By default PEFT uses alpha=rank=1')
    parser.add_argument('--drop-lpips', action='store_true', default=True, help='Delete the lpips.* keys from the final checkpoint')
    args = parser.parse_args()

    print(f"Loading: {args.input}")
    sd = load_state_dict(args.input)
    print(f"{len(sd)} keys initial\n")

    print(f"Merging LoRA with alpha={args.alpha}...\n")
    merged = merge_peft(sd, alpha=args.alpha)

    if args.drop_lpips:
        before = len(merged)
        merged = {k: v for k, v in merged.items() if not k.startswith('lpips')}
        print(f"\nFiltered {before - len(merged)} keys lpips.*")

    print(f"\nFinal: {len(merged)} keys")
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    torch.save({'state_dict': merged}, args.output)
    print("Done!\n")

    # Sanity check : check if some expected keys are present in the merged dict
    expected_samples = [
        'model.diffusion_model.time_embed.0.weight',
        'model.diffusion_model.time_embed.2.weight',
    ]
    print("Sanity check :")
    for k in expected_samples:
        status = "✅" if k in merged else "❌ MISSING"
        print(f"  {status}  {k}")