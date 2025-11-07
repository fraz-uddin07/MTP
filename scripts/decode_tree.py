import os, json, itertools, torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

class _MedusaSingleHead(nn.Module):
    def __init__(self, d_model, vocab_size, lm_head=None):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_model, bias=True)
        use_bias_w2 = bool(getattr(lm_head, "bias", None) is not None) if lm_head is not None else False
        self.w2 = nn.Linear(d_model, vocab_size, bias=use_bias_w2)
    def forward(self, h):
        return self.w2(F.silu(self.w1(h)) + h)

class InferenceMedusa(nn.Module):
    def __init__(self, d_model, vocab_size, offsets, lm_head=None):
        super().__init__()
        self.offsets = sorted(int(o) for o in offsets)
        self.heads = nn.ModuleDict({str(o): _MedusaSingleHead(d_model, vocab_size, lm_head) for o in self.offsets})
    def forward(self, h):
        outs = []
        for o in self.offsets:
            outs.append(self.heads[str(o)](h))
        return outs

def _topk_ids_per_head(head_logits, s_list):
    out = []
    for k, logits in enumerate(head_logits):
        s = s_list[k]
        ids = torch.topk(logits, k=s, dim=-1).indices
        out.append(ids)
    return out

def _enumerate_nodes(s_list):
    nodes = []
    for depth in range(1, len(s_list)+1):
        for path in itertools.product(*[range(s_list[i]) for i in range(depth)]):
            nodes.append((depth, path))
    return nodes

def _build_branch_index(nodes):
    by_node, chain = {}, {}
    for idx, (depth, path) in enumerate(nodes):
        by_node[(depth, path)] = idx
        if depth == 1: chain[idx] = []
        else:
            parent = (depth-1, path[:-1])
            chain[idx] = chain[by_node[parent]] + [by_node[parent]]
    return by_node, chain

def _build_tree_tokens(nodes, topk_ids_list, device):
    tok = []
    for depth, path in nodes:
        choice = path[-1]
        tok.append(topk_ids_list[depth-1][0, choice].item())
    return torch.tensor(tok, device=device, dtype=torch.long)

def _build_masks_and_positions(prefix_len, nodes, chain, device, dtype):
    N = len(nodes)
    L = prefix_len + N
    neg = torch.finfo(dtype).min if dtype.is_floating_point else -1e4
    m = torch.full((1, 1, L, L), neg, device=device, dtype=dtype)
    for q in range(prefix_len):
        m[0, 0, q, :q+1] = 0
    for i, _ in enumerate(nodes):
        qi = prefix_len + i
        m[0, 0, qi, :prefix_len] = 0
        for pj in chain[i]:
            m[0, 0, qi, prefix_len + pj] = 0
    pos = torch.arange(prefix_len, device=device).unsqueeze(0)
    depth_pos = [prefix_len + depth - 1 for depth, _ in nodes]
    pos_all = torch.cat([pos, torch.tensor(depth_pos, device=device).unsqueeze(0)], dim=1)
    return m, pos_all

def _greedy_accept_from_logits_topk(tree_logits, nodes, cand_tokens, chain, acceptance_k=5):
    accepted = {}
    for i, _ in enumerate(nodes):
        logits = tree_logits[i]
        top_k_preds = torch.topk(logits, k=acceptance_k, dim=-1).indices
        accepted[i] = (int(cand_tokens[i].item()) in top_k_preds)
    best_depth = 0
    best_last_index = None
    idx_by_depth = {}
    for i, (depth, _) in enumerate(nodes):
        idx_by_depth.setdefault(depth, []).append(i)
    for depth in range(1, max(d for d, _ in nodes)+1):
        ok_any = False
        for i in idx_by_depth[depth]:
            ok = True
            for j in chain[i]:
                if not accepted[j]: ok = False; break
            if ok and accepted[i]:
                ok_any = True
                best_last_index = i
        if ok_any: best_depth = depth
        else: break
    if best_depth == 0: return 0, []
    seq = []
    curr = best_last_index
    path = []
    while curr is not None:
        path.append(curr)
        if len(chain[curr]) == 0: break
        curr = chain[curr][-1]
    path = list(reversed(path))
    for i in path:
        seq.append(int(cand_tokens[i].item()))
    return best_depth, seq

class TreeDecoder:
    def __init__(self, base_model_id, adapter_dir, mtp_ckpt_path, tokenizer_path=None, quantization="4bit"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tok = AutoTokenizer.from_pretrained(tokenizer_path or adapter_dir or base_model_id, use_fast=True, trust_remote_code=True)
        if self.tok.pad_token is None: self.tok.pad_token = self.tok.eos_token
        self.tok.padding_side = "left"
        quant_cfg = None
        if quantization == "4bit":
            quant_cfg = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                           bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
                                           bnb_4bit_use_double_quant=True)
        base = AutoModelForCausalLM.from_pretrained(base_model_id, quantization_config=quant_cfg, device_map="auto", trust_remote_code=True)
        self.model = PeftModel.from_pretrained(base, adapter_dir)
        self.model.eval()
        d_model = getattr(self.model.config, "hidden_size", None)
        vocab = getattr(self.model.config, "vocab_size", None)
        if d_model is None or vocab is None: raise RuntimeError("hidden_size/vocab_size not found")
        lm_head = getattr(self.model, "lm_head", None)
        sd_raw = torch.load(mtp_ckpt_path, map_location=self.device)
        sd = sd_raw if isinstance(sd_raw, dict) else sd_raw.get("state_dict", sd_raw)
        keys = [k for k in sd.keys() if k.startswith("heads.")]
        idxs = sorted({k.split(".")[1] for k in keys})
        if not idxs: raise RuntimeError("No heads.* keys in mtp checkpoint")
        try:
            _ = sd[f"heads.{idxs[0]}.w1.weight"]
        except:
            raise RuntimeError("Expected keys like heads.<off>.w1.weight")
        self.medusa = InferenceMedusa(d_model, vocab, offsets=idxs, lm_head=lm_head).to(self.device)
        res = self.medusa.load_state_dict(sd, strict=True)
        self.model_dtype = self.model.lm_head.weight.dtype
        self.model.to(self.model_dtype)
        self.medusa.to(self.model_dtype)
        self.medusa.eval()

    @torch.no_grad()
    def generate(self, prompt, max_new_tokens=64, s_list=None, acceptance_k=5, temperature=0.0, verbose=False):
        if s_list is None: raise ValueError("s_list required")
        if len(s_list) > len(self.medusa.offsets): raise ValueError("s_list longer than available heads")
        ids = self.tok(prompt, return_tensors="pt").input_ids.to(self.device)
        out = []
        medusa_tokens_accepted = 0
        backbone_tokens_accepted = 0
        total_steps = 0
        for _ in range(max_new_tokens):
            total_steps += 1
            base_out = self.model(input_ids=ids, output_hidden_states=True, use_cache=False, return_dict=True)
            last_h = base_out.hidden_states[-1][:, -1, :].to(self.model_dtype)
            base_logits = base_out.logits[:, -1, :]
            if temperature and temperature > 0:
                probs = torch.softmax(base_logits / temperature, dim=-1)
                base_next = torch.multinomial(probs, num_samples=1)
            else:
                base_next = torch.argmax(base_logits, dim=-1, keepdim=True)
            head_logits_all = self.medusa(last_h)
            head_logits = head_logits_all[:len(s_list)]
            topk = _topk_ids_per_head(head_logits, s_list)
            nodes = _enumerate_nodes(s_list)
            by_node, chain = _build_branch_index(nodes)
            cand_tokens = _build_tree_tokens(nodes, topk, self.device)
            prefix_len = ids.size(1)
            if verbose:
                print(f"\n--- Step {total_steps} Proposals ---")
                base_next_token_id = int(base_next.item())
                print(f"  Backbone LM Head (Top-1): {self.tok.decode(base_next_token_id, skip_special_tokens=True)!r}")
                proposals_by_depth = {}
                for node_idx, (depth, path) in enumerate(nodes):
                    if depth not in proposals_by_depth: proposals_by_depth[depth] = []
                    token_id = cand_tokens[node_idx].item()
                    proposals_by_depth[depth].append(f"  MTP Path({ '->'.join(map(str, path)) }): {self.tok.decode(token_id, skip_special_tokens=True)!r}")
                for depth, proposals in sorted(proposals_by_depth.items()):
                    print(f"MTP Depth {depth}:"); print("\n".join(proposals)); print("------------------------------")
            attn_mask, pos_ids = _build_masks_and_positions(prefix_len, nodes, chain, self.device, self.model_dtype)
            all_ids = torch.cat([ids, cand_tokens.view(1, -1)], dim=1)
            embedding_device = self.model.get_input_embeddings().weight.device
            embeds = self.model.get_input_embeddings()(all_ids.to(embedding_device))
            logits_all = self.model(inputs_embeds=embeds, attention_mask=attn_mask, position_ids=pos_ids, use_cache=False, return_dict=True).logits
            tree_logits_slice = logits_all[0, prefix_len:, :]
            depth_accepted, seq = _greedy_accept_from_logits_topk(tree_logits_slice, nodes, cand_tokens, chain, acceptance_k=acceptance_k)
            if depth_accepted == 0:
                ids = torch.cat([ids, base_next], dim=1)
                new_token_id = int(base_next.item())
                out.append(new_token_id)
                backbone_tokens_accepted += 1
                if verbose: print(f"Judge (k={acceptance_k}): REJECTED. Using 1 Backbone token: {self.tok.decode(new_token_id, skip_special_tokens=True)!r}")
            else:
                add = torch.tensor(seq, device=self.device, dtype=torch.long).view(1, -1)
                ids = torch.cat([ids, add], dim=1)
                out.extend(seq)
                medusa_tokens_accepted += len(seq)
                if verbose:
                    decoded_seq = [self.tok.decode(t, skip_special_tokens=True) for t in seq]
                    print(f"Judge (k={acceptance_k}): ACCEPTED. Using {len(seq)} MTP tokens: {decoded_seq}")
            if ids[0, -1].item() == self.tok.eos_token_id: break
        if verbose:
            print("\n--- Generation Stats ---")
            print(f"Total steps (forward passes): {total_steps}")
            print(f"Tokens from Medusa heads (accepted): {medusa_tokens_accepted}")
            print(f"Tokens from Backbone (fallback): {backbone_tokens_accepted}")
            total_gen = medusa_tokens_accepted + backbone_tokens_accepted
            print(f"Total tokens generated: {total_gen}")
            if total_steps > 0: print(f"Average tokens per step (speedup): {total_gen / total_steps:.2f}")
            print("------------------------\n")
        return self.tok.decode(out, skip_special_tokens=True), ids

if __name__ == "__main__":
    base = "NousResearch/Llama-2-7b-chat-hf"
    adapter_dir = "/content/drive/MyDrive/a/checkpoints/llama7b_wikitext_selfdistill_lwarmup/epoch_002"
    mtp_path    = "/content/drive/MyDrive/a/checkpoints/llama7b_wikitext_selfdistill_lwarmup/epoch_002/mtp_head.pt"
    decoder = TreeDecoder(base_model_id=base, adapter_dir=adapter_dir, mtp_ckpt_path=mtp_path, tokenizer_path=adapter_dir, quantization="4bit")
    text, _ = decoder.generate(
        "Write a short summary about Fourier transforms:",
        max_new_tokens=64,
        s_list=[2,3],
        acceptance_k=20,
        temperature=0.0,
        verbose=True
    )
    print(text)