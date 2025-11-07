import os, json, time, itertools, numpy as np, torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_from_disk
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
import evaluate as hf_eval
from tqdm.auto import tqdm

BASE_MODEL = "NousResearch/Llama-2-7b-chat-hf"
ADAPTER_DIR = "/content/drive/MyDrive/a/checkpoints/llama7b_wikitext_selfdistill_lwarmup/epoch_002"
MTP_HEAD = "/content/drive/MyDrive/a/checkpoints/llama7b_wikitext_selfdistill_lwarmup/epoch_002/mtp_head.pt"
DATA_DIR = "/content/drive/MyDrive/a/data/processed_wikitext103/test"
S_LIST = [2, 3]
ACCEPTANCE_K = 10
NUM_SAMPLES = 10
MAX_NEW_TOKENS = 64
TEMPERATURE = 0.0
PPL_STRIDE = 512
PPL_BLOCK = 1024
PPL_MAX_TOKENS = 50000

class _MedusaHead(nn.Module):
    def __init__(self, d_model, vocab_size, lm_head=None, dtype=torch.float16):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_model, bias=True, dtype=dtype)
        use_bias_w2 = bool(getattr(lm_head, "bias", None) is not None) if lm_head is not None else False
        self.w2 = nn.Linear(d_model, vocab_size, bias=use_bias_w2, dtype=dtype)
    def forward(self, h):
        return self.w2(F.silu(self.w1(h)) + h)

class InferenceMedusa(nn.Module):
    def __init__(self, d_model, vocab_size, offsets, lm_head=None, dtype=torch.float16):
        super().__init__()
        self.offsets = [int(o) for o in sorted(offsets, key=lambda x:int(x))]
        self.heads = nn.ModuleDict({str(o): _MedusaHead(d_model, vocab_size, lm_head, dtype=dtype) for o in self.offsets})
    def forward(self, h, use_n=None):
        offs = self.offsets if use_n is None else self.offsets[:use_n]
        return [self.heads[str(o)](h) for o in offs]

def _topk_ids_per_head(head_logits, s_list):
    out = []
    for i, logits in enumerate(head_logits):
        s = s_list[i]
        out.append(torch.topk(logits, k=s, dim=-1).indices)
    return out

def _enumerate_nodes(s_list):
    nodes = []
    for depth in range(1, len(s_list)+1):
        for path in itertools.product(*[range(s_list[i]) for i in range(depth)]):
            nodes.append((depth, path))
    return nodes

def _build_chain(nodes):
    idx = {}
    chain = {}
    for i,(d,p) in enumerate(nodes):
        idx[(d,p)] = i
    for i,(d,p) in enumerate(nodes):
        if d==1: chain[i]=[]
        else:
            parent=(d-1,p[:-1])
            chain[i]=chain[idx[parent]]+[idx[parent]]
    return chain

def _cand_tokens(nodes, topk_ids_list, device):
    tok=[]
    for depth,path in nodes:
        c=path[-1]
        tok.append(topk_ids_list[depth-1][0,c].item())
    return torch.tensor(tok, device=device, dtype=torch.long)

def _mask_and_pos(prefix_len, nodes, chain, device, dtype):
    N=len(nodes)
    L=prefix_len+N
    neg=torch.finfo(dtype).min if torch.is_floating_point(torch.empty((),dtype=dtype)) else -1e4
    m=torch.full((1,1,L,L),neg,device=device,dtype=dtype)
    for q in range(prefix_len):
        m[0,0,q,:q+1]=0
    for i,_ in enumerate(nodes):
        qi=prefix_len+i
        m[0,0,qi,:prefix_len]=0
        for pj in chain[i]:
            m[0,0,qi,prefix_len+pj]=0
    pos=torch.arange(prefix_len,device=device).unsqueeze(0)
    depth_pos=[prefix_len+depth-1 for depth,_ in nodes]
    pos_all=torch.cat([pos,torch.tensor(depth_pos,device=device).unsqueeze(0)],dim=1)
    return m,pos_all

def _greedy_accept(tree_logits, nodes, cand_tokens, chain, k):
    ok={}
    for i in range(len(nodes)):
        topk=torch.topk(tree_logits[i],k=k,dim=-1).indices
        ok[i]=int(cand_tokens[i].item()) in topk
    best_depth=0
    best_idx=None
    idx_by_depth={}
    for i,(d,_) in enumerate(nodes):
        idx_by_depth.setdefault(d,[]).append(i)
    for d in range(1,max(d for d,_ in nodes)+1):
        any_ok=False
        for i in idx_by_depth[d]:
            good=ok[i] and all(ok[j] for j in chain[i])
            if good:
                any_ok=True
                best_idx=i
        if any_ok: best_depth=d
        else: break
    if best_depth==0: return 0,[]
    path=[]
    cur=best_idx
    while True:
        path.append(cur)
        if len(chain[cur])==0: break
        cur=chain[cur][-1]
    path=list(reversed(path))
    seq=[int(cand_tokens[i].item()) for i in path]
    return best_depth,seq

class TreeJudgeDecoder:
    def __init__(self, base_model, adapter_dir, mtp_path, tokenizer_src=None, s_list=None, acceptance_k=5, max_new_tokens=64, temperature=0.0, quant="4bit"):
        self.device="cuda" if torch.cuda.is_available() else "cpu"
        self.s_list=list(s_list)
        self.acceptance_k=int(acceptance_k)
        self.max_new_tokens=int(max_new_tokens)
        self.temperature=float(temperature)
        qcfg=None
        if quant=="4bit":
            qcfg=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type="nf4",bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,bnb_4bit_use_double_quant=True)
        self.tok=AutoTokenizer.from_pretrained(tokenizer_src or base_model, use_fast=True, trust_remote_code=True)
        if self.tok.pad_token is None: self.tok.pad_token=self.tok.eos_token
        self.tok.padding_side="left"
        base=AutoModelForCausalLM.from_pretrained(base_model, quantization_config=qcfg, device_map="auto", trust_remote_code=True)
        self.model=PeftModel.from_pretrained(base, adapter_dir)
        self.model.eval()
        d_model=int(self.model.config.hidden_size)
        vocab=int(self.model.config.vocab_size)
        lm_head=getattr(self.model,"lm_head",None)
        sd=torch.load(mtp_path,map_location=self.device)
        if "state_dict" in sd and isinstance(sd["state_dict"],dict): sd=sd["state_dict"]
        keys=[k for k in sd.keys() if k.startswith("heads.")]
        offs=sorted({k.split(".")[1] for k in keys})
        self.medusa=InferenceMedusa(d_model,vocab,offs,lm_head,dtype=self.model.lm_head.weight.dtype).to(self.device)
        self.medusa.load_state_dict(sd,strict=True)
        if len(self.s_list) > len(self.medusa.offsets):
            self.s_list = self.s_list[:len(self.medusa.offsets)]
        self.model_dtype=self.model.lm_head.weight.dtype
        self.model.to(self.model_dtype)
        self.medusa.to(self.model_dtype)

    @torch.no_grad()
    def generate(self, prompt):
        ids=self.tok(prompt,return_tensors="pt").input_ids.to(self.device)
        out=[]
        medusa_accept=0
        backbone_accept=0
        steps=0
        total_tokens_generated=0
        while total_tokens_generated < self.max_new_tokens:
            steps+=1
            o=self.model(input_ids=ids,output_hidden_states=True,use_cache=False,return_dict=True)
            h=o.hidden_states[-1][:, -1, :].to(self.model_dtype)
            base_logits=o.logits[:, -1, :]
            if self.temperature>0:
                probs=torch.softmax(base_logits/self.temperature,dim=-1)
                base_next=torch.multinomial(probs,num_samples=1)
            else:
                base_next=torch.argmax(base_logits,dim=-1,keepdim=True)
            head_logits=self.medusa(h,use_n=len(self.s_list))
            topk=_topk_ids_per_head(head_logits,self.s_list)
            nodes=_enumerate_nodes(self.s_list)
            chain=_build_chain(nodes)
            cand=_cand_tokens(nodes,topk,self.device)
            pref=ids.size(1)
            attn,pos=_mask_and_pos(pref,nodes,chain,self.device,self.model_dtype)
            all_ids=torch.cat([ids,cand.view(1,-1)],dim=1)
            emb_dev=self.model.get_input_embeddings().weight.device
            attn=attn.to(emb_dev)
            pos=pos.to(emb_dev)
            embeds=self.model.get_input_embeddings()(all_ids.to(emb_dev))
            logits=self.model(inputs_embeds=embeds,attention_mask=attn,position_ids=pos,use_cache=False,return_dict=True).logits
            tree_slice=logits[0,pref:,:]
            depth,seq=_greedy_accept(tree_slice,nodes,cand,chain,self.acceptance_k)
            if depth==0:
                ids=torch.cat([ids,base_next],dim=1)
                new_token_id=int(base_next.item())
                out.append(new_token_id)
                backbone_accept+=1
                total_tokens_generated+=1
            else:
                seq=seq[:self.max_new_tokens-total_tokens_generated]
                if not seq:
                    ids=torch.cat([ids,base_next],dim=1)
                    new_token_id=int(base_next.item())
                    out.append(new_token_id)
                    backbone_accept+=1
                    total_tokens_generated+=1
                else:
                    add=torch.tensor(seq,device=self.device,dtype=torch.long).view(1,-1)
                    ids=torch.cat([ids,add],dim=1)
                    out.extend(seq)
                    medusa_accept+=len(seq)
                    total_tokens_generated+=len(seq)
            if ids[0,-1].item()==self.tok.eos_token_id: break
        return self.tok.decode(out,skip_special_tokens=True), {
            "medusa_tokens": medusa_accept,
            "backbone_tokens": backbone_accept,
            "total_tokens": len(out),
            "steps": steps,
        }

def _load_backbone(base_model, adapter_dir):
    qcfg=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type="nf4",bnb_4bit_compute_dtype=torch.float16,bnb_4bit_use_double_quant=True)
    tok=AutoTokenizer.from_pretrained(base_model,use_fast=True)
    if tok.pad_token is None: tok.pad_token=tok.eos_token
    base=AutoModelForCausalLM.from_pretrained(base_model,device_map="auto",quantization_config=qcfg,trust_remote_code=True)
    model=PeftModel.from_pretrained(base,adapter_dir)
    model.eval()
    return tok,model

@torch.no_grad()
def _greedy_backbone(tok, model, prompt, max_new_tokens):
    dev=next(model.parameters()).device
    ids=tok(prompt,return_tensors="pt").input_ids.to(dev)
    t0=time.perf_counter()
    out=model.generate(input_ids=ids,do_sample=False,max_new_tokens=max_new_tokens,pad_token_id=tok.eos_token_id)
    dt=time.perf_counter()-t0
    gen=tok.decode(out[0, ids.size(1):], skip_special_tokens=True)
    return gen, out.size(1)-ids.size(1), dt

@torch.no_grad()
def perplexity(tok, model, texts, stride=512, block=1024, max_tokens=50000):
    dev=next(model.parameters()).device
    text="\n\n".join(texts)
    ids=tok(text,return_tensors="pt",truncation=True,max_length=max_tokens).input_ids.to(dev)
    nlls=[]
    total=0
    for i in tqdm(range(0,ids.size(1),stride),desc="Computing Perplexity"):
        begin=max(i+stride-block,0)
        end=min(i+stride,ids.size(1))
        trg_len=end-i
        if trg_len<=0: continue
        inp=ids[:,begin:end]
        tgt=inp.clone()
        tgt[:,:-trg_len]=-100
        out=model(input_ids=inp,labels=tgt)
        nlls.append(out.loss*trg_len)
        total+=trg_len
    if total==0: return 0.0
    return torch.exp(torch.stack(nlls).sum()/total).item()

def _decode_texts(ds, tok, n):
    cols=set(ds.column_names)
    if "text" in cols:
        raw=ds["text"]
    elif "document" in cols:
        raw=ds["document"]
    elif "input_ids" in cols:
        ids_list=ds["input_ids"]
        raw=[tok.decode(list(x),skip_special_tokens=True) for x in ids_list]
    else:
        raise ValueError(str(cols))
    raw=[t for t in raw if isinstance(t,str) and t.strip()]
    return raw[:n]

def main():
    tok_tmp=AutoTokenizer.from_pretrained(BASE_MODEL,use_fast=True)
    if tok_tmp.pad_token is None: tok_tmp.pad_token=tok_tmp.eos_token
    ds=load_from_disk(DATA_DIR)
    texts=_decode_texts(ds,tok_tmp,NUM_SAMPLES)
    decoder=TreeJudgeDecoder(
        BASE_MODEL,
        ADAPTER_DIR,
        MTP_HEAD,
        tokenizer_src=None,
        s_list=S_LIST,
        acceptance_k=ACCEPTANCE_K,
        max_new_tokens=MAX_NEW_TOKENS,
        temperature=TEMPERATURE,
        quant="4bit"
    )
    tok_base,base=_load_backbone(BASE_MODEL,ADAPTER_DIR)
    preds_base=[]
    preds_med=[]
    base_tps_list=[]
    med_tps_list=[]
    base_lat_list=[]
    med_lat_list=[]
    total_medusa_tokens=0
    total_backbone_tokens=0
    total_steps=0
    for p in tqdm(texts,desc="Evaluating"):
        g_b,b_tok,b_dt=_greedy_backbone(tok_base,base,p,MAX_NEW_TOKENS)
        t0=time.perf_counter()
        g_m, stats = decoder.generate(p)
        m_dt=time.perf_counter()-t0
        preds_base.append(g_b.strip())
        preds_med.append(g_m.strip())
        base_tps_list.append(b_tok/max(b_dt,1e-6))
        med_tps_list.append(stats["total_tokens"]/max(m_dt,1e-6))
        base_lat_list.append(b_dt)
        med_lat_list.append(m_dt)
        total_medusa_tokens+=stats["medusa_tokens"]
        total_backbone_tokens+=stats["backbone_tokens"]
        total_steps+=stats["steps"]
    bleu=hf_eval.load("sacrebleu").compute(predictions=preds_med,references=[[r] for r in preds_base])["score"]
    ppl=perplexity(tok_base,base,texts,stride=PPL_STRIDE,block=PPL_BLOCK,max_tokens=PPL_MAX_TOKENS)
    total_gen_tokens=total_medusa_tokens+total_backbone_tokens
    out={
        "config":{
            "base_model":BASE_MODEL,
            "adapter":ADAPTER_DIR,
            "mtp_head":MTP_HEAD,
            "s_list":S_LIST,
            "num_samples":NUM_SAMPLES,
            "max_new_tokens":MAX_NEW_TOKENS,
            "acceptance_k":ACCEPTANCE_K
        },
        "quality_metrics":{
            "bleu_vs_backbone_fidelity":float(bleu),
            "perplexity":float(ppl)
        },
        "aggregate_stats":{
            "avg_acceptance_length":float(total_gen_tokens/max(total_steps,1)),
            "mtp_token_ratio":float(total_medusa_tokens/max(total_gen_tokens,1)),
            "total_medusa_tokens":int(total_medusa_tokens),
            "total_backbone_tokens":int(total_backbone_tokens),
            "total_steps":int(total_steps)
        },
        "speed":{
            "base_tps_mean":float(np.mean(base_tps_list)),
            "medusa_tps_mean":float(np.mean(med_tps_list)),
            "speedup":float(np.mean(med_tps_list)/max(np.mean(base_tps_list),1e-9))
        },
        "latency":{
            "base_mean_s":float(np.mean(base_lat_list)),
            "medusa_mean_s":float(np.mean(med_lat_list)),
            "speedup_latency":float(np.mean(base_lat_list)/max(np.mean(med_lat_list),1e-9))
        }
    }
    print(json.dumps(out,indent=2))

if __name__=="__main__":
    main()
