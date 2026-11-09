"""LumiNet v2: ~99k params. Подкоманды: label | train | match.
pip install torch python-chess numpy ; Stockfish нужен только для label и match."""
import argparse, json, math, random, time
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
import chess, chess.engine

# ---------- кодирование: доска всегда глазами стороны, которая ходит ----------
def encode(board):
    b = board if board.turn else board.mirror()          # чёрные -> зеркало, "мы" всегда белые
    t = np.zeros(64, np.int64)
    for sq, p in b.piece_map().items():
        t[sq] = p.piece_type + (0 if p.color else 6)     # 0 пусто, 1-6 свои, 7-12 чужие
    g = np.zeros(12, np.float32)
    g[0], g[1] = b.has_kingside_castling_rights(True), b.has_queenside_castling_rights(True)
    g[2], g[3] = b.has_kingside_castling_rights(False), b.has_queenside_castling_rights(False)
    if b.ep_square is not None: g[4 + b.ep_square % 8] = 1
    return t, g

def midx(m, flip):
    f, t = m.from_square, m.to_square
    if flip: f ^= 56; t ^= 56
    return f * 64 + t

def legal_idx(board):                                    # idx -> ход (превращения только в ферзя)
    flip = not board.turn
    return {midx(m, flip): m for m in board.legal_moves if not m.promotion or m.promotion == chess.QUEEN}

# ---------- модель ----------
class RMS(nn.Module):
    def __init__(s, d): super().__init__(); s.g = nn.Parameter(torch.ones(d))
    def forward(s, x): return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6) * s.g

class Block(nn.Module):
    def __init__(s, d, h, ff):
        super().__init__()
        s.h, s.n1, s.n2 = h, RMS(d), RMS(d)
        s.qkv, s.o = nn.Linear(d, 3 * d, bias=False), nn.Linear(d, d, bias=False)
        s.rb = nn.Parameter(torch.zeros(h, 226))         # смещение внимания по (dx,dy) на доске, свой на голову
        s.ff = nn.Sequential(nn.Linear(d, ff), nn.GELU(), nn.Linear(ff, d))
    def forward(s, x, rel):
        B, L, D = x.shape
        q, k, v = s.qkv(s.n1(x)).view(B, L, 3, s.h, D // s.h).permute(2, 0, 3, 1, 4)
        a = F.scaled_dot_product_attention(q, k, v, attn_mask=s.rb[:, rel].unsqueeze(0))  # без причинной маски
        x = x + s.o(a.transpose(1, 2).reshape(B, L, D))
        return x + s.ff(s.n2(x))

class LumiNet(nn.Module):
    def __init__(s, d=64, h=4, ff=192, nblk=2, maxloop=12):
        super().__init__()
        s.cfg, s.maxloop = dict(d=d, h=h, ff=ff, nblk=nblk, maxloop=maxloop), maxloop
        s.emb, s.glob = nn.Embedding(13, d), nn.Linear(12, d, bias=False)
        s.pos, s.loop = nn.Parameter(torch.randn(65, d) * .02), nn.Parameter(torch.zeros(maxloop, d))
        s.blocks = nn.ModuleList(Block(d, h, ff) for _ in range(nblk))   # ОДИН набор весов, крутится T раз
        s.nin, s.nout = RMS(d), RMS(d)
        s.pf, s.pt = nn.Linear(d, 32, bias=False), nn.Linear(d, 32, bias=False)  # указатель from->to
        s.v = nn.Sequential(nn.Linear(2 * d, 32), nn.GELU(), nn.Linear(32, 1))
        f, r = torch.arange(64) % 8, torch.arange(64) // 8
        rel = torch.full((65, 65), 225)
        rel[:64, :64] = (f[:, None] - f[None] + 7) * 15 + (r[:, None] - r[None] + 7)
        s.register_buffer("rel", rel)
    def heads(s, x):
        x = s.nout(x); sq = x[:, :64]
        pol = (s.pf(sq) @ s.pt(sq).transpose(1, 2)).flatten(1) / math.sqrt(32)   # (B,4096)
        val = torch.tanh(s.v(torch.cat([sq.mean(1), x[:, 64]], -1))).squeeze(-1)
        return pol, val
    def forward(s, t, g, T, all_iters=False):
        e = torch.cat([s.emb(t), s.glob(g).unsqueeze(1)], 1) + s.pos
        x, outs = e, []
        for i in range(T):
            x = s.nin(x) + e + s.loop[min(i, s.maxloop - 1)]   # инъекция входа на каждой итерации
            for b in s.blocks: x = b(x, s.rel)
            if all_iters or i == T - 1: outs.append(s.heads(x))
        return outs

def load(path, dev="cpu"):
    ck = torch.load(path, map_location=dev)
    net = LumiNet(**ck["cfg"]).to(dev); net.load_state_dict(ck["model"]); return net.eval()

@torch.no_grad()
def run(net, boards, T, bs=2048):
    dev = next(net.parameters()).device; P, V = [], []
    for i in range(0, len(boards), bs):
        enc = [encode(b) for b in boards[i:i + bs]]
        t = torch.from_numpy(np.stack([e[0] for e in enc])).to(dev)
        g = torch.from_numpy(np.stack([e[1] for e in enc])).to(dev)
        (p, v), = net(t, g, T); P.append(p); V.append(v)
    return torch.cat(P), torch.cat(V)

# ---------- поиск: сеть выбирает кандидатов, 2-плай полный перебор, листья оценивает value-голова ----------
@torch.no_grad()
def pick_move(net, board, T=12, topk=10):
    legal = legal_idx(board)
    if len(legal) == 1: return next(iter(legal.values()))
    pol, _ = run(net, [board], T)
    ids = list(legal); order = pol[0, ids].argsort(descending=True)[:topk].tolist()
    cand = [legal[ids[i]] for i in order]
    score, leaves, owner = [2.0] * len(cand), [], []
    for ci, m in enumerate(cand):
        board.push(m)
        if board.is_checkmate(): board.pop(); return m
        if board.is_game_over(): score[ci] = 0.0
        else:
            for r in board.legal_moves:
                if r.promotion and r.promotion != chess.QUEEN: continue
                board.push(r)
                if board.is_checkmate(): score[ci] = -1.0
                elif board.is_game_over(): score[ci] = min(score[ci], 0.0)
                else: leaves.append(board.copy(stack=False)); owner.append(ci)
                board.pop()
        board.pop()
    if leaves:
        _, v = run(net, leaves, T)                        # значение с точки зрения нас (мы снова ходим)
        for ci, val in zip(owner, v.tolist()): score[ci] = min(score[ci], val)
    return cand[max(range(len(cand)), key=lambda i: score[i])]

# ---------- разметка Stockfish: лучший ход + оценка; + внепартийные позиции ----------
def _label_chunk(job):
    items, sf, depth = job
    eng = chess.engine.SimpleEngine.popen_uci(sf); out = []
    for fen, src in items:
        b = chess.Board(fen)
        if b.is_game_over(): continue
        info = eng.analyse(b, chess.engine.Limit(depth=depth))
        cp = info["score"].pov(b.turn).score(mate_score=1500)
        out.append((fen, info["pv"][0].uci(), max(-1500, min(1500, cp)), src))
    eng.quit(); return out

def label(a):
    from concurrent.futures import ProcessPoolExecutor
    pool, rng, items = json.load(open(a.data)), random.Random(0), []
    for src, (fen, _) in enumerate(pool):                 # ход Карлсена не нужен: цель - ход Stockfish
        items.append((fen, src)); b = chess.Board(fen)
        for _ in range(a.variants - 1):                   # +1, +2 случайных хода: позиции, где надо наказывать ошибки
            if b.is_game_over(): break
            b.push(rng.choice(list(b.legal_moves))); items.append((b.fen(), src))
    jobs = [(items[i:i + 500], a.sf, a.depth) for i in range(0, len(items), 500)]
    rows = []
    with ProcessPoolExecutor(a.workers) as ex:
        for k, r in enumerate(ex.map(_label_chunk, jobs)):
            rows += r; print(f"labeled {len(rows)}/{len(items)}", flush=True)
    json.dump(rows, open(a.out, "w"))

# ---------- обучение: случайная глубина T + глубокая супервизия на каждой итерации ----------
def build(rows, dev):
    N = len(rows)
    T, G, P, V = np.zeros((N, 64), np.int8), np.zeros((N, 12), np.float32), np.zeros(N, np.int64), np.zeros(N, np.float32)
    L = -np.ones((N, 256), np.int16)
    for i, (fen, uci, cp, _) in enumerate(rows):
        b = chess.Board(fen); T[i], G[i] = encode(b)
        P[i] = midx(chess.Move.from_uci(uci), not b.turn); V[i] = math.tanh(cp / 400)
        li = list(legal_idx(b)); L[i, :len(li)] = li
    return [torch.from_numpy(x).to(dev) for x in (T, G, P, V, L)]

def batch(D, idx):
    t, g, p, v, l = (x[idx] for x in D)
    mask = torch.zeros(len(idx), 4096, dtype=torch.bool, device=t.device)
    mask.scatter_(1, l.clamp(min=0).long(), l >= 0)
    return t.long(), g, p, v, mask

@torch.no_grad()
def evaluate(net, D, Ts=(1, 2, 4, 8, 12, 16, 24)):
    net.eval(); n = len(D[0]); res = {}
    for T in Ts:
        acc = mse = 0
        for i in range(0, n, 1024):
            t, g, p, v, m = batch(D, torch.arange(i, min(i + 1024, n), device=D[0].device))
            (pol, val), = net(t, g, T)
            acc += (pol.masked_fill(~m, -1e9).argmax(-1) == p).sum().item(); mse += ((val - v) ** 2).sum().item()
        res[T] = (round(acc / n, 3), round(mse / n, 4))
    net.train(); return res                               # {T: (top1 vs Stockfish, value MSE)}

def train(a):
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    rows = json.load(open(a.labels)); ns = max(r[3] for r in rows) + 1
    tr, va = build([r for r in rows if r[3] < ns - a.val_src], dev), build([r for r in rows if r[3] >= ns - a.val_src], dev)
    net = LumiNet().to(dev); print("params:", sum(p.numel() for p in net.parameters()), "train:", len(tr[0]), "val:", len(va[0]))
    dec = [p for p in net.parameters() if p.dim() >= 2]; nod = [p for p in net.parameters() if p.dim() < 2]
    opt = torch.optim.AdamW([{"params": dec, "weight_decay": 0.05}, {"params": nod, "weight_decay": 0}], lr=a.lr, betas=(.9, .98))
    t0 = time.time()
    for step in range(1, a.steps + 1):
        lr = a.lr * min(1, step / 300) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * step / a.steps)))
        for gp in opt.param_groups: gp["lr"] = lr
        t, g, p, v, m = batch(tr, torch.randint(0, len(tr[0]), (a.bs,), device=dev))
        outs = net(t, g, random.randint(a.tmin, a.tmax), all_iters=True)
        loss = sum((i + 1) * (F.cross_entropy(pol.masked_fill(~m, -1e9), p) + 2 * F.mse_loss(val, v))
                   for i, (pol, val) in enumerate(outs)) / sum(range(1, len(outs) + 1))
        opt.zero_grad(set_to_none=True); loss.backward()
        nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step()
        if step % 500 == 0: print(f"step {step} loss {loss.item():.4f} {time.time() - t0:.0f}s", flush=True)
        if step % a.eval_every == 0 or step == a.steps:
            print("  eval {T: (top1, valMSE)}:", evaluate(net, va), flush=True)
            torch.save({"model": net.state_dict(), "cfg": net.cfg}, a.out)

# ---------- настоящая метрика силы: матч против Stockfish с ограничением Elo ----------
def match(a):
    dev = "cuda" if torch.cuda.is_available() else "cpu"; net = load(a.ckpt, dev)
    eng = chess.engine.SimpleEngine.popen_uci(a.sf)
    eng.configure({"UCI_LimitStrength": True, "UCI_Elo": a.elo}); sc = 0
    for gm in range(a.games):
        b, us = chess.Board(), (chess.WHITE if gm % 2 == 0 else chess.BLACK)
        for _ in range(4): b.push(random.choice(list(b.legal_moves)))   # разнообразие дебютов
        while not b.is_game_over(claim_draw=True) and b.fullmove_number < 150:
            b.push(pick_move(net, b, a.T, a.topk) if b.turn == us else eng.play(b, chess.engine.Limit(time=0.02)).move)
        w = {"1-0": 1., "0-1": 0.}.get(b.result(claim_draw=True), .5); sc += w if us else 1 - w
        print(f"game {gm + 1}: {sc}/{gm + 1}", flush=True)
    s = min(max(sc / a.games, .01), .99)
    print(f"score {sc}/{a.games}  ~Elo {a.elo + 400 * math.log10(s / (1 - s)):.0f} (грубо)"); eng.quit()

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("label"); p.add_argument("--data", required=True); p.add_argument("--sf", required=True)
    p.add_argument("--out", default="labels.json"); p.add_argument("--depth", type=int, default=8)
    p.add_argument("--variants", type=int, default=3); p.add_argument("--workers", type=int, default=4)
    p = sub.add_parser("train"); p.add_argument("--labels", default="labels.json"); p.add_argument("--out", default="lumi2.pt")
    p.add_argument("--steps", type=int, default=20000); p.add_argument("--bs", type=int, default=256)
    p.add_argument("--lr", type=float, default=2e-3); p.add_argument("--tmin", type=int, default=6)
    p.add_argument("--tmax", type=int, default=12); p.add_argument("--val_src", type=int, default=1000)
    p.add_argument("--eval_every", type=int, default=2500)
    p = sub.add_parser("match"); p.add_argument("--ckpt", default="lumi2.pt"); p.add_argument("--sf", required=True)
    p.add_argument("--elo", type=int, default=1350); p.add_argument("--games", type=int, default=20)
    p.add_argument("--T", type=int, default=12); p.add_argument("--topk", type=int, default=10)
    a = ap.parse_args(); {"label": label, "train": train, "match": match}[a.cmd](a)
