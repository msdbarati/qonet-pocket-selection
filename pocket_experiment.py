"""
Data-aware pocket-selection for QONet initialization.

Implements the two improvement directions over arXiv:2606.18515 for a Quantum
DeepONet (QONet) operator-learning model:

  (#3) DATA-AVERAGED first-moment operator.  The paper's first-moment operator is
       averaged over the parameter-init ensemble only.  For an operator-learning
       model the readout is also driven by the *input-function ensemble*.  We
       define the data-averaged first-moment operator and estimate its non-trivial
       (off-fixed-point) weight by
          D(P) = E_{theta~P} Var_{u~D}[ b_k(u;theta) ]   averaged over k,
       i.e. how much the branch readouts <Z_k> already vary across the data at init.
       D is the data-averaged operator gap xi-bar from the BP fixed point.

  (#2) POCKET-SELECTION criterion.  Among first-moment-VALID (BP-avoiding)
       initialization families, rank by a single score combining trainability and
       data alignment:
          T(P) = Var over inits of  d<Z_0>/d theta_1     (second-moment / trainability)
          S(P) = sqrt( D(P) * T(P) )                      (pocket score)
       The criterion PREDICTS the best init without training; we validate it by
       training QONet with every family and correlating S(P) with final test error.

Circuit = repo's SU1 (RY + triangle CX) + angle reuploading, local Z_k readout.
Self-contained (PennyLane + JAX).  Antiderivative operator G(u)(y)=int_0^y u.

Usage:
  python pocket_experiment.py diag            # diagnostic table (D, T, S) for all families
  python pocket_experiment.py train FAMILY SEED   # train one family/seed -> json
"""
import os, sys, json, time
import numpy as np
import jax, jax.numpy as jnp, jax.random as jr
import pennylane as qml
import optax

N      = 4          # qubits
LREUP  = 2          # data-reuploading encodings (theta has LREUP+1 SU1 blocks)
M      = 12         # query / sensor grid
NTR    = 40
NTE    = 30
OUT    = os.path.dirname(os.path.abspath(__file__))

# ----------------------------------------------------------------------------- data
def cheb_functions(ns, m, n_terms=5, rng=None):
    x = np.linspace(0., 1., m); xx = 2*x - 1
    out = np.zeros((ns, m))
    for s in range(ns):
        a = rng.uniform(-1, 1, n_terms)
        out[s] = np.polynomial.chebyshev.chebval(xx, a)
    return x, out

def cosine_functions(ns, m, rng=None):
    x = np.linspace(0., 1., m); out = np.zeros((ns, m))
    for s in range(ns):
        A = rng.uniform(0, 1); B = rng.uniform(-0.5, 0.5); P = rng.uniform(0, 2*np.pi)
        out[s] = A*np.cos(2*np.pi*x + P) + B
    return x, out

def antideriv(x, U):
    dx = x[1]-x[0]; S = np.zeros_like(U)
    S[:, 1:] = np.cumsum(0.5*(U[:, 1:]+U[:, :-1])*dx, axis=1)
    return S

def diffusion_step(x, U0, D=0.05, dt=0.05):
    m = x.shape[0]; k = 2*np.pi*np.fft.fftfreq(m, d=(x[1]-x[0]))
    decay = np.exp(-D*(k**2)*dt)
    return np.real(np.fft.ifft(np.fft.fft(U0, axis=1)*decay[None, :], axis=1))

def burgers_step(x, U0, nu=0.04, dt=0.04, nsub=300):
    m = x.shape[0]; k = 2*np.pi*np.fft.fftfreq(m, d=(x[1]-x[0]))
    u = U0.astype(float).copy(); h = dt/nsub
    for _ in range(nsub):
        uh = np.fft.fft(u, axis=1)
        ux = np.real(np.fft.ifft(1j*k[None, :]*uh, axis=1))
        uxx = np.real(np.fft.ifft(-(k[None, :]**2)*uh, axis=1))
        u = u + h*(-u*ux + nu*uxx)
    return u

def make_data(pde="antideriv"):
    rng = np.random.default_rng(0)
    if pde == "antideriv":
        x, U = cheb_functions(NTR+NTE, M, rng=rng); Y = antideriv(x, U)
    elif pde == "diffusion":
        x, U = cosine_functions(NTR+NTE, M, rng=rng); Y = diffusion_step(x, U)
    elif pde == "burgers":
        x, U = cosine_functions(NTR+NTE, M, rng=rng); Y = burgers_step(x, U)
    else:
        raise ValueError(pde)
    U = U/(np.abs(U).max()+1e-12); Y = Y/(np.abs(Y).max()+1e-12)
    return (jnp.array(x), jnp.array(U[:NTR]), jnp.array(Y[:NTR]),
            jnp.array(U[NTR:]), jnp.array(Y[NTR:]))

# ----------------------------------------------------------------------------- circuit
dev = qml.device("default.qubit", wires=N)

def _su1(theta_block):
    for q in range(N): qml.RY(theta_block[q], wires=q)
    for a in range(N):
        for b in range(a+1, N): qml.CNOT(wires=[a, b])   # triangle entangler

@qml.qnode(dev, interface="jax", diff_method="backprop")
def branch(theta, v):                 # theta (LREUP+1, N) ; v (M,)
    for l in range(LREUP):
        _su1(theta[l])
        for q in range(N): qml.RZ(jnp.pi/2 - v[(l*N+q) % M], wires=q)   # data reuploading
    _su1(theta[LREUP])
    return [qml.expval(qml.PauliZ(q)) for q in range(N)]

@qml.qnode(dev, interface="jax", diff_method="backprop")
def trunk(theta, y):                  # theta (LREUP+1, N) ; y scalar
    for l in range(LREUP):
        _su1(theta[l])
        for q in range(N): qml.RZ(jnp.pi/2 - y*(q+1), wires=q)
    _su1(theta[LREUP])
    return [qml.expval(qml.PauliZ(q)) for q in range(N)]

bvec = jax.jit(lambda th, v: jnp.stack(branch(th, v)))   # (N,)
tvec = jax.jit(lambda th, y: jnp.stack(trunk(th, y)))     # (N,)
B_all = jax.jit(jax.vmap(bvec, in_axes=(None, 0)))         # (batch,N)
T_all = jax.jit(jax.vmap(tvec, in_axes=(None, 0)))         # (batch,N)

# ----------------------------------------------------------------------------- init families
SHP = (LREUP+1, N)
FAMILIES = {  # name: (sampler(key)->theta, E[cos theta], valid?)
    "identity N(0,.05)":   (lambda k: 0.05*jr.normal(k, SHP),                 float(np.exp(-0.05**2/2)), True),
    "small-Gauss N(0,.3)": (lambda k: 0.30*jr.normal(k, SHP),                 float(np.exp(-0.30**2/2)), True),
    "calc_a U[-.4pi,.4pi]":(lambda k: jr.uniform(k, SHP, minval=-0.4*np.pi, maxval=0.4*np.pi), float(np.sin(0.4*np.pi)/(0.4*np.pi)), True),
    "shifted N(pi/4,.3)":  (lambda k: np.pi/4 + 0.30*jr.normal(k, SHP),       float(np.cos(np.pi/4)*np.exp(-0.30**2/2)), True),
    "BP U[-pi,pi]":        (lambda k: jr.uniform(k, SHP, minval=-np.pi, maxval=np.pi), 0.0, False),
}

# ----------------------------------------------------------------------------- diagnostics
def diagnostics(Utr, S=60):
    """T = Var over inits of d<Z_0>/d theta_1 ; D = data-averaged operator gap."""
    dC = jax.jit(jax.grad(lambda th, v: bvec(th, v)[0]))   # d<Z_0>/d theta
    rows = {}
    for name, (samp, ecos, valid) in FAMILIES.items():
        keys = jr.split(jr.key(7), S)
        gs = []
        for i in range(S):
            th = samp(keys[i]); v = Utr[i % Utr.shape[0]]
            gs.append(float(dC(th, v)[0, 0]))
        T = float(np.var(gs))
        ds = []
        for i in range(S):
            th = samp(keys[i])
            Bm = B_all(th, Utr)                 # (NTR, N)
            ds.append(float(jnp.mean(jnp.var(Bm, axis=0))))
        D = float(np.mean(ds))
        rows[name] = dict(Ecos=ecos, valid=valid, T=T, D=D, S=float(np.sqrt(max(D,0)*max(T,0))))
    return rows

# ----------------------------------------------------------------------------- scaling (stage-1 gate)
def branch_builder(n, lreup):
    d = qml.device("default.qubit", wires=n)
    def su1(tb):
        for q in range(n): qml.RY(tb[q], wires=q)
        for a in range(n):
            for b in range(a+1, n): qml.CNOT(wires=[a, b])
    @qml.qnode(d, interface="jax", diff_method="backprop")
    def circ(theta, v):
        for l in range(lreup):
            su1(theta[l])
            for q in range(n): qml.RZ(jnp.pi/2 - v[(l*n+q) % v.shape[0]], wires=q)
        su1(theta[lreup])
        return qml.expval(qml.PauliZ(0))
    return circ

def depth_scan(n0=6, depths=(2, 6, 10, 16, 22), S=40):
    """Stage-1 gate via first-moment contraction: Var[d<Z_0>/d theta_1] vs reuploading depth.
       Predicted: ~ E[cos]^(2L).  BP family (E[cos]=0) collapses; valid families retain."""
    fams = {"identity N(0,.05)": (lambda k, sh: 0.05*jr.normal(k, sh), float(np.exp(-0.05**2/2))),
            "small-Gauss N(0,.3)": (lambda k, sh: 0.30*jr.normal(k, sh), float(np.exp(-0.30**2/2))),
            "calc_a U[-.4pi,.4pi]": (lambda k, sh: jr.uniform(k, sh, minval=-0.4*np.pi, maxval=0.4*np.pi), float(np.sin(0.4*np.pi)/(0.4*np.pi))),
            "BP U[-pi,pi]": (lambda k, sh: jr.uniform(k, sh, minval=-np.pi, maxval=np.pi), 0.0)}
    out = {f: [] for f in fams}; ecos = {f: e for f, (s, e) in fams.items()}
    for L in depths:
        circ = branch_builder(n0, L)
        dC = jax.jit(jax.grad(lambda th, v: circ(th, v)))
        sh = (L+1, n0)
        for f, (samp, e) in fams.items():
            keys = jr.split(jr.key(3), 2*S); gs = []
            for i in range(S):
                th = samp(keys[i], sh)
                v = jr.uniform(keys[S+i], (n0,), minval=-1, maxval=1)
                gs.append(float(dC(th, v)[0, 0]))
            out[f].append(float(np.var(gs)))
    return {"depths": list(depths), "n": n0, "var": out, "ecos": ecos}

def scaling(ns=(4, 6, 8, 10), S=40):
    """Stage-1 validity gate: Var over inits of d<Z_0>/d theta_1 vs n (depth=lreup fixed)."""
    fams = {"identity N(0,.05)": lambda k, sh: 0.05*jr.normal(k, sh),
            "small-Gauss N(0,.3)": lambda k, sh: 0.30*jr.normal(k, sh),
            "calc_a U[-.4pi,.4pi]": lambda k, sh: jr.uniform(k, sh, minval=-0.4*np.pi, maxval=0.4*np.pi),
            "BP U[-pi,pi]": lambda k, sh: jr.uniform(k, sh, minval=-np.pi, maxval=np.pi)}
    out = {f: [] for f in fams}
    for n in ns:
        circ = branch_builder(n, LREUP)
        dC = jax.jit(jax.grad(lambda th, v: circ(th, v)))
        sh = (LREUP+1, n)
        for f, samp in fams.items():
            keys = jr.split(jr.key(3), 2*S)
            gs = []
            for i in range(S):
                th = samp(keys[i], sh)
                v = jr.uniform(keys[S+i], (n,), minval=-1, maxval=1)
                gs.append(float(dC(th, v)[0, 0]))
            out[f].append(float(np.var(gs)))
    return {"ns": list(ns), "var": out}

# ----------------------------------------------------------------------------- training
def rel_l2(pred, tgt):
    return float(jnp.linalg.norm(pred-tgt)/(jnp.linalg.norm(tgt)+1e-12))

def train_family(name, seed, epochs=300):
    Utr, Ytr, Ute, Yte, xq = TR_U, TR_Y, TE_U, TE_Y, XQ
    samp, ecos, valid = FAMILIES[name]
    k = jr.key(1000+seed); k1, k2 = jr.split(k)
    params = {"b": samp(k1), "t": samp(k2), "bias": jnp.array(0.0)}

    def predict(p, U, Y):
        Bm = B_all(p["b"], U)            # (Nu,N)
        Tm = T_all(p["t"], Y)            # (Ny,N)
        return Bm @ Tm.T + p["bias"]     # (Nu,Ny)

    def loss(p):
        return jnp.mean((predict(p, Utr, xq) - Ytr)**2)

    opt = optax.adam(2e-2); st = opt.init(params)
    vg = jax.jit(jax.value_and_grad(loss))
    pred_te = jax.jit(lambda p: predict(p, Ute, xq))
    for ep in range(epochs):
        L, g = vg(params); u, st = opt.update(g, st); params = optax.apply_updates(params, u)
    te = rel_l2(pred_te(params), Yte)
    tr = rel_l2(predict(params, Utr, xq), Ytr)
    return dict(family=name, seed=seed, Ecos=ecos, valid=valid,
                train_relL2=tr, test_relL2=te, final_loss=float(loss(params)))

# ----------------------------------------------------------------------------- main
XQ, TR_U, TR_Y, TE_U, TE_Y = make_data()

if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "diag":
        t0 = time.time()
        rows = diagnostics(TR_U)
        json.dump(rows, open(f"{OUT}/diag.json", "w"), indent=2)
        print(f"{'family':24s}{'E[cos]':>8s}{'valid':>7s}{'T(train)':>12s}{'D(data-gap)':>13s}{'S(pocket)':>12s}")
        for n, r in rows.items():
            print(f"{n:24s}{r['Ecos']:8.3f}{str(r['valid']):>7s}{r['T']:12.3e}{r['D']:13.3e}{r['S']:12.3e}", flush=True)
        print(f"[diag {time.time()-t0:.1f}s]")
    elif mode == "train":
        fam = sys.argv[2]; seed = int(sys.argv[3])
        t0 = time.time(); r = train_family(fam, seed)
        r["seconds"] = time.time()-t0
        fn = f"{OUT}/train_{fam.split()[0].replace('(','').replace('/','')}_{seed}.json"
        json.dump(r, open(fn, "w"), indent=2)
        print(f"{fam:24s} seed{seed} test={r['test_relL2']:.4f} train={r['train_relL2']:.4f} [{r['seconds']:.1f}s]", flush=True)
    elif mode == "all":
        pde = sys.argv[2] if len(sys.argv) > 2 else "antideriv"
        epochs = int(sys.argv[3]) if len(sys.argv) > 3 else 400
        seeds = [0, 1, 2, 3, 4]
        XQ, TR_U, TR_Y, TE_U, TE_Y = make_data(pde)
        diag = diagnostics(TR_U)
        out = {}
        for fam in FAMILIES:
            tes = []
            for s in seeds:
                r = train_family(fam, s, epochs=epochs); tes.append(r["test_relL2"])
            d = diag[fam]
            out[fam] = dict(Ecos=d["Ecos"], valid=d["valid"], T=d["T"], D=d["D"], S=d["S"],
                            test_mean=float(np.mean(tes)), test_std=float(np.std(tes)),
                            test_all=tes)
            print(f"[{pde}] {fam:22s} valid={str(d['valid']):>5s} S={d['S']:.3e} "
                  f"testRMSE={np.mean(tes):.4f}+/-{np.std(tes):.4f}", flush=True)
        fn = f"{OUT}/results_{pde}.json"
        json.dump(out, open(fn, "w"), indent=2)
        print(f"saved {fn}", flush=True)
    elif mode == "scaling":
        sc = scaling()
        json.dump(sc, open(f"{OUT}/scaling.json", "w"), indent=2)
        for f, vs in sc["var"].items():
            print(f"{f:24s} " + " ".join(f"{v:9.2e}" for v in vs), flush=True)
        print("ns =", sc["ns"], "saved scaling.json", flush=True)
    elif mode == "depth":
        sc = depth_scan()
        json.dump(sc, open(f"{OUT}/depth.json", "w"), indent=2)
        print("depths =", sc["depths"], "n =", sc["n"])
        for f, vs in sc["var"].items():
            print(f"{f:24s} E[cos]={sc['ecos'][f]:.3f}  " + " ".join(f"{v:9.2e}" for v in vs), flush=True)
        print("saved depth.json", flush=True)
