#!/usr/bin/env python3
"""Model definitions for the two architecture families (Branch A: direct; Branch B: modular). All models are local, size-extensive, conservative (F = -dE/dx,
stress = (1/V) dE/d eps by autograd through MACE's displacement), and periodic-invariant by construction (MACE periodic graphs).

Branch A: `DirectMACE`  -- plain ScaleShiftMACE on all edges within r_max (A-direct / A-delta differ only in the training target).
Branch B: `ModularIBX`  -- E = E0 + sum_i e_intra_i + sum_i e_inter_i + E_backstop
   e_intra : MACE channel restricted to INTRAMOLECULAR edges = pairs of the same molecule at the molecule's OWN image (fixed 18-bond
             topology, molecule = atom // 17); classification uses per-atom image offsets from ibx.unwrap_molecules, so it is a geometric,
             representation-independent property (contacts between a molecule and its own periodic images are intermolecular).
   e_inter : MACE channel on ALL other edges (other molecules AND periodic images of the same molecule), initial node features seeded by a
             learned linear map of the intramolecular channel's final scalar features (molecule-aware environment).
   E_backstop : sum over intermolecular I...O pairs of A (r0 - r)^3_+, A = A_max sigmoid(a), r0 fixed (DFT-derived, <= 2.20 A).
Edges carry a boolean `edge_is_intra` computed in `graph_from_atoms` (data prep and ASE calculator use the same code)."""
import os
os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")   # e3nn ships trusted constants.pt
import numpy as np
import torch
from e3nn import o3
from mace.modules.blocks import RealAgnosticResidualInteractionBlock, RealAgnosticInteractionBlock, LinearNodeEmbeddingBlock
from mace.modules.models import ScaleShiftMACE
from mace.modules.utils import get_outputs, prepare_graph
from mace.tools.scatter import scatter_sum
from mace.data import AtomicData
from mace.data.utils import Configuration
from mace.tools.utils import AtomicNumberTable
from mace.tools import torch_geometric

Z_TABLE = AtomicNumberTable([1, 6, 8, 53])
APM = 17


def _perm(x):
    return x[..., torch.LongTensor([1, 2, 0]).to(x.device)]


def image_offsets(positions, cell, mol_atoms):
    """Integer lattice offsets n_i with u_i = r_i + n_i @ cell (u = molecules unwrapped through the fixed topology mol_atoms)."""
    from . import geometry as ibx
    u = ibx.unwrap_with_topology(positions, cell, mol_atoms)
    return np.round((u - positions) @ np.linalg.inv(cell)).astype(np.int64)


def graph_from_atoms(atoms, r_max, energy=None, forces=None, stress=None, need_intra=True, weight=1.0, stress_weight=1.0, mol_atoms=None):
    pbc = tuple(bool(x) for x in atoms.pbc); cell = np.array(atoms.cell.array)
    cfg = Configuration(atomic_numbers=np.array(atoms.numbers), positions=np.array(atoms.positions), properties={}, property_weights={},
                        cell=cell, pbc=pbc, weight=weight, config_type="Default", head="Default")
    d = AtomicData.from_config(cfg, z_table=Z_TABLE, cutoff=r_max, heads=["Default"])
    d.energy_ref = torch.tensor([0.0 if energy is None else float(energy)], dtype=torch.get_default_dtype())
    d.forces_ref = torch.zeros(len(atoms), 3, dtype=torch.get_default_dtype()) if forces is None else torch.tensor(forces, dtype=torch.get_default_dtype())
    d.stress_ref = torch.zeros(1, 3, 3, dtype=torch.get_default_dtype()) if stress is None else torch.tensor(np.asarray(stress).reshape(1, 3, 3), dtype=torch.get_default_dtype())
    d.has_stress = torch.tensor([0.0 if stress is None else 1.0], dtype=torch.get_default_dtype())
    d.sample_weight = torch.tensor([float(weight)], dtype=torch.get_default_dtype())
    d.n_atoms = torch.tensor([len(atoms)], dtype=torch.get_default_dtype())
    if need_intra:
        from . import geometry as ibx
        if mol_atoms is None: mol_atoms = ibx.block_topology(len(atoms))
        n = image_offsets(atoms.positions, cell, mol_atoms); mol = np.empty(len(atoms), dtype=np.int64)
        for k, m in enumerate(mol_atoms): mol[m] = k
        s, r = d.edge_index.numpy(); us = d.unit_shifts.numpy().round().astype(np.int64)
        intra = (mol[s] == mol[r]) & np.all(us == (n[r] - n[s]), axis=1)
        d.edge_is_intra = torch.tensor(intra, dtype=torch.bool)
    return d


def build_mace(r_max, hidden_irreps, max_ell=3, correlation=3, num_interactions=2, avg_num_neighbors=30.0, atomic_energies=None,
               scale=1.0, shift=0.0, num_bessel=8, num_polynomial_cutoff=5, mlp_irreps="16x0e", radial_mlp=(64, 64, 64), zbl=False):
    ae = np.zeros(4) if atomic_energies is None else np.asarray(atomic_energies, dtype=float)
    return ScaleShiftMACE(r_max=r_max, num_bessel=num_bessel, num_polynomial_cutoff=num_polynomial_cutoff, max_ell=max_ell,
                          interaction_cls=RealAgnosticResidualInteractionBlock, interaction_cls_first=RealAgnosticInteractionBlock,
                          num_interactions=num_interactions, num_elements=4, hidden_irreps=o3.Irreps(hidden_irreps), MLP_irreps=o3.Irreps(mlp_irreps),
                          atomic_energies=ae, avg_num_neighbors=avg_num_neighbors, atomic_numbers=[1, 6, 8, 53], correlation=correlation,
                          gate=torch.nn.functional.silu, radial_MLP=list(radial_mlp), atomic_inter_scale=scale, atomic_inter_shift=shift,
                          pair_repulsion=bool(zbl))


_ZORDER = ["H", "C", "O", "I"]


def clamp_table(r0_table):
    t = torch.zeros(4, 4)
    for key, v in r0_table.items():
        a, b = key.split("-"); i, j = _ZORDER.index(a), _ZORDER.index(b); t[i, j] = t[j, i] = float(v)
    return t


def clamp_edges(vectors, lengths, zs, zr, table, delta):
    """Soft lower clamp of edge lengths seen by the LEARNED channels : r_eff = r for r >= r_c; r_c - delta tanh((r_c - r)/delta)
    below (C1 at r_c, bounded below by r_c - delta). r_c = per-element-pair guard onset (below every DFT training distance), so exactly inactive on
    all data. Direction unchanged. ZBL / guard / backstop always use the true geometry."""
    L = lengths.squeeze(-1) if lengths.dim() > 1 else lengths
    rc = table.to(L.dtype)[zs, zr]
    act = (rc > 0) & (L < rc)
    Leff = torch.where(act, rc - delta * torch.tanh((rc - L) / delta), L)
    v = vectors * (Leff / torch.clamp(L, min=1e-9)).unsqueeze(-1)
    return v, Leff.unsqueeze(-1) if lengths.dim() > 1 else Leff


class DirectMACE(torch.nn.Module):
    kind = "A"

    def __init__(self, **kw):
        super().__init__(); self.cfg = dict(kw); self.net = build_mace(**kw); self.r_max = kw["r_max"]; self.clamp = None

    def set_clamp(self, r0_table, delta=0.15):
        self.clamp = (clamp_table(r0_table), float(delta))

    def forward(self, batch, compute_force=True, compute_stress=True, training=False):
        if self.clamp is None:
            out = self.net(batch, training=training, compute_force=compute_force, compute_stress=compute_stress)
            return dict(energy=out["energy"], forces=out["forces"], stress=out["stress"])
        net = self.net; table, delta = self.clamp
        ctx = prepare_graph(batch, compute_virials=False, compute_stress=compute_stress, compute_displacement=False, lammps_mliap=False)
        arange = ctx.num_atoms_arange.to(torch.int64); ng = ctx.num_graphs; heads = ctx.node_heads.to(torch.int64); na = batch["node_attrs"]
        z = na.argmax(-1); s_, r_ = batch["edge_index"]
        v_eff, L_eff = clamp_edges(ctx.vectors, ctx.lengths, z[s_], z[r_], table, delta)
        node_e0 = net.atomic_energies_fn(na)[arange, heads]; e0 = scatter_sum(src=node_e0, index=batch["batch"], dim=0, dim_size=ng).to(ctx.vectors.dtype)
        node_feats = net.node_embedding(na); edge_attrs = net.spherical_harmonics(v_eff)
        edge_feats, cutoff = net.radial_embedding(L_eff, na, batch["edge_index"], net.atomic_numbers)
        pair = net.pair_repulsion_fn(ctx.lengths, na, batch["edge_index"], net.atomic_numbers) if hasattr(net, "pair_repulsion") else torch.zeros_like(node_e0)
        es = [pair]; feats = []
        for i, (inter, prod) in enumerate(zip(net.interactions, net.products)):
            node_feats, sc = inter(node_attrs=na, node_feats=node_feats, edge_attrs=edge_attrs, edge_feats=edge_feats, edge_index=batch["edge_index"], cutoff=cutoff)
            node_feats = prod(node_feats=node_feats, sc=sc, node_attrs=na); feats.append(node_feats)
        for i, ro in enumerate(net.readouts):
            es.append(ro(feats[-1 if len(net.readouts) == 1 else i], heads)[arange, heads])
        node_inter = net.scale_shift(torch.sum(torch.stack(es, 0), 0), heads)
        inter_e = scatter_sum(node_inter, batch["batch"], dim=-1, dim_size=ng)
        forces, virials, stress, _, _ = get_outputs(energy=inter_e, positions=ctx.positions, displacement=ctx.displacement, vectors=ctx.vectors, cell=ctx.cell,
                                                    training=training, compute_force=compute_force, compute_virials=False, compute_stress=compute_stress)
        return dict(energy=e0 + inter_e, forces=forces, stress=stress)


class ModularIBX(torch.nn.Module):
    kind = "B"

    def __init__(self, r_intra=4.5, r_inter=5.5, hidden_intra="64x0e", hidden_inter="64x0e", max_ell=3, correlation=3, n_int_intra=2, n_int_inter=2,
                 avg_nn_intra=12.0, avg_nn_inter=30.0, atomic_energies=None, scale=1.0, backstop=True, r0_backstop=2.20, A_max=27.0, seed_inter=True, zbl=False):
        super().__init__()
        self.cfg = dict(r_intra=r_intra, r_inter=r_inter, hidden_intra=hidden_intra, hidden_inter=hidden_inter, max_ell=max_ell, correlation=correlation,
                        n_int_intra=n_int_intra, n_int_inter=n_int_inter, avg_nn_intra=avg_nn_intra, avg_nn_inter=avg_nn_inter,
                        atomic_energies=None if atomic_energies is None else list(map(float, atomic_energies)), scale=scale, backstop=backstop,
                        r0_backstop=r0_backstop, A_max=A_max, seed_inter=seed_inter, zbl=zbl)
        self.r_max = max(r_intra, r_inter); self.r_intra, self.r_inter = r_intra, r_inter
        self.intra = build_mace(r_intra, hidden_intra, max_ell, correlation, n_int_intra, avg_nn_intra, None, scale, 0.0)
        self.inter = build_mace(r_inter, hidden_inter, max_ell, correlation, n_int_inter, avg_nn_inter, None, scale, 0.0)
        ae = np.zeros(4) if atomic_energies is None else np.asarray(atomic_energies, dtype=float)
        self.register_buffer("atomic_energies", torch.tensor(ae, dtype=torch.get_default_dtype()))
        self.seed_inter = seed_inter
        if seed_inter:
            n_sc_intra = o3.Irreps(hidden_intra).count(o3.Irrep("0e")); n_sc_inter = o3.Irreps(hidden_inter).count(o3.Irrep("0e"))
            self.seed = o3.Linear(o3.Irreps(f"{n_sc_intra}x0e"), o3.Irreps(f"{n_sc_inter}x0e"))
            self.n_sc_intra = n_sc_intra
        self.backstop = backstop; self.r0 = r0_backstop; self.A_max = A_max
        self.a_backstop = torch.nn.Parameter(torch.tensor(0.0))
        self.clamp = None
        self.zbl = zbl
        if zbl:
            from mace.modules.radial import ZBLBasis
            self.zbl_fn = ZBLBasis(p=6)

    def set_clamp(self, r0_table, delta=0.15):
        self.clamp = (clamp_table(r0_table), float(delta))

    def _channel(self, ch, node_attrs, node_feats, edge_index, vectors, lengths, num_atoms_arange, node_heads):
        edge_attrs = ch.spherical_harmonics(vectors)
        edge_feats, cutoff = ch.radial_embedding(lengths, node_attrs, edge_index, ch.atomic_numbers)
        es, feats = [], None
        for i, (inter, prod) in enumerate(zip(ch.interactions, ch.products)):
            node_feats, sc = inter(node_attrs=node_attrs, node_feats=node_feats, edge_attrs=edge_attrs, edge_feats=edge_feats, edge_index=edge_index, cutoff=cutoff)
            node_feats = prod(node_feats=node_feats, sc=sc, node_attrs=node_attrs)
            es.append(ch.readouts[i](node_feats, node_heads)[num_atoms_arange, node_heads])
        e = ch.scale_shift(torch.sum(torch.stack(es, 0), 0), node_heads)
        return e, node_feats

    def forward(self, batch, compute_force=True, compute_stress=True, training=False, return_parts=False):
        ctx = prepare_graph(batch, compute_virials=False, compute_stress=compute_stress, compute_displacement=False, lammps_mliap=False)
        vectors, lengths = ctx.vectors, ctx.lengths.squeeze(-1) if ctx.lengths.dim() > 1 else ctx.lengths
        node_attrs = batch["node_attrs"]; ng = ctx.num_graphs; b = batch["batch"]; arange = ctx.num_atoms_arange; heads = ctx.node_heads
        intra = batch["edge_is_intra"]
        m_in = intra & (lengths <= self.r_intra); m_out = (~intra) & (lengths <= self.r_inter)
        L = lengths.unsqueeze(-1)
        ei, eo = batch["edge_index"][:, m_in], batch["edge_index"][:, m_out]
        vc, Lc = vectors, L
        if self.clamp is not None:
            zz = node_attrs.argmax(-1); s0, r0_ = batch["edge_index"]; vc, Lc = clamp_edges(vectors, L, zz[s0], zz[r0_], self.clamp[0], self.clamp[1])
        e_intra, f_intra = self._channel(self.intra, node_attrs, self.intra.node_embedding(node_attrs), ei, _perm(vc[m_in]), Lc[m_in], arange, heads)
        h0 = self.inter.node_embedding(node_attrs)
        if self.seed_inter:
            h0 = h0 + self.seed(f_intra[:, :self.n_sc_intra])
        e_inter, _ = self._channel(self.inter, node_attrs, h0, eo, _perm(vc[m_out]), Lc[m_out], arange, heads)
        e0 = (node_attrs @ self.atomic_energies)
        E_intra = scatter_sum(e_intra, b, dim=-1, dim_size=ng); E_inter = scatter_sum(e_inter, b, dim=-1, dim_size=ng)
        E0 = scatter_sum(e0, b, dim=-1, dim_size=ng)
        E = E0 + E_intra + E_inter
        if self.backstop:
            s, r = batch["edge_index"]; z = batch["node_attrs"].argmax(-1)       # Z_TABLE order [H, C, O, I]
            sel = (~intra) & (z[s] == 3) & (z[r] == 2) & (lengths < self.r0)
            A = self.A_max * torch.sigmoid(self.a_backstop)
            eb = A * (self.r0 - lengths[sel]) ** 3
            E_b = scatter_sum(eb, b[s[sel]], dim=-1, dim_size=ng); E = E + E_b
        else:
            E_b = torch.zeros_like(E)
        if self.zbl:
            E_zbl = scatter_sum(self.zbl_fn(L, node_attrs, batch["edge_index"], self.intra.atomic_numbers), b, dim=-1, dim_size=ng); E = E + E_zbl
        forces, virials, stress, _, _ = get_outputs(energy=E, positions=ctx.positions, displacement=ctx.displacement, vectors=vectors, cell=ctx.cell,
                                                    training=training, compute_force=compute_force, compute_virials=False, compute_stress=compute_stress)
        out = dict(energy=E, forces=forces, stress=stress)
        if return_parts: out.update(E_intra=E_intra, E_inter=E_inter, E_backstop=E_b, E0=E0)
        return out


def make_loader(graphs, batch_size, shuffle):
    return torch_geometric.dataloader.DataLoader(graphs, batch_size=batch_size, shuffle=shuffle, drop_last=False)


def batch_to_dict(batch):
    d = batch.to_dict(); d["num_graphs"] = batch.num_graphs
    return d


class RBCalculator:
    """ASE calculator for DirectMACE / ModularIBX (+ optional baseline calculator added on top, e.g. a baseline potential for delta-learning comparisons)."""

    def __new__(cls, *a, **k):
        from ase.calculators.calculator import Calculator, all_changes

        class _C(Calculator):
            implemented_properties = ["energy", "free_energy", "forces", "stress"]

            def __init__(self, model, baseline=None, topology="auto", fix_topology=False, **kw):
                super().__init__(**kw); self.model = model.eval(); self.baseline = baseline; self.topology = topology; self.fix_topology = fix_topology; self._top = None

            def calculate(self, atoms=None, properties=None, system_changes=all_changes):
                super().calculate(atoms, properties, system_changes)
                top = None
                if self.model.kind == "B":
                    from . import geometry as ibx
                    if self.topology == "blocks": top = ibx.block_topology(len(atoms))
                    elif self.fix_topology and self._top is not None and len(self._top) * APM == len(atoms): top = self._top
                    else: top = ibx.topology_from_atoms(atoms)
                    if self.fix_topology: self._top = top
                g = graph_from_atoms(atoms, self.model.r_max, need_intra=(self.model.kind == "B"), mol_atoms=top)
                bd = batch_to_dict(torch_geometric.batch.Batch.from_data_list([g]))
                out = self.model(bd, compute_force=True, compute_stress=bool(atoms.pbc.any()))
                e = float(out["energy"].detach()[0]); f = out["forces"].detach().numpy().copy(); s = out["stress"].detach().numpy()[0]
                if self.baseline is not None:
                    a2 = atoms.copy(); a2.calc = self.baseline
                    e += a2.get_potential_energy(); f += a2.get_forces(); s = s + a2.get_stress(voigt=False)
                self.results = dict(energy=e, free_energy=e, forces=f, stress=np.array([s[0, 0], s[1, 1], s[2, 2], s[1, 2], s[0, 2], s[0, 1]]))
        return _C(*a, **k)


class PairGuard(torch.nn.Module):
    """Extrapolation guard : E = sum_{i<j} K (r0(Zi,Zj) - r)^3_+ with r0 = (minimum distance of that element pair in the DFT TRAIN
    data) - margin. Exactly zero on every training structure (by construction), steeply repulsive only where no DFT data exist. Fixed, not trained.
    Returns energy per graph, forces and stress (ASE convention (1/V) dE/d eps, from the pair virial)."""
    ORDER = ["H", "C", "O", "I"]                 # Z_TABLE order [1, 6, 8, 53]

    def __init__(self, r0_table, K=500.0, form="cubic"):
        super().__init__(); self.form = form       # "cubic": K (r0-r)^3 ; "inv3": K ((r0-r)/r)^3 (diverges as r -> 0; )
        t = torch.zeros(4, 4)
        for key, v in r0_table.items():
            a, b = key.split("-"); i, j = self.ORDER.index(a), self.ORDER.index(b); t[i, j] = t[j, i] = float(v)
        self.register_buffer("r0", t); self.K = float(K)

    def forward(self, batch, compute_force=True, compute_stress=True):
        pos = batch["positions"]; s, r = batch["edge_index"]; z = batch["node_attrs"].argmax(-1); ng = int(batch["num_graphs"])
        p = pos.detach().clone().requires_grad_(True)
        v = p[r] - p[s] + batch["shifts"]; L = torch.linalg.norm(v, dim=-1); r0 = self.r0[z[s], z[r]]
        act = (L < r0) & (r0 > 0)
        x = torch.clamp(r0 - L, min=0.0); Ls = torch.clamp(L, min=1e-6)
        if self.form == "inv3":
            e_edge = torch.where(act, 0.5 * self.K * (x / Ls) ** 3, torch.zeros_like(L))
        else:
            e_edge = torch.where(act, 0.5 * self.K * x ** 3, torch.zeros_like(L))      # directed edges -> 1/2
        E = scatter_sum(e_edge, batch["batch"][s], dim=-1, dim_size=ng)
        F = torch.zeros_like(pos); S = torch.zeros(ng, 3, 3, dtype=pos.dtype, device=pos.device)
        if bool(act.any()):
            g = torch.autograd.grad(E.sum(), p, create_graph=False)[0]; F = -g
            if self.form == "inv3":   # d/dL [0.5 K ((r0-L)/L)^3] = 1.5 K ((r0-L)/L)^2 * (-r0/L^2)
                dEdL = torch.where(act, -1.5 * self.K * (x / Ls) ** 2 * r0 / Ls ** 2, torch.zeros_like(L))
            else:
                dEdL = torch.where(act, -1.5 * self.K * x ** 2, torch.zeros_like(L))
            w = (dEdL / torch.clamp(L, min=1e-9)).unsqueeze(-1).unsqueeze(-1) * (v.unsqueeze(-1) * v.unsqueeze(-2))
            S = scatter_sum(w, batch["batch"][s], dim=0, dim_size=ng)
            cell = batch["cell"].view(-1, 3, 3); vol = torch.abs(torch.linalg.det(cell)).view(-1, 1, 1); S = S / vol
        return E.detach(), F.detach(), S.detach()


class GuardedModel(torch.nn.Module):
    """model + PairGuard (energy, forces, stress added)."""

    def __init__(self, model, guard):
        super().__init__(); self.model = model; self.guard = guard; self.kind = model.kind; self.r_max = model.r_max

    def forward(self, batch, compute_force=True, compute_stress=True, training=False, **kw):
        out = dict(self.model(batch, compute_force=compute_force, compute_stress=compute_stress, training=training))
        Eg, Fg, Sg = self.guard(batch)
        out["energy"] = out["energy"] + Eg
        if out.get("forces") is not None: out["forces"] = out["forces"] + Fg
        if out.get("stress") is not None: out["stress"] = out["stress"] + Sg
        return out
