"""Phase 3 diagnostic: background matchers (two independent generators).

Given a (15,25) dB window T (NaN = missing minute), search the released
minutes (data/phase3_minutes.npz) for candidate background windows B such
that B <= T approximately cell-wise (a controlled addition can only add
energy). Released and test values both sit exactly on the 0.1 dB grid
(measured in phase3_index.py), so tolerances are tight and principled.

Matcher A (KD): candidate window starts come from k nearest released
minutes to T's anchor minutes under Manhattan distance on the >=1 kHz
bands, filtered by the hard one-sided constraint at the anchor minute.

Matcher B (hash): candidate starts come from released minutes whose two
highest band values (5011.9, 7943.3 Hz) exactly equal the anchor minute's
(a genuinely different generation mechanism: exact grid hashing, no
distance ranking), filtered the same way.

Both share one verification: vectorised violation fraction (cells with
B > T + 0.06 dB) and mean positive excess; final ranking is fixed a
priori: fewest violations, then smallest mean excess. Nothing is fitted
on Test.csv.
"""
import numpy as np
from scipy.spatial import cKDTree
from common import DATA, CENTRES, LOW, db_to_lin

HB = np.where(CENTRES >= 1000.0)[0]        # bands >= 1 kHz (10)
TOL_DB = 0.06                              # grid tolerance
K_ANCHOR = 4000                            # KD neighbours per anchor
HASH_BANDS = np.array([23, 24])            # 5011.9, 7943.3 Hz


def passage_profile():
    pas = np.load(DATA / "passages.npz", allow_pickle=True)
    rel, kinds = list(pas["rel_db"]), pas["kinds"]
    profs = []
    for c, k in zip(rel, kinds):
        if k != "vessel":
            continue
        e = db_to_lin(c).sum(axis=0)
        profs.append(e / e.sum())
    return np.mean(profs, axis=0)


PASS_PROF = passage_profile()


def full_metrics(T_db, B_db):
    """Rich metrics for one verified candidate (residual structure)."""
    obs = np.isfinite(T_db)
    Tl = db_to_lin(np.where(obs, T_db, np.nan))
    Bl = db_to_lin(B_db)
    resid = np.where(obs, np.maximum(Tl - Bl, 0.0), 0.0)
    tot = float(np.nansum(np.where(obs, Tl, 0.0)))
    rsum = float(resid.sum())
    share = rsum / tot if tot > 0 else 0.0
    low_frac = float(resid[:, LOW].sum() / rsum) if rsum > 0 else 0.0
    mt = resid.sum(axis=1)
    mo = obs.any(axis=1) & (mt > 0)
    ac = np.nan
    if mo.sum() >= 5:
        x = mt[mo]
        ac = float(np.corrcoef(x[:-1], x[1:])[0, 1])
    bp = resid.sum(axis=0)
    sc = float(np.corrcoef(bp / bp.sum(), PASS_PROF)[0, 1]) if bp.sum() > 0 else np.nan
    return dict(resid_share=float(share), resid_low_frac=low_frac,
                resid_autocorr=ac, resid_spec_corr=sc)


def gain_aligned_viol(T_db, B_db):
    """Diagnostic: violation fraction after the best scalar gain offset.

    Offset = median high-band gap (T - B), clipped to +-3 dB. Distinguishes
    'background not in the release' from 'in the release but gain-shifted'.
    Returns (offset_db, viol_after).
    """
    obs = np.isfinite(T_db)
    gaps = (T_db[:, HB] - B_db[:, HB])[obs[:, HB]]
    off = float(np.clip(np.median(gaps), -3.0, 3.0))
    viol = float(np.mean((B_db + off)[obs] > T_db[obs] + TOL_DB))
    return off, viol


class _Base:
    def __init__(self, exclude_sites=()):
        d = np.load(DATA / "phase3_minutes.npz", allow_pickle=True)
        self.V, self.site, self.t, self.runok = d["V"], d["site"], d["t"], d["runok"]
        self.exclude_sites = set(exclude_sites)
        self.sites = [s for s in np.unique(self.site) if s not in self.exclude_sites]

    def anchors(self, T_db):
        obs_min = np.where(np.isfinite(T_db).any(axis=1))[0]
        if len(obs_min) == 0:
            return []
        tot = np.nansum(db_to_lin(np.where(np.isfinite(T_db), T_db, np.nan)), axis=1)
        out = []
        for a in (obs_min[np.argmin(tot[obs_min])], obs_min[len(obs_min) // 2], obs_min[-1]):
            if a not in out:
                out.append(int(a))
        return out

    def verify(self, T_db, starts):
        """Vectorised metrics for many candidate starts.

        Ranking (fixed a priori, physically motivated):
          1. fewest violating cells (B above T + 0.06 dB);
          2. most EXACT minutes: observed minutes where >=24 of 25 bands
             sit on the same 0.1 dB grid value as T (|T-B| <= 0.051).
             The true background equals T exactly in every minute the
             addition does not touch; a merely similar window (temporal
             neighbour, same sea state) essentially never does;
          3. highest overall exact-cell fraction;
          4. smallest median high-band gap; 5. smallest mean excess.
        """
        starts = np.asarray(sorted(set(int(k) for k in starts
                                       if 0 <= k < len(self.runok) and self.runok[k])))
        if len(starts) == 0:
            return []
        obs = np.isfinite(T_db)
        Td = np.where(obs, T_db, -np.inf)
        Tf = np.where(obs, T_db, 9999.0)
        B = np.stack([self.V[k:k + 15] for k in starts]).astype(np.float64)
        viol = (B[:, obs] > Td[obs][None, :] + TOL_DB).mean(axis=1)
        excess = np.maximum(B - Td[None] - TOL_DB, 0.0)[:, obs].mean(axis=1)
        hb_obs = obs[:, HB]
        hb_gap = np.array([np.median((Td[:, HB] - b[:, HB])[hb_obs]) for b in B])
        eq = (np.abs(B - Tf[None]) <= 0.051) & obs[None]
        eq_frac = eq.sum(axis=(1, 2)) / obs.sum()
        eq_per_min = eq.sum(axis=2)                       # (K,15)
        min_obs = obs.any(axis=1)
        n_exact_min = ((eq_per_min >= 24) & min_obs[None]).sum(axis=1)
        order = np.lexsort((excess, hb_gap, -eq_frac, -n_exact_min, viol))
        return [dict(start_idx=int(starts[i]), site=str(self.site[starts[i]]),
                     t_start=int(self.t[starts[i]]), viol=float(viol[i]),
                     n_exact_min=int(n_exact_min[i]), eq_frac=float(eq_frac[i]),
                     hb_med_gap_db=float(hb_gap[i]),
                     mean_excess_db=float(excess[i])) for i in order]

    def match(self, T_db, topn=3, enrich=True):
        cands = self.verify(T_db, self.candidate_starts(T_db))
        out = cands[:topn]
        if enrich:
            for c in out:
                B = self.V[c["start_idx"]:c["start_idx"] + 15].astype(np.float64)
                c.update(full_metrics(T_db, B))
        return out


class Matcher(_Base):
    """Matcher A: KD-tree anchor generation."""

    def __init__(self, exclude_sites=()):
        super().__init__(exclude_sites)
        self.trees, self.members = {}, {}
        for s in self.sites:
            idx = np.where(self.site == s)[0]
            self.members[s] = idx
            self.trees[s] = cKDTree(self.V[idx][:, HB].astype(np.float64))

    def candidate_starts(self, T_db, max_starts=6000):
        starts = set()
        for a in self.anchors(T_db):
            q = T_db[a, HB].astype(np.float64)
            for s, tree in self.trees.items():
                _, ii = tree.query(q, k=min(K_ANCHOR, len(self.members[s])), p=1)
                for gi in self.members[s][np.atleast_1d(ii)]:
                    if np.all(self.V[gi] <= T_db[a] + TOL_DB):
                        starts.add(int(gi) - a)
                    if len(starts) >= max_starts:
                        return list(starts)
        return list(starts)


class HashMatcher(_Base):
    """Matcher B: exact grid-hash anchor generation on the top two bands."""

    def __init__(self, exclude_sites=()):
        super().__init__(exclude_sites)
        keep = np.array([s not in self.exclude_sites for s in self.site])
        keys = (np.round(self.V[:, HASH_BANDS[0]] * 10).astype(np.int64) * 100000
                + np.round(self.V[:, HASH_BANDS[1]] * 10).astype(np.int64))
        keys = np.where(keep, keys, -1)
        order = np.argsort(keys, kind="stable")
        self._h_order = order
        uk, first = np.unique(keys[order], return_index=True)
        self._h_bounds = {int(k): (int(first[i]),
                                    int(first[i + 1]) if i + 1 < len(first) else len(order))
                          for i, k in enumerate(uk) if k >= 0}

    def _hash_members(self, key):
        b = self._h_bounds.get(int(key))
        if b is None:
            return ()
        return self._h_order[b[0]:b[1]]

    def candidate_starts(self, T_db, max_starts=6000):
        starts = set()
        for a in self.anchors(T_db):
            base = (int(round(T_db[a, HASH_BANDS[0]] * 10)) * 100000
                    + int(round(T_db[a, HASH_BANDS[1]] * 10)))
            # B <= T: hash keys at or below the anchor's, within 0.3 dB
            for d0 in range(0, 4):
                for d1 in range(0, 4):
                    for gi in self._hash_members(base - d0 * 100000 - d1):
                        if np.all(self.V[gi] <= T_db[a] + TOL_DB):
                            starts.add(int(gi) - a)
                        if len(starts) >= max_starts:
                            return list(starts)
        return list(starts)
