# VulReasoner conversion status

Updated: 2026-08-19

## Result

The supplied minimal VulReasoner program now runs unchanged through the
`pyreason_gpu` monkey patch and the application-neutral
`srdatalog.pyreason` compiler. The compiler contains no VulReasoner predicate
names or rule-specific branches. It recognizes a proven fragment of the
registered Python annotation function and rejects callbacks outside that
fragment rather than guessing their meaning.

The checked-in `CWE_121_MVP2.graphml` case has exact temporal-result parity
with PyReason. Larger generated cases have exact digest parity through 12,288
transitions. The 32,768-transition case is an SRDatalog-only scaling result and
is not presented as a parity result.

## How the program is converted

The companion facade is installed as `pyreason`, so VulReasoner's existing
`import pyreason as pr`, graph load, facts, six CSV rules, and registered
`paired_minimum_bounds_ann_fn` callback are not rewritten by hand. The facade
captures those calls as a neutral `SourceProgram`; the compiler in this
repository owns all subsequent lowering.

For example, one original rule is:

```text
analystAt(CB2):paired_minimum_bounds_ann_fn <-1
  analystAt(CB1):[0.25,1],
  hasLabel(CB1,Lcause):[0.1,1],
  hasLabel(CB2,Leffect):[0.1,1],
  can_cause(Lcause,Leffect):[0.1,1],
  stepFrom(CB1,CB2)
```

The target representation makes four previously implicit semantics explicit:

1. **Intervals are value columns.** A logical row
   `P(key,time):[lower,upper]` becomes
   `P(key,time,lower_bits,upper_bits)`. Duplicate logical keys use interval
   intersection:

   ```text
   [l1,u1] join [l2,u2] = [max(l1,l2), min(u1,u2)]
   ```

2. **The callback aggregate is rule-local.** Each satisfying body grounding is
   a derivation witness. The callback computes a candidate interval for each
   connector, then performs `ARG MAX(candidate.lower, -admission_rank)` and
   carries the upper endpoint from the same winning witness. The compiler
   therefore emits the equivalent staged rules:

   ```text
   Candidate(rule,head,time,rank,l,u) <-
     joined body, l=min(label1.l,label2.l,connector.l),
                  u=min(label1.u,label2.u,connector.u).

   Selected(rule,head,time,rank,l,u) <-
     ARG MAX Candidate BY (head,time) ON (l,-rank).

   Effective(rule,head,time,l,u) <- Selected(...,l,u), post-winner checks.
   AnalystAt(head,time,l,u)       <- Effective(...,l,u).
   ```

   The split is required. Intersecting every raw candidate would combine losing
   witnesses and does not implement the Python `ARG MAX`. Selection happens
   within one source rule; only the selected outputs of different source rules
   meet through interval intersection in `AnalystAt`.

3. **`<-1` is a temporal join.** A `Successor(time,next_time)` relation
   moves the selected head to the next logical time. Recursive evaluation then
   uses ordinary multi-delta semi-naive variants; it does not assume that only
   one recursive body relation can change.

4. **Order identity is logical, not physical.** PyReason's callback can observe
   predicate-map admission order through its first-match scans and strict
   maximum tie behavior. The lowering carries a stable logical admission rank.
   It does not use a GPU row address or a physical tuple ID.

This selection evidence is provenance-aware but is not semiring provenance.
Traditional semiring provenance annotates relational derivations algebraically
and retains alternative proofs. Here the user callback inspects an ordered
collection of witnesses and deliberately selects one witness's payload.
Ordinary rule explanations are reconstructed later by a demand rewrite and
witness replay; callback explanations need an additional logical
aggregate-change event and are not yet claimed.

## Validation matrix

| Input or behavior | Status | Evidence |
|---|---|---|
| Supplied `CWE_121_MVP2.graphml` | **Exact parity** | 188 nodes, 403 edges, 138 connector facts, six analyst rules |
| Complete informative history | **Exact parity** | `b1@0`, `b1@1`, `b2@2`, `b3@3`, `b4@4`, each `[1,1]` |
| Runner digest for that history | **Exact parity** | `f5a80e6021576a19536f80b16bd6d0388a0ac35bfbb1e7f27312d49d3a8a1a00` |
| Generated 4×4×2 workload | **Exact parity** | 32 transitions; all target bounds and digest match |
| Generated 8×32×4 workload | **Exact parity** | 1,024 transitions; all target bounds and digest match |
| Generated 12×128×8 workload | **Exact parity** | 12,288 transitions; all target bounds and digest match |
| Generated 16×256×8 workload | **GPU only** | 32,768 transitions; PyReason was intentionally not run |
| Current repository regression suite | **Pass** | 1,202 passed, 8 skipped; mypy clean on 26 configured source files |

The recorded warm benchmark on an RTX 6000 Ada was 29.39 seconds in PyReason
versus 32.7 milliseconds for SRDatalog at 12,288 transitions. The notebook's
live PyReason suite has a 120-second timeout and stops at that size; compilation
time is reported separately from cached GPU execution.

## Cases where no correct compatibility answer is claimed

These cases are either rejected explicitly or supported only as a named replay
of PyReason's operational behavior. No approximate result is presented as an
order-independent Datalog model:

| Case | Why it is not answered |
|---|---|
| Extended rules that derive `~stepFrom` against an existing `stepFrom` | PyReason performs an order-sensitive conflict repair/retraction. The supplied extended trace contains one such inconsistent edge. This is not monotone interval-lattice materialization, and a small source-order variant can loop at timestep 0. |
| A callback that reads a conflict-repaired `[0,1]` row | SRDatalog retains the crossed internal interval to detect conflict. Exact support needs a snapshot-local repaired read view before callback grouping. |
| One admission rank with divergent nested first-match payloads | Python's `break` requires an inner `ARG MIN(admission_rank)` before the outer maximum. The current runtime functional-dependency guard rejects the ambiguous case. |
| Exact callback provenance or `get_rule_trace` operation chronology | Materialized candidates do not reconstruct the same-snapshot group or PyReason's global operation numbers. Ordinary constant-head rule explanations are supported on demand; callback trace requests are rejected. |
| Inconsistent-predicate-list co-updates and `set_static` rule heads | These require atomic side effects with the primary head update and are not represented by the current Core rules. |
| Binary `infer_edges` that grows PyReason's graph domain | Later grounding depends on a mutable active domain. The finite target domain cannot silently approximate it. |
| Phase-sensitive closed-world initialization | Delayed initialization and domain-changing cases are rejected because the target lacks source-round/time-stratified worlds. |
| Fixed-domain recursive “CWA” | PyReason coerces missing/`[0,1]` values to `[0,0]` at read time. The staged target can reproduce covered finite cases, but this is explicitly operational compatibility—not stratified negation or well-founded semantics. Self-negating/order-sensitive programs have no order-independent Datalog answer to claim. |

The extended conflict artifacts are used only as local diagnostic evidence.
They are not committed because the raw API response includes embeddings and
application identifiers, and the trace exports are not needed to reproduce the
supported minimal result.

## Reproduction

```bash
# Independent PyReason/SRDatalog comparison; cached CUDA artifact.
uv run --project minimal_vulreasoner --frozen \
  python minimal_vulreasoner/validate_minimal_parity.py --no-compile

# From a companion pyreason-gpu checkout: sync the example's pinned Python
# 3.10 environment, then import the unchanged setup after installing the
# facade as the module named "pyreason".
uv sync --project vendor/srdatalog-python/minimal_vulreasoner --frozen
PYTHONPATH=src:vendor/srdatalog-python/src \
  PYREASON_GPU_TIMEOUT_SECONDS=600 \
  vendor/srdatalog-python/minimal_vulreasoner/.venv/bin/python \
  examples/06_vulreasoner_monkey_patch.py
```

The detailed semantic derivation, original rules, generated rules, small
counterexamples, parity cells, and scaling plot remain in
`vulreasoner_srdatalog_semantics.ipynb`.
