# What we are trying to establish with CRAG

**CRAG is the assembly-plus-generation method we did not build** (`ai4ce/CRAG`,
ICML 2026). It couples a GARF-lineage assembly backbone with a TripoSG shape head:
assembly gives part-level structure to generation, generation gives whole-shape
context back to assembly — including synthesising missing geometry. The questions
here are whether that second half helps on **worn, incomplete excavated pottery**,
where TORA and GARF both fail on the Juglet.

This folder is **state, not a log**. Edit a line when it turns out wrong; git holds the
history. The runs themselves live in `docs/notes/` (created on first use).

Prefix **`R`**, permanent. Numbers are never reused. **R2 is next.**

| # | Question | Status | Blocked by |
|---|---|---|---|
| [R1](R1-does-generation-help-the-juglet.md) | Does CRAG's generative shape prior help on the Juglet where pure assembly fails? | open — blocked on upstream checkpoint release | none |

## Related

Workspace questions in [`../../intent/`](../../intent/) — `U1` (judging without an answer
key), `U2` (perception or placement), `U7` (what is actually being compared). The
comparison partners have their own folders: `../../tora/intent/`,
`../../GARF/intent/`.
