# Plan review — revised 2026-09-29

## Critical

None. The revised master contract now specifies serializable mappings, project/snapshot/run paths, `load_snapshot`, run discovery/loading, verified run selection, forecast fields, and worker status/stop identity. Subtasks 01 and 02 point to those contracts and a shared fixture.

## Warnings

None. Generic signal handling now has an explicit canonical column, causal input allowlist, signed output domain, and missing-row segment boundaries. New XJTU/HSE projects use scoped raw directories and snapshots; HSE author-test and official RUL restrictions are explicit. The one-manual-folder ratio is defined over automatic groups with realized counts shown. Worker launch, cancellation, and archive checks have concrete job identity rules.

## Suggestions

- Align the linked legacy deletion wording in the master plan's early architecture and final decision bullets, and in subtasks 01/04. The detailed interface gives linked projects a small owned metadata root that can contain new signal runs. Deletion should tombstone the link and move **only that owned root** to recoverable trash; original legacy raw/processed/research paths remain untouched. The intended behavior is clear from the revised contract and does not block implementation.
- In C's UI test, explicitly assert that Results stays locked for a linked legacy project until it has a verified new signal run. This follows the master contract and protects against treating an old classifier/RUL run as a signal forecast.

## Verdict

**APPROVED.** Proceed with A first; publish its fixture and DTOs before B/C integration. Review each implementation slice against the stated isolation, causal, and archive acceptance criteria.
