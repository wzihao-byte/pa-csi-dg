# Paper and implementation review

- User preference: when a manuscript expression is ambiguous or questioned by a review, inspect the actual implementation and the configuration of the cited experiment before interpreting or revising it. Do not treat a reviewer's characterization as implementation evidence.
- Distinguish software conventions from physical claims. In particular, the historical `AntennaViewBuilder` configuration `mode=tx, num_rx=1, num_tx=3, num_subcarriers=30` defines three **pseudo-Tx feature-group views**. It does not assert three physical transmitting antennas. Preserve its slicing semantics and explain the convention in prose.
- Tie each reported result to its own code/configuration and stored experiment evidence. If an expression remains unresolved after checking those sources, record the uncertainty instead of guessing its meaning.
