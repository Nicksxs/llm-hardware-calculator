# Capacity policy v1

These configurable allowances are planning choices, not measured backend requirements:

| Component | Default GiB |
| --- | ---: |
| GPU runtime buffers | 0.5 |
| GPU safety margin | 1.0 |
| Host runtime buffers | 1.0 |
| OS and other host applications | 4.0 |
| Host loading workspace | 1.0 |
| Additional GPU loading staging | 0.25 |

The core accepts a custom `BudgetPolicy`, including zero allowances for synthetic tests. CLI overrides use GiB/GB units. Capacity inputs represent budgets **before** these allowances; if using already-free RAM, adjust the OS reserve to avoid counting the same consumption twice.

At split `n` of `L` uniform layers, both weights `W` and KV payload `K` follow the GPU fraction `n/L`. Non-layer tensors and backend-specific placement can violate this approximation.

- GPU steady budget: `W × n/L + K × n/L + GPU runtime + GPU margin`.
- GPU loading peak: GPU steady budget plus GPU staging.
- Host steady budget: `W × (1−n/L) + K × (1−n/L) + host runtime + OS reserve`.
- Host loading peak: `W + K × (1−n/L) + host runtime + OS reserve + loading workspace`.

The loading policy conservatively retains **one full host copy of the weights** during loading, replacing the steady CPU weight component. It does not add another full copy on top of CPU weights. This is a policy scenario, not a claim about every mmap or GPU loader. OS page residency, initialization graph workspace, fragmentation, actual buffer size and tensor placement remain runtime dependent. All GPU-only allowances become zero when `n = 0`.

A placement is acceptable only when both steady and loading budgets fit their respective GPU and host capacities. No capacity estimate guarantees successful loading or acceptable token throughput.
