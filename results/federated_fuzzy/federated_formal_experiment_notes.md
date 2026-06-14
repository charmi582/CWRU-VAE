# Federated Fuzzy Formal Experiment Notes

Setting: Paderborn normal-only clients, client partitions by bearing and by operating condition, seeds 42/202/777, FedAvg rounds=10, local epochs=2, centralized epochs=20, max files per condition=1. Fault windows are audit-only.

## Research questions

- RQ1: Can FedAvg approach centralized training without sharing raw windows?
- RQ2: Does FedAvg improve or stabilize local-only training?
- RQ3: Which clients show false alarm or miss-rate drift?
- RQ4: Can fuzzy health index expose client-to-client health-score drift?

## Main interpretation

- CNN-AE is the main viable federated model in this setting.
- FedAvg is privacy-preserving but does not automatically outperform local-only or centralized training in F1.
- FedAvg can reduce some client-stability variation, especially for miss-rate variation in condition clients, but the tradeoff depends on the partition and policy.
- VAE remains unstable under the current non-IID federated setting and should be reported as supplementary evidence that model choice matters.
- The strongest contribution is the comparison framework: local-only vs centralized vs federated, client stability, and fuzzy health drift under normal-only training.
