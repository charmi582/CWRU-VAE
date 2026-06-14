# IMS External Validation Plan

IMS Bearing is selected as the next external validation dataset for the federated journal paper. The dataset is hosted by NASA PCoE and contains run-to-failure bearing experiments donated by the Center for Intelligent Maintenance Systems, University of Cincinnati.

## Why IMS

- It provides run-to-failure bearing degradation records rather than only artificially induced fault classes.
- It is aligned with industrial health monitoring and predictive-maintenance scenarios.
- It supports a normal-only anomaly-monitoring protocol by treating early-life files as healthy and late-life files as audit faults.
- It complements Paderborn, which is currently the main external federated benchmark in the project.

## Official Source

- NASA PCoE IMS Bearings download: `https://phm-datasets.s3.amazonaws.com/NASA/4.+Bearings.zip`
- Approximate archive size: 1.07 GB
- NASA citation: J. Lee, H. Qiu, G. Yu, J. Lin, and Rexnord Technical Services (2007). IMS, University of Cincinnati. “Bearing Data Set”, NASA Prognostics Data Repository, NASA Ames Research Center, Moffett Field, CA.

## Planned Protocol

- Clients by bearing: each IMS test-bearing channel is treated as a client.
- Clients by test condition: each run-to-failure experiment is treated as a condition client.
- Normal-only training: earliest 20% of files from each test.
- Fault audit: latest 20% of files from each test.
- Middle degradation region: excluded from the first external anomaly-monitoring protocol.
- Main comparison: local-only, centralized, FedAvg, FedProx, FedBN and personalized variants.
- Calibration policies: client-specific, pooled and adaptive drift-aware calibration.

## Reproducible Command

```bash
python ims_external_validation.py --download --extract --summarize
```

The local readiness report is written to `results/external/ims/ims_external_validation_readiness.md`, but `results/external/` is intentionally ignored by git because the external datasets are large.
