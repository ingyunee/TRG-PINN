# Known limitations and disclosures

1. **gPINN helper reconstruction.** The saved gPINN method blocks are exact,
   but two helper definitions were absent from the final notebooks. The public
   helper implementation is reconstructed and disclosed. Reported gPINN
   metrics come from frozen artifacts.

2. **Checkpoint-extra metadata.** Checkpoint `extra` fields can contain earlier
   evaluation values. Final reporting uses `metrics_final.json` and
   `all_metrics_final.csv`.

3. **2D shallow-water reference sensitivity.** FV512 and FV1024 preserve the
   reported method ranking, but the reference-resolution discrepancy is
   material and is reported in the manuscript.

4. **Conservation trade-offs.** Improved discontinuity reconstruction does not
   imply uniform improvement of every conservation diagnostic.

5. **Smoke-test scope.** Every benchmark runtime imports successfully and all
   eight specialized methods pass an isolated 1D Burgers smoke test. The
   complete 48 benchmark-method fresh-training matrix is not claimed to replace
   the frozen reported artifacts.
