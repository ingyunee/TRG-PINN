# Release checklist

- [ ] Select and approve the final software license.
- [ ] Confirm institutional intellectual-property requirements.
- [ ] Add the final `LICENSE` file.
- [ ] Confirm the author spelling and order in `CITATION.cff`.
- [ ] Insert the final article DOI when available.
- [ ] Insert the software/Zenodo DOI after the external artifact upload.
- [ ] Run `python scripts/verify_source_release.py` on the private/source checkout.
- [ ] Run `python scripts/verify_release.py` after external artifacts are installed.
- [ ] Confirm no Git-tracked file exceeds the host size limit.
- [ ] Upload specialized-baseline checkpoints and FV1024 reference separately.
- [ ] Verify external archives against the SHA-256 manifests.
- [ ] Use the STEP 9B curated release directory/ZIP, not the research workspace.
