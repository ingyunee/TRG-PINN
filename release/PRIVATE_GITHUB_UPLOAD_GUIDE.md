# Private GitHub upload guide

This release notebook does not connect to or upload anything to GitHub.

## 1. Create a private repository manually

Sign in to GitHub as `ingyunee` and create:

```text
Repository name: TRG-PINN
Visibility: Private
```

Do not initialize it with another README or `.gitignore`; the curated source
ZIP already contains those files. Leave the GitHub license selector empty
for now because the private pre-submission package intentionally has no
approved `LICENSE` yet.

## 2. Extract the local ZIP

After STEP 9B finishes, extract:

```text
TRG-PINN-v1.0.0-source.zip
```

The extracted folder is:

```text
TRG-PINN-v1.0.0
```

## 3. Upload manually

You may use GitHub Desktop or Git from the extracted folder.

Example Git commands after the private repository exists:

```bash
git init
git add .
git commit -m "Initial private reproducibility package v1.0.0"
git branch -M main
git remote add origin https://github.com/ingyunee/TRG-PINN.git
git push -u origin main
```

These commands are instructions only. The STEP 9B notebook does not execute
them.

## 3. Verify the private source package

Run immediately after upload:

```bash
python scripts/verify_source_release.py
```

`verify_release.py` also works in source-only mode and skips only the full
external-artifact check until those files are installed.

## 4. Keep the repository private before submission

The repository may remain private during manuscript preparation and submission.
Only invited collaborators can access it.

## 5. Before making it public

Complete these items first:

- choose and add the final software `LICENSE`;
- confirm `CITATION.cff`;
- add article/software DOI metadata when available;
- run `python scripts/verify_source_release.py`;
- review `docs/KNOWN_LIMITATIONS.md`;
- verify the external binary archive.

## 6. Later change visibility manually

When the authors decide the code can be public, change the repository
visibility from **Private** to **Public** in the GitHub repository settings.

No automatic visibility change is performed by this package.
