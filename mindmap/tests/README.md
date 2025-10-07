# Tests for Nvblox Diffusion Policy

## Regression tests
Regression tests compare an algorithm’s output data with stored reference data to detect unintended changes in behavior. If a regression test fails, first determine whether the new behavior is indeed expected. If so, regenerate the baseline data by using the `--generate_baseline` flag in pytest, then commit the updated dataset to Git LFS. Note that Git LFS has limited storage, so be mindful to keep file sizes reasonable.
