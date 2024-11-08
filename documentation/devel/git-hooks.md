# Developer documentation: how to install (software for) git hooks

We use [pre-commit](https://pre-commit.com) to check coding conventions, including
white-space formatting rules.

Unfortunately, exact formatting used depends on the `clang-format` (and others?) version. Therefore,
you might need to install the same version as what is currently used in our
[.pre-commit-config.yaml](../../.pre-commit-config.yaml) for your editor to pick up
the same version.

## Installation of software

### TL;DR

```sh
pip/conda install pre-commit
git add blabla
pre-commit run
# optionally check files
# "add" changes made by the hooks
git add blabla
git commit
```

### More detail

We highly recommend to use `conda`:
```sh
cd /whereever/STIR
conda env create --file pre-commit-environment.yml
conda activate pre-commit-env
```

Alternative:

1. Install Python and pip

2. Install [pre-commit](https://pre-commit.com). See https://pre-commit.com/#install but the following might work.
   ```sh
   pip install pre-commit
   ```
   If this fails with a permission error, try adding `--user` to the command. If that fails with a message about `PyYAML` and `distutils`, try
   ```sh
   pip install --ignore-installed PyYAML
   ```

3. Install clang-format (but use correct version). At the time of writing, on Ubuntu 22.04, you can do
   ```sh
   sudo apt install clang-format
   ```

## Enable pre-commit hooks

```sh
cd /whereever/STIR
pre-commit install
```

If you need to work with a branch that was forked prior to our use of `clang-format`, you will need to temporarily disable/uninstall pre-commit again:

    pre-commit uninstall

or one-off with

    git commit --no-verify

You will then need to run `pre-commit run --all-files` when done before we will merge your pull request.
