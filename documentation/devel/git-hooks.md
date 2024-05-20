# Developer documentation: how to install (software for) git hooks

We use [pre-commit](https://pre-commit.com) to check coding conventions, including
white-space formatting rules.

Unfortunately, exact formatting used depends on the `clang-format` version. Therefore,
you do have to install the same version as what is currently used in our GitHub Action.

## Installation of software

We highly recommend to use `conda`:
```sh
cd /whereever/STIR
conda env create --file pre-commit-environment.yml
conda activate pre-commit-env
```

### Alternative
You first need to have Python and pip

### Install [pre-commit](https://pre-commit.com)
See https://pre-commit.com/#install but the following might work.

    pip install pre-commit

If this fails with a permission error, try adding `--user` to the command.

If that fails with a message about `PyYAML` and `distutils`, try

    pip install --ignore-installed PyYAML

### Install clang-format (correct version)
At the time of writing, the `clang-format` version in `pre-commit-environment.yml`
is the same default version in Ubuntu 22.04, so if you are using that, you could do

    sudo apt install clang-format

## Enable pre-commit hooks
```sh
cd /whereever/STIR
pre-commit install
```

If you need to work with a branch that was forked prior to in inclusion of clang-format, you will need to temporarily disable/uninstall pre-commit again:

    pre-commit uninstall

or one-off with

    git commit --no-verify
