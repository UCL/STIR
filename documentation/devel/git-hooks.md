# Developer documentation: how to install (software for) git hooks

You first need to have Python and pip

## Install [pre-commit](https://pre-commit.com)
See https://pre-commit.com/#install but the following might work.

    pip install pre-commit

If this fails with a permission error, try adding `--user` to the command.

If that fails with a message about `PyYAML` and `distutils`, try

    pip install --ignore-installed PyYAML

## Install clang-format
### debian/Ubuntu
    sudo apt install clang-format
### MacOS
    brew install clang-format
### Others
search the internet and tell us

## Enable pre-commit hooks
```sh
cd /whereever/STIR
pre-commit install
```

If you need to work with a branch that was forked prior to in inclusion of clang-format, you will need to temporarily disable/uninstall pre-commit again:

    pre-commit uninstall

or once-off with

    git commit --no-verify
