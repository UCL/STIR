# Developer documentation: how to install (software for) git hooks

You first need to have Python and pip

## Install [pre-commit](https://pre-commit.com)
See https://pre-commit.com/#install but the following might work.

    sudo -H pip install pre-commit

If that fails with a message about `PyYAML` and `distutils, try

    sudo -H pip install --ignore-installed PyYAML

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

