Copyright (C) 2013 University College London
This file is part of STIR.

SPDX-License-Identifier: Apache-2.0

See STIR/LICENSE.txt for details

This directory contains test scripts for calling STIR from Python.

INSTALLATION
============

See also the README.txt in the swig sub-directory.

You will need `py.test`. If there is no package for it on your system, it should be
easy to install, see http://pytest.org.

TESTING
=======
Running the tests is simple as `pytest` will discover test files itself
(i.e. all files called `test_*.py`). 
```bash
$ python -m pytest
```
Or you can run a single test file, for example
```bash
$ python -m pytest test_buildblock.py 
```
You should then see something like
```
============================= test session starts ==============================
platform linux2 -- Python 2.7.2 -- pytest-1.3.4
test path 1: test_buildblock.py

test_buildblock.py ...........

========================== 11 passed in 0.10 seconds ===========================
```

If there is a failure, you would see this clearly indicated. For instance
```
============================= test session starts ==============================
platform linux2 -- Python 2.7.2 -- pytest-1.3.4
test path 1: test_buildblock.py

test_buildblock.py ...........F

=================================== FAILURES ===================================
___________________________ test_illustrate_failure ____________________________

    def test_illustrate_failure():
>       assert 0
E       assert 0

test_buildblock.py:183: AssertionError
===================== 1 failed, 11 passed in 0.18 seconds ======================

```
