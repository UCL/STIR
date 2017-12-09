Copyright (C) 2013 University College London
This file is part of STIR.

This file is free software; you can redistribute it and/or modify
it under the terms of the GNU Lesser General Public License as published by
the Free Software Foundation; either version 2.1 of the License, or
(at your option) any later version.

This file is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Lesser General Public License for more details.

See STIR/LICENSE.txt for details

This directory contains test scripts for calling STIR from Python.

INSTALLATION
============

See the README.txt in the swig sub-directory.

TESTING
=======
You will need py.test. If there is no package for it on your system, it should be
easy to install, see http://pytest.org.

Then, running the tests is simple. For example
```
$ py.test tests_buildblock.py 
```
You should then see something like
```
============================= test session starts ==============================
platform linux2 -- Python 2.7.2 -- pytest-1.3.4
test path 1: tests_buildblock.py

tests_buildblock.py ...........

========================== 11 passed in 0.10 seconds ===========================
```

If there is a failure, you would see this clearly indicated. For instance
```
============================= test session starts ==============================
platform linux2 -- Python 2.7.2 -- pytest-1.3.4
test path 1: tests_buildblock.py

tests_buildblock.py ...........F

=================================== FAILURES ===================================
___________________________ test_illustrate_failure ____________________________

    def test_illustrate_failure():
>       assert 0
E       assert 0

tests_buildblock.py:183: AssertionError
===================== 1 failed, 11 passed in 0.18 seconds ======================

```
