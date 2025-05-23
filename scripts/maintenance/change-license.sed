# sed script to replace the GNU license statement with an Apache 2.0 one.
# Copyright (C) 2021 University College London
# SPDX-License-Identifier: Apache-2.0
# Author: Kris Thielemans
#
# Thanks to https://www.grymoire.com/Unix/Sed.html
#
/.*This file is free software; you can redistribute it and\/or modify.*/ {
# first put 10 lines into the pattern space (as the whole GNU license statement takes ~8 lines)
N;
N;
N;
N;
N;
N;
N;
N;
N;
N;
# now replace the GNU license statement with an Apache one
# preserving any characters used at the start of the line
  s#(.*)This file is free software; you can redistribute it and/or modify.*\n\1it under the terms.*GNU .*General Public License for more details.#\1SPDX-License-Identifier: Apache-2.0#;
}
