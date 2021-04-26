#!/usr/bin/env bash

# This is an example script for use with clang-tidy. 
# Use after coniguring the stir build in a build
# folder, with -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
# so that 
#
#    compile_commands.json
#
# exists. Then run this with with clang-tidy like this:
#
# ./modernize-use-override.sh [FOLDER-WITH-COMPILE_COMMANDS.json]
#
# see:
#
# https://www.kdab.com/clang-tidy-part-1-modernize-source-code-using-c11c14/
#
# Licensed under the Apache License, Version 2.0
# See STIR/LICENSE.txt for details
#
# Copyright 2021 Positrigo, Max Ahnen
#

run-clang-tidy -p $1 -header-filter=.* -checks='-*,modernize-use-override' -fix