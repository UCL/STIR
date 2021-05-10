# sed script to add PARAPET license to the spdx identifier.
# Copyright (C) 2021 University College London
# SPDX-License-Identifier: Apache-2.0
# Author: Kris Thielemans
#
s/(SPDX-License-Identifier: .*)$/\1 AND License-ref-PARAPET-license/
s/( AND License-ref-PARAPET-license)\1/\1/
