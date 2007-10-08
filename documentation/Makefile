#
# $Id$
#
# Author Kris Thielemans
# Copyright 2004- $Date$ Hammersmith Imanet Ltd
# This file is part of STIR, and is distributed under the 
# terms of the GNU Lesser General Public Licence (LGPL) Version 2.1.


all: STIR-glossary.pdf STIR-general-overview.pdf \
     STIR-developers-overview.pdf \
     STIR-UsersGuide.pdf \
     STIR-version2-vs-version1.pdf

clean:
	$(RM) STIR-*.pdf *.log *.dvi *.aux *.toc *.out

.PHONE: clean
# rule to make pdf from tex source
# note: need to run twice to get page refs etc ok.
%.pdf: %.tex
	pdflatex $<
	pdflatex $<
