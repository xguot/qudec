PYTHON ?= python3
LATEXMK ?= latexmk

.PHONY: all data notes clean

all: data
	$(LATEXMK) -pdf -interaction=nonstopmode -halt-on-error -file-line-error -outdir=build qudec_phd_slides.tex

data:
	$(PYTHON) build_data.py

notes: data
	$(LATEXMK) -pdf -interaction=nonstopmode -halt-on-error -file-line-error -outdir=build qudec_presenter.tex

clean:
	$(RM) -r build
