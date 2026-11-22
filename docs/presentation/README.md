# Decoder benchmark slides

Five Beamer slides based on the [audited benchmark results](../results/21501046/README.md).
Edit the slides in `qudec_phd_slides.tex` and the timed speaker notes in `narration.json`.

Build from the repository root with Python 3, Beamer, PGFPlots, Latin Modern, and `latexmk`.

```sh
make -C docs/presentation        # build/qudec_phd_slides.pdf
make -C docs/presentation notes  # build/qudec_presenter.pdf with speaker notes
make -C docs/presentation clean  # remove generated files
```

Both builds write their PDFs, plot data, and recording script to `docs/presentation/build/`.
