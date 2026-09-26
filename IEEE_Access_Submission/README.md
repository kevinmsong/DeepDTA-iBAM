# IEEE Access submission

**DeepDTA-iBAM: Cross-Attention Interaction Mapping for Affinity Prediction,
Ligand Retrieval, and Analog Generation**

## Deliverables

- [Main manuscript](output/pdf/main.pdf)
- [Supplementary material](output/pdf/supplementary.pdf)
- [Combined reading copy](output/pdf/combined.pdf)
- [Self-contained submission source](output/ieee_submission_source.zip)
- [Figure package: vector PDFs and 600 dpi PNGs](output/figures_600dpi.zip)

The main uses the official IEEE Access author template. See
[template provenance](TEMPLATE_SOURCE.md) for its source and the limited
submission-specific preamble adjustments. The final analog structure figure
uses saved model predictions and archived docking results; it does not report
experimental potency or synthetic feasibility.

## Rebuild

Install pdfLaTeX and BibTeX in a current TeX distribution, plus Python with
PyMuPDF and Pillow. From the repository root:

```bash
python -m pip install pymupdf pillow
python analysis/package_ieee_submission.py
```

The script builds the main, supplement, and combined PDF; checks abstract and
table-heading word limits, em dashes, cross-references, and layout warnings;
exports included figures at 600 dpi at their actual display widths; and rebuilds
the self-contained source archive to verify matching pages and extracted text.
It does not run inference or regenerate scientific results.

`output/qa/validation.json` contains automated checks and file hashes. A separate
visual review checks pages and figures after each final build.

The repository also includes a [minimal reference implementation](../reference/README.md).
Original trained weights and protein caches are not distributed. The explicit
random demonstration checks architecture execution and does not estimate affinity.
