# Final submission review

Reviewed September 26, 2026.

- Inspected all 15 main-manuscript pages and all 12 supplementary pages as rendered images.
- Checked all 11 figures and 14 tables for clipping, overlap, margins, captions, labels, units, and readable text. No layout defects requiring correction were found.
- The new structure figure shows the dasatinib seed and five analogs selected from the audited pool by saved predicted KIBA score. Structures are monochrome, with wavy bonds preserving unspecified stereochemistry.
- The other figures use Okabe-Ito working colors and cividis with labels, shapes, line styles, or patterns. The palette screen passes for the colors actually used together. This practical review is not a formal accessibility certification.
- All included figures have vector PDFs and PNGs at at least 600 dpi at their actual manuscript display widths. The smallest recorded font spans are mathematical subscripts or exponents.
- Final logs contain no overfull or underfull boxes, undefined references, missing assets, or other detected TeX/pdfTeX warnings. No rendered em dashes were found.
- The abstract has 218 words, or 230 with hyphenated terms split. All 14 table headings are shorter than 20 words; the longest has 17.
- All 38 checked headline values match their saved artifacts. Eight reference-inference and atom-alignment tests passed, including a local trained-checkpoint forward pass.
- The self-contained source archive rebuilt independently with identical extracted text and page counts for the main, supplement, and combined PDF. Title and author metadata match the final sources.

The minimal reference reuses the production model. Original weights and protein caches are not distributed; its explicit random mode checks tensor flow and does not estimate affinity.
