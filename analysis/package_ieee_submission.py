"""Build and package the active IEEE Access manuscript reproducibly.

Requires pdflatex, bibtex, PyMuPDF, and Pillow. Run from any directory:
    python analysis/package_ieee_submission.py

The default manuscript directory is IEEE_Access_Submission. Builds and source
bundle verification use isolated directories under its tmp/ folder. Only files
actually consumed by LaTeX, the active bibliography/style files, and a rebuild
README enter the source ZIPs. Checkpoints, caches, raw data, and build auxiliaries
are never included. Archived manuscript sources are not modified.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

import fitz
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent.parent
DOCUMENTS = ("main", "supplementary", "combined")
AUXILIARY_SUFFIXES = {".aux", ".bbl", ".blg", ".log", ".out", ".fls", ".toc",
                      ".lof", ".lot", ".synctex", ".fdb_latexmk"}
SOURCE_ASSET_SUFFIXES = {".tex", ".bib", ".bst", ".cls", ".sty", ".pdf", ".png",
                         ".jpg", ".jpeg", ".eps", ".pfb", ".pfa", ".tfm", ".vf",
                         ".fd", ".map", ".enc", ".def", ".clo"}
WARNING_PATTERN = re.compile(r"Overfull|Underfull|undefined|LaTeX Warning|Package \S+ Warning|pdfTeX warning")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def strip_comments(text: str) -> str:
    return re.sub(r"(?<!\\)%[^\n]*", "", text)


def braced(text: str, start: int) -> tuple[str, int]:
    if text[start] != "{":
        raise ValueError("Expected opening brace")
    depth = 1
    for i in range(start + 1, len(text)):
        if text[i] == "{" and text[i - 1] != "\\":
            depth += 1
        elif text[i] == "}" and text[i - 1] != "\\":
            depth -= 1
            if depth == 0:
                return text[start + 1:i], i + 1
    raise ValueError("Unbalanced TeX braces")


def prose(text: str) -> str:
    text = strip_comments(text).replace("{,}", ",")
    text = text.replace(r"\%", "%").replace(r"\,", " ").replace("~", " ")
    text = re.sub(r"\\[A-Za-z]+\*?", "", text)
    return re.sub(r"[{}$]", "", text).strip()


def words(text: str, split_hyphens: bool = True) -> list[str]:
    pattern = r"[A-Za-z0-9]+(?:[,.][0-9]+)*"
    if not split_hyphens:
        pattern += r"(?:-[A-Za-z0-9]+)*"
    return re.findall(pattern, prose(text).replace("--", "-"))


def source_editorial_checks(manuscript: Path) -> dict:
    main = strip_comments((manuscript / "main.tex").read_text(encoding="utf-8"))
    abstract = main.split(r"\begin{abstract}", 1)[1].split(r"\end{abstract}", 1)[0]
    abstract_counts = {"words": len(words(abstract, False)),
                       "words_splitting_hyphenated_terms": len(words(abstract))}
    if max(abstract_counts.values()) >= 250:
        raise ValueError(f"Abstract must be shorter than 250 words: {abstract_counts}")
    tables = []
    for document in ("main", "supplementary"):
        text = strip_comments((manuscript / f"{document}.tex").read_text(encoding="utf-8"))
        body = text.split(r"\begin{document}", 1)[1]
        for number, match in enumerate(re.finditer(
                r"\\begin\{(table\*?|smalltable)\}(.*?)\\end\{\1\}", body, re.S), 1):
            caption_match = re.search(r"\\caption(?:\[[^\]]*\])?\s*\{", match.group(2))
            if not caption_match:
                raise ValueError(f"Missing caption in {document} table {number}")
            caption, _ = braced(match.group(2), caption_match.end() - 1)
            count = len(words(caption))
            if count >= 20:
                raise ValueError(f"Table heading must be shorter than 20 words: {document} {number}: {caption}")
            tables.append({"document": document, "table_number": number,
                           "heading": prose(caption), "words": count,
                           "words_including_table_number_label": count + 2})
        if "\u2014" in body or r"\textemdash" in body or "---" in body:
            raise ValueError(f"Em dash found in authored {document} body")
    return {"abstract": abstract_counts, "table_headings": tables,
            "table_count": len(tables), "maximum_table_heading_words": max(x["words"] for x in tables)}


def run(command: list[str], cwd: Path, transcript: Path, environment: dict) -> None:
    result = subprocess.run(command, cwd=cwd, env=environment, stdin=subprocess.DEVNULL,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=240)
    transcript.write_bytes(result.stdout)
    if result.returncode:
        tail = result.stdout.decode(errors="replace")[-6000:]
        raise RuntimeError(f"Command failed ({result.returncode}): {' '.join(command)}\n{tail}")


def stage_build_inputs(manuscript: Path, staging: Path) -> tuple[Path, dict[str, str]]:
    """Copy source assets, never auxiliaries, into a fresh relative layout.

    The repository manuscript uses ../results. An extracted self-contained
    archive instead contains all assets below its own root and must not borrow
    any files from the surrounding repository during verification.
    """
    manuscript = manuscript.resolve()
    repository_layout = manuscript.parent == ROOT
    source_root = ROOT if repository_layout else manuscript
    directories = [manuscript]
    if repository_layout:
        directories.append(ROOT / "results")
    excluded = {"tmp", "output", ".build", ".git", "__pycache__", "cache",
                "raw", "checkpoints", "J_Cheminform_latex_bundle"}
    generated_pdfs = {f"{stem}.pdf" for stem in DOCUMENTS}
    mapping = {}
    for directory in directories:
        for current, subdirectories, filenames in os.walk(directory):
            subdirectories[:] = [name for name in subdirectories if name not in excluded]
            for name in filenames:
                source = Path(current) / name
                if source.suffix.lower() not in SOURCE_ASSET_SUFFIXES or name in generated_pdfs:
                    continue
                target = staging / source.relative_to(source_root)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, target)
                mapping[str(target.resolve())] = str(source.resolve())
    return staging / manuscript.relative_to(source_root), mapping


def build_documents(manuscript: Path, build: Path, engines: dict) -> dict:
    build = build.resolve()
    build.mkdir(parents=True, exist_ok=True)
    commands = {}
    with tempfile.TemporaryDirectory(prefix="sources_", dir=build) as temporary:
        staging = Path(temporary).resolve()
        if not staging.is_relative_to(build):
            raise ValueError("Unsafe temporary build path")
        cwd, mapping = stage_build_inputs(manuscript, staging)
        (build / "staged_inputs.json").write_text(json.dumps(mapping, indent=2) + "\n", encoding="utf-8")
        environment = dict(os.environ)
        # The empty component retains standard TeX paths. Do not add the
        # original source directory: it may contain stale .aux/.out/.bbl files.
        for name in ("TEXINPUTS", "BIBINPUTS", "BSTINPUTS", "TEXFONTMAPS", "TFMFONTS", "T1FONTS"):
            environment[name] = str(cwd) + os.pathsep
        try:
            for stem in DOCUMENTS:
                source = cwd / f"{stem}.tex"
                if not source.exists():
                    raise FileNotFoundError(source)
                command = [engines["pdflatex"], "-interaction=nonstopmode", "-halt-on-error",
                           "-file-line-error", "-recorder", source.name]
                commands[stem] = command
                all_passes = build / f"{stem}.all_passes.fls"
                all_passes.write_text("", encoding="utf-8")

                def tex_pass(iteration: int) -> None:
                    run(command, cwd, build / f"{stem}.pass{iteration}.stdout.txt", environment)
                    # Unsettled citations/page breaks can exercise fonts that
                    # the final pass never loads. Retain every pass's inputs.
                    with all_passes.open("a", encoding="utf-8") as handle:
                        handle.write((cwd / f"{stem}.fls").read_text(errors="replace") + "\n")

                tex_pass(1)
                auxiliary = cwd / f"{stem}.aux"
                if r"\bibdata{" in auxiliary.read_text(errors="replace"):
                    run([engines["bibtex"], stem], cwd, build / f"{stem}.bibtex.stdout.txt", environment)
                    for iteration in (2, 3):
                        tex_pass(iteration)
                else:
                    tex_pass(2)
                # Some class/longtable combinations need an additional pass.
                for iteration in (4, 5):
                    log = (cwd / f"{stem}.log").read_text(errors="replace")
                    if not re.search(r"Rerun to get|Label\(s\) may have changed|Please rerun", log):
                        break
                    tex_pass(iteration)
                print(f"Built {stem}.pdf", flush=True)
        finally:
            # Preserve outputs and diagnostic logs after the fresh tree is
            # removed. Combined.pdf was built from the new staged PDFs.
            for stem in DOCUMENTS:
                for path in cwd.glob(f"{stem}.*"):
                    if path.suffix.lower() in AUXILIARY_SUFFIXES | {".pdf"}:
                        shutil.copy2(path, build / path.name)
    return commands


def inspect_documents(build: Path) -> tuple[dict, list[str]]:
    reports, errors = {}, []
    for stem in DOCUMENTS:
        log = (build / f"{stem}.log").read_text(errors="replace")
        warnings = [line.strip() for line in log.splitlines() if WARNING_PATTERN.search(line)]
        with fitz.open(build / f"{stem}.pdf") as document:
            text = "\n".join(page.get_text() for page in document)
            report = {"pages": len(document), "sha256": sha(build / f"{stem}.pdf"),
                      "text_sha256": hashlib.sha256(text.encode()).hexdigest(),
                      "em_dash_count": text.count("\u2014"), "unresolved_marker_count": text.count("??"),
                      "pending_asset_count": len(re.findall(r"Pending\s+(?:figure|table)", text, re.I)),
                      "log_warnings": warnings,
                      "page_dimensions_points": [[page.rect.width, page.rect.height] for page in document]}
        if warnings or report["em_dash_count"] or report["unresolved_marker_count"] or report["pending_asset_count"]:
            errors.append(f"{stem}: {len(warnings)} log warnings, {report['em_dash_count']} em dashes, "
                          f"{report['unresolved_marker_count']} unresolved markers, "
                          f"{report['pending_asset_count']} missing-asset placeholders")
        reports[stem] = report
    if reports["combined"]["pages"] != reports["main"]["pages"] + reports["supplementary"]["pages"]:
        errors.append("Combined page count differs from main plus supplement")
    return reports, errors


def recorder_inputs(build: Path, manuscript: Path) -> set[Path]:
    inputs = set()
    manifest = build / "staged_inputs.json"
    staged = {Path(key): Path(value) for key, value in
              json.loads(manifest.read_text(encoding="utf-8")).items()} if manifest.exists() else {}
    for stem in DOCUMENTS:
        recorder_path = build / f"{stem}.all_passes.fls"
        if not recorder_path.exists():
            recorder_path = build / f"{stem}.fls"
        recorder = recorder_path.read_text(errors="replace")
        cwd = manuscript
        for line in recorder.splitlines():
            if line.startswith("PWD "):
                cwd = Path(line[4:]).resolve()
            elif line.startswith("INPUT "):
                token = line[6:].strip().strip('"')
                path = Path(token)
                path = (cwd / path).resolve() if not path.is_absolute() else path.resolve()
                path = staged.get(path, path)
                if path.is_file() and path.is_relative_to(ROOT) and not path.is_relative_to(build):
                    if path.suffix.lower() in SOURCE_ASSET_SUFFIXES:
                        inputs.add(path)
    # BibTeX inputs are not necessarily recorded by pdfTeX.
    for stem in ("main", "supplementary"):
        text = strip_comments((manuscript / f"{stem}.tex").read_text(encoding="utf-8"))
        for command, extension in (("bibliography", ".bib"), ("bibliographystyle", ".bst")):
            for match in re.finditer(r"\\" + command + r"\{([^}]+)\}", text):
                for name in match.group(1).split(","):
                    path = manuscript / (name.strip() + extension)
                    if path.exists():
                        inputs.add(path.resolve())
                    elif extension == ".bib":
                        raise FileNotFoundError(path)
    inputs.update((manuscript / f"{stem}.tex").resolve() for stem in DOCUMENTS)
    for name in ("TEMPLATE_SOURCE.md", "template_assets.json"):
        if (manuscript / name).is_file():
            inputs.add((manuscript / name).resolve())
    inputs = {p for p in inputs if p.name not in {f"{stem}.pdf" for stem in DOCUMENTS}}
    if not any(p.suffix == ".bib" for p in inputs):
        raise ValueError("No bibliography included in source dependency set")
    # Enforce the minimal-source boundary even if an accidental TeX input exists.
    for path in inputs:
        relative = path.relative_to(ROOT)
        if any(part in {"checkpoints", "cache", "raw", ".venv", "tmp", "output"} for part in relative.parts):
            raise ValueError(f"Unexpected private, cache, or build dependency: {relative}")
    return inputs


def tabular_heading_checks(inputs: set[Path]) -> list[dict]:
    headings = []
    for path in sorted(inputs):
        if path.suffix != ".tex":
            continue
        text = strip_comments(path.read_text(encoding="utf-8"))
        # Column headings lie between the first top rule and midrule in each
        # booktabs table. Caption checks are separate and handle balanced braces.
        for number, block in enumerate(re.findall(r"\\toprule(.*?)\\midrule", text, re.S), 1):
            for row in re.split(r"\\\\", block):
                for cell in re.split(r"(?<!\\)&", row):
                    if not prose(cell):
                        continue
                    count = len(words(cell))
                    if count >= 20:
                        raise ValueError(f"Column heading exceeds limit: {path.name}: {prose(cell)}")
                    headings.append({"source": str(path.relative_to(ROOT)), "table_block": number,
                                     "heading": prose(cell), "words": count})
    return headings


def geometry(log: str, document: str) -> dict:
    result = {}
    for key in ("TEXT", "COLUMN"):
        match = re.search(r"MANUSCRIPT_" + key + r"_WIDTH_PT\s*[=:]\s*([0-9.]+)(?:pt)?", log)
        if match:
            result[key.lower()] = float(match.group(1)) / 72.27
    if "text" not in result:
        match = re.search(r"\\textwidth\s*=\s*([0-9.]+)pt", log)
        if match:
            result["text"] = float(match.group(1)) / 72.27
    if "text" not in result:
        raise ValueError(f"Missing text-width marker for {document}; add \\typeout{{MANUSCRIPT_TEXT_WIDTH_PT=\\the\\textwidth}}")
    if "column" not in result:
        result["column"] = result["text"]
        if document == "main":
            raise ValueError("Missing main column-width marker")
    return result


def included_figures(manuscript: Path, build: Path, inputs: set[Path]) -> list[dict]:
    figures = []
    for document in ("main", "supplementary"):
        text = strip_comments((manuscript / f"{document}.tex").read_text(encoding="utf-8"))
        body = text.split(r"\begin{document}", 1)[1]
        widths = geometry((build / f"{document}.log").read_text(errors="replace"), document)
        for number, match in enumerate(re.finditer(r"\\begin\{(figure\*?)\}(.*?)\\end\{\1\}", body, re.S), 1):
            graphic_matches = list(re.finditer(r"\\includegraphics(?:\[([^\]]*)\])?\{([^}]+)\}", match.group(2)))
            if not graphic_matches:
                raise ValueError(f"No explicit figure file in {document} figure {number}")
            for image_number, graphic in enumerate(graphic_matches, 1):
                token = graphic.group(2)
                candidates = [(manuscript / token).resolve(), (ROOT / "results" / token).resolve()]
                path = next((p for p in candidates if p in inputs), None)
                if path is None:
                    matches = [p for p in inputs if p.name == Path(token).name]
                    if len(matches) == 1:
                        path = matches[0]
                if path is None or path.suffix.lower() != ".pdf":
                    raise ValueError(f"A used vector PDF is required for figure {document}/{number}: {token}")
                options = graphic.group(1) or ""
                width_match = re.search(r"width\s*=\s*([0-9.]*)\s*\\(textwidth|columnwidth|linewidth)", options)
                if width_match:
                    factor = float(width_match.group(1) or 1)
                    macro = width_match.group(2)
                    base = widths["text"] if macro == "textwidth" or (macro == "linewidth" and
                           (match.group(1) == "figure*" or document == "supplementary")) else widths["column"]
                    inches = factor * base
                else:
                    absolute = re.search(r"width\s*=\s*([0-9.]+)(in|cm|mm|pt)", options)
                    if not absolute:
                        raise ValueError(f"Unsupported or missing explicit figure width: {options}")
                    inches = float(absolute.group(1)) / {"in": 1, "cm": 2.54, "mm": 25.4, "pt": 72.27}[absolute.group(2)]
                label = str(number) if document == "main" else f"S{number}"
                if len(graphic_matches) > 1:
                    label += f".{image_number}"
                figures.append({"figure": label, "document": document, "source": path,
                                "display_width_inches": inches, "include_options": options})
    return figures


def export_figures(figures: list[dict], destination: Path) -> tuple[list[dict], list[Path]]:
    destination.mkdir(parents=True, exist_ok=True)
    reports, paths, names = [], [], set()
    for figure in figures:
        source = figure["source"]
        if source.stem in names:
            raise ValueError(f"Duplicate exported figure filename: {source.stem}")
        names.add(source.stem)
        vector = destination / source.name
        raster = destination / f"{source.stem}.png"
        shutil.copy2(source, vector)
        with fitz.open(source) as document:
            if len(document) != 1:
                raise ValueError(f"Figure must be a one-page PDF: {source}")
            page = document[0]
            target_pixels = math.ceil(600 * figure["display_width_inches"])
            scale = target_pixels / page.rect.width
            pixmap = page.get_pixmap(matrix=fitz.Matrix(scale, scale), alpha=False)
            image = Image.frombytes("RGB", [pixmap.width, pixmap.height], pixmap.samples)
            image.save(raster, dpi=(600, 600))
            spans = [span for block in page.get_text("dict")["blocks"] if "lines" in block
                     for line in block["lines"] for span in line["spans"] if span["text"].strip()]
            native_width = page.rect.width / 72
            with Image.open(raster) as exported:
                density = list(exported.info.get("dpi", (0, 0)))
            actual_dpi = pixmap.width / figure["display_width_inches"]
            if actual_dpi < 600 or not all(abs(value - 600) < 0.01 for value in density):
                raise ValueError(f"Raster resolution check failed for {source.name}")
            report = {key: value for key, value in figure.items() if key != "source"}
            report.update({"source": str(source.relative_to(ROOT)), "file_stem": source.stem,
                           "native_pdf_width_inches": native_width,
                           "pixels": [pixmap.width, pixmap.height], "png_dpi_metadata": density,
                           "effective_dpi_at_display_width": actual_dpi,
                           "vector_path_count": len(page.get_drawings()),
                           "pdf_image_objects": len(page.get_images(full=True)),
                           "minimum_text_points_at_display_width": min((s["size"] for s in spans), default=0)
                               * figure["display_width_inches"] / native_width,
                           "pdf_sha256": sha(vector), "png_sha256": sha(raster)})
            reports.append(report)
        paths.extend([vector, raster])
    metadata = destination / "figure_validation.json"
    metadata.write_text(json.dumps(reports, indent=2) + "\n", encoding="utf-8")
    readme = destination / "README.md"
    readme.write_text("# Submission figures\n\nVector PDFs are the actual figures included in the manuscript. "
                      "PNG copies are rendered from those PDFs at at least 600 pixels per inch at their "
                      "LaTeX display widths. PNG density metadata may read 599.9988 because of integer "
                      "pixels-per-meter storage. Captions remain in the manuscripts.\n\n"
                      "| Figure | File stem |\n|---|---|\n" +
                      "".join(f"| {x['figure']} | {x['file_stem']} |\n" for x in reports) +
                      "\nSee figure_validation.json for dimensions, text sizes, and SHA-256 hashes. "
                      "Automated packaging does not replace visual review.\n", encoding="utf-8")
    paths.extend([metadata, readme])
    return reports, paths


def render_qa(build: Path, destination: Path) -> list[str]:
    destination.mkdir(parents=True, exist_ok=True)
    sheets = []
    for stem in ("main", "supplementary"):
        with fitz.open(build / f"{stem}.pdf") as document:
            for start in range(0, len(document), 4):
                sheet = Image.new("RGB", (1420, 1920), "#dddddd")
                draw = ImageDraw.Draw(sheet)
                for i in range(start, min(start + 4, len(document))):
                    pixmap = document[i].get_pixmap(matrix=fitz.Matrix(1.5, 1.5), alpha=False)
                    image = Image.frombytes("RGB", [pixmap.width, pixmap.height], pixmap.samples)
                    image.save(destination / f"{stem}_{i + 1:02d}.png")
                    image.thumbnail((680, 910))
                    x, y = 15 + 710 * ((i - start) % 2), 25 + 955 * ((i - start) // 2)
                    sheet.paste(image, (x, y))
                    draw.text((x, y - 17), f"{stem}, page {i + 1}", fill="black")
                target = destination / f"{stem}_sheet_{start // 4 + 1:02d}.jpg"
                sheet.save(target, quality=90)
                sheets.append(target.name)
    return sheets


def deterministic_zip(destination: Path, members: dict[str, bytes]) -> None:
    with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for name, payload in sorted(members.items()):
            information = zipfile.ZipInfo(name.replace("\\", "/"), (1980, 1, 1, 0, 0, 0))
            information.compress_type = zipfile.ZIP_DEFLATED
            information.external_attr = 0o644 << 16
            archive.writestr(information, payload)


def bundle_destination(path: Path, manuscript: Path) -> Path:
    if path.is_relative_to(manuscript):
        return path.relative_to(manuscript)
    if path.is_relative_to(ROOT / "results"):
        relative = path.relative_to(ROOT / "results")
        return Path("tables" if path.suffix == ".tex" else "figures") / relative
    return Path("assets") / path.relative_to(ROOT)


def rewrite_file_arguments(text: str, original: Path, mapping: dict[Path, Path]) -> str:
    pattern = re.compile(r"(\\(?:input|include|includegraphics|IfFileExists|bibliography|bibliographystyle))"
                         r"(\[[^\]]*\])?\{([^}]+)\}")
    def replace(match):
        command, options, token = match.group(1), match.group(2) or "", match.group(3)
        extension = {r"\bibliography": ".bib", r"\bibliographystyle": ".bst",
                     r"\input": ".tex", r"\include": ".tex"}.get(command)
        new_tokens = []
        for item in token.split(",") if command == r"\bibliography" else [token]:
            candidate = (original.parent / item).resolve()
            choices = [candidate]
            if extension and candidate.suffix != extension:
                choices.insert(0, Path(str(candidate) + extension))
            found = next((p for p in choices if p in mapping), None)
            if found is None:
                matches = [p for p in mapping if p.name == Path(item).name]
                found = matches[0] if len(matches) == 1 else None
            if found is None:
                new_tokens.append(item)
                continue
            target = mapping[found].as_posix()
            if extension and not item.endswith(extension):
                target = target[:-len(extension)]
            new_tokens.append(target)
        return command + options + "{" + ",".join(new_tokens) + "}"
    return pattern.sub(replace, text)


def rebuild_readme(working_directory: str) -> str:
    return ("# IEEE Access submission source\n\n"
            "This archive contains only the active manuscript, supplement, combined wrapper, bibliography, "
            "required local template assets, tables, and included vector figures. It excludes checkpoints, "
            "training data, caches, obsolete drafts, and build auxiliaries.\n\n"
            "Install a current TeX distribution with pdfLaTeX, BibTeX, and the packages used by the sources. "
            "Official IEEE template assets are retained unchanged.\n\n"
            f"Run the following from `{working_directory}` after extracting:\n\n```text\n"
            "pdflatex -interaction=nonstopmode -halt-on-error main.tex\n"
            "bibtex main\npdflatex -interaction=nonstopmode -halt-on-error main.tex\n"
            "pdflatex -interaction=nonstopmode -halt-on-error main.tex\n"
            "pdflatex -interaction=nonstopmode -halt-on-error supplementary.tex\n"
            "bibtex supplementary\npdflatex -interaction=nonstopmode -halt-on-error supplementary.tex\n"
            "pdflatex -interaction=nonstopmode -halt-on-error supplementary.tex\n"
            "pdflatex -interaction=nonstopmode -halt-on-error combined.tex\n"
            "pdflatex -interaction=nonstopmode -halt-on-error combined.tex\n```\n\n"
            "Outputs are main.pdf, supplementary.pdf, and combined.pdf. Run BibTeX again if the bibliography "
            "changes. Render and visually inspect the final PDFs before submission. This source archive "
            "rebuilds the documents from saved figures; it does not rerun model analyses.\n")


def package_sources(inputs: set[Path], manuscript: Path, output: Path, engines: dict,
                    original_reports: dict, verify: bool) -> dict:
    mapping = {path: bundle_destination(path, manuscript) for path in inputs}
    if len(set(mapping.values())) != len(mapping):
        raise ValueError("Source bundle paths collide")
    self_members, repo_members = {}, {}
    for path, destination in sorted(mapping.items(), key=lambda item: item[1].as_posix()):
        payload = path.read_bytes()
        repo_members[path.relative_to(ROOT).as_posix()] = payload
        if path.suffix == ".tex":
            payload = rewrite_file_arguments(payload.decode("utf-8"), path, mapping).encode("utf-8")
        self_members[destination.as_posix()] = payload
    self_members["README.md"] = rebuild_readme("the archive root").encode()
    repo_members["README_REBUILD.md"] = rebuild_readme(manuscript.relative_to(ROOT).as_posix()).encode()
    verification = {"performed": verify}
    if verify:
        temporary_parent = (manuscript / "tmp").resolve()
        temporary_parent.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="ieee_source_verify_", dir=temporary_parent) as temporary:
            staging = Path(temporary).resolve()
            if not staging.is_relative_to(temporary_parent):
                raise ValueError("Unsafe temporary verification path")
            for name, payload in self_members.items():
                target = staging / name
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(payload)
            verified_build = staging / ".build"
            build_documents(staging, verified_build, engines)
            verified_reports, errors = inspect_documents(verified_build)
            if errors:
                raise ValueError(f"Self-contained source rebuild failed checks: {errors}")
            verification["documents"] = {}
            for stem in DOCUMENTS:
                same = verified_reports[stem]["text_sha256"] == original_reports[stem]["text_sha256"]
                if not same or verified_reports[stem]["pages"] != original_reports[stem]["pages"]:
                    raise ValueError(f"Self-contained rebuild differs from active-source build: {stem}")
                verification["documents"][stem] = {"pages": verified_reports[stem]["pages"],
                                                     "identical_extracted_text": same}
    archives = {}
    for name, members in (("ieee_submission_source.zip", self_members),
                          ("ieee_submission_source_repo_layout.zip", repo_members)):
        target = output / name
        deterministic_zip(target, members)
        archives[name] = {"sha256": sha(target), "files": sorted(members), "bytes": target.stat().st_size}
    return {"archives": archives, "self_contained_rebuild": verification,
            "source_dependencies": [{"path": str(p.relative_to(ROOT)), "sha256": sha(p)} for p in sorted(inputs)]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manuscript-dir", type=Path, default=ROOT / "IEEE_Access_Submission")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--build-only", action="store_true", help="Build/check documents without producing release ZIPs")
    parser.add_argument("--skip-bundle-verification", action="store_true", help="Skip isolated source rebuild; reported in QA")
    args = parser.parse_args()
    manuscript = args.manuscript_dir.resolve()
    output = (args.output_dir or manuscript / "output").resolve()
    if not manuscript.is_relative_to(ROOT) or not output.is_relative_to(ROOT):
        raise ValueError("Manuscript and output directories must remain within this repository")
    engines = {name: shutil.which(name) for name in ("pdflatex", "bibtex")}
    if not all(engines.values()):
        raise RuntimeError("pdflatex and bibtex must be available on PATH")
    output.mkdir(parents=True, exist_ok=True)
    qa = output / "qa"
    qa.mkdir(exist_ok=True)
    build = manuscript / "tmp" / "ieee_package_build"
    source_hashes = {stem: sha(manuscript / f"{stem}.tex") for stem in DOCUMENTS}
    report = {"editorial": source_editorial_checks(manuscript), "engines": engines,
              "visual_review": "Not performed by this packager; page renders are supplied for review."}
    build_documents(manuscript, build, engines)
    documents, errors = inspect_documents(build)
    report["documents"] = documents
    report["errors"] = errors
    inputs = recorder_inputs(build, manuscript)
    report["column_heading_checks"] = tabular_heading_checks(inputs)
    for stem in DOCUMENTS:
        shutil.copy2(build / f"{stem}.log", qa / f"{stem}.log")
    report["qa_contact_sheets"] = render_qa(build, qa / "pages")
    if source_hashes != {stem: sha(manuscript / f"{stem}.tex") for stem in DOCUMENTS}:
        errors.append("Active sources changed during packaging; rebuild after edits finish")
    (qa / "validation.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if errors:
        raise ValueError("Publication checks failed: " + "; ".join(errors))
    if args.build_only:
        print(json.dumps({"build_directory": str(build), "checks_passed": True,
                          "pages": {k: v["pages"] for k, v in documents.items()}}))
        return
    figures = included_figures(manuscript, build, inputs)
    report["figures"], figure_paths = export_figures(figures, output / "figures")
    deterministic_zip(output / "figures_600dpi.zip", {p.name: p.read_bytes() for p in figure_paths})
    report["source_packaging"] = package_sources(inputs, manuscript, output, engines,
                                                 documents, not args.skip_bundle_verification)
    pdf_output = output / "pdf"
    pdf_output.mkdir(exist_ok=True)
    for stem in DOCUMENTS:
        shutil.copy2(build / f"{stem}.pdf", pdf_output / f"{stem}.pdf")
    report["all_automated_checks_passed"] = True
    report["figures_zip_sha256"] = sha(output / "figures_600dpi.zip")
    (qa / "validation.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"pages": {k: v["pages"] for k, v in documents.items()},
                      "figures": len(figures), "source_files": len(inputs),
                      "output": str(output), "all_automated_checks_passed": True}))


if __name__ == "__main__":
    main()
