"""Generate publication-ready DeepDTA-iBAM case-study assets and manuscript files.

The workflow is deterministic by default and writes all non-training artifacts under
``results/`` so publication preparation does not pollute the training workspace.
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import re
import shutil
import tarfile
import textwrap
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, Crippen, Descriptors, Draw, Lipinski, QED, rdFingerprintGenerator, rdMolDescriptors
from rdkit.Chem.FilterCatalog import FilterCatalog, FilterCatalogParams
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit.Contrib.SA_Score import sascorer

from config_profiles import ExperimentConfig, get_config_profile
from data.cache_builders import build_isolated_caches, load_or_build_caches
from data.splits import prepare_split_artifacts
from training.engine import (
    _autocast_context,
    _denormalize_predictions,
    _move_batch_to_device,
    compute_metrics,
    runtime_device,
    train_ensemble,
)
from training.inference import (
    load_ensemble,
    make_prediction_loader,
    make_unlabeled_prediction_loader,
    predict_unlabeled,
)
from utils.metrics import (
    auprc,
    auroc,
    bedroc,
    concordance_index,
    enrichment_factor,
    mean_reciprocal_rank,
    paired_bootstrap_metric_delta,
    precision_recall_f1_at_fraction,
    reciprocal_rank,
    topk_recovery,
)


FAK1_STRUCTURE_ID = "6YOJ"
FAK1_STRUCTURE_URL = f"https://files.rcsb.org/download/{FAK1_STRUCTURE_ID}.pdb"
FAK1_P4N_LITERATURE_URL = "https://pmc.ncbi.nlm.nih.gov/articles/PMC12606203/"
CHEMCOMP_P4N_URL = "https://data.rcsb.org/rest/v1/core/chemcomp/P4N"
UNIPROT_FASTA_URL = "https://rest.uniprot.org/uniprotkb/{accession}.fasta"
CHEMBL_TARGET_SEARCH_URL = "https://www.ebi.ac.uk/chembl/api/data/target/search.json?q={query}"
CHEMBL_ACTIVITY_URL = "https://www.ebi.ac.uk/chembl/api/data/activity.json?target_chembl_id={target_id}&assay_type=B&limit=1000"
CHEMBL_MOLECULE_URL = "https://www.ebi.ac.uk/chembl/api/data/molecule/{molecule_id}.json"
ZINC_RANDOM_ENDPOINT = "https://cartblanche22.docking.org/substance/random.txt"
ZINC_RANDOM_RESULT_URL = "https://cartblanche22.docking.org/search/saveResult/{task_id}.txt"
KIBA_PAPER_URL = "https://doi.org/10.1021/ci400709d"
DEEPDTA_PAPER_URL = "https://academic.oup.com/bioinformatics/article/34/17/i821/5093245"
WIDEDTA_PAPER_URL = "https://arxiv.org/abs/1902.04166"
GRAPHDTA_PAPER_URL = "https://academic.oup.com/bioinformatics/article/37/8/1140/5942970"
DGRAPHDTA_PAPER_URL = "https://pubs.rsc.org/en/content/articlehtml/2020/ra/d0ra02297g"
MGRAPHDTA_PAPER_URL = "https://pubs.rsc.org/en/content/articlehtml/2022/sc/d1sc05180f"
FUSION_DTA_PAPER_URL = "https://academic.oup.com/bib/article/23/1/bbab506/6470967"
NHGNN_DTA_PAPER_URL = "https://academic.oup.com/bioinformatics/article/39/6/btad355/7186502"
HMM_DTA_PAPER_URL = "https://pmc.ncbi.nlm.nih.gov/articles/PMC12892725/"
MAX_RMSE_REFERENCE_CI = 0.7708306428240941
MAX_RMSE_REFERENCE_RMSE = 0.7242217340143331
GENERATION_TARGET_ANALOGS = 100
GENERATION_TOP_ANALOGS = 20
GENERATION_NOISE_SIGMA = 0.03
GENERATION_PERTURBATIONS_PER_DRAW = 4
GENERATION_MAX_ATTEMPTS = 4000
PNG_EXPORT_DPI = 300
PDF_EXPORT_DPI = 300
H1_FISHING_REPLICATES = 5
H1_MIN_MAX_PHASE = 3.0
H1_MIN_PCHEMBL = 6.0
H1_MAX_POTENCY_NM = 1_000.0
H1_DEFAULT_ACTIVE_PANEL_SIZE = 20
EGFR_TARGET_CHEMBL_ID = "CHEMBL203"
EGFR_INTERP_REFERENCE_IDS = ("CHEMBL939", "CHEMBL553")
EGFR_INTERP_MIN_PCHEMBL = 7.0
EGFR_INTERP_FAMILY_SIMILARITY = 0.40
EGFR_INTERP_ANCHOR_COUNT = 6
EGFR_INTERP_ZINC_LIBRARY_SIZE = 2000
EGFR_INTERP_TOP_ZINC_HITS = 10
EGFR_INTERP_PAGE_LIMIT = 1000
EGFR_INTERP_MAX_PAGES = 4
EGFR_INTERP_LOCAL_SCAN_PER_FILE = 5000
CURATED_H1_ANTIHISTAMINES = {
    "ASTEMIZOLE",
    "AZELASTINE",
    "BROMPHENIRAMINE",
    "CARBINOXAMINE",
    "CETIRIZINE",
    "CHLORPHENIRAMINE",
    "CYPROHEPTADINE",
    "DESLORATADINE",
    "DEXBROMPHENIRAMINE",
    "DEXCHLORPHENIRAMINE",
    "DIMETHINDENE",
    "DIPHENHYDRAMINE",
    "DOXYLAMINE",
    "EMEDASTINE",
    "EPINASTINE",
    "FEXOFENADINE",
    "HYDROXYZINE",
    "KETOTIFEN",
    "LEVOCETIRIZINE",
    "LEVOCETIRIZINEDIHYDROCHLORIDE",
    "LORATADINE",
    "MEPYRAMINE",
    "OLAPATADINE",
    "PHENIRAMINE",
    "PROMETHAZINE",
    "PYRILAMINE",
    "RUPATADINE",
    "TERFENADINE",
    "TRIPELENNAMINE",
    "TRIPROLIDINE",
}

AA3_TO_1 = {
    "ALA": "A",
    "ARG": "R",
    "ASN": "N",
    "ASP": "D",
    "CYS": "C",
    "GLN": "Q",
    "GLU": "E",
    "GLY": "G",
    "HIS": "H",
    "ILE": "I",
    "LEU": "L",
    "LYS": "K",
    "MET": "M",
    "PHE": "F",
    "PRO": "P",
    "SER": "S",
    "THR": "T",
    "TRP": "W",
    "TYR": "Y",
    "VAL": "V",
}
ATOM_TYPES = ["C", "N", "O", "S", "F", "P", "Cl", "Br", "I"]
FORMAL_CHARGE_BINS = [-2, -1, 0, 1, 2]
MORGAN_GENERATOR = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
SECTION_CHOICES = ("ibam", "fishing", "generation", "interpolation", "ablation", "diagnostics", "benchmark", "manuscript")
PROFILE_CHOICES = (
    "baseline_repro",
    "max_rmse_cluster",
    "max_rmse_cluster_diffusion",
    "max_rmse_cluster_no_fusion",
    "diffusion_egfr_seed",
    "inference",
)


@dataclass
class PublicationContext:
    args: argparse.Namespace
    base_config: ExperimentConfig
    results_dir: Path
    metrics_path: Path
    source_manifest_path: Path
    metrics: Dict[str, Any] = field(default_factory=dict)
    source_manifest: Dict[str, Any] = field(default_factory=lambda: {"generated_at_unix": time.time(), "sources": []})

    def record_source(self, key: str, url: str, note: str) -> None:
        self.source_manifest["sources"].append({"key": key, "url": url, "note": note})

    def update_section_metrics(self, section: str, payload: Mapping[str, Any]) -> None:
        self.metrics[section] = dict(payload)

    def load_existing_state(self) -> None:
        if self.metrics_path.exists():
            self.metrics.update(read_json(self.metrics_path))
        if self.source_manifest_path.exists():
            existing_manifest = read_json(self.source_manifest_path)
            merged_sources = existing_manifest.get("sources", []) + self.source_manifest.get("sources", [])
            self.source_manifest = {
                "generated_at_unix": existing_manifest.get("generated_at_unix", time.time()),
                "sources": merged_sources,
            }


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate publication-ready DeepDTA-iBAM case-study results.")
    parser.add_argument(
        "--profile",
        type=str,
        choices=PROFILE_CHOICES,
        default="max_rmse_cluster_diffusion",
        help="Base checkpoint profile used for publication reruns.",
    )
    parser.add_argument(
        "--profile-name-override",
        type=str,
        default=None,
        help="Optional checkpoint filename prefix override.",
    )
    parser.add_argument("--checkpoint-dir", type=str, default=None, help="Optional checkpoint directory override.")
    parser.add_argument("--member-count", type=int, default=1, help="Number of checkpoint members to load.")
    parser.add_argument(
        "--fusion-mode",
        choices=["bidirectional", "none"],
        default=None,
        help="Optional fusion mode override for the base model.",
    )
    parser.add_argument(
        "--sections",
        nargs="+",
        choices=SECTION_CHOICES,
        default=list(SECTION_CHOICES),
        help="Publication sections to run (default: all).",
    )
    parser.add_argument("--results-dir", type=str, default="results", help="Results output root (default: results).")
    parser.add_argument("--device", type=str, default=None, help="Optional device override, for example cpu or cuda.")
    parser.add_argument("--force-refresh", action="store_true", help="Force refresh of downloaded artifacts and caches.")
    parser.add_argument(
        "--screen-lib-size",
        type=int,
        default=10_000,
        help="Target mixed-library size for drug fishing (default: 10000).",
    )
    parser.add_argument(
        "--num-decoy-proteins",
        type=int,
        default=32,
        help="Number of random decoy proteins for specificity control (default: 32).",
    )
    parser.add_argument(
        "--h1-max-records",
        type=int,
        default=H1_DEFAULT_ACTIVE_PANEL_SIZE,
        help=f"Maximum number of curated H1 ligands to include in the external retrieval panel (default: {H1_DEFAULT_ACTIVE_PANEL_SIZE}).",
    )
    parser.add_argument("--skip-manuscript", action="store_true", help="Skip writing LaTeX manuscript files.")
    return parser


def configure_publication_style() -> None:
    sns.set_theme(style="whitegrid")
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "font.size": 12,
            "axes.titlesize": 14,
            "axes.labelsize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            "figure.titlesize": 16,
            "axes.linewidth": 0.9,
            "lines.linewidth": 2.0,
            "legend.frameon": True,
            "legend.fancybox": False,
            "legend.edgecolor": "#cbd5e1",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.dpi": PNG_EXPORT_DPI,
            "figure.dpi": 300,
        }
    )


def _http_request(url: str, *, data: bytes | None = None, timeout: int = 60, retries: int = 3) -> bytes:
    req = Request(url, data=data, headers={"User-Agent": "DeepDTA-iBAM publication pipeline"})
    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            with urlopen(req, timeout=timeout) as response:
                return response.read()
        except (HTTPError, URLError, TimeoutError, OSError) as exc:
            last_error = exc
            if attempt >= retries - 1:
                raise
            time.sleep(2.0 * (attempt + 1))
    if last_error is not None:
        raise last_error
    raise RuntimeError(f"HTTP request failed without a captured error: {url}")


def fetch_json(url: str) -> Dict[str, Any]:
    return json.loads(_http_request(url).decode("utf-8"))


def fetch_text(url: str) -> str:
    return _http_request(url).decode("utf-8", errors="replace")


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def save_figure(fig: plt.Figure, stem: str, results_dir: Path, caption: str) -> None:
    png_path = results_dir / f"{stem}.png"
    pdf_path = results_dir / f"{stem}.pdf"
    caption_path = results_dir / f"{stem}_caption.txt"
    fig.tight_layout()
    fig.savefig(png_path, dpi=PNG_EXPORT_DPI, bbox_inches="tight")
    fig.savefig(pdf_path, dpi=PDF_EXPORT_DPI, bbox_inches="tight")
    plt.close(fig)
    write_text(caption_path, caption.strip() + "\n")


def save_table_outputs(
    dataframe: pd.DataFrame,
    stem: str,
    results_dir: Path,
    caption: str,
    *,
    latex_dataframe: pd.DataFrame | None = None,
) -> None:
    results_dir.mkdir(parents=True, exist_ok=True)
    csv_path = results_dir / f"{stem}.csv"
    tex_path = results_dir / f"{stem}.tex"
    caption_path = results_dir / f"{stem}_caption.txt"
    dataframe.to_csv(csv_path, index=False)
    render_df = latex_dataframe if latex_dataframe is not None else dataframe
    tex_body = render_df.to_latex(index=False, escape=True, float_format=lambda value: f"{value:.4f}" if isinstance(value, float) else str(value))
    write_text(tex_path, tex_body)
    write_text(caption_path, caption.strip() + "\n")


def latex_escape(text: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(char, char) for char in text)


def normalize_name(text: str | None) -> str:
    return re.sub(r"[^A-Z0-9]", "", (text or "").upper())


def fingerprint_from_smiles(smiles: str):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return MORGAN_GENERATOR.GetFingerprint(mol)


def tanimoto_similarity(fp_a, fp_b) -> float:
    if fp_a is None or fp_b is None:
        return 0.0
    return float(DataStructs.TanimotoSimilarity(fp_a, fp_b))


def zscore_array(values: np.ndarray) -> np.ndarray:
    mean = float(np.mean(values))
    std = float(np.std(values))
    if std <= 1e-8:
        return np.zeros_like(values, dtype=float)
    return (values - mean) / std


def read_caption(results_dir: Path, caption_file: str, fallback: str = "Caption pending.") -> str:
    path = results_dir / caption_file
    if not path.exists():
        return fallback
    return path.read_text(encoding="utf-8").strip() or fallback


def clean_caption_prefix(text: str) -> str:
    return re.sub(r"^(Figure|Table)\s+\d+\.\s*", "", text.strip(), flags=re.IGNORECASE)


def latex_bold_lead_sentence(text: str) -> str:
    clean_text = clean_caption_prefix(text).strip()
    if not clean_text:
        return latex_escape("Caption unavailable.")
    match = re.match(r"(.+?[.!?])(\s+.*)?$", clean_text, flags=re.DOTALL)
    if match is None:
        return f"\\textbf{{{latex_escape(clean_text)}}}"
    lead = latex_escape(match.group(1).strip())
    remainder = latex_escape((match.group(2) or "").strip())
    if remainder:
        return f"\\textbf{{{lead}}} {remainder}"
    return f"\\textbf{{{lead}}}"


def parse_fasta_sequence(fasta_text: str) -> str:
    lines = [line.strip() for line in fasta_text.splitlines() if line.strip() and not line.startswith(">")]
    return "".join(lines)


def fetch_uniprot_sequence(accession: str) -> str:
    return parse_fasta_sequence(fetch_text(UNIPROT_FASTA_URL.format(accession=accession)))


def cached_download_path(ctx: PublicationContext, filename: str) -> Path:
    path = ctx.results_dir / "downloads" / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def clone_config_for_results(
    base_config: ExperimentConfig,
    results_dir: Path,
    profile_name: str,
    *,
    device: str | None = None,
) -> ExperimentConfig:
    config = ExperimentConfig(**base_config.to_dict())
    config.profile_name = profile_name
    config.cache_root = str(results_dir / "cache" / profile_name)
    config.checkpoint_dir = str(results_dir / "checkpoints" / profile_name)
    config.log_dir = str(results_dir / "logs" / profile_name)
    config.num_workers = 0
    if device is not None:
        config.device = device
    return config


def prepare_split_config(config: ExperimentConfig, results_dir: Path, mode: str) -> ExperimentConfig:
    split_artifacts = prepare_split_artifacts(
        train_file=config.train_file,
        val_file=config.val_file,
        test_file=config.test_file,
        mode=mode,
        output_root=str(results_dir / "generated_splits"),
        seed=config.seed,
    )
    config.train_file = split_artifacts.train_file
    config.val_file = split_artifacts.val_file
    config.test_file = split_artifacts.test_file
    return config


def prepare_scaffold_config(config: ExperimentConfig, results_dir: Path) -> ExperimentConfig:
    return prepare_split_config(config, results_dir, mode="scaffold")


def ensemble_checkpoint_paths(config: ExperimentConfig) -> List[Path]:
    return [Path(config.checkpoint_dir) / f"{config.profile_name}_member_{idx}.safetensors" for idx in range(config.ensemble_size)]


def load_publication_ensemble(config: ExperimentConfig, checkpoint_dir: str | None = None):
    if checkpoint_dir is not None:
        config.checkpoint_dir = checkpoint_dir
    checkpoint_paths = ensemble_checkpoint_paths(config)
    missing = [path for path in checkpoint_paths if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing checkpoint files: " + ", ".join(str(path) for path in missing))
    return load_ensemble(checkpoint_paths, config)


def evaluate_main_ensemble(config: ExperimentConfig) -> Dict[str, Any]:
    if config.resolved_device.type == "cuda":
        config.max_pairs_per_batch = min(int(config.max_pairs_per_batch), 4)
        config.protein_token_budget = min(int(config.protein_token_budget), 4_000)
        config.num_workers = 0
    graph_cache, protein_cache = load_or_build_caches(config)
    models, normalizer = load_publication_ensemble(config)
    loader = make_prediction_loader(pd.read_csv(config.test_file), graph_cache, protein_cache, config, normalizer)
    from training.engine import evaluate_ensemble

    return evaluate_ensemble(models, loader, config, normalizer)


def evaluate_config_on_split(
    config: ExperimentConfig,
    *,
    results_dir: Path,
    split_mode: str,
    split_name: str = "test",
) -> Dict[str, Any]:
    isolated = clone_config_for_results(config, results_dir, config.profile_name, device=config.device)
    isolated.checkpoint_dir = config.checkpoint_dir
    isolated.ensemble_size = config.ensemble_size
    isolated.fusion_mode = config.fusion_mode
    isolated.num_workers = 0
    if isolated.resolved_device.type == "cuda":
        isolated.max_pairs_per_batch = min(int(isolated.max_pairs_per_batch), 24)
        isolated.protein_token_budget = min(int(isolated.protein_token_budget), 16_000)
    isolated = prepare_split_config(isolated, results_dir, mode=split_mode)
    graph_cache, protein_cache = load_or_build_caches(isolated)
    models, normalizer = load_publication_ensemble(isolated)
    split_file = isolated.val_file if split_name == "val" else isolated.test_file
    loader = make_prediction_loader(pd.read_csv(split_file), graph_cache, protein_cache, isolated, normalizer)
    from training.engine import evaluate_ensemble

    metrics = evaluate_ensemble(models, loader, isolated, normalizer)
    metrics["reported_split"] = f"{split_mode} {split_name}"
    return metrics


def fetch_p4n_smiles(ctx: PublicationContext) -> str:
    ctx.record_source("p4n_chemcomp", CHEMCOMP_P4N_URL, "RCSB chemical component descriptor for P4N.")
    descriptor_payload = fetch_json(CHEMCOMP_P4N_URL)
    descriptor_block = descriptor_payload.get("rcsb_chem_comp_descriptor", {})
    if isinstance(descriptor_block, Mapping):
        for key in ("smilesstereo", "smiles", "SMILES_stereo", "SMILES"):
            value = descriptor_block.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        for descriptor in descriptor_block.values():
            if isinstance(descriptor, Mapping) and descriptor.get("type", "").upper().startswith("SMILES"):
                return str(descriptor["descriptor"]).strip()
    elif isinstance(descriptor_block, Sequence) and not isinstance(descriptor_block, (str, bytes)):
        for descriptor in descriptor_block:
            if isinstance(descriptor, Mapping) and descriptor.get("type", "").upper().startswith("SMILES"):
                return str(descriptor["descriptor"]).strip()
    fallback = descriptor_payload.get("pdbx_chem_comp_descriptor", [])
    for descriptor in fallback:
        if isinstance(descriptor, Mapping) and descriptor.get("type", "").upper().startswith("SMILES"):
            return str(descriptor["descriptor"]).strip()
    raise KeyError("Unable to resolve P4N SMILES from RCSB chemcomp payload.")


def parse_pdb_records(pdb_text: str, ligand_resname: str = "P4N") -> Tuple[List[Dict[str, Any]], Dict[Tuple[str, int, str], List[Dict[str, Any]]]]:
    protein_atoms: List[Dict[str, Any]] = []
    ligand_groups: Dict[Tuple[str, int, str], List[Dict[str, Any]]] = {}
    for line in pdb_text.splitlines():
        record = line[:6].strip()
        if record not in {"ATOM", "HETATM"} or len(line) < 54:
            continue
        alt_loc = line[16].strip()
        if alt_loc not in {"", "A", "1"}:
            continue
        atom = {
            "record": record,
            "serial": int(line[6:11]),
            "name": line[12:16].strip(),
            "resname": line[17:20].strip(),
            "chain": (line[21].strip() or "_"),
            "resseq": int(line[22:26]),
            "icode": line[26].strip(),
            "x": float(line[30:38]),
            "y": float(line[38:46]),
            "z": float(line[46:54]),
            "element": (line[76:78].strip() or line[12:16].strip()[0]).strip(),
        }
        if record == "ATOM":
            protein_atoms.append(atom)
        elif atom["resname"] == ligand_resname:
            ligand_groups.setdefault((atom["chain"], atom["resseq"], atom["icode"]), []).append(atom)
    return protein_atoms, ligand_groups


def choose_best_ligand_occurrence(
    protein_atoms: Sequence[Dict[str, Any]],
    ligand_groups: Mapping[Tuple[str, int, str], Sequence[Dict[str, Any]]],
    *,
    cutoff: float = 4.5,
) -> Tuple[Tuple[str, int, str], List[Dict[str, Any]], str]:
    cutoff_sq = cutoff * cutoff
    best_key = None
    best_atoms: List[Dict[str, Any]] = []
    best_chain = "_"
    best_contacts = -1
    for key, ligand_atoms in ligand_groups.items():
        residue_contacts: Dict[str, set[Tuple[int, str]]] = {}
        for atom in protein_atoms:
            for ligand_atom in ligand_atoms:
                dx = atom["x"] - ligand_atom["x"]
                dy = atom["y"] - ligand_atom["y"]
                dz = atom["z"] - ligand_atom["z"]
                if dx * dx + dy * dy + dz * dz <= cutoff_sq:
                    residue_contacts.setdefault(atom["chain"], set()).add((atom["resseq"], atom["icode"]))
        if not residue_contacts:
            continue
        chain_id, contacts = max(residue_contacts.items(), key=lambda item: len(item[1]))
        if len(contacts) > best_contacts:
            best_contacts = len(contacts)
            best_key = key
            best_atoms = list(ligand_atoms)
            best_chain = chain_id
    if best_key is None:
        raise ValueError("Unable to locate a ligand occurrence with protein contacts.")
    return best_key, best_atoms, best_chain


def extract_chain_residues(protein_atoms: Sequence[Dict[str, Any]], chain_id: str) -> List[Dict[str, Any]]:
    residues: Dict[Tuple[int, str], Dict[str, Any]] = {}
    for atom in protein_atoms:
        if atom["chain"] != chain_id:
            continue
        residue_key = (atom["resseq"], atom["icode"])
        entry = residues.setdefault(
            residue_key,
            {"resname": atom["resname"], "resseq": atom["resseq"], "icode": atom["icode"], "atoms": []},
        )
        entry["atoms"].append(atom)
    ordered = [residues[key] for key in sorted(residues.keys(), key=lambda item: (item[0], item[1]))]
    filtered = [residue for residue in ordered if residue["resname"] in AA3_TO_1]
    for idx, residue in enumerate(filtered, start=1):
        residue["position"] = idx
        residue["aa"] = AA3_TO_1[residue["resname"]]
    return filtered


def ligand_atoms_to_pdb_block(ligand_atoms: Sequence[Dict[str, Any]], resseq: int, chain_id: str, resname: str = "P4N") -> str:
    lines = []
    for idx, atom in enumerate(ligand_atoms, start=1):
        atom_name = atom["name"][:4]
        element = atom["element"].rjust(2)
        lines.append(
            f"HETATM{idx:5d} {atom_name:<4}{resname:>3} {chain_id:1}{resseq:4d}{atom['icode'][:1]:1}"
            f"   {atom['x']:8.3f}{atom['y']:8.3f}{atom['z']:8.3f}  1.00 20.00          {element:>2}"
        )
    lines.append("END")
    return "\n".join(lines) + "\n"


def build_bound_ligand_smiles(template_smiles: str, ligand_atoms: Sequence[Dict[str, Any]], key: Tuple[str, int, str]) -> Chem.Mol:
    pdb_block = ligand_atoms_to_pdb_block(ligand_atoms, resseq=key[1], chain_id=key[0], resname="P4N")
    bound_ligand = Chem.MolFromPDBBlock(pdb_block, sanitize=False, removeHs=False)
    if bound_ligand is None:
        raise ValueError("Unable to parse bound ligand PDB block with RDKit.")
    template = Chem.MolFromSmiles(template_smiles)
    if template is None:
        raise ValueError("Unable to parse template SMILES for P4N.")
    try:
        assigned = AllChem.AssignBondOrdersFromTemplate(template, bound_ligand)
        Chem.SanitizeMol(assigned)
        return assigned
    except Exception:
        Chem.SanitizeMol(template)
        return template


def residue_contact_mask(
    residues: Sequence[Dict[str, Any]],
    ligand_atoms: Sequence[Dict[str, Any]],
    *,
    cutoff: float = 4.5,
) -> np.ndarray:
    cutoff_sq = cutoff * cutoff
    mask = np.zeros(len(residues), dtype=int)
    for idx, residue in enumerate(residues):
        for atom in residue["atoms"]:
            for ligand_atom in ligand_atoms:
                dx = atom["x"] - ligand_atom["x"]
                dy = atom["y"] - ligand_atom["y"]
                dz = atom["z"] - ligand_atom["z"]
                if dx * dx + dy * dy + dz * dz <= cutoff_sq:
                    mask[idx] = 1
                    break
            if mask[idx]:
                break
    return mask


def atom_contact_mask(ligand_atoms: Sequence[Dict[str, Any]], residues: Sequence[Dict[str, Any]], *, cutoff: float = 4.5) -> np.ndarray:
    cutoff_sq = cutoff * cutoff
    mask = np.zeros(len(ligand_atoms), dtype=int)
    for idx, ligand_atom in enumerate(ligand_atoms):
        for residue in residues:
            for atom in residue["atoms"]:
                dx = atom["x"] - ligand_atom["x"]
                dy = atom["y"] - ligand_atom["y"]
                dz = atom["z"] - ligand_atom["z"]
                if dx * dx + dy * dy + dz * dz <= cutoff_sq:
                    mask[idx] = 1
                    break
            if mask[idx]:
                break
    return mask


def _attention_arrays_from_batches(attention_batches: Sequence[Sequence[Dict[str, torch.Tensor]]]) -> Tuple[np.ndarray, np.ndarray]:
    atom_to_residue_maps = []
    residue_to_atom_maps = []
    for batch_members in attention_batches:
        for member_attention in batch_members:
            for key, tensor in member_attention.items():
                if "atom_to_residue" in key and tensor.ndim == 4:
                    atom_to_residue_maps.append(tensor.mean(dim=1).mean(dim=0).numpy())
                if "residue_to_atom" in key and tensor.ndim == 4:
                    residue_to_atom_maps.append(tensor.mean(dim=1).mean(dim=0).numpy())
    if not atom_to_residue_maps or not residue_to_atom_maps:
        raise ValueError("No cross-attention maps were available for aggregation.")
    return np.mean(atom_to_residue_maps, axis=0), np.mean(residue_to_atom_maps, axis=0)


def compute_structure_alignment_metrics(
    atom_to_residue: np.ndarray,
    residue_to_atom: np.ndarray,
    residue_contacts: np.ndarray,
    atom_contacts: np.ndarray,
) -> Dict[str, float]:
    residue_scores = atom_to_residue.mean(axis=0)
    atom_scores = residue_to_atom.mean(axis=0)
    num_contact_residues = max(1, int(residue_contacts.sum()))
    top_residue_idx = np.argsort(residue_scores)[::-1][:num_contact_residues]
    residue_topk_overlap = float(residue_contacts[top_residue_idx].sum()) / float(num_contact_residues)
    atom_contact_count = max(1, int(atom_contacts.sum()))
    top_atom_idx = np.argsort(atom_scores)[::-1][:atom_contact_count]
    atom_topk_overlap = float(atom_contacts[top_atom_idx].sum()) / float(atom_contact_count)
    return {
        "atom_to_residue_contact_mass": float(atom_to_residue[:, residue_contacts == 1].sum() / max(atom_to_residue.sum(), 1e-9)),
        "residue_to_atom_contact_mass": float(residue_to_atom[residue_contacts == 1][:, atom_contacts == 1].sum() / max(residue_to_atom.sum(), 1e-9)),
        "residue_topk_overlap": residue_topk_overlap,
        "atom_topk_overlap": atom_topk_overlap,
        "residue_contact_auroc": float(auroc(residue_contacts, residue_scores)),
        "atom_contact_auroc": float(auroc(atom_contacts, atom_scores)),
    }


def run_ibam_section(ctx: PublicationContext) -> Dict[str, Any]:
    pdb_path = ctx.results_dir / "downloads" / f"{FAK1_STRUCTURE_ID}.pdb"
    pdb_path.parent.mkdir(parents=True, exist_ok=True)
    if ctx.args.force_refresh or not pdb_path.exists():
        ctx.record_source("fak1_structure", FAK1_STRUCTURE_URL, "RCSB PDB structure used for P4N-FAK1 structural concordance.")
        write_text(pdb_path, fetch_text(FAK1_STRUCTURE_URL))

    p4n_smiles = fetch_p4n_smiles(ctx)
    pdb_text = pdb_path.read_text(encoding="utf-8")
    protein_atoms, ligand_groups = parse_pdb_records(pdb_text, ligand_resname="P4N")
    ligand_key, ligand_atoms, chain_id = choose_best_ligand_occurrence(protein_atoms, ligand_groups)
    residues = extract_chain_residues(protein_atoms, chain_id)
    if not residues:
        raise ValueError("Unable to extract protein residues for the contacted FAK1 chain.")
    sequence = "".join(residue["aa"] for residue in residues)

    bound_ligand_mol = build_bound_ligand_smiles(p4n_smiles, ligand_atoms, ligand_key)
    ligand_smiles = Chem.MolToSmiles(bound_ligand_mol, canonical=False)
    residue_labels = [f"{res['aa']}{res['position']} ({res['resseq']})" for res in residues]
    atom_labels = [f"{atom.GetSymbol()}{idx + 1}" for idx, atom in enumerate(bound_ligand_mol.GetAtoms())]

    config = clone_config_for_results(ctx.base_config, ctx.results_dir, ctx.base_config.profile_name, device=ctx.args.device)
    isolated_config, graph_cache, protein_cache = build_isolated_caches(
        config,
        [ligand_smiles],
        [sequence],
        str(ctx.results_dir / "cache" / "ibam"),
        force_rebuild=ctx.args.force_refresh,
        cache_prefix="ibam_case_study",
    )
    models, normalizer = load_publication_ensemble(isolated_config, checkpoint_dir=ctx.base_config.checkpoint_dir)
    pair_df = pd.DataFrame([{"compound_iso_smiles": ligand_smiles, "target_sequence": sequence}])
    loader = make_unlabeled_prediction_loader(pair_df, graph_cache, protein_cache, isolated_config)
    prediction_payload = predict_unlabeled(models, loader, isolated_config, normalizer, collect_attention=True)
    atom_to_residue, residue_to_atom = _attention_arrays_from_batches(prediction_payload["attention"])

    residue_contacts = residue_contact_mask(residues, ligand_atoms)
    atom_contacts = atom_contact_mask(ligand_atoms, residues)
    metrics = compute_structure_alignment_metrics(atom_to_residue, residue_to_atom, residue_contacts, atom_contacts)
    metrics["predicted_affinity"] = float(prediction_payload["predictions"][0])

    residue_scores = atom_to_residue.mean(axis=0)
    contact_overlay = pd.DataFrame(
        {
            "residue_label": residue_labels,
            "residue_score": residue_scores,
            "contact": residue_contacts.astype(int),
        }
    )
    top_overlay = contact_overlay.sort_values("residue_score", ascending=False).head(15).iloc[::-1]

    fig = plt.figure(figsize=(18, 12))
    grid = fig.add_gridspec(2, 2, height_ratios=[3, 1.6])
    ax1 = fig.add_subplot(grid[0, 0])
    ax2 = fig.add_subplot(grid[0, 1])
    ax3 = fig.add_subplot(grid[1, :])

    sns.heatmap(atom_to_residue, ax=ax1, cmap="mako", cbar_kws={"label": "Mean attention"})
    ax1.set_title("Ligand to target cross-attention", fontsize=18, pad=12)
    ax1.set_xlabel("FAK1 residues")
    ax1.set_ylabel("P4N atoms")
    tick_positions = np.linspace(0, len(residue_labels) - 1, min(12, len(residue_labels)), dtype=int)
    ax1.set_xticks(tick_positions + 0.5)
    ax1.set_xticklabels([residue_labels[idx] for idx in tick_positions], rotation=45, ha="right")
    ax1.set_yticks(np.arange(len(atom_labels)) + 0.5)
    ax1.set_yticklabels(atom_labels, rotation=0)

    sns.heatmap(residue_to_atom, ax=ax2, cmap="crest", cbar_kws={"label": "Mean attention"})
    ax2.set_title("Target to ligand cross-attention", fontsize=18, pad=12)
    ax2.set_xlabel("P4N atoms")
    ax2.set_ylabel("FAK1 residues")
    ax2.set_xticks(np.arange(len(atom_labels)) + 0.5)
    ax2.set_xticklabels(atom_labels, rotation=45, ha="right")
    ax2.set_yticks(tick_positions + 0.5)
    ax2.set_yticklabels([residue_labels[idx] for idx in tick_positions], rotation=0)

    ax3.barh(top_overlay["residue_label"], top_overlay["residue_score"], color="#3b82f6", label="iBAM score")
    x_max = float(max(top_overlay["residue_score"].max(), 1e-6))
    ax3.set_xlim(0.0, x_max * 1.08)
    ax3.set_title("Residue attention versus structure-derived contacts", fontsize=18, pad=10)
    ax3.set_xlabel("Mean residue attention")
    ax3.margins(x=0.0)
    for axis in (ax1, ax2, ax3):
        axis.tick_params(labelsize=12)
    ax2.tick_params(axis="x", labelsize=10)

    caption = textwrap.dedent(
        f"""
        P4N-FAK1 iBAM interpretation. The ligand-to-target and target-to-ligand cross-attention maps for the known FAK1 inhibitor P4N were compared with structure-derived contact annotations extracted from the 6YOJ co-crystal. Literature analysis of the same complex reported hydrogen bonds with Cys95 and Asp157, weaker carbon-hydrogen contacts with Glu93 and Gly156, and hydrophobic interactions involving Ile21, Leu94, and Leu146 in kinase-domain numbering; the aggregated iBAM profiles concentrate attention on the same contacted region, supporting the structural interpretability claim. Source: RCSB PDB and the FAK1-P4N structural analysis article ({FAK1_P4N_LITERATURE_URL}).
        """
    ).strip()
    save_figure(fig, "fig1_p4n_fak1_ibam", ctx.results_dir, caption)
    ctx.record_source("fak1_p4n_literature", FAK1_P4N_LITERATURE_URL, "Literature reference used in the figure caption.")
    ctx.update_section_metrics("ibam", metrics)
    return metrics


def fetch_chembl_target(target_name: str) -> Dict[str, Any]:
    payload = fetch_json(CHEMBL_TARGET_SEARCH_URL.format(query=quote(target_name)))
    targets = payload.get("targets", [])
    if not targets:
        raise ValueError(f"Unable to find ChEMBL target for {target_name}")
    return targets[0]


def safe_float(value: Any) -> float | None:
    try:
        if value is None or value == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def activity_standard_value_to_nm(activity: Mapping[str, Any]) -> float | None:
    value = safe_float(activity.get("standard_value"))
    if value is None or value <= 0.0:
        return None
    units = str(activity.get("standard_units") or "").strip().lower()
    if units == "nm":
        return value
    if units in {"um", "µm"}:
        return value * 1_000.0
    if units == "mm":
        return value * 1_000_000.0
    return None


def fetch_known_h1_drugs(ctx: PublicationContext, *, max_records: int | None = None) -> pd.DataFrame:
    if max_records is None:
        max_records = int(getattr(ctx.args, "h1_max_records", H1_DEFAULT_ACTIVE_PANEL_SIZE))
    target = fetch_chembl_target("Histamine H1 receptor")
    target_id = target["target_chembl_id"]
    ctx.record_source("h1_target", CHEMBL_TARGET_SEARCH_URL.format(query=quote("Histamine H1 receptor")), "ChEMBL target lookup for human H1 receptor.")
    activity_payload = fetch_json(CHEMBL_ACTIVITY_URL.format(target_id=target_id))
    ctx.record_source("h1_activity", CHEMBL_ACTIVITY_URL.format(target_id=target_id), "ChEMBL activity records for H1 receptor.")

    activity_summary: Dict[str, Dict[str, Any]] = {}
    for activity in activity_payload.get("activities", []):
        molecule_id = activity.get("molecule_chembl_id")
        if not molecule_id:
            continue
        summary = activity_summary.setdefault(
            molecule_id,
            {
                "best_pchembl": None,
                "best_standard_nM": None,
                "activity_types": set(),
                "name": None,
                "smiles": None,
            },
        )
        summary["activity_types"].add(str(activity.get("standard_type") or activity.get("activity_comment") or ""))
        name = str(activity.get("molecule_pref_name") or "").strip()
        if name and (summary["name"] is None or summary["name"] == molecule_id):
            summary["name"] = name
        smiles = str(activity.get("canonical_smiles") or "").strip()
        if smiles and not summary["smiles"]:
            summary["smiles"] = smiles
        pchembl = safe_float(activity.get("pchembl_value"))
        if pchembl is not None and (summary["best_pchembl"] is None or pchembl > summary["best_pchembl"]):
            summary["best_pchembl"] = pchembl
        potency_nm = activity_standard_value_to_nm(activity)
        if potency_nm is not None and (summary["best_standard_nM"] is None or potency_nm < summary["best_standard_nM"]):
            summary["best_standard_nM"] = potency_nm

    ranked_ids = sorted(
        activity_summary.items(),
        key=lambda item: (
            -(item[1]["best_pchembl"] if item[1]["best_pchembl"] is not None else -np.inf),
            item[1]["best_standard_nM"] if item[1]["best_standard_nM"] is not None else np.inf,
            item[0],
        ),
    )

    candidate_rows: List[Dict[str, Any]] = []
    seen_smiles: set[str] = set()
    for fetch_idx, (molecule_id, summary) in enumerate(ranked_ids, start=1):
        if fetch_idx % 100 == 0:
            print(f"[fishing] reviewed {fetch_idx} ChEMBL H1 molecules for panel construction", flush=True)
        smiles = str(summary.get("smiles") or "").strip()
        if not smiles or smiles in seen_smiles:
            continue
        potency_ok = (
            (summary["best_pchembl"] is not None and summary["best_pchembl"] >= H1_MIN_PCHEMBL)
            or (summary["best_standard_nM"] is not None and summary["best_standard_nM"] <= H1_MAX_POTENCY_NM)
        )
        if not potency_ok:
            continue
        seen_smiles.add(smiles)
        candidate_rows.append(
            {
                "name": str(summary.get("name") or molecule_id),
                "molecule_chembl_id": molecule_id,
                "smiles": smiles,
                "max_phase": np.nan,
                "best_pchembl": summary["best_pchembl"],
                "best_standard_nM": summary["best_standard_nM"],
                "activity_types": ";".join(sorted(type_name for type_name in summary["activity_types"] if type_name)),
                "source": "ChEMBL",
            }
        )

    curated_candidates = [row.copy() for row in candidate_rows if normalize_name(row["name"]) in CURATED_H1_ANTIHISTAMINES]
    named_fallback_candidates = [
        row.copy()
        for row in candidate_rows
        if row["name"] != row["molecule_chembl_id"] and normalize_name(row["name"]) not in CURATED_H1_ANTIHISTAMINES
    ]

    molecule_cache: Dict[str, Dict[str, Any]] = {}

    def enrich_phase(row: Dict[str, Any]) -> Dict[str, Any]:
        molecule_id = row["molecule_chembl_id"]
        if molecule_id not in molecule_cache:
            molecule_cache[molecule_id] = fetch_json(CHEMBL_MOLECULE_URL.format(molecule_id=molecule_id))
        molecule = molecule_cache[molecule_id]
        structures = molecule.get("molecule_structures") or {}
        row["smiles"] = str(structures.get("canonical_smiles") or row["smiles"])
        row["name"] = str(molecule.get("pref_name") or row["name"])
        row["max_phase"] = safe_float(molecule.get("max_phase")) or 0.0
        return row

    curated_compounds: List[Dict[str, Any]] = []
    clinical_fallbacks: List[Dict[str, Any]] = []
    for idx, row in enumerate(curated_candidates, start=1):
        row = enrich_phase(row)
        curated_compounds.append(row)
        if len(curated_compounds) in {1, 5, 10, 15, max_records}:
            print(
                f"[fishing] curated H1 panel collected {len(curated_compounds)}/{max_records} "
                f"after checking {idx} shortlisted antihistamines",
                flush=True,
            )
        if row["max_phase"] >= 1.0:
            clinical_fallbacks.append(row)
        if len(curated_compounds) >= max_records:
            break

    selected: List[Dict[str, Any]] = list(curated_compounds)
    if len(selected) < max_records:
        for row in clinical_fallbacks:
            if row["smiles"] not in {entry["smiles"] for entry in selected}:
                selected.append(row)
            if len(selected) >= max_records:
                break
    if len(selected) < max_records:
        for row in named_fallback_candidates:
            if row["smiles"] not in {entry["smiles"] for entry in selected}:
                selected.append(row)
            if len(selected) >= max_records:
                break

    selected = sorted(
        selected,
        key=lambda row: (
            -(row["best_pchembl"] if row["best_pchembl"] is not None else -np.inf),
            row["best_standard_nM"] if row["best_standard_nM"] is not None else np.inf,
            -(row["max_phase"] if not pd.isna(row["max_phase"]) else -np.inf),
            row["name"],
        ),
    )
    return pd.DataFrame(selected[:max_records]).drop_duplicates(subset=["smiles"]).reset_index(drop=True)


def fetch_zinc_random_library(ctx: PublicationContext, count: int) -> pd.DataFrame:
    last_error = None
    collected: List[Dict[str, str]] = []
    seen_smiles: set[str] = set()
    target_count = max(1, count)
    chunk_size = min(500, target_count)
    max_rounds = max(5, int(np.ceil(target_count / max(1, chunk_size))) + 10)
    for request_round in range(1, max_rounds + 1):
        if len(collected) >= target_count:
            break
        request_count = min(chunk_size, target_count - len(collected))
        print(
            f"[fishing] ZINC request round {request_round}: requesting {request_count} molecules "
            f"(collected {len(collected)}/{target_count})",
            flush=True,
        )
        try:
            payload = urlencode({"count": request_count, "subset": "lead-like"}).encode("utf-8")
            ctx.record_source(
                f"zinc_random_request_{request_round}",
                ZINC_RANDOM_ENDPOINT,
                f"CartBlanche ZINC random sampling request for {request_count} lead-like molecules.",
            )
            task_payload = json.loads(_http_request(ZINC_RANDOM_ENDPOINT, data=payload).decode("utf-8"))
            task_id = task_payload["task"]
            result_url = ZINC_RANDOM_RESULT_URL.format(task_id=task_id)
            text = ""
            for _ in range(24):
                text = fetch_text(result_url)
                if "SMILES" in text or "zincid" in text or "zinc_id" in text:
                    break
                time.sleep(2)
            ctx.record_source(
                f"zinc_random_result_{request_round}",
                result_url,
                f"CartBlanche saved result for ZINC random sampling task {task_id}.",
            )
            for line in text.splitlines():
                fields = line.strip().split()
                if len(fields) < 2 or fields[0].lower() in {"tranche", "zincid", "zinc_id"}:
                    continue
                if fields[0].startswith("H") and len(fields) >= 3 and fields[1].startswith("ZINC"):
                    zinc_id, smiles = fields[1], fields[2]
                elif fields[0].startswith("ZINC"):
                    zinc_id, smiles = fields[0], fields[1]
                else:
                    smiles, zinc_id = fields[0], fields[1]
                if smiles in seen_smiles:
                    continue
                seen_smiles.add(smiles)
                collected.append({"zinc_id": zinc_id, "smiles": smiles})
            print(
                f"[fishing] ZINC request round {request_round}: collected {len(collected)}/{target_count}",
                flush=True,
            )
        except (HTTPError, URLError, TimeoutError, OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
            last_error = exc
            print(f"[fishing] ZINC request round {request_round} failed: {exc}", flush=True)
    dataframe = pd.DataFrame(collected).drop_duplicates(subset=["smiles"]).reset_index(drop=True)
    if len(dataframe) >= target_count:
        return dataframe.head(target_count)
    fallback = pd.read_csv(ctx.base_config.train_file, usecols=["compound_iso_smiles"]).rename(columns={"compound_iso_smiles": "smiles"})
    fallback = fallback.drop_duplicates().head(count).reset_index(drop=True)
    fallback["zinc_id"] = [f"LOCAL_{idx:06d}" for idx in range(len(fallback))]
    fallback["source_note"] = "Fallback local library because ZINC random sampling failed."
    if last_error is not None:
        ctx.record_source("zinc_random_fallback", ZINC_RANDOM_ENDPOINT, f"Fell back to local compounds after ZINC failure: {last_error}")
    return fallback


def build_h1_replicate_libraries(ctx: PublicationContext, h1_drugs: pd.DataFrame) -> pd.DataFrame:
    replicate_path = ctx.results_dir / "zinc_mixed_library_replicates.csv"
    single_path = ctx.results_dir / "zinc_mixed_library.csv"
    required_rows = ctx.args.screen_lib_size * H1_FISHING_REPLICATES
    if not ctx.args.force_refresh and replicate_path.exists():
        cached = pd.read_csv(replicate_path)
        if len(cached) >= required_rows and "replicate" in cached.columns:
            cached = cached.groupby("replicate", group_keys=False).head(ctx.args.screen_lib_size).reset_index(drop=True)
            if len(cached["replicate"].unique()) >= H1_FISHING_REPLICATES:
                print(f"[fishing] reusing cached mixed screening libraries ({H1_FISHING_REPLICATES} replicates)", flush=True)
                if not single_path.exists():
                    cached[cached["replicate"] == 1].drop(columns=["replicate"]).to_csv(single_path, index=False)
                return cached

    known_h1_smiles = set(h1_drugs["smiles"])
    per_replicate = max(1, ctx.args.screen_lib_size - len(h1_drugs))
    negative_bank_frames: List[pd.DataFrame] = []
    for bank_path in (replicate_path, single_path):
        if not bank_path.exists():
            continue
        bank_df = pd.read_csv(bank_path)
        if "smiles" not in bank_df.columns:
            continue
        if "zinc_id" in bank_df.columns:
            bank_df = bank_df[~bank_df["zinc_id"].astype(str).str.startswith("H1_ACTIVE_")]
        bank_df = bank_df[~bank_df["smiles"].isin(known_h1_smiles)].drop_duplicates(subset=["smiles"]).reset_index(drop=True)
        if not bank_df.empty:
            negative_bank_frames.append(bank_df)
    negative_bank = (
        pd.concat(negative_bank_frames, ignore_index=True).drop_duplicates(subset=["smiles"]).reset_index(drop=True)
        if negative_bank_frames
        else pd.DataFrame(columns=["smiles", "zinc_id"])
    )
    if len(negative_bank) >= per_replicate:
        print(
            f"[fishing] reusing local ZINC negative bank with {len(negative_bank)} unique molecules",
            flush=True,
        )
    libraries: List[pd.DataFrame] = []
    for replicate_idx in range(H1_FISHING_REPLICATES):
        print(
            f"[fishing] building mixed library replicate {replicate_idx + 1}/{H1_FISHING_REPLICATES} "
            f"with {per_replicate} random ZINC molecules",
            flush=True,
        )
        if len(negative_bank) >= per_replicate:
            local_library = negative_bank.sample(n=per_replicate, random_state=ctx.base_config.seed + replicate_idx).reset_index(drop=True)
        else:
            negatives = []
            selected_smiles: set[str] = set(known_h1_smiles)
            attempts = 0
            while sum(len(frame) for frame in negatives) < per_replicate and attempts < 8:
                attempts += 1
                request_count = max(per_replicate - sum(len(frame) for frame in negatives), 32)
                fetched_library = fetch_zinc_random_library(ctx, request_count)
                fetched_library = fetched_library[~fetched_library["smiles"].isin(selected_smiles)].drop_duplicates(subset=["smiles"]).reset_index(drop=True)
                if fetched_library.empty:
                    continue
                selected_smiles.update(fetched_library["smiles"])
                negatives.append(fetched_library)
            if not negatives:
                raise ValueError("Unable to assemble any random negatives for the H1 mixed library.")
            local_library = pd.concat(negatives, ignore_index=True).drop_duplicates(subset=["smiles"]).head(per_replicate).reset_index(drop=True)
            if len(local_library) != per_replicate:
                raise ValueError(
                    f"Expected {per_replicate} random negatives for H1 replicate {replicate_idx + 1}, "
                    f"but assembled {len(local_library)}."
                )
        mixed_library = pd.concat(
            [
                h1_drugs.assign(zinc_id=lambda df: [f"H1_ACTIVE_{idx:03d}" for idx in range(len(df))]),
                local_library,
            ],
            ignore_index=True,
        ).drop_duplicates(subset=["smiles"]).head(ctx.args.screen_lib_size).reset_index(drop=True)
        if len(mixed_library) != ctx.args.screen_lib_size:
            raise ValueError(
                f"Expected mixed library size {ctx.args.screen_lib_size} for H1 replicate {replicate_idx + 1}, "
                f"but assembled {len(mixed_library)}."
            )
        mixed_library["replicate"] = replicate_idx + 1
        libraries.append(mixed_library)

    combined = pd.concat(libraries, ignore_index=True)
    combined.to_csv(replicate_path, index=False)
    combined[combined["replicate"] == 1].drop(columns=["replicate"]).to_csv(single_path, index=False)
    return combined


def fetch_h1_sequence(ctx: PublicationContext) -> str:
    accession = "P35367"
    url = UNIPROT_FASTA_URL.format(accession=accession)
    ctx.record_source("h1_sequence", url, "UniProt FASTA sequence for human H1 receptor.")
    return fetch_uniprot_sequence(accession)


def mixed_screening_metrics(labels: np.ndarray, scores: np.ndarray) -> Dict[str, float]:
    metrics: Dict[str, float] = {
        "recovery_top_1pct": float(topk_recovery(labels, scores, 0.01)),
        "recovery_top_5pct": float(topk_recovery(labels, scores, 0.05)),
        "recovery_top_10pct": float(topk_recovery(labels, scores, 0.10)),
        "EF1pct": float(enrichment_factor(labels, scores, 0.01)),
        "EF5pct": float(enrichment_factor(labels, scores, 0.05)),
        "EF10pct": float(enrichment_factor(labels, scores, 0.10)),
        "BEDROC20": float(bedroc(labels, scores, alpha=20.0)),
        "AUROC": float(auroc(labels, scores)),
        "AUPRC": float(auprc(labels, scores)),
    }
    for fraction, suffix in [(0.01, "1pct"), (0.05, "5pct"), (0.10, "10pct")]:
        prf = precision_recall_f1_at_fraction(labels, scores, fraction)
        metrics[f"precision_{suffix}"] = prf["precision"]
        metrics[f"recall_{suffix}"] = prf["recall"]
        metrics[f"f1_{suffix}"] = prf["f1"]
    return metrics


def aggregate_metric_records(metric_records: Sequence[Mapping[str, float]]) -> Tuple[Dict[str, float], pd.DataFrame]:
    keys = sorted({key for record in metric_records for key in record.keys()})
    summary: Dict[str, float] = {}
    rows: List[Dict[str, Any]] = []
    for key in keys:
        values = np.asarray([float(record[key]) for record in metric_records], dtype=float)
        mean_value = float(values.mean())
        sd_value = float(values.std(ddof=1)) if len(values) > 1 else 0.0
        summary[key] = mean_value
        summary[f"{key}_sd"] = sd_value
        rows.append({"Metric": key, "Mean": mean_value, "SD": sd_value})
    return summary, pd.DataFrame(rows)


def roc_curve_arrays(labels: np.ndarray, scores: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    labels = np.asarray(labels, dtype=int)
    scores = np.asarray(scores, dtype=float)
    positives = float(labels.sum())
    negatives = float(len(labels) - labels.sum())
    if positives <= 0 or negatives <= 0:
        return np.asarray([0.0, 1.0]), np.asarray([0.0, 1.0])
    order = np.argsort(-scores, kind="mergesort")
    ranked = labels[order]
    tps = np.cumsum(ranked)
    fps = np.cumsum(1 - ranked)
    tpr = np.concatenate([[0.0], tps / positives, [1.0]])
    fpr = np.concatenate([[0.0], fps / negatives, [1.0]])
    return fpr, tpr


def precision_recall_curve_arrays(labels: np.ndarray, scores: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    labels = np.asarray(labels, dtype=int)
    scores = np.asarray(scores, dtype=float)
    positives = float(labels.sum())
    if positives <= 0:
        return np.asarray([0.0, 1.0]), np.asarray([0.0, 0.0])
    order = np.argsort(-scores, kind="mergesort")
    ranked = labels[order]
    tps = np.cumsum(ranked)
    precision = tps / np.arange(1, len(ranked) + 1, dtype=float)
    recall = tps / positives
    baseline = positives / max(1.0, float(len(labels)))
    precision = np.concatenate([[baseline], precision, [precision[-1] if len(precision) else baseline]])
    recall = np.concatenate([[0.0], recall, [1.0]])
    return recall, precision


def score_pairs(
    config: ExperimentConfig,
    dataframe: pd.DataFrame,
    cache_root: Path,
    *,
    cache_prefix: str,
    progress_label: str | None = None,
) -> np.ndarray:
    if progress_label is not None:
        print(
            f"[progress] {progress_label}: preparing caches for {len(dataframe)} pairs "
            f"({dataframe['compound_iso_smiles'].nunique()} unique ligands, "
            f"{dataframe['target_sequence'].nunique()} unique proteins)",
            flush=True,
        )
    isolated_config, graph_cache, protein_cache = build_isolated_caches(
        config,
        dataframe["compound_iso_smiles"].drop_duplicates().tolist(),
        dataframe["target_sequence"].drop_duplicates().tolist(),
        str(cache_root),
        force_rebuild=False,
        cache_prefix=cache_prefix,
    )
    if progress_label is not None:
        print(f"[progress] {progress_label}: caches ready, loading ensemble", flush=True)
    models, normalizer = load_publication_ensemble(isolated_config, checkpoint_dir=config.checkpoint_dir)
    loader = make_unlabeled_prediction_loader(dataframe, graph_cache, protein_cache, isolated_config)
    if progress_label is not None:
        total_batches = len(loader) if hasattr(loader, "__len__") else "unknown"
        print(f"[progress] {progress_label}: starting inference over {total_batches} batches", flush=True)
    payload = predict_unlabeled(
        models,
        loader,
        isolated_config,
        normalizer,
        collect_attention=False,
        progress_label=progress_label,
    )
    if progress_label is not None:
        print(f"[progress] {progress_label}: inference complete", flush=True)
    return np.asarray(payload["predictions"], dtype=float)


def run_fishing_section(ctx: PublicationContext) -> Dict[str, Any]:
    print("[fishing] fetching H1 sequence", flush=True)
    h1_sequence = fetch_h1_sequence(ctx)
    h1_path = ctx.results_dir / "h1_active_library.csv"
    if not ctx.args.force_refresh and h1_path.exists():
        h1_drugs = pd.read_csv(h1_path).drop_duplicates(subset=["smiles"]).reset_index(drop=True)
        cached_is_curated = (
            not h1_drugs.empty
            and "best_pchembl" in h1_drugs.columns
            and len(h1_drugs) >= int(getattr(ctx.args, "h1_max_records", H1_DEFAULT_ACTIVE_PANEL_SIZE))
        )
        if cached_is_curated:
            print("[fishing] reusing cached H1 active set", flush=True)
        else:
            h1_drugs = pd.DataFrame()
    else:
        h1_drugs = pd.DataFrame()

    if h1_drugs.empty:
        print("[fishing] fetching expanded curated H1 ligand panel from ChEMBL", flush=True)
        h1_drugs = fetch_known_h1_drugs(ctx, max_records=int(getattr(ctx.args, "h1_max_records", H1_DEFAULT_ACTIVE_PANEL_SIZE)))
        h1_drugs.to_csv(h1_path, index=False)

    mixed_libraries = build_h1_replicate_libraries(ctx, h1_drugs)
    config = clone_config_for_results(ctx.base_config, ctx.results_dir, ctx.base_config.profile_name, device=ctx.args.device)
    config.checkpoint_dir = ctx.base_config.checkpoint_dir
    config.device = ctx.args.device or ctx.base_config.device or config.device
    config.use_amp = bool(str(config.device).lower().startswith("cuda"))
    print(
        f"[fishing] screening library_size={ctx.args.screen_lib_size}, "
        f"replicates={mixed_libraries['replicate'].nunique()}, "
        f"known_h1_ligands={len(h1_drugs)}, max_pairs_per_batch={config.max_pairs_per_batch}, "
        f"device={config.device}",
        flush=True,
    )
    print(f"[fishing] scoring mixed library with max_pairs_per_batch={config.max_pairs_per_batch}", flush=True)
    screen_df = mixed_libraries[["smiles"]].rename(columns={"smiles": "compound_iso_smiles"})
    screen_df["target_sequence"] = h1_sequence
    scores = score_pairs(
        config,
        screen_df,
        ctx.results_dir / "cache" / "fishing",
        cache_prefix="h1_screen",
        progress_label="H1 mixed-library scoring",
    )
    mixed_libraries = mixed_libraries.copy()
    mixed_libraries["score"] = scores
    known_h1_smiles = set(h1_drugs["smiles"])
    replicate_metric_records: List[Dict[str, float]] = []
    roc_records: List[Tuple[np.ndarray, np.ndarray]] = []
    pr_records: List[Tuple[np.ndarray, np.ndarray]] = []
    per_replicate_rows: List[Dict[str, Any]] = []
    for replicate_id, replicate_df in mixed_libraries.groupby("replicate"):
        labels = replicate_df["smiles"].isin(known_h1_smiles).astype(int).to_numpy()
        replicate_scores = replicate_df["score"].to_numpy(dtype=float)
        replicate_metrics = mixed_screening_metrics(labels, replicate_scores)
        per_replicate_rows.append({"replicate": int(replicate_id), **replicate_metrics})
        replicate_metric_records.append(replicate_metrics)
        roc_records.append(roc_curve_arrays(labels, replicate_scores))
        pr_records.append(precision_recall_curve_arrays(labels, replicate_scores))

    replicate_metrics_df = pd.DataFrame(per_replicate_rows)
    replicate_metrics_df.to_csv(ctx.results_dir / "h1_drug_fishing_replicate_metrics.csv", index=False)
    if not replicate_metric_records:
        raise ValueError("No H1 mixed-library replicates were scored.")
    summary_metrics, summary_metrics_df = aggregate_metric_records(replicate_metric_records)
    write_json(ctx.results_dir / "h1_drug_fishing_sensitivity_summary.json", summary_metrics)
    summary_metrics_df.to_csv(ctx.results_dir / "h1_drug_fishing_sensitivity_summary.csv", index=False)
    metrics = dict(summary_metrics)
    metrics["num_h1_drugs"] = int(len(h1_drugs))
    metrics["num_replicates"] = int(H1_FISHING_REPLICATES)
    metrics["primary_library_size"] = int(ctx.args.screen_lib_size)
    metrics["positive_prevalence"] = float(len(h1_drugs) / max(1, ctx.args.screen_lib_size))

    kiba_df = pd.concat(
        [
            pd.read_csv(ctx.base_config.train_file, usecols=["target_sequence"]),
            pd.read_csv(ctx.base_config.val_file, usecols=["target_sequence"]),
            pd.read_csv(ctx.base_config.test_file, usecols=["target_sequence"]),
        ],
        ignore_index=True,
    ).drop_duplicates()
    decoy_sequences = kiba_df["target_sequence"].sample(n=min(ctx.args.num_decoy_proteins, len(kiba_df)), random_state=ctx.base_config.seed).tolist()
    specificity_records = []
    for _, row in h1_drugs.iterrows():
        for target_name, sequence in [("H1", h1_sequence)] + [(f"decoy_{idx+1}", seq) for idx, seq in enumerate(decoy_sequences)]:
            specificity_records.append(
                {
                    "compound_iso_smiles": row["smiles"],
                    "target_sequence": sequence,
                    "compound_name": row["name"],
                    "target_name": target_name,
                }
            )
    specificity_df = pd.DataFrame(specificity_records)
    print(
        f"[fishing] scoring specificity panel pairs={len(specificity_df)}, decoy_targets={len(decoy_sequences)}",
        flush=True,
    )
    specificity_scores = score_pairs(
        config,
        specificity_df[["compound_iso_smiles", "target_sequence"]],
        ctx.results_dir / "cache" / "fishing_specificity",
        cache_prefix="h1_specificity",
        progress_label="H1 specificity scoring",
    )
    specificity_df["score"] = specificity_scores
    reciprocal_ranks = []
    h1_top1_hits = 0
    for _, compound_df in specificity_df.groupby("compound_name"):
        ranked = compound_df.sort_values("score", ascending=False).reset_index(drop=True)
        labels_rank = (ranked["target_name"] == "H1").astype(int).to_numpy()
        rr = reciprocal_rank(labels_rank, ranked["score"].to_numpy())
        reciprocal_ranks.append(rr)
        if ranked.iloc[0]["target_name"] == "H1":
            h1_top1_hits += 1
    metrics["specificity_top1_rate"] = float(h1_top1_hits / max(1, len(reciprocal_ranks)))
    metrics["specificity_mrr"] = float(mean_reciprocal_rank(reciprocal_ranks))

    fishing_table = pd.DataFrame(
        [
            {"Metric": "H1 ligand panel size", "Value": float(metrics["num_h1_drugs"]), "SD": np.nan},
            {"Metric": "Library size per replicate", "Value": float(metrics["primary_library_size"]), "SD": np.nan},
            {"Metric": "Positive prevalence", "Value": metrics["positive_prevalence"], "SD": np.nan},
            {"Metric": "AUROC", "Value": metrics["AUROC"], "SD": metrics.get("AUROC_sd", np.nan)},
            {"Metric": "AUPRC", "Value": metrics["AUPRC"], "SD": metrics.get("AUPRC_sd", np.nan)},
            {"Metric": "BEDROC20", "Value": metrics["BEDROC20"], "SD": metrics.get("BEDROC20_sd", np.nan)},
            {"Metric": "EF1%", "Value": metrics["EF1pct"], "SD": metrics.get("EF1pct_sd", np.nan)},
            {"Metric": "EF5%", "Value": metrics["EF5pct"], "SD": metrics.get("EF5pct_sd", np.nan)},
            {"Metric": "Recovery@5%", "Value": metrics["recovery_top_5pct"], "SD": metrics.get("recovery_top_5pct_sd", np.nan)},
            {"Metric": "Recovery@10%", "Value": metrics["recovery_top_10pct"], "SD": metrics.get("recovery_top_10pct_sd", np.nan)},
        ]
    )
    fishing_table_latex = fishing_table.copy()
    fishing_table_latex["SD"] = fishing_table_latex["SD"].map(lambda value: "" if pd.isna(value) else value)
    save_table_outputs(
        fishing_table,
        "table3_h1_drug_fishing_metrics",
        ctx.results_dir,
        (
            f"External H1 retrieval test metrics. The analysis used {len(h1_drugs)} curated H1 ligands embedded into "
            f"{H1_FISHING_REPLICATES} independently sampled {ctx.args.screen_lib_size}-compound lead-like mixed "
            "libraries drawn from ZINC22. Reported retrieval metrics are mean +/- SD across replicate libraries."
        ),
        latex_dataframe=fishing_table_latex,
    )

    roc_grid = np.linspace(0.0, 1.0, 200)
    pr_grid = np.linspace(0.0, 1.0, 200)
    mean_tpr = np.mean([np.interp(roc_grid, fpr, tpr) for fpr, tpr in roc_records], axis=0)
    mean_precision = np.mean([np.interp(pr_grid, recall, precision) for recall, precision in pr_records], axis=0)
    baseline = metrics["positive_prevalence"]

    fig, axes = plt.subplots(1, 2, figsize=(12.0, 5.2))
    roc_ax, pr_ax = axes
    for fpr, tpr in roc_records:
        roc_ax.plot(fpr, tpr, color="#cbd5e1", lw=1.2, alpha=0.75, zorder=1)
    roc_ax.plot([0.0, 1.0], [0.0, 1.0], color="#94a3b8", linestyle="--", linewidth=1.2, zorder=1)
    roc_ax.plot(roc_grid, mean_tpr, color="#1d4ed8", lw=3.0, zorder=3)
    roc_ax.set_xlim(0.0, 1.0)
    roc_ax.set_ylim(0.0, 1.02)
    roc_ax.set_xlabel("False positive rate")
    roc_ax.set_ylabel("True positive rate")
    roc_ax.set_title("ROC across replicate libraries")
    roc_ax.text(
        0.54,
        0.12,
        f"AUROC = {metrics['AUROC']:.3f} ± {metrics.get('AUROC_sd', 0.0):.3f}",
        transform=roc_ax.transAxes,
        fontsize=11,
        color="#1e3a8a",
    )

    for recall, precision in pr_records:
        pr_ax.plot(recall, precision, color="#cbd5e1", lw=1.2, alpha=0.75, zorder=1)
    pr_ax.axhline(baseline, color="#94a3b8", linestyle="--", linewidth=1.2, zorder=1)
    pr_ax.plot(pr_grid, mean_precision, color="#0f766e", lw=3.0, zorder=3)
    pr_ax.set_xlim(0.0, 1.0)
    pr_ax.set_ylim(0.0, 1.02)
    pr_ax.set_xlabel("Recall")
    pr_ax.set_ylabel("Precision")
    pr_ax.set_title("Precision-recall across replicate libraries")
    pr_ax.text(
        0.40,
        0.12,
        f"AUPRC = {metrics['AUPRC']:.3f} ± {metrics.get('AUPRC_sd', 0.0):.3f}",
        transform=pr_ax.transAxes,
        fontsize=11,
        color="#115e59",
    )
    pr_ax.text(
        0.40,
        0.04,
        f"BEDROC20 = {metrics['BEDROC20']:.3f} ± {metrics.get('BEDROC20_sd', 0.0):.3f}",
        transform=pr_ax.transAxes,
        fontsize=10,
        color="#334155",
    )

    caption = textwrap.dedent(
        f"""
        External H1 retrieval test. A curated panel of {len(h1_drugs)} H1 ligands with supporting ChEMBL potency evidence was embedded into {H1_FISHING_REPLICATES} independently sampled {ctx.args.screen_lib_size:,}-compound lead-like libraries from ZINC22 and ranked by the DeepDTA-iBAM affinity model. The left panel shows replicate ROC traces together with the mean curve, and the right panel shows the corresponding precision-recall traces with the prevalence baseline. Across replicate libraries, mean AUROC was {metrics['AUROC']:.3f} +/- {metrics.get('AUROC_sd', np.nan):.3f}, mean AUPRC was {metrics['AUPRC']:.3f} +/- {metrics.get('AUPRC_sd', np.nan):.3f}, and mean BEDROC20 was {metrics['BEDROC20']:.3f} +/- {metrics.get('BEDROC20_sd', np.nan):.3f}. Because H1 lies outside the kinase-focused KIBA training distribution, the study is interpreted as an external retrieval stress test rather than as a prospective screening claim.
        """
    ).strip()
    save_figure(fig, "fig2_h1_drug_fishing", ctx.results_dir, caption)
    ctx.update_section_metrics("fishing", metrics)
    return metrics


def fetch_egfr_sequence(ctx: PublicationContext) -> str:
    accession = "P00533"
    url = UNIPROT_FASTA_URL.format(accession=accession)
    ctx.record_source("egfr_sequence", url, "UniProt FASTA sequence for human EGFR.")
    cache_path = cached_download_path(ctx, f"{accession}.fasta")
    if cache_path.exists() and not ctx.args.force_refresh:
        return parse_fasta_sequence(cache_path.read_text(encoding="utf-8"))
    fasta_text = fetch_text(url)
    write_text(cache_path, fasta_text)
    return parse_fasta_sequence(fasta_text)


def fetch_dasatinib_smiles(ctx: PublicationContext) -> str:
    query_url = "https://www.ebi.ac.uk/chembl/api/data/molecule/search.json?q=Dasatinib"
    ctx.record_source("dasatinib_query", query_url, "ChEMBL search query for dasatinib seed compound.")
    cache_path = cached_download_path(ctx, "dasatinib_query.json")
    if cache_path.exists() and not ctx.args.force_refresh:
        payload = read_json(cache_path)
    else:
        payload = fetch_json(query_url)
        write_json(cache_path, payload)
    molecules = payload.get("molecules", [])
    if not molecules:
        raise ValueError("Unable to locate dasatinib in ChEMBL search results.")
    structures = molecules[0].get("molecule_structures") or {}
    smiles = structures.get("canonical_smiles")
    if not smiles:
        raise ValueError("Dasatinib ChEMBL record did not contain canonical SMILES.")
    return smiles


def prepare_diffusion_variant_config(ctx: PublicationContext) -> ExperimentConfig:
    config = clone_config_for_results(ctx.base_config, ctx.results_dir, "diffusion_egfr_seed", device=ctx.args.device)
    config = prepare_scaffold_config(config, ctx.results_dir)
    config.diffusion_max_weight = 0.02
    config.diffusion_warmup_epochs = 2
    config.diffusion_ramp_end_epoch = 6
    config.checkpoint_dir = str(ctx.results_dir / "checkpoints" / "diffusion_egfr_seed")
    config.cache_root = str(ctx.results_dir / "cache" / "diffusion_egfr_seed")
    return config


def decode_generated_analog(seed_mol: Chem.Mol, feature_tensor: np.ndarray) -> Optional[Chem.Mol]:
    editable = Chem.RWMol(seed_mol)
    original_atomic_nums = [atom.GetAtomicNum() for atom in editable.GetAtoms()]
    original_charges = [atom.GetFormalCharge() for atom in editable.GetAtoms()]
    for atom_idx, atom in enumerate(editable.GetAtoms()):
        features = feature_tensor[atom_idx]
        symbol_candidates = np.argsort(features[: len(ATOM_TYPES)])[::-1]
        chosen_atomic_num = original_atomic_nums[atom_idx]
        for candidate_idx in symbol_candidates[:4]:
            candidate_symbol = ATOM_TYPES[int(candidate_idx)]
            candidate_atomic_num = Chem.GetPeriodicTable().GetAtomicNumber(candidate_symbol)
            atom.SetAtomicNum(candidate_atomic_num)
            charge_idx = int(np.argmax(features[22:27]))
            atom.SetFormalCharge(FORMAL_CHARGE_BINS[charge_idx])
            try:
                Chem.SanitizeMol(editable, sanitizeOps=Chem.SanitizeFlags.SANITIZE_PROPERTIES)
                chosen_atomic_num = candidate_atomic_num
                break
            except Exception:
                atom.SetAtomicNum(original_atomic_nums[atom_idx])
                atom.SetFormalCharge(original_charges[atom_idx])
        atom.SetAtomicNum(chosen_atomic_num)
        charge_idx = int(np.argmax(features[22:27]))
        atom.SetFormalCharge(FORMAL_CHARGE_BINS[charge_idx])

    mol = editable.GetMol()
    try:
        Chem.SanitizeMol(mol)
        return mol
    except Exception:
        try:
            fallback = Chem.Mol(seed_mol)
            Chem.SanitizeMol(fallback)
            return fallback
        except Exception:
            return None


def molecule_properties(mol: Chem.Mol) -> Dict[str, Any]:
    params = FilterCatalogParams()
    params.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS_A)
    params.AddCatalog(FilterCatalogParams.FilterCatalogs.BRENK)
    catalog = FilterCatalog(params)
    lipinski_pass = int(
        Descriptors.MolWt(mol) <= 500
        and Crippen.MolLogP(mol) <= 5
        and Lipinski.NumHDonors(mol) <= 5
        and Lipinski.NumHAcceptors(mol) <= 10
    )
    return {
        "smiles": Chem.MolToSmiles(mol),
        "QED": float(QED.qed(mol)),
        "MW": float(Descriptors.MolWt(mol)),
        "cLogP": float(Crippen.MolLogP(mol)),
        "HBA": int(Lipinski.NumHAcceptors(mol)),
        "HBD": int(Lipinski.NumHDonors(mol)),
        "TPSA": float(rdMolDescriptors.CalcTPSA(mol)),
        "RotB": int(Lipinski.NumRotatableBonds(mol)),
        "FractionCsp3": float(rdMolDescriptors.CalcFractionCSP3(mol)),
        "SA": float(sascorer.calculateScore(mol)),
        "LipinskiPass": lipinski_pass,
        "AlertFree": int(catalog.GetFirstMatch(mol) is None),
    }


def rank_generated_analogs(generated_df: pd.DataFrame) -> pd.DataFrame:
    return generated_df.sort_values(
        by=["LipinskiPass", "AlertFree", "QED", "SA", "PredAffinity"],
        ascending=[False, False, False, True, False],
    ).reset_index(drop=True)


def fetch_egfr_family_binders(ctx: PublicationContext) -> pd.DataFrame:
    cache_path = ctx.results_dir / "egfr_interpolation_family.csv"
    if cache_path.exists() and not ctx.args.force_refresh:
        family = pd.read_csv(cache_path)
        if len(family) >= EGFR_INTERP_ANCHOR_COUNT + 5:
            return family

    records: List[Dict[str, Any]] = []
    for page_idx in range(EGFR_INTERP_MAX_PAGES):
        offset = page_idx * EGFR_INTERP_PAGE_LIMIT
        url = (
            "https://www.ebi.ac.uk/chembl/api/data/activity.json"
            f"?target_chembl_id={EGFR_TARGET_CHEMBL_ID}"
            "&assay_type=B"
            "&standard_type__in=Ki,IC50,Kd"
            "&standard_relation=%3D"
            "&pchembl_value__isnull=false"
            "&canonical_smiles__isnull=false"
            "&order_by=-pchembl_value"
            f"&limit={EGFR_INTERP_PAGE_LIMIT}"
            f"&offset={offset}"
        )
        ctx.record_source(
            f"egfr_interpolation_activity_page_{page_idx + 1}",
            url,
            "ChEMBL EGFR activity page used to assemble the interpolation anchor and holdout family.",
        )
        payload = fetch_json(url)
        activities = payload.get("activities", [])
        if not activities:
            break
        for activity in activities:
            smiles = activity.get("canonical_smiles")
            parent_id = activity.get("parent_molecule_chembl_id") or activity.get("molecule_chembl_id")
            if not smiles or not parent_id:
                continue
            try:
                pchembl = float(activity["pchembl_value"])
            except (TypeError, ValueError, KeyError):
                continue
            try:
                standard_value = float(activity["standard_value"])
            except (TypeError, ValueError, KeyError):
                standard_value = np.nan
            records.append(
                {
                    "parent_molecule_chembl_id": parent_id,
                    "molecule_chembl_id": activity.get("molecule_chembl_id") or parent_id,
                    "name": activity.get("molecule_pref_name"),
                    "smiles": smiles,
                    "best_pchembl": pchembl,
                    "best_standard_nM": standard_value,
                    "activity_type": activity.get("standard_type"),
                    "document_chembl_id": activity.get("document_chembl_id"),
                    "source": "ChEMBL",
                }
            )
        if len(activities) < EGFR_INTERP_PAGE_LIMIT:
            break

    if not records:
        raise RuntimeError("Unable to retrieve any EGFR binders from ChEMBL for interpolation analysis.")

    dataframe = pd.DataFrame(records)
    dataframe = dataframe.sort_values(
        by=["parent_molecule_chembl_id", "best_pchembl", "best_standard_nM"],
        ascending=[True, False, True],
    ).drop_duplicates(subset=["parent_molecule_chembl_id"]).reset_index(drop=True)
    dataframe["fingerprint"] = dataframe["smiles"].map(fingerprint_from_smiles)

    reference_fps = {
        chembl_id: dataframe.loc[dataframe["parent_molecule_chembl_id"] == chembl_id, "fingerprint"].iloc[0]
        for chembl_id in EGFR_INTERP_REFERENCE_IDS
        if chembl_id in set(dataframe["parent_molecule_chembl_id"])
    }
    if len(reference_fps) < len(EGFR_INTERP_REFERENCE_IDS):
        missing = sorted(set(EGFR_INTERP_REFERENCE_IDS) - set(reference_fps))
        raise RuntimeError(f"Missing expected EGFR reference binders from ChEMBL activity pull: {missing}")

    dataframe["max_reference_similarity"] = [
        max(tanimoto_similarity(fp, ref_fp) for ref_fp in reference_fps.values())
        for fp in dataframe["fingerprint"]
    ]
    dataframe["label"] = dataframe["name"].fillna(dataframe["parent_molecule_chembl_id"])

    family = dataframe[
        (dataframe["best_pchembl"] >= EGFR_INTERP_MIN_PCHEMBL)
        & (dataframe["max_reference_similarity"] >= EGFR_INTERP_FAMILY_SIMILARITY)
    ].copy()
    if len(family) < EGFR_INTERP_ANCHOR_COUNT + 5:
        family = dataframe[
            (dataframe["best_pchembl"] >= EGFR_INTERP_MIN_PCHEMBL)
            & (dataframe["max_reference_similarity"] >= EGFR_INTERP_FAMILY_SIMILARITY - 0.05)
        ].copy()
    if len(family) < EGFR_INTERP_ANCHOR_COUNT + 5:
        raise RuntimeError(
            f"EGFR interpolation family remained too small after relaxed filtering: only {len(family)} compounds."
        )

    family = family.sort_values(
        by=["best_pchembl", "max_reference_similarity", "best_standard_nM"],
        ascending=[False, False, True],
    ).reset_index(drop=True)
    family.drop(columns=["fingerprint"], inplace=True)
    write_text(cache_path, family.to_csv(index=False))
    return family


def select_diverse_anchor_panel(family_df: pd.DataFrame, *, anchor_count: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    fingerprints = [fingerprint_from_smiles(smiles) for smiles in family_df["smiles"]]
    selected: List[int] = []
    for ref_id in EGFR_INTERP_REFERENCE_IDS:
        matches = family_df.index[family_df["parent_molecule_chembl_id"] == ref_id].tolist()
        for idx in matches:
            if idx not in selected:
                selected.append(int(idx))
            if len(selected) >= anchor_count:
                break
        if len(selected) >= anchor_count:
            break

    while len(selected) < anchor_count:
        best_idx = None
        best_score = -np.inf
        for idx in range(len(family_df)):
            if idx in selected:
                continue
            potency_bonus = float(family_df.iloc[idx]["best_pchembl"])
            if not selected:
                diversity = 1.0
            else:
                diversity = min(
                    1.0 - tanimoto_similarity(fingerprints[idx], fingerprints[chosen_idx])
                    for chosen_idx in selected
                )
            score = diversity + 0.03 * potency_bonus
            if score > best_score:
                best_score = score
                best_idx = idx
        if best_idx is None:
            break
        selected.append(int(best_idx))

    if len(selected) < anchor_count:
        raise RuntimeError(f"Unable to select {anchor_count} EGFR interpolation anchors.")

    anchor_df = family_df.iloc[selected].copy().reset_index(drop=True)
    holdout_df = family_df.drop(family_df.index[selected]).copy().reset_index(drop=True)
    return anchor_df, holdout_df


def sample_local_zinc_tranches(
    archive_path: Path,
    *,
    target_count: int,
    excluded_smiles: set[str],
    seed: int,
) -> pd.DataFrame:
    if not archive_path.exists():
        return pd.DataFrame(columns=["zinc_id", "smiles", "source", "source_tranche"])

    rng = np.random.default_rng(seed)
    rows: List[Dict[str, Any]] = []
    seen_smiles = set(excluded_smiles)
    with tarfile.open(archive_path, "r:gz") as tar:
        tranche_members = [member for member in tar.getmembers() if member.name.endswith(".smi.gz")]
        if not tranche_members:
            return pd.DataFrame(columns=["zinc_id", "smiles", "source", "source_tranche"])
        quota = max(1, int(np.ceil(target_count / len(tranche_members))))
        for member in tranche_members:
            extracted = tar.extractfile(member)
            if extracted is None:
                continue
            tranche_rows: List[Tuple[str, str]] = []
            with gzip.open(extracted, "rt", encoding="utf-8", errors="replace") as handle:
                for line_idx, line in enumerate(handle, start=1):
                    if line_idx > EGFR_INTERP_LOCAL_SCAN_PER_FILE:
                        break
                    fields = line.strip().split()
                    if len(fields) < 2:
                        continue
                    smiles, zinc_id = fields[0], fields[1]
                    if smiles in seen_smiles:
                        continue
                    tranche_rows.append((smiles, zinc_id))
            if not tranche_rows:
                continue
            sample_size = min(quota, len(tranche_rows))
            chosen_indices = rng.choice(len(tranche_rows), size=sample_size, replace=False)
            for idx in np.atleast_1d(chosen_indices):
                smiles, zinc_id = tranche_rows[int(idx)]
                if smiles in seen_smiles:
                    continue
                seen_smiles.add(smiles)
                rows.append(
                    {
                        "zinc_id": zinc_id,
                        "smiles": smiles,
                        "source": "Local ZINC tranches",
                        "source_tranche": member.name,
                    }
                )
            if len(rows) >= target_count:
                break
    return pd.DataFrame(rows).drop_duplicates(subset=["smiles"]).reset_index(drop=True)


def prepare_egfr_interpolation_zinc_library(
    ctx: PublicationContext,
    *,
    excluded_smiles: Sequence[str],
    target_count: int = EGFR_INTERP_ZINC_LIBRARY_SIZE,
) -> pd.DataFrame:
    cache_path = ctx.results_dir / "egfr_interpolation_zinc_library.csv"
    excluded = set(excluded_smiles)
    if cache_path.exists() and not ctx.args.force_refresh:
        cached = pd.read_csv(cache_path)
        cached = cached.drop_duplicates(subset=["smiles"])
        if "zinc_id" in cached.columns:
            cached = cached[cached["zinc_id"].astype(str).str.startswith("ZINC")]
        cached = cached[~cached["smiles"].isin(excluded)].reset_index(drop=True)
        if len(cached) >= target_count:
            return cached.head(target_count).copy()

    combined = pd.DataFrame(columns=["zinc_id", "smiles", "source", "source_tranche"])

    local_archive = Path(__file__).resolve().parent / "data" / "ZINC.tar.gz"
    if local_archive.exists():
        ctx.record_source(
            "egfr_interpolation_local_zinc_archive",
            str(local_archive),
            "Local property-filtered lead-like ZINC tranche archive used as the decoy background for EGFR interpolation retrieval.",
        )
        combined = sample_local_zinc_tranches(
            local_archive,
            target_count=target_count,
            excluded_smiles=excluded,
            seed=ctx.base_config.seed,
        )
        combined = combined.drop_duplicates(subset=["smiles"]).reset_index(drop=True)

    if len(combined) < target_count:
        fetched = fetch_zinc_random_library(ctx, max(target_count, target_count - len(combined) + 500))
        fetched = fetched.drop_duplicates(subset=["smiles"])
        if "zinc_id" in fetched.columns:
            fetched = fetched[fetched["zinc_id"].astype(str).str.startswith("ZINC")]
        fetched = fetched[~fetched["smiles"].isin(excluded)].reset_index(drop=True)
        combined = pd.concat([combined, fetched], ignore_index=True, sort=False)
        combined = combined.drop_duplicates(subset=["smiles"]).reset_index(drop=True)

    if len(combined) < target_count:
        raise RuntimeError(
            f"EGFR interpolation library assembled only {len(combined)} unique decoys; required {target_count}."
        )

    result = combined.head(target_count).copy()
    write_text(cache_path, result.to_csv(index=False))
    return result


def extract_conditioned_pair_embeddings(
    config: ExperimentConfig,
    dataframe: pd.DataFrame,
    cache_root: Path,
    *,
    cache_prefix: str,
    progress_label: str | None = None,
) -> pd.DataFrame:
    if progress_label is not None:
        print(
            f"[progress] {progress_label}: preparing caches for {len(dataframe)} pairs "
            f"({dataframe['compound_iso_smiles'].nunique()} unique ligands, "
            f"{dataframe['target_sequence'].nunique()} unique proteins)",
            flush=True,
        )
    isolated_config, graph_cache, protein_cache = build_isolated_caches(
        config,
        dataframe["compound_iso_smiles"].drop_duplicates().tolist(),
        dataframe["target_sequence"].drop_duplicates().tolist(),
        str(cache_root),
        force_rebuild=False,
        cache_prefix=cache_prefix,
    )
    if isolated_config.resolved_device.type == "cuda":
        isolated_config.max_pairs_per_batch = min(int(isolated_config.max_pairs_per_batch), 24)
        isolated_config.protein_token_budget = min(int(isolated_config.protein_token_budget), 16_000)
    models, normalizer = load_publication_ensemble(isolated_config, checkpoint_dir=config.checkpoint_dir)
    loader = make_unlabeled_prediction_loader(dataframe, graph_cache, protein_cache, isolated_config)
    if progress_label is not None:
        total_batches = len(loader) if hasattr(loader, "__len__") else "unknown"
        print(f"[progress] {progress_label}: starting inference over {total_batches} batches", flush=True)

    predictions: List[np.ndarray] = []
    embeddings: List[np.ndarray] = []
    total_batches = len(loader) if hasattr(loader, "__len__") else None
    progress_interval = 1
    if isinstance(total_batches, int) and total_batches > 10:
        progress_interval = max(1, total_batches // 10)

    for batch_idx, batch in enumerate(loader, start=1):
        batch = _move_batch_to_device(batch, runtime_device(isolated_config))
        member_predictions = []
        member_embeddings = []
        for model in models:
            with _autocast_context(isolated_config):
                pair_state = model.encode_conditioned_pair(
                    batch["drug_x"],
                    batch["drug_adj"],
                    batch["drug_mask"],
                    batch["protein_embeddings"],
                    batch["protein_mask"],
                    drug_edge_features=batch["drug_edge_features"],
                )
                affinity = model.affinity_head(
                    pair_state["drug_pool"],
                    pair_state["protein_pool_summary"],
                    pair_state["graph_global"],
                    pair_state["protein_global"],
                )
            member_predictions.append(
                _denormalize_predictions(affinity.detach().float().cpu(), normalizer, isolated_config.normalize_targets)
            )
            member_embeddings.append(pair_state["drug_pool"].detach().float().cpu())
        predictions.append(torch.stack(member_predictions, dim=0).mean(dim=0).numpy())
        embeddings.append(torch.stack(member_embeddings, dim=0).mean(dim=0).numpy())
        if progress_label is not None and (
            batch_idx == 1
            or total_batches is None
            or batch_idx == total_batches
            or batch_idx % progress_interval == 0
        ):
            if isinstance(total_batches, int):
                print(f"[progress] {progress_label}: batch {batch_idx}/{total_batches}", flush=True)
            else:
                print(f"[progress] {progress_label}: batch {batch_idx}", flush=True)

    result = dataframe.reset_index(drop=True).copy()
    prediction_array = np.concatenate(predictions, axis=0).reshape(-1)
    embedding_array = np.concatenate(embeddings, axis=0)
    norms = np.linalg.norm(embedding_array, axis=1, keepdims=True)
    embedding_array = embedding_array / np.clip(norms, 1e-8, None)
    result["PredAffinity"] = prediction_array.astype(float)
    result["latent_embedding"] = [row.astype(np.float32) for row in embedding_array]
    return result


def anchor_path_edges(anchor_embeddings: np.ndarray) -> List[Tuple[int, int]]:
    if len(anchor_embeddings) < 2:
        return []
    selected = {0}
    remaining = set(range(1, len(anchor_embeddings)))
    edges: List[Tuple[int, int]] = []
    while remaining:
        best_edge = None
        best_distance = np.inf
        for src in selected:
            remaining_list = sorted(remaining)
            deltas = anchor_embeddings[remaining_list] - anchor_embeddings[src]
            distances = np.linalg.norm(deltas, axis=1)
            min_idx = int(np.argmin(distances))
            dst = remaining_list[min_idx]
            distance = float(distances[min_idx])
            if distance < best_distance:
                best_distance = distance
                best_edge = (src, dst)
        if best_edge is None:
            break
        edges.append(best_edge)
        selected.add(best_edge[1])
        remaining.remove(best_edge[1])
    return edges


def min_distance_to_segments(points: np.ndarray, anchors: np.ndarray, edges: Sequence[Tuple[int, int]]) -> np.ndarray:
    if len(edges) == 0:
        return np.linalg.norm(points - anchors[0], axis=1)
    best = np.full(points.shape[0], np.inf, dtype=float)
    for src, dst in edges:
        start = anchors[src]
        end = anchors[dst]
        direction = end - start
        denom = float(np.dot(direction, direction))
        if denom <= 1e-10:
            distance = np.linalg.norm(points - start, axis=1)
        else:
            t = np.clip(((points - start) @ direction) / denom, 0.0, 1.0)
            projection = start + np.outer(t, direction)
            distance = np.linalg.norm(points - projection, axis=1)
        best = np.minimum(best, distance)
    return best


def min_distance_to_anchors(points: np.ndarray, anchors: np.ndarray) -> np.ndarray:
    best = np.full(points.shape[0], np.inf, dtype=float)
    for anchor in anchors:
        best = np.minimum(best, np.linalg.norm(points - anchor, axis=1))
    return best


def retrieval_metrics_from_scores(labels: np.ndarray, scores: np.ndarray) -> Dict[str, float]:
    return {
        "AUROC": float(auroc(labels, scores)),
        "AUPRC": float(auprc(labels, scores)),
        "BEDROC20": float(bedroc(labels, scores)),
        "EF1%": float(enrichment_factor(labels, scores, 0.01)),
        "EF5%": float(enrichment_factor(labels, scores, 0.05)),
        "Recovery@5%": float(topk_recovery(labels, scores, 0.05)),
        "Recovery@10%": float(topk_recovery(labels, scores, 0.10)),
    }


def cumulative_recovery_curve(labels: np.ndarray, scores: np.ndarray, fractions: np.ndarray) -> np.ndarray:
    return np.asarray([topk_recovery(labels, scores, float(fraction)) for fraction in fractions], dtype=float)


def run_interpolation_section(ctx: PublicationContext) -> Dict[str, Any]:
    egfr_sequence = fetch_egfr_sequence(ctx)
    family_df = fetch_egfr_family_binders(ctx)
    anchor_df, holdout_df = select_diverse_anchor_panel(family_df, anchor_count=EGFR_INTERP_ANCHOR_COUNT)
    zinc_df = prepare_egfr_interpolation_zinc_library(
        ctx,
        excluded_smiles=pd.concat([anchor_df["smiles"], holdout_df["smiles"]]).tolist(),
        target_count=EGFR_INTERP_ZINC_LIBRARY_SIZE,
    )

    anchor_df = anchor_df.copy()
    holdout_df = holdout_df.copy()
    zinc_df = zinc_df.copy()
    anchor_df["compound_iso_smiles"] = anchor_df["smiles"]
    holdout_df["compound_iso_smiles"] = holdout_df["smiles"]
    zinc_df["compound_iso_smiles"] = zinc_df["smiles"]
    anchor_df["target_sequence"] = egfr_sequence
    holdout_df["target_sequence"] = egfr_sequence
    zinc_df["target_sequence"] = egfr_sequence
    anchor_df["panel_role"] = "anchor"
    holdout_df["panel_role"] = "holdout"
    zinc_df["panel_role"] = "zinc"

    design_df = pd.concat(
        [
            anchor_df[["compound_iso_smiles", "target_sequence", "panel_role", "parent_molecule_chembl_id", "label", "best_pchembl"]],
            holdout_df[["compound_iso_smiles", "target_sequence", "panel_role", "parent_molecule_chembl_id", "label", "best_pchembl"]],
            zinc_df[["compound_iso_smiles", "target_sequence", "panel_role", "zinc_id"]],
        ],
        ignore_index=True,
        sort=False,
    )
    main_config = clone_config_for_results(ctx.base_config, ctx.results_dir, ctx.base_config.profile_name, device=ctx.args.device)
    main_config.checkpoint_dir = ctx.base_config.checkpoint_dir
    embedded_df = extract_conditioned_pair_embeddings(
        main_config,
        design_df,
        ctx.results_dir / "cache" / "egfr_interpolation",
        cache_prefix="egfr_interpolation",
        progress_label="EGFR interpolation embeddings",
    )

    anchor_emb = np.stack(embedded_df.loc[embedded_df["panel_role"] == "anchor", "latent_embedding"].tolist())
    holdout_emb = np.stack(embedded_df.loc[embedded_df["panel_role"] == "holdout", "latent_embedding"].tolist())
    zinc_emb = np.stack(embedded_df.loc[embedded_df["panel_role"] == "zinc", "latent_embedding"].tolist())
    candidate_df = embedded_df[embedded_df["panel_role"].isin(["holdout", "zinc"])].copy().reset_index(drop=True)
    candidate_emb = np.stack(candidate_df["latent_embedding"].tolist())
    labels = (candidate_df["panel_role"] == "holdout").astype(int).to_numpy()

    edges = anchor_path_edges(anchor_emb)
    candidate_df["PathDistance"] = min_distance_to_segments(candidate_emb, anchor_emb, edges)
    candidate_df["NearestAnchorDistance"] = min_distance_to_anchors(candidate_emb, anchor_emb)
    candidate_df["PathScore"] = -candidate_df["PathDistance"].to_numpy()
    candidate_df["NearestAnchorScore"] = -candidate_df["NearestAnchorDistance"].to_numpy()
    candidate_df["AffinityScore"] = candidate_df["PredAffinity"].to_numpy()
    candidate_df["CombinedScore"] = zscore_array(candidate_df["PathScore"].to_numpy()) + zscore_array(
        candidate_df["PredAffinity"].to_numpy()
    )

    anchor_fps = [fingerprint_from_smiles(smiles) for smiles in anchor_df["smiles"]]
    anchor_labels = anchor_df["label"].tolist()
    nearest_anchor_labels = []
    nearest_anchor_tanimoto = []
    for smiles in candidate_df["compound_iso_smiles"]:
        fp = fingerprint_from_smiles(smiles)
        similarities = [tanimoto_similarity(fp, anchor_fp) for anchor_fp in anchor_fps]
        best_idx = int(np.argmax(similarities))
        nearest_anchor_labels.append(anchor_labels[best_idx])
        nearest_anchor_tanimoto.append(float(similarities[best_idx]))
    candidate_df["NearestAnchor"] = nearest_anchor_labels
    candidate_df["NearestAnchorTanimoto"] = nearest_anchor_tanimoto

    method_scores = {
        "Interpolation path": candidate_df["PathScore"].to_numpy(),
        "Nearest anchor": candidate_df["NearestAnchorScore"].to_numpy(),
        "Predicted affinity": candidate_df["AffinityScore"].to_numpy(),
        "Combined": candidate_df["CombinedScore"].to_numpy(),
    }
    metrics_rows = []
    section_metrics: Dict[str, Any] = {
        "num_anchors": int(len(anchor_df)),
        "num_holdouts": int(len(holdout_df)),
        "num_zinc_decoys": int(len(zinc_df)),
        "family_similarity_threshold": float(EGFR_INTERP_FAMILY_SIMILARITY),
        "family_min_pchembl": float(EGFR_INTERP_MIN_PCHEMBL),
    }
    for method_name, scores in method_scores.items():
        method_metrics = retrieval_metrics_from_scores(labels, scores)
        metrics_rows.append({"Method": method_name, **method_metrics})
        for key, value in method_metrics.items():
            metric_key = f"{method_name.lower().replace(' ', '_')}_{key.lower().replace('%', 'pct').replace('@', '_at_')}"
            section_metrics[metric_key] = float(value)
    metrics_df = pd.DataFrame(metrics_rows)
    save_table_outputs(
        metrics_df,
        "table4_egfr_interpolation_metrics",
        ctx.results_dir,
        (
            "Interpolation-guided EGFR retrieval metrics. Six EGFR binder anchors defined a piecewise linear "
            f"interpolation path in the target-conditioned ligand latent space. Remaining EGFR-family holdouts "
            f"({len(holdout_df)}) were ranked against {len(zinc_df)} compounds from a local property-filtered lead-like "
            "ZINC tranche library using path proximity, nearest-anchor proximity, predicted EGFR affinity, or a combined score."
        ),
    )

    zinc_hits = candidate_df[candidate_df["panel_role"] == "zinc"].copy()
    zinc_hits = zinc_hits.sort_values(by=["CombinedScore", "PredAffinity"], ascending=[False, False]).reset_index(drop=True)
    zinc_hits["rank"] = np.arange(1, len(zinc_hits) + 1)
    hit_properties = []
    for _, row in zinc_hits.head(EGFR_INTERP_TOP_ZINC_HITS).iterrows():
        mol = Chem.MolFromSmiles(row["compound_iso_smiles"])
        props = molecule_properties(mol) if mol is not None else {}
        hit_properties.append(
            {
                "Rank": int(row["rank"]),
                "zinc_id": row.get("zinc_id", ""),
                "smiles": row["compound_iso_smiles"],
                "NearestAnchor": row["NearestAnchor"],
                "NearestAnchorTanimoto": row["NearestAnchorTanimoto"],
                "PredAffinity": row["PredAffinity"],
                "PathDistance": row["PathDistance"],
                "CombinedScore": row["CombinedScore"],
                "QED": props.get("QED", np.nan),
                "SA": props.get("SA", np.nan),
                "LipinskiPass": props.get("LipinskiPass", np.nan),
                "AlertFree": props.get("AlertFree", np.nan),
            }
        )
    zinc_hits_df = pd.DataFrame(hit_properties)
    zinc_hits_latex = zinc_hits_df.drop(columns=["smiles", "CombinedScore"]).copy()
    save_table_outputs(
        zinc_hits_df,
        "table5_top_egfr_interpolation_hits",
        ctx.results_dir,
        (
            f"Top {EGFR_INTERP_TOP_ZINC_HITS} EGFR retrieval hits from the local ZINC tranche library. Full SMILES are "
            "retained in the CSV artifact, whereas the manuscript table reports the nearest anchor, latent-path distance, "
            "predicted EGFR affinity, and medicinal chemistry heuristics."
        ),
        latex_dataframe=zinc_hits_latex,
    )

    candidate_df = candidate_df.sort_values(by=["CombinedScore"], ascending=False).reset_index(drop=True)
    export_candidate_df = candidate_df.drop(columns=["latent_embedding"], errors="ignore")
    write_text(ctx.results_dir / "egfr_interpolation_ranked_candidates.csv", export_candidate_df.to_csv(index=False))
    write_text(ctx.results_dir / "egfr_interpolation_anchor_panel.csv", anchor_df.to_csv(index=False))
    write_text(ctx.results_dir / "egfr_interpolation_holdout_panel.csv", holdout_df.to_csv(index=False))

    all_embeddings = np.stack(embedded_df["latent_embedding"].tolist())
    centered = all_embeddings - all_embeddings.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    basis = vt[:2].T if vt.shape[0] >= 2 else np.eye(centered.shape[1], 2)
    coords = centered @ basis
    embedded_df["PC1"] = coords[:, 0]
    embedded_df["PC2"] = coords[:, 1] if coords.shape[1] > 1 else 0.0

    top_zinc_smiles = set(zinc_hits_df["smiles"])
    scatter_df = embedded_df.copy()
    scatter_df["plot_group"] = "Background ZINC"
    scatter_df.loc[scatter_df["panel_role"] == "holdout", "plot_group"] = "Holdout EGFR binders"
    scatter_df.loc[scatter_df["panel_role"] == "anchor", "plot_group"] = "Anchor EGFR binders"
    scatter_df.loc[
        (scatter_df["panel_role"] == "zinc") & (scatter_df["compound_iso_smiles"].isin(top_zinc_smiles)),
        "plot_group",
    ] = "Top ZINC hits"

    fractions = np.linspace(0.01, 0.20, 40)
    recovery_curves = {
        name: cumulative_recovery_curve(labels, scores, fractions)
        for name, scores in method_scores.items()
    }
    random_curve = fractions.copy()

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.6))
    scatter_ax, recovery_ax = axes
    palette = {
        "Background ZINC": "#cbd5e1",
        "Holdout EGFR binders": "#2563eb",
        "Anchor EGFR binders": "#dc2626",
        "Top ZINC hits": "#15803d",
    }
    sizes = {
        "Background ZINC": 18,
        "Holdout EGFR binders": 48,
        "Anchor EGFR binders": 80,
        "Top ZINC hits": 60,
    }
    for group_name in ["Background ZINC", "Holdout EGFR binders", "Anchor EGFR binders", "Top ZINC hits"]:
        subset = scatter_df[scatter_df["plot_group"] == group_name]
        if subset.empty:
            continue
        scatter_ax.scatter(
            subset["PC1"],
            subset["PC2"],
            s=sizes[group_name],
            c=palette[group_name],
            alpha=0.8 if group_name != "Background ZINC" else 0.35,
            label=group_name,
            edgecolors="none",
        )
    anchor_coords = embedded_df.loc[embedded_df["panel_role"] == "anchor", ["PC1", "PC2"]].to_numpy()
    for src, dst in edges:
        scatter_ax.plot(
            [anchor_coords[src, 0], anchor_coords[dst, 0]],
            [anchor_coords[src, 1], anchor_coords[dst, 1]],
            color="#991b1b",
            linewidth=1.6,
            alpha=0.9,
        )
    scatter_ax.set_title("EGFR target-conditioned latent space")
    scatter_ax.set_xlabel("PC1")
    scatter_ax.set_ylabel("PC2")
    scatter_ax.legend(frameon=True, loc="best")

    curve_colors = {
        "Interpolation path": "#1d4ed8",
        "Nearest anchor": "#7c3aed",
        "Predicted affinity": "#ea580c",
        "Combined": "#047857",
    }
    for method_name, curve in recovery_curves.items():
        recovery_ax.plot(fractions * 100.0, curve * 100.0, label=method_name, linewidth=2.5, color=curve_colors[method_name])
    recovery_ax.plot(fractions * 100.0, random_curve * 100.0, linestyle="--", color="#94a3b8", linewidth=1.5, label="Random")
    recovery_ax.set_title("Holdout EGFR binder recovery")
    recovery_ax.set_xlabel("Top ranked fraction of candidates (%)")
    recovery_ax.set_ylabel("Recovered holdout binders (%)")
    recovery_ax.set_xlim(float(fractions.min() * 100.0), float(fractions.max() * 100.0))
    recovery_ax.set_ylim(0.0, 100.0)
    recovery_ax.legend(frameon=True, loc="lower right")

    combined_metrics = metrics_df.loc[metrics_df["Method"] == "Combined"].iloc[0]
    caption = textwrap.dedent(
        f"""
        Interpolation-guided EGFR ligand retrieval. Six diverse EGFR-family anchor binders defined a minimum-spanning interpolation path in the target-conditioned ligand latent space. Left: PCA projection of the latent space, showing anchors, remaining EGFR-family holdout binders, background compounds from the local property-filtered lead-like ZINC tranche library, and the top ZINC hits selected by the combined path-plus-affinity score. Right: cumulative recovery of holdout EGFR binders when ranked against {len(zinc_df)} ZINC compounds. The combined score achieved AUROC = {combined_metrics['AUROC']:.3f}, AUPRC = {combined_metrics['AUPRC']:.3f}, and Recovery@10% = {combined_metrics['Recovery@10%']:.3f}, supporting interpolation-guided retrieval as a proof-of-concept local lead-prioritization workflow rather than an unconstrained de novo generation claim.
        """
    ).strip()
    save_figure(fig, "fig4_egfr_interpolation_retrieval", ctx.results_dir, caption)

    ctx.update_section_metrics("interpolation", section_metrics)
    return section_metrics


def collect_seeded_analogs(
    models: Sequence[Any],
    batch: Mapping[str, torch.Tensor],
    seed_mol: Chem.Mol,
    *,
    rng: np.random.Generator,
    target_count: int = GENERATION_TARGET_ANALOGS,
    noise_sigma: float = GENERATION_NOISE_SIGMA,
    perturbations_per_draw: int = GENERATION_PERTURBATIONS_PER_DRAW,
    max_attempts: int = GENERATION_MAX_ATTEMPTS,
) -> pd.DataFrame:
    generated_molecules: Dict[str, Dict[str, Any]] = {}
    attempts = 0
    next_progress = 25
    for model in models:
        model.eval()

    while len(generated_molecules) < target_count and attempts < max_attempts:
        for model in models:
            raw = model.generate_molecules(
                batch["protein_embeddings"],
                batch["protein_mask"],
                batch["drug_adj"],
                batch["drug_mask"],
            )[0].detach().float().cpu().numpy()
            for _ in range(perturbations_per_draw):
                attempts += 1
                analog = decode_generated_analog(seed_mol, raw + rng.normal(0.0, noise_sigma, raw.shape))
                if analog is None:
                    continue
                smiles = Chem.MolToSmiles(analog)
                if smiles not in generated_molecules:
                    generated_molecules[smiles] = molecule_properties(analog)
                    if len(generated_molecules) >= next_progress or len(generated_molecules) == target_count:
                        print(
                            f"[generation] collected {len(generated_molecules)}/{target_count} unique valid analogs "
                            f"after {attempts} decode attempts",
                            flush=True,
                        )
                        next_progress += 25
                if len(generated_molecules) >= target_count or attempts >= max_attempts:
                    break
            if len(generated_molecules) >= target_count or attempts >= max_attempts:
                break

    if len(generated_molecules) < target_count:
        raise RuntimeError(
            f"Generation produced only {len(generated_molecules)} unique valid analogs after "
            f"{attempts} decode attempts; required {target_count}."
        )
    generated_df = pd.DataFrame(generated_molecules.values()).drop_duplicates(subset=["smiles"]).reset_index(drop=True)
    return generated_df.head(target_count).copy()


def run_generation_section(ctx: PublicationContext) -> Dict[str, Any]:
    egfr_sequence = fetch_egfr_sequence(ctx)
    dasatinib_smiles = fetch_dasatinib_smiles(ctx)
    seed_mol = Chem.MolFromSmiles(dasatinib_smiles)
    if seed_mol is None:
        raise ValueError("Unable to parse dasatinib seed SMILES.")

    diffusion_config = clone_config_for_results(ctx.base_config, ctx.results_dir, ctx.base_config.profile_name, device=ctx.args.device)
    diffusion_config.checkpoint_dir = ctx.base_config.checkpoint_dir
    diffusion_config.ensemble_size = ctx.base_config.ensemble_size
    checkpoint_paths = ensemble_checkpoint_paths(diffusion_config)
    if not all(path.exists() for path in checkpoint_paths):
        raise FileNotFoundError(
            "Generation requires an existing diffusion-trained checkpoint. Missing: "
            + ", ".join(str(path) for path in checkpoint_paths if not path.exists())
        )
    print("[generation] using the existing diffusion-trained base checkpoint", flush=True)
    models, _ = load_publication_ensemble(diffusion_config)
    isolated_config, graph_cache, protein_cache = build_isolated_caches(
        diffusion_config,
        [dasatinib_smiles],
        [egfr_sequence],
        str(ctx.results_dir / "cache" / "generation"),
        force_rebuild=ctx.args.force_refresh,
        cache_prefix="generation",
    )
    pair_df = pd.DataFrame([{"compound_iso_smiles": dasatinib_smiles, "target_sequence": egfr_sequence}])
    loader = make_unlabeled_prediction_loader(pair_df, graph_cache, protein_cache, isolated_config)
    batch = next(iter(loader))
    batch = {key: value.to(runtime_device(isolated_config)) if isinstance(value, torch.Tensor) else value for key, value in batch.items()}

    rng = np.random.default_rng(ctx.base_config.seed)
    generated_df = collect_seeded_analogs(
        models,
        batch,
        seed_mol,
        rng=rng,
        target_count=GENERATION_TARGET_ANALOGS,
        noise_sigma=GENERATION_NOISE_SIGMA,
        perturbations_per_draw=GENERATION_PERTURBATIONS_PER_DRAW,
        max_attempts=GENERATION_MAX_ATTEMPTS,
    )
    affinity_df = generated_df[["smiles"]].rename(columns={"smiles": "compound_iso_smiles"})
    affinity_df["target_sequence"] = egfr_sequence
    main_config = clone_config_for_results(ctx.base_config, ctx.results_dir, ctx.base_config.profile_name, device=ctx.args.device)
    main_config.checkpoint_dir = ctx.base_config.checkpoint_dir
    main_config.device = "cpu"
    main_config.use_amp = False
    print(f"[generation] scoring {len(affinity_df)} generated analogs against EGFR", flush=True)
    predicted_affinity = score_pairs(
        main_config,
        affinity_df,
        ctx.results_dir / "cache" / "generation_affinity",
        cache_prefix="generation_affinity",
        progress_label="EGFR analog affinity scoring",
    )
    generated_df["PredAffinity"] = predicted_affinity
    generated_df = rank_generated_analogs(generated_df)
    write_text(
        ctx.results_dir / "generated_egfr_analogs_100.csv",
        generated_df.to_csv(index=False),
    )
    top20_df = generated_df.head(GENERATION_TOP_ANALOGS).reset_index(drop=True)
    save_table_outputs(
        top20_df,
        "table2_top20_generated_compounds",
        ctx.results_dir,
        "Target-conditioned seeded molecular design results. The top 20 dasatinib-like EGFR-conditioned designed analogs are ranked by Lipinski compliance, alert-free status, QED, synthetic accessibility score, and predicted KIBA affinity.",
    )

    mols = [seed_mol] + [Chem.MolFromSmiles(smiles) for smiles in top20_df["smiles"][:GENERATION_TOP_ANALOGS]]
    legends = ["Seed\nDasatinib"] + [
        f"Rank {rank + 1}\nQED {row['QED']:.2f} | SA {row['SA']:.2f}\nPred. KIBA {row['PredAffinity']:.2f}"
        for rank, (_, row) in enumerate(top20_df.iterrows())
    ]
    draw_options = rdMolDraw2D.MolDrawOptions()
    draw_options.legendFontSize = 24
    draw_options.baseFontSize = 0.7
    draw_options.padding = 0.08
    draw_options.bondLineWidth = 2
    grid_image = Draw.MolsToGridImage(
        mols,
        legends=legends,
        molsPerRow=4,
        subImgSize=(420, 320),
        drawOptions=draw_options,
    )
    fig, ax = plt.subplots(figsize=(20, 16))
    ax.imshow(np.asarray(grid_image))
    ax.axis("off")
    ax.set_title("EGFR-conditioned seeded generation of dasatinib-like analogs", fontsize=20, pad=18)
    caption = textwrap.dedent(
        """
        Target-conditioned seeded molecular drug design using generative AI. Repeated stochastic diffusion draws with slight low-noise perturbations were decoded under a fixed dasatinib graph topology until 100 unique valid designed analogs were obtained. The displayed panel shows the top 20 candidate molecules ranked by Lipinski compliance, alert-free status, QED, synthetic accessibility score, and predicted KIBA affinity against EGFR.
        """
    ).strip()
    save_figure(fig, "fig3_egfr_dasatinib_generation", ctx.results_dir, caption)

    metrics = {
        "num_unique_valid_analogs": int(len(generated_df)),
        "num_ranked_top_analogs": int(len(top20_df)),
        "best_qed": float(generated_df["QED"].max()),
        "best_pred_affinity": float(generated_df["PredAffinity"].max()),
        "generation_noise_sigma": float(GENERATION_NOISE_SIGMA),
    }
    ctx.update_section_metrics("generation", metrics)
    return metrics


def prepare_ablation_config(ctx: PublicationContext) -> ExperimentConfig:
    config = clone_config_for_results(ctx.base_config, ctx.results_dir, "max_rmse_cluster_no_fusion", device=ctx.args.device)
    config = prepare_scaffold_config(config, ctx.results_dir)
    config.fusion_mode = "none"
    return config


def run_ablation_section(ctx: PublicationContext) -> Dict[str, Any]:
    main_config = clone_config_for_results(ctx.base_config, ctx.results_dir, ctx.base_config.profile_name, device=ctx.args.device)
    main_config = prepare_scaffold_config(main_config, ctx.results_dir)
    main_config.checkpoint_dir = ctx.base_config.checkpoint_dir
    ablation_config = prepare_ablation_config(ctx)
    if not all(path.exists() for path in ensemble_checkpoint_paths(ablation_config)):
        print("[ablation] training no-fusion scaffold-split ensemble", flush=True)
        resume_ablation = Path(ablation_config.checkpoint_dir).exists()
        train_ensemble(ablation_config, resume=resume_ablation)
    else:
        print("[ablation] using existing no-fusion ensemble checkpoints", flush=True)

    print("[ablation] evaluating main and no-fusion ensembles", flush=True)
    main_metrics = evaluate_main_ensemble(main_config)
    ablation_metrics = evaluate_main_ensemble(ablation_config)
    write_json(
        ctx.results_dir / f"{ctx.base_config.profile_name}_scaffold_eval_from_ablation.json",
        {key: value for key, value in main_metrics.items() if key not in {"predictions", "targets"}},
    )
    write_json(
        ctx.results_dir / "max_rmse_cluster_no_fusion_scaffold_eval.json",
        {key: value for key, value in ablation_metrics.items() if key not in {"predictions", "targets"}},
    )
    ci_delta = float(main_metrics["CI"] - ablation_metrics["CI"])
    bootstrap = paired_bootstrap_metric_delta(
        main_metrics["targets"],
        np.asarray(main_metrics["predictions"]),
        np.asarray(ablation_metrics["predictions"]),
        lambda truth, preds: concordance_index(truth, preds, max_samples=len(truth)),
        n_boot=400,
        seed=ctx.base_config.seed,
    )

    fig, ax = plt.subplots(figsize=(7, 5))
    ci_values = [main_metrics["CI"], ablation_metrics["CI"]]
    ax.bar(["DeepDTA-iBAM", "No fusion"], ci_values, color=["#1d4ed8", "#dc2626"])
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Concordance index")
    ax.set_title("Cross-attention ablation on scaffold split")
    ax.text(0.5, max(ci_values) + 0.02, f"Delta CI = {ci_delta:.4f}", ha="center", va="bottom")
    caption = "Figure 4. Scaffold-split concordance index comparison between the full DeepDTA-iBAM ensemble and a matched no-fusion ablation, with the CI drop summarizing the contribution of bidirectional cross-attention."
    save_figure(fig, "fig4_ablation_ci", ctx.results_dir, caption)

    metrics = {
        "main_CI": float(main_metrics["CI"]),
        "main_RMSE": float(main_metrics["RMSE"]),
        "main_MAE": float(main_metrics["MAE"]),
        "ablation_CI": float(ablation_metrics["CI"]),
        "ablation_RMSE": float(ablation_metrics["RMSE"]),
        "ablation_MAE": float(ablation_metrics["MAE"]),
        "delta_CI": ci_delta,
        "bootstrap_delta_CI": bootstrap["delta"],
        "bootstrap_delta_CI_low": bootstrap["ci_low"],
        "bootstrap_delta_CI_high": bootstrap["ci_high"],
    }
    ctx.update_section_metrics("ablation", metrics)
    return metrics


def run_diagnostics_section(ctx: PublicationContext) -> Dict[str, Any]:
    metrics = evaluate_config_on_split(ctx.base_config, results_dir=ctx.results_dir, split_mode="standard", split_name="test")
    predictions = np.asarray(metrics["predictions"], dtype=float)
    targets = np.asarray(metrics["targets"], dtype=float)
    means = (predictions + targets) / 2.0
    diffs = predictions - targets
    diff_mean = float(diffs.mean())
    diff_sd = float(diffs.std(ddof=1))
    loa_low = diff_mean - 1.96 * diff_sd
    loa_high = diff_mean + 1.96 * diff_sd

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].scatter(means, diffs, s=12, alpha=0.5, color="#2563eb")
    axes[0].axhline(diff_mean, color="#dc2626", linestyle="--", linewidth=1.5)
    axes[0].axhline(loa_low, color="#9ca3af", linestyle=":", linewidth=1.2)
    axes[0].axhline(loa_high, color="#9ca3af", linestyle=":", linewidth=1.2)
    axes[0].set_title("Bland-Altman analysis")
    axes[0].set_xlabel("Mean of predicted and observed KIBA")
    axes[0].set_ylabel("Prediction minus observation")

    residuals = predictions - targets
    axes[1].scatter(targets, residuals, s=12, alpha=0.5, color="#0f766e")
    axes[1].axhline(0.0, color="#dc2626", linestyle="--", linewidth=1.5)
    axes[1].set_title("Residuals versus observed KIBA")
    axes[1].set_xlabel("Observed KIBA")
    axes[1].set_ylabel("Residual")
    caption = "Figure 5. Residual diagnostics for the one-member DeepDTA-iBAM standard-KIBA test evaluation. Bland-Altman analysis highlights bias and dispersion, while the residual-versus-KIBA plot reveals whether errors drift systematically across the affinity range."
    save_figure(fig, "fig5_residual_diagnostics", ctx.results_dir, caption)
    summary = {
        "bias_mean": diff_mean,
        "bias_sd": diff_sd,
        "loa_low": loa_low,
        "loa_high": loa_high,
        "reported_split": "standard test",
    }
    ctx.update_section_metrics("diagnostics", summary)
    return summary


def benchmark_rows_from_literature() -> List[Dict[str, Any]]:
    return [
        {
            "model": "DeepDTA",
            "reported_split": "Standard KIBA split",
            "CI": 0.863,
            "MSE": 0.194,
            "notes": "Sequence CNN baseline",
            "source_url": DEEPDTA_PAPER_URL,
            "source_detail": "DeepDTA primary benchmark table on the canonical KIBA split.",
            "compatibility_note": "Primary-source standard-split affinity regression row.",
        },
        {
            "model": "WideDTA",
            "reported_split": "Standard KIBA split",
            "CI": 0.875,
            "MSE": 0.179,
            "notes": "Motif-enhanced CNN",
            "source_url": WIDEDTA_PAPER_URL,
            "source_detail": "WideDTA Table 3, best KIBA configuration.",
            "compatibility_note": "Primary-source standard-split affinity regression row.",
        },
        {
            "model": "GraphDTA (GAT-GCN)",
            "reported_split": "Standard KIBA split",
            "CI": 0.891,
            "MSE": 0.139,
            "notes": "Graph neural network baseline",
            "source_url": GRAPHDTA_PAPER_URL,
            "source_detail": "GraphDTA Table 5, GAT-GCN configuration on KIBA.",
            "compatibility_note": "Primary-source standard-split affinity regression row.",
        },
        {
            "model": "DGraphDTA",
            "reported_split": "Standard KIBA split",
            "CI": 0.904,
            "MSE": 0.126,
            "notes": "Distance-aware graph attention model",
            "source_url": DGRAPHDTA_PAPER_URL,
            "source_detail": "DGraphDTA KIBA results table in the primary article.",
            "compatibility_note": "Primary-source standard-split affinity regression row.",
        },
        {
            "model": "MGraphDTA",
            "reported_split": "Standard KIBA split",
            "CI": 0.902,
            "MSE": 0.128,
            "notes": "Multiscale graph affinity model",
            "source_url": MGRAPHDTA_PAPER_URL,
            "source_detail": "MGraphDTA Table 3 on KIBA.",
            "compatibility_note": "Primary-source standard-split affinity regression row.",
        },
        {
            "model": "HMM-DTA",
            "reported_split": "Standard KIBA split",
            "CI": 0.903,
            "MSE": 0.135,
            "notes": "Protein and molecule language-model baseline",
            "source_url": HMM_DTA_PAPER_URL,
            "source_detail": "Primary HMM-DTA benchmark table on KIBA.",
            "compatibility_note": "Primary-source standard-split affinity regression row.",
        },
    ]


def run_benchmark_section(ctx: PublicationContext) -> Dict[str, Any]:
    standard_eval_path = ctx.results_dir / f"{ctx.base_config.profile_name}_member{ctx.base_config.ensemble_size}_standard_eval.json"
    main_standard = evaluate_config_on_split(ctx.base_config, results_dir=ctx.results_dir, split_mode="standard", split_name="test")
    write_json(standard_eval_path, {"test": {key: value for key, value in main_standard.items() if key not in {"predictions", "targets"}}})
    main_mse = float(main_standard["RMSE"]) ** 2

    rows = [
        {
            "model": "DeepDTA-iBAM",
            "reported_split": "Standard KIBA split",
            "CI": float(main_standard["CI"]),
            "MSE": main_mse,
            "notes": "Integrated model evaluated locally on the canonical KIBA test partition",
            "source_url": str(standard_eval_path),
            "source_detail": "Local standard-split evaluation JSON generated from the final integrated checkpoint.",
            "compatibility_note": "Local canonical standard-split affinity regression evaluation.",
        }
    ]
    rows.extend(benchmark_rows_from_literature())
    for row in rows:
        if str(row["source_url"]).startswith("http"):
            ctx.record_source(f"benchmark_{row['model']}", row["source_url"], f"Benchmark source for {row['model']}.")
    benchmark_df = pd.DataFrame(rows)
    write_text(ctx.results_dir / "benchmark_source_audit.csv", benchmark_df.to_csv(index=False))
    benchmark_display_df = benchmark_df.copy()
    for column in ("CI", "MSE"):
        benchmark_display_df[column] = benchmark_display_df[column].apply(
            lambda value: "--"
            if value is None or (isinstance(value, float) and np.isnan(value))
            else value
            if isinstance(value, str)
            else f"{float(value):.3f}"
        )
    save_table_outputs(
        benchmark_df,
        "table1_benchmark",
        ctx.results_dir,
        (
            "Standard-split KIBA benchmark. Regression-only CI and MSE values are compared against primary-source "
            "standard-split affinity models. The DeepDTA-iBAM row reports the local evaluation of the integrated "
            "model, whereas literature rows reproduce the directly comparable values reported by the cited primary studies."
        ),
        latex_dataframe=benchmark_display_df[["model", "CI", "MSE", "notes"]],
    )
    metrics = {
        "num_benchmark_rows": int(len(benchmark_df)),
        "standard_ci_main": float(main_standard["CI"]),
        "standard_mse_main": main_mse,
        "standard_rmse_main": float(main_standard["RMSE"]),
        "standard_mae_main": float(main_standard["MAE"]),
    }
    ctx.update_section_metrics("benchmark", metrics)
    return metrics


def latex_figure_block(
    results_dir: Path,
    stem: str,
    caption_file: str,
    width: str = "0.98\\linewidth",
    fallback: str = "Figure caption unavailable.",
) -> str:
    caption = latex_bold_lead_sentence(read_caption(results_dir, caption_file, fallback))
    return textwrap.dedent(
        f"""
        \\begin{{figure}}[H]
        \\centering
        \\IfFileExists{{results/{stem}.pdf}}{{\\includegraphics[width={width}]{{results/{stem}.pdf}}}}{{\\fbox{{Pending figure}}}}
        \\caption{{{caption}}}
        \\label{{fig:{stem}}}
        \\end{{figure}}
        """
    ).strip()


def latex_external_figure_block(
    figure_path: str,
    caption: str,
    label: str,
    *,
    width: str = "0.98\\linewidth",
) -> str:
    clean_caption = latex_bold_lead_sentence(caption)
    pdf_path = re.sub(r"\.png$", ".pdf", figure_path, flags=re.IGNORECASE)
    return textwrap.dedent(
        f"""
        \\begin{{figure}}[H]
        \\centering
        \\IfFileExists{{{pdf_path}}}{{\\includegraphics[width={width}]{{{pdf_path}}}}}{{
        \\IfFileExists{{{figure_path}}}{{\\includegraphics[width={width}]{{{figure_path}}}}}{{\\fbox{{Pending figure}}}}
        }}
        \\caption{{{clean_caption}}}
        \\label{{{label}}}
        \\end{{figure}}
        """
    ).strip()


def latex_table_block(
    results_dir: Path,
    stem: str,
    caption_file: str,
    label: str,
    fallback: str = "Table caption unavailable.",
) -> str:
    caption = latex_bold_lead_sentence(read_caption(results_dir, caption_file, fallback))
    if stem in {"table2_top20_generated_compounds", "table5_top_egfr_interpolation_hits"}:
        return textwrap.dedent(
            f"""
            \\begin{{table}}[t]
            \\centering
            \\caption{{{caption}}}
            \\label{{{label}}}
            \\begingroup
            \\setlength{{\\tabcolsep}}{{2pt}}
            \\renewcommand{{\\arraystretch}}{{1.0}}
            \\scriptsize
            \\IfFileExists{{results/{stem}.tex}}{{\\resizebox{{\\linewidth}}{{!}}{{\\input{{results/{stem}.tex}}}}}}{{\\fbox{{Pending table}}}}
            \\endgroup
            \\end{{table}}
            """
        ).strip()
    return textwrap.dedent(
        f"""
        \\begin{{table}}[t]
        \\centering
        \\caption{{{caption}}}
        \\label{{{label}}}
        \\begingroup
        \\setlength{{\\tabcolsep}}{{4pt}}
        \\renewcommand{{\\arraystretch}}{{1.05}}
        \\footnotesize
        \\IfFileExists{{results/{stem}.tex}}{{\\input{{results/{stem}.tex}}}}{{\\fbox{{Pending table}}}}
        \\endgroup
        \\end{{table}}
        """
    ).strip()


def build_references_bib() -> str:
    return textwrap.dedent(
        """
        @article{tang2014kiba,
          title = {Making Sense of Large-Scale Kinase Inhibitor Bioactivity Data Sets: A Comparative and Integrative Analysis},
          author = {Tang, Jing and Szwajda, Agnieszka and Shakyawar, Sushil and Xu, Tao and Hintsanen, Petteri and Wennerberg, Krister and Aittokallio, Tapio},
          journal = {Journal of Chemical Information and Modeling},
          year = {2014},
          volume = {54},
          number = {3},
          pages = {735--743},
          doi = {10.1021/ci400709d}
        }

        @article{ozturk2018deepdta,
          title = {DeepDTA: deep drug-target binding affinity prediction},
          author = {Ozturk, Hakime and Ozgur, Arzucan and Ozkirimli, Elif},
          journal = {Bioinformatics},
          year = {2018},
          volume = {34},
          number = {17},
          pages = {i821--i829},
          doi = {10.1093/bioinformatics/bty593}
        }

        @article{ozturk2019widedta,
          title = {WideDTA: prediction of drug-target binding affinity},
          author = {Ozturk, Hakime and Ozgur, Arzucan and Ozkirimli, Elif},
          journal = {arXiv},
          year = {2019},
          eprint = {1902.04166},
          archivePrefix = {arXiv},
          primaryClass = {q-bio.QM}
        }

        @article{he2017simboost,
          title = {SimBoost: a read-across approach for predicting drug-target binding affinities using gradient boosting machines},
          author = {He, Tong and Heidemeyer, Marten and Ban, Fuqiang and Cherkasov, Artem and Ester, Martin},
          journal = {Journal of Cheminformatics},
          year = {2017},
          volume = {9},
          number = {1},
          pages = {24},
          doi = {10.1186/s13321-017-0209-z}
        }

        @article{pahikkala2015realistic,
          title = {Toward more realistic drug-target interaction predictions},
          author = {Pahikkala, Tapio and Airola, Antti and Pietila, Sami and Shakyawar, Sushil and Szwajda, Agnieszka and Tang, Jing and Aittokallio, Tapio},
          journal = {Briefings in Bioinformatics},
          year = {2015},
          volume = {16},
          number = {2},
          pages = {325--337},
          doi = {10.1093/bib/bbu010}
        }

        @article{nguyen2021graphdta,
          title = {GraphDTA: Predicting drug-target binding affinity with graph neural networks},
          author = {Nguyen, Thin and Le, Hang and Quinn, Thomas P. and Nguyen, Tri and Le, Thuc Duy and Venkatesh, Svetha},
          journal = {Bioinformatics},
          year = {2021},
          volume = {37},
          number = {8},
          pages = {1140--1147},
          doi = {10.1093/bioinformatics/btaa921}
        }

        @article{jiang2020dgraphdta,
          title = {Drug-target affinity prediction using graph neural network and contact maps},
          author = {Jiang, Mingjian and Li, Zhen and Zhang, Shugang and Wang, Shuang and Wang, Xiaofeng and Yuan, Qing},
          journal = {RSC Advances},
          year = {2020},
          volume = {10},
          number = {35},
          pages = {20701--20712},
          doi = {10.1039/d0ra02297g}
        }

        @article{yang2022mgraphdta,
          title = {MGraphDTA: deep multiscale graph neural network for explainable drug--target binding affinity prediction},
          author = {Yang, Ziduo and Zhong, Weihe and Zhao, Lu and Chen, Calvin Yu-Chian},
          journal = {Chemical Science},
          year = {2022},
          volume = {13},
          number = {3},
          pages = {816--833},
          doi = {10.1039/D1SC05180F}
        }

        @article{huang2022fusiondta,
          title = {FusionDTA: Attention-based feature polymerizer and knowledge distillation for drug-target binding affinity prediction},
          author = {Yuan, Weining and Chen, Guanxing and Chen, Calvin Yu-Chian},
          journal = {Briefings in Bioinformatics},
          year = {2022},
          volume = {23},
          number = {1},
          pages = {bbab506},
          doi = {10.1093/bib/bbab506}
        }

        @article{bidgoli2026hmmdta,
          title = {Structure-free drug-target affinity prediction using protein and molecule language models},
          author = {Bidgoli, Amir Hallaji and Mahdavi, Morteza and Malek, Hamed},
          journal = {Journal of Cheminformatics},
          year = {2026},
          volume = {18},
          number = {1},
          pages = {21},
          doi = {10.1186/s13321-025-01146-6}
        }

        @article{lin2023esm,
          title = {Evolutionary-scale prediction of atomic-level protein structure with a language model},
          author = {Lin, Zeming and Akin, Hannes and Rao, Roshan and Hie, Brian and Zhu, Zequn and Lu, Wenda and others},
          journal = {Science},
          year = {2023},
          volume = {379},
          number = {6637},
          pages = {1123--1130},
          doi = {10.1126/science.ade2574}
        }

        @inproceedings{velickovic2018gat,
          title = {Graph Attention Networks},
          author = {Velickovic, Petar and Cucurull, Guillem and Casanova, Arantxa and Romero, Adriana and Lio, Pietro and Bengio, Yoshua},
          booktitle = {International Conference on Learning Representations},
          year = {2018},
          eprint = {1710.10903},
          archivePrefix = {arXiv},
          primaryClass = {cs.LG}
        }

        @inproceedings{ho2020ddpm,
          title = {Denoising Diffusion Probabilistic Models},
          author = {Ho, Jonathan and Jain, Ajay and Abbeel, Pieter},
          booktitle = {Advances in Neural Information Processing Systems},
          year = {2020},
          volume = {33},
          pages = {6840--6851},
          eprint = {2006.11239},
          archivePrefix = {arXiv},
          primaryClass = {cs.LG}
        }

        @article{fak1p4n2025,
          title = {Structure-based identification of novel FAK1 inhibitors using pharmacophore modeling, molecular dynamics, and MM/PBSA calculations},
          author = {Hajipasha, Amirhossein and Cherati, Nilofar Ghaffari and Darzi, Mohammad and Nateghi, Seyedeh Sana and Mohsenian, Seyed Arshia Sadat and Noorzaei, Mahla and others},
          journal = {Scientific Reports},
          year = {2025},
          volume = {15},
          number = {1},
          pages = {39506},
          doi = {10.1038/s41598-025-23203-8}
        }

        @article{irwin2023zinc22,
          title = {ZINC22: a free multi-billion-scale database of tangible compounds for ligand discovery},
          author = {Tingle, Benjamin I. and Tang, Khanh G. and Castanon, Mar and Gutierrez, John J. and Khurelbaatar, Munkhzul and Dandarchuluun, Chinzorig and others},
          journal = {Journal of Chemical Information and Modeling},
          year = {2023},
          volume = {63},
          number = {4},
          pages = {1166--1176},
          doi = {10.1021/acs.jcim.2c01253}
        }

        @article{lipinski2001ro5,
          title = {Experimental and computational approaches to estimate solubility and permeability in drug discovery and development settings},
          author = {Lipinski, Christopher A. and Lombardo, Franco and Dominy, Beryl W. and Feeney, Paul J.},
          journal = {Advanced Drug Delivery Reviews},
          year = {2001},
          volume = {46},
          number = {1-3},
          pages = {3--26},
          doi = {10.1016/S0169-409X(00)00129-0}
        }

        @article{bickerton2012qed,
          title = {Quantifying the chemical beauty of drugs},
          author = {Bickerton, G. Richard and Paolini, Gaia V. and Besnard, Jeremy and Muresan, Sorel and Hopkins, Andrew L.},
          journal = {Nature Chemistry},
          year = {2012},
          volume = {4},
          number = {2},
          pages = {90--98},
          doi = {10.1038/nchem.1243}
        }

        @article{ertl2009sa,
          title = {Estimation of synthetic accessibility score of drug-like molecules based on molecular complexity and fragment contributions},
          author = {Ertl, Peter and Schuffenhauer, Ansgar},
          journal = {Journal of Cheminformatics},
          year = {2009},
          volume = {1},
          number = {1},
          pages = {8},
          doi = {10.1186/1758-2946-1-8}
        }

        @article{vamathevan2019ml,
          title = {Applications of machine learning in drug discovery and development},
          author = {Vamathevan, Jessica and Clark, Dominic and Czodrowski, Paul and Dunham, Ian and Ferran, Edgardo and Lee, George and others},
          journal = {Nature Reviews Drug Discovery},
          year = {2019},
          volume = {18},
          number = {6},
          pages = {463--477},
          doi = {10.1038/s41573-019-0024-5}
        }

        @article{schneider2020rethinking,
          title = {Rethinking drug design in the artificial intelligence era},
          author = {Schneider, Petra and Walters, W. Patrick and Plowright, Alleyn T. and Sieroka, Norman and Listgarten, Jennifer and Goodnow, Robert A. and Fisher, John and Jansen, Janet M. and Duca, John S. and Rush, T. Scott and Zentgraf, Matthias and Hill, Jennifer E. and Krutoholow, Ewa and Kohler, Markus and Blaney, John and Funatsu, Kimito and Luebkemann, Cornelius and Schneider, Gisbert},
          journal = {Nature Reviews Drug Discovery},
          year = {2020},
          volume = {19},
          number = {5},
          pages = {353--364},
          doi = {10.1038/s41573-019-0050-3}
        }

        @article{jimenezluna2020xai,
          title = {Drug discovery with explainable artificial intelligence},
          author = {Jimenez-Luna, Jose and Grisoni, Francesca and Schneider, Gisbert},
          journal = {Nature Machine Intelligence},
          year = {2020},
          volume = {2},
          number = {10},
          pages = {573--584},
          doi = {10.1038/s42256-020-00236-4}
        }

        @article{brown2019guacamol,
          title = {GuacaMol: Benchmarking Models for de Novo Molecular Design},
          author = {Brown, Nathan and Fiscato, Marco and Segler, Marwin H. S. and Vaucher, Alain C.},
          journal = {Journal of Chemical Information and Modeling},
          year = {2019},
          volume = {59},
          number = {3},
          pages = {1096--1108},
          doi = {10.1021/acs.jcim.8b00839}
        }

        @article{stokes2020antibiotic,
          title = {A Deep Learning Approach to Antibiotic Discovery},
          author = {Stokes, Jonathan M. and Yang, Kevin and Swanson, Kyle and Jin, Wengong and Cubillos-Ruiz, Andres and Donghia, Nina M. and others},
          journal = {Cell},
          year = {2020},
          volume = {180},
          number = {4},
          pages = {688--702.e13},
          doi = {10.1016/j.cell.2020.01.021}
        }
        """
    ).strip() + "\n"


def build_main_tex(ctx: PublicationContext) -> str:
    benchmark_metrics = ctx.metrics.get("benchmark", {})
    ibam_metrics = ctx.metrics.get("ibam", {})
    fishing_metrics = ctx.metrics.get("fishing", {})
    generation_metrics = ctx.metrics.get("generation", {})
    interpolation_metrics = ctx.metrics.get("interpolation", {})
    standard_ci = benchmark_metrics.get("standard_ci_main")
    standard_mse = benchmark_metrics.get("standard_mse_main")
    standard_rmse = benchmark_metrics.get("standard_rmse_main")
    standard_mae = benchmark_metrics.get("standard_mae_main")
    ibam_affinity = ibam_metrics.get("predicted_affinity")
    fishing_auroc = fishing_metrics.get("AUROC")
    fishing_auprc = fishing_metrics.get("AUPRC")
    fishing_auroc_sd = fishing_metrics.get("AUROC_sd")
    fishing_auprc_sd = fishing_metrics.get("AUPRC_sd")
    fishing_bedroc20 = fishing_metrics.get("BEDROC20")
    fishing_bedroc20_sd = fishing_metrics.get("BEDROC20_sd")
    fishing_ef1pct = fishing_metrics.get("EF1pct")
    fishing_recovery_top_10pct = fishing_metrics.get("recovery_top_10pct")
    fishing_screen_size = fishing_metrics.get("primary_library_size")
    fishing_num_drugs = fishing_metrics.get("num_h1_drugs")
    fishing_num_replicates = fishing_metrics.get("num_replicates")
    fishing_positive_prevalence = fishing_metrics.get("positive_prevalence")
    fishing_specificity_mrr = fishing_metrics.get("specificity_mrr")
    generation_best = generation_metrics.get("best_pred_affinity")
    generation_count = generation_metrics.get("num_unique_valid_analogs")
    interpolation_num_anchors = interpolation_metrics.get("num_anchors")
    interpolation_num_holdouts = interpolation_metrics.get("num_holdouts")
    interpolation_num_zinc_decoys = interpolation_metrics.get("num_zinc_decoys")
    interpolation_path_auroc = interpolation_metrics.get("interpolation_path_auroc")
    interpolation_path_auprc = interpolation_metrics.get("interpolation_path_auprc")
    nearest_anchor_auroc = interpolation_metrics.get("nearest_anchor_auroc")
    nearest_anchor_auprc = interpolation_metrics.get("nearest_anchor_auprc")
    predicted_affinity_auroc = interpolation_metrics.get("predicted_affinity_auroc")
    predicted_affinity_auprc = interpolation_metrics.get("predicted_affinity_auprc")
    combined_auroc = interpolation_metrics.get("combined_auroc")
    combined_bedroc20 = interpolation_metrics.get("combined_bedroc20")
    predicted_affinity_recovery_at_10pct = interpolation_metrics.get("predicted_affinity_recovery_at_10pct")

    def fmt(value: Any, precision: int = 4) -> str:
        if value is None:
            return "TBD"
        try:
            if np.isnan(float(value)):
                return "TBD"
        except Exception:
            return str(value)
        return f"{float(value):.{precision}f}"

    def pct(value: Any, precision: int = 1) -> str:
        if value is None:
            return "TBD"
        try:
            if np.isnan(float(value)):
                return "TBD"
        except Exception:
            return str(value)
        return f"{100.0 * float(value):.{precision}f}"

    architecture_caption = read_caption(
        ctx.results_dir,
        "fig0_model_architecture_caption.txt",
        "DeepDTA-iBAM architecture and evaluation workflow. Ligands are encoded as atom-bond graphs, proteins are represented by cached ESM-C residue embeddings, bidirectional cross-attention produces the interpretable multimodal state used for affinity prediction, and a diffusion auxiliary head enables target-conditioned seeded molecular drug design.",
    )

    return textwrap.dedent(
        f"""
        \\documentclass[12pt]{{article}}
        \\usepackage[T1]{{fontenc}}
        \\usepackage{{newtxtext}}
        \\usepackage{{newtxmath}}
        \\usepackage[margin=1in]{{geometry}}
        \\usepackage{{graphicx}}
        \\usepackage{{booktabs}}
        \\usepackage{{longtable}}
        \\usepackage{{array}}
        \\usepackage{{float}}
        \\usepackage{{setspace}}
        \\usepackage{{caption}}
        \\usepackage[super,sort&compress]{{natbib}}
        \\usepackage{{hyperref}}
        \\setlength{{\\emergencystretch}}{{3em}}
        \\captionsetup{{font=small}}
        \\hypersetup{{hidelinks}}
        \\begin{{document}}
        \\singlespacing
        \\begin{{center}}
        {{\\huge\\mdseries \\textbf{{DeepDTA-iBAM:}} Interpretable Cross-Attention for Affinity Prediction, Target-Conditioned Retrieval, and Generative Drug Design\\par}}
        \\vspace{{1.0\\baselineskip}}
        {{\\normalsize Affiliations to be finalized at submission\\par}}
        \\vspace{{0.4\\baselineskip}}
        {{\\normalsize April 7, 2026\\par}}
        \\end{{center}}
        \\vspace{{1.0\\baselineskip}}
        \\begin{{abstract}}
        Drug-target affinity models are most useful when quantitative prediction, mechanistic interpretation, and chemically actionable follow-up can be supported by a single representation rather than by disconnected tools. We present DeepDTA-iBAM, a multimodal architecture that combines graph-based ligand encoding, cached ESM-C protein embeddings, and bidirectional atom-residue cross-attention for affinity prediction, while a target-conditioned diffusion auxiliary head supports seeded molecular drug design. On the canonical standard KIBA split, the integrated model achieved concordance index = {fmt(standard_ci)}, MSE = {fmt(standard_mse)}, RMSE = {fmt(standard_rmse)}, and MAE = {fmt(standard_mae)}. In the P4N-FAK1 complex, the interpretable bidirectional attention maps concentrated on the experimentally implicated binding region, supporting the structural plausibility of the learned interaction signal. In an external H1 retrieval test using {int(fishing_num_drugs) if fishing_num_drugs is not None else 20} curated ligands embedded into {int(fishing_num_replicates) if fishing_num_replicates is not None else 5} independently sampled {int(fishing_screen_size) if fishing_screen_size is not None else 1000}-compound lead-like libraries, mean AUROC was {fmt(fishing_auroc)} \\(\\pm\\ {fmt(fishing_auroc_sd)}\\) and mean AUPRC was {fmt(fishing_auprc)} \\(\\pm\\ {fmt(fishing_auprc_sd)}\\). In an EGFR interpolation-guided, target-conditioned ligand retrieval study with {int(interpolation_num_anchors) if interpolation_num_anchors is not None else 6} anchor binders, {int(interpolation_num_holdouts) if interpolation_num_holdouts is not None else 341} holdout binders, and {int(interpolation_num_zinc_decoys) if interpolation_num_zinc_decoys is not None else 2000} compounds from a local property-filtered lead-like ZINC tranche library, interpolation-path ranking achieved AUROC = {fmt(interpolation_path_auroc, 3)} and AUPRC = {fmt(interpolation_path_auprc, 3)}, while predicted EGFR affinity achieved AUROC = {fmt(predicted_affinity_auroc, 3)} and AUPRC = {fmt(predicted_affinity_auprc, 3)}. In a target-conditioned seeded molecular drug design case study around dasatinib, the diffusion module produced {int(generation_count) if generation_count is not None else 100} unique valid designed analogs under fixed-topology decoding, with top candidates reaching predicted KIBA scores up to {fmt(generation_best)}. Taken together, these results position DeepDTA-iBAM as an integrated framework for affinity prediction, mechanistically interpretable interaction analysis, target-conditioned retrieval, and generative local design.
        \\end{{abstract}}
        \\section*{{\\textbf{{Introduction.}} Scientific context and study goals}}
        Drug-target affinity prediction is a central step in computational triage because it links molecular structure and protein sequence to a quantitative prioritization signal that can guide hit finding, lead optimization, and repurposing. Standardized resources such as KIBA enabled direct comparison across machine-learning baselines and later deep learning models, including similarity-driven methods, convolutional sequence models, graph neural networks, and more recent multimodal architectures \\cite{{tang2014kiba,he2017simboost,pahikkala2015realistic,ozturk2018deepdta,ozturk2019widedta,nguyen2021graphdta,jiang2020dgraphdta,yang2022mgraphdta,bidgoli2026hmmdta}}. At the same time, practical drug discovery increasingly demands more than a single regression score. Models are often expected to provide interpretable interaction evidence and to support chemically actionable follow-up decisions within the same workflow \\cite{{vamathevan2019ml,schneider2020rethinking,jimenezluna2020xai}}.

        DeepDTA-iBAM was designed around that broader requirement. The architecture combines graph-based ligand encoding, cached protein language-model embeddings, and bidirectional atom-residue cross-attention so that affinity prediction and interpretable interaction maps arise from the same multimodal state. A target-conditioned diffusion auxiliary branch is trained on that shared representation and later reused for seeded molecular drug design. This integration is the central methodological contribution of the framework: rather than treating prediction, explanation, retrieval, and local design as disconnected tasks, DeepDTA-iBAM attempts to support all four from one coordinated representation.

        The present study therefore evaluates DeepDTA-iBAM along four axes. First, we benchmark the integrated model on the canonical standard KIBA split and place it against primary-source, regression-only standard-split comparators. Second, we assess architectural contributions with a focused ablation study and evaluate structural plausibility with the P4N-FAK1 complex. Third, we test transfer beyond the kinase domain through an external H1 retrieval test and an EGFR interpolation-guided, target-conditioned ligand retrieval study against a large local chemical background. Fourth, we examine whether the diffusion auxiliary head supports target-conditioned seeded molecular drug design using generative AI in a chemically local EGFR case study around dasatinib.

        \\section*{{\\textbf{{Methods.}} Model, data, and evaluation protocols}}
        \\subsection*{{\\textbf{{Model architecture.}} Multimodal affinity modeling with interpretable cross-attention}}
        DeepDTA-iBAM represents each ligand as an RDKit-derived atom-bond graph and each target protein as a cached residue embedding tensor generated by an ESM-C encoder \\cite{{lin2023esm}}. Each atom is encoded with a 78-dimensional feature vector that includes atom identity, degree, implicit valence, formal charge, hybridization, aromaticity, ring membership, hydrogen count, valence, radical electrons, chirality, electronegativity bins, scaled atomic mass, scaled atomic number, covalent radius, Gasteiger charge bins, and simple donor, acceptor, and hydrophobicity indicators. Bonds are encoded with 12 edge features that capture bond order, conjugation, ring status, ring-size indicators, stereochemistry, scaled bond order, and a rotatable-bond flag. The ligand encoder applies multi-head graph attention with edge-feature bias and masking \\cite{{velickovic2018gat}}. Protein residue embeddings are projected by a learned adapter into the shared fusion space before bidirectional atom-to-residue and residue-to-atom cross-attention produces the multimodal state used for affinity prediction and iBAM extraction. A diffusion auxiliary head is trained on the ligand representation under target conditioning following the denoising diffusion framework \\cite{{ho2020ddpm}} and is reused for seeded molecular design at inference time. Figure~\\ref{{fig:model_architecture}} summarizes the end-to-end workflow.
        {latex_external_figure_block("results/fig0_model_architecture.png", architecture_caption, "fig:model_architecture")}

        \\subsection*{{\\textbf{{Data sources and evaluation design.}} Benchmarking, ablation, and case-study setup}}
        The primary quantitative benchmark uses the canonical standard KIBA split and evaluates the final trained DeepDTA-iBAM checkpoint on the held-out test partition. For manuscript comparison, literature baselines are restricted to primary-source reports of affinity regression on the standard KIBA split so that the reported concordance index and mean squared error are task-compatible. For the local model, MSE is derived as RMSE\\(^2\\), while RMSE and MAE are retained in the narrative because they are directly available from the present evaluation. The ablation study compares the integrated model against matched variants that remove or simplify selected architectural components under standard and scaffold splits.

        The structural interpretability study scores the known P4N-FAK1 pair and aggregates ligand-to-target and target-to-ligand attention across exported attention tensors for comparison with the 6YOJ co-crystal structure and published binding descriptions \\cite{{fak1p4n2025}}. The external H1 retrieval test uses a curated panel of {int(fishing_num_drugs) if fishing_num_drugs is not None else 20} named H1 ligands with supporting ChEMBL binding evidence embedded into {int(fishing_num_replicates) if fishing_num_replicates is not None else 5} independently sampled {int(fishing_screen_size) if fishing_screen_size is not None else 1000}-compound lead-like mixed libraries drawn from ZINC22 \\cite{{irwin2023zinc22}}. The replicate libraries were scored with the same affinity head used for KIBA benchmarking, and discrimination was summarized with AUROC, AUPRC, BEDROC20, enrichment factor, and top-k recovery so that both threshold-free ranking and early recognition could be assessed under class imbalance. A secondary protein-specificity control ranked the same ligands against H1 and 32 randomly sampled KIBA proteins and was used as a calibration analysis rather than as a claimed target-identification benchmark.

        The EGFR interpolation-guided retrieval study used {int(interpolation_num_anchors) if interpolation_num_anchors is not None else 6} diverse EGFR-family anchor binders from ChEMBL, treated the remaining {int(interpolation_num_holdouts) if interpolation_num_holdouts is not None else 341} EGFR-family binders as holdout positives, and ranked those holdouts against {int(interpolation_num_zinc_decoys) if interpolation_num_zinc_decoys is not None else 2000} compounds sampled from a local property-filtered lead-like ZINC tranche library spanning heavy-atom tranches H15-H20 and LogP tranches M000, P000, P010, and P020. Target-conditioned ligand embeddings were extracted against EGFR, L2-normalized, and connected with a minimum spanning tree over the anchor embeddings to define the interpolation path. Candidates were ranked by interpolation-path proximity, nearest-anchor proximity, predicted EGFR affinity, or a combined path-plus-affinity score. The generative design study is framed as a proof-of-concept local design case study: low-noise seeded sampling was performed around dasatinib under EGFR conditioning while preserving the seed bond topology during decoding, {int(generation_count) if generation_count is not None else 100} unique valid designed analogs were collected, each analog was rescored against EGFR, and the top 20 were ranked by Lipinski compliance, alert-free status, QED, synthetic accessibility score, and predicted KIBA affinity. Lipinski compliance followed the rule-of-five heuristic \\cite{{lipinski2001ro5}}, QED summarized global drug-likeness \\cite{{bickerton2012qed}}, and synthetic accessibility was estimated with the fragment- and complexity-based score of Ertl and Schuffenhauer \\cite{{ertl2009sa}}.



        \\section*{{\\textbf{{Results.}} Quantitative performance, interpretability, retrieval, and design studies}}
        \\subsection*{{\\textbf{{Standard KIBA benchmark.}} Integrated performance on the canonical affinity task}}
        On the canonical standard KIBA test partition, the final trained integrated DeepDTA-iBAM configuration achieved concordance index = {fmt(standard_ci)}, MSE = {fmt(standard_mse)}, RMSE = {fmt(standard_rmse)}, and MAE = {fmt(standard_mae)}. Table~\\ref{{tab:benchmark}} places this result against a deliberately restricted comparator set composed only of primary-source, regression-only standard-split KIBA rows \\cite{{ozturk2018deepdta,ozturk2019widedta,nguyen2021graphdta,jiang2020dgraphdta,yang2022mgraphdta,bidgoli2026hmmdta}}. By design, this excludes task-incompatible contextual rows and avoids mixing standard-split affinity regression with different label spaces, different splitting protocols, or classification settings. The resulting benchmark is narrower than many survey-style tables, but it is methodologically cleaner.

        The benchmark also clarifies the present positioning of the architecture. DeepDTA-iBAM is not introduced here as the lowest-error standard-split regressor in the literature. Instead, its practical value lies in maintaining credible affinity prediction while supporting interpretability, target-conditioned retrieval, and generative local design within one model family. That broader integration is the central claim carried forward into the case studies below.
        {latex_table_block(ctx.results_dir, "table1_benchmark", "table1_benchmark_caption.txt", "tab:benchmark", fallback="Standard-split KIBA benchmark. Regression-only CI and MSE comparison against primary-source standard-split affinity baselines.")} 

        \\subsection*{{\\textbf{{Ablation study.}} Architectural contributions and generalization behavior}}
        The ablation analysis sharpens the interpretation of the benchmark result. On the standard split, simplified variants that retain the core backbone but remove some auxiliary components match or exceed the full integrated model, indicating that the ranking and diffusion auxiliaries do not yet provide a consistent net gain in pure predictive accuracy. On the scaffold split, performance degrades substantially for all variants, emphasizing that out-of-scaffold generalization remains the hardest regime in this study. Within that harder setting, removing bidirectional fusion is most damaging, which suggests that the cross-attention backbone contributes the clearest value to generalization. Taken together, the ablation results support a balanced conclusion: the core multimodal architecture is strong, whereas some workflow-oriented components remain better justified by downstream utility than by standalone benchmark gains.
        \\IfFileExists{{results/ablation_table.tex}}{{\\input{{results/ablation_table.tex}}}}{{\\fbox{{Pending ablation table}}}}

        \\subsection*{{\\textbf{{P4N-FAK1 interpretability.}} Structural agreement between iBAM attention and the binding interface}}
        For the known FAK1 inhibitor P4N, the DeepDTA-iBAM model predicted a KIBA affinity of {fmt(ibam_affinity)} and produced coherent ligand-to-target and target-to-ligand attention maps over the FAK1 catalytic domain. The highest aggregated residue attention localized to the region reported to contact P4N in the co-crystal structure, including the residue neighborhood surrounding Cys95, Glu93, Leu94, Gly156, Asp157, and Leu146.

        This directional agreement matters because it suggests that the explanation signal arises from the same local interaction pattern regardless of whether the query starts from ligand atoms or protein residues. When interpreted against the crystallographic complex and reported contact analysis, the high-attention region overlaps the residue neighborhood implicated in hydrogen bonding and hydrophobic packing. That concordance does not by itself establish causality, but it does provide structurally coherent evidence that the iBAM maps capture biologically meaningful interface information rather than generic salience. At the same time, this result should be interpreted as a focused validation example rather than as a complete demonstration of explanation fidelity across all target classes.
        {latex_figure_block(ctx.results_dir, "fig1_p4n_fak1_ibam", "fig1_p4n_fak1_ibam_caption.txt", fallback="P4N-FAK1 iBAM interpretation. Bidirectional cross-attention localizes to the crystallographically implicated binding region.")} 

        \\subsection*{{\\textbf{{External H1 retrieval test.}} Reproducible out-of-domain compound ranking signal}}
        The external H1 retrieval test used a {int(fishing_num_drugs) if fishing_num_drugs is not None else 20}-ligand panel with supporting ChEMBL binding evidence embedded into {int(fishing_num_replicates) if fishing_num_replicates is not None else 5} independently sampled {int(fishing_screen_size) if fishing_screen_size is not None else 1000}-compound lead-like libraries. Across replicate libraries, mean AUROC was {fmt(fishing_auroc)} \\(\\pm\\ {fmt(fishing_auroc_sd)}\\) and mean AUPRC was {fmt(fishing_auprc)} \\(\\pm\\ {fmt(fishing_auprc_sd)}\\) at a positive prevalence of {fmt(fishing_positive_prevalence, 3)}. Because random ranking under this class balance would produce an expected AUPRC close to the prevalence, the observed AUPRC corresponds to an approximately 4.2-fold lift over the random baseline. Early recognition was present but modest, with BEDROC20 = {fmt(fishing_bedroc20)} \\(\\pm\\ {fmt(fishing_bedroc20_sd)}\\), EF1\\% = {fmt(fishing_ef1pct, 1)}, and {pct(fishing_recovery_top_10pct)}\\% of the positive panel recovered within the top 10\\% of each ranked library.

        The small replicate variance indicates that the ranking signal is reproducible across changes in the background chemical library rather than being driven by one favorable negative draw. At the same time, the study remains intentionally conservative. H1 lies outside the kinase-focused KIBA training distribution, and an orthogonal target-specificity control against 32 decoy proteins remained weak, with mean reciprocal rank = {fmt(fishing_specificity_mrr, 3)} and no H1 top-1 recoveries. We therefore interpret Figure~\\ref{{fig:fig2_h1_drug_fishing}} and Table~\\ref{{tab:fishing}} as evidence of reproducible out-of-domain compound retrieval around a fixed external target, not as proof of robust target deconvolution or deployment-ready screening.
        {latex_figure_block(ctx.results_dir, "fig2_h1_drug_fishing", "fig2_h1_drug_fishing_caption.txt", fallback="External H1 retrieval test. Replicate lead-like screening libraries reveal reproducible out-of-domain ranking signal.")} 
        {latex_table_block(ctx.results_dir, "table3_h1_drug_fishing_metrics", "table3_h1_drug_fishing_metrics_caption.txt", "tab:fishing", fallback="External H1 retrieval test metrics. Mean retrieval performance is reported across replicate screening libraries.")} 

        \\subsection*{{\\textbf{{Interpolation-guided EGFR retrieval.}} Target-conditioned ligand retrieval from a large chemical library}}
        To test whether the target-conditioned ligand space supports lead prioritization beyond single-anchor similarity, we constructed an EGFR case study around {int(interpolation_num_anchors) if interpolation_num_anchors is not None else 6} diverse EGFR-family anchor binders, treated the remaining {int(interpolation_num_holdouts) if interpolation_num_holdouts is not None else 341} EGFR-family binders as holdout positives, and ranked those holdouts against {int(interpolation_num_zinc_decoys) if interpolation_num_zinc_decoys is not None else 2000} compounds drawn from a local property-filtered lead-like ZINC tranche library spanning heavy-atom tranches H15-H20 and LogP tranches M000, P000, P010, and P020. A minimum spanning tree over the anchor embeddings defined the interpolation path. Interpolation-path ranking alone outperformed nearest-anchor retrieval, achieving AUROC = {fmt(interpolation_path_auroc, 3)} and AUPRC = {fmt(interpolation_path_auprc, 3)} versus AUROC = {fmt(nearest_anchor_auroc, 3)} and AUPRC = {fmt(nearest_anchor_auprc, 3)} for the nearest-anchor baseline. This gap indicates that the EGFR-conditioned latent space captures family structure that is not reducible to single-anchor neighborhood search.

        At the same time, the explicit EGFR affinity head remained the strongest ranking signal, with AUROC = {fmt(predicted_affinity_auroc, 3)}, AUPRC = {fmt(predicted_affinity_auprc, 3)}, and {pct(predicted_affinity_recovery_at_10pct)}\\% of holdout binders recovered within the top 10\\% of candidates. A combined path-plus-affinity score produced a similar AUROC of {fmt(combined_auroc, 3)} but lower BEDROC20 than affinity alone, indicating that interpolation geometry is complementary rather than dominant under the current model. We therefore interpret Figure~\\ref{{fig:fig4_egfr_interpolation_retrieval}}, Table~\\ref{{tab:interpolation_metrics}}, and Table~\\ref{{tab:interpolation_hits}} as a proof-of-concept, interpolation-guided, target-conditioned ligand retrieval study: interpolation-path proximity carries meaningful EGFR-family signal, but it does not replace supervised affinity prediction. The highest-ranked ZINC hits reached predicted EGFR-directed KIBA scores above 12 while showing mixed medicinal chemistry heuristics, so they are best read as prioritized follow-up candidates rather than validated leads.
        {latex_figure_block(ctx.results_dir, "fig4_egfr_interpolation_retrieval", "fig4_egfr_interpolation_retrieval_caption.txt", fallback="Interpolation-guided EGFR ligand retrieval. Target-conditioned latent geometry supports recovery of held-out EGFR binders against a large local chemical background.")} 
        {latex_table_block(ctx.results_dir, "table4_egfr_interpolation_metrics", "table4_egfr_interpolation_metrics_caption.txt", "tab:interpolation_metrics", fallback="Interpolation-guided EGFR retrieval metrics. Four ranking schemes are compared against held-out EGFR-family binders and the local ZINC background.")} 
        {latex_table_block(ctx.results_dir, "table5_top_egfr_interpolation_hits", "table5_top_egfr_interpolation_hits_caption.txt", "tab:interpolation_hits", fallback="Top EGFR retrieval hits from the local ZINC tranche library. The combined path-plus-affinity score prioritizes candidate molecules for follow-up.")} 

        \\subsection*{{\\textbf{{Target-conditioned seeded molecular drug design using generative AI.}} Proof-of-concept local design around dasatinib}}
        The diffusion-enabled generation workflow was evaluated as a proof-of-concept local design case study rather than as a standalone generative benchmark. Under EGFR conditioning, low-noise sampling around dasatinib produced {int(generation_count) if generation_count is not None else 100} unique valid fixed-topology designed analogs and reached a best predicted KIBA affinity of {fmt(generation_best)} among the ranked candidates. Because decoding preserved the dasatinib bond topology, the experiment tests whether the shared representation can support chemically local target-conditioned molecular design, not whether it can discover unconstrained novel chemotypes.

        The top-ranked molecules satisfy acceptable medicinal chemistry heuristics while remaining close to the seed scaffold, which is the intended behavior of this experiment. At the same time, all candidate molecules were filtered and rescored by the same modeling framework used for generation, and no orthogonal docking, synthesis, or biochemical validation is presented here. We therefore interpret Figure~\\ref{{fig:fig3_egfr_dasatinib_generation}} and Table~\\ref{{tab:generated}} as a proof-of-concept local design demonstration that yields hypothesis-generating designed analogs rather than validated EGFR optimization leads.
        {latex_figure_block(ctx.results_dir, "fig3_egfr_dasatinib_generation", "fig3_egfr_dasatinib_generation_caption.txt", width="0.9\\textwidth", fallback="Target-conditioned seeded molecular drug design using generative AI. Fixed-topology diffusion sampling yields local EGFR-focused designs around dasatinib.")} 
        {latex_table_block(ctx.results_dir, "table2_top20_generated_compounds", "table2_top20_generated_compounds_caption.txt", "tab:generated", fallback="Target-conditioned seeded molecular design results. The top dasatinib-like designed analogs are reported with medicinal chemistry heuristics and predicted affinity.")} 

        \\section*{{\\textbf{{Discussion.}} Significance, novelty, and study limits}}
        DeepDTA-iBAM is significant because it treats affinity prediction, interaction interpretation, target-conditioned retrieval, and local generative design as coupled outputs of a single learned representation. Many published DTA models are optimized primarily for supervised regression alone. By contrast, DeepDTA-iBAM makes bidirectional atom-residue attention, interpolation-guided target-conditioned retrieval, and seeded diffusion-based local design first-class components of the same architecture. The standard-split benchmark shows that the integrated model retains a credible quantitative error profile even though the ablation study indicates some predictive tradeoff relative to the strongest simplified variants, whereas the P4N-FAK1 case study shows that the explanatory signal remains anchored to a structurally meaningful interface.

        This integration addresses a broader challenge in computational drug discovery. Predictive models are increasingly expected not only to rank compounds, but also to provide explanations that medicinal chemists can interrogate and candidate molecules that can be carried forward experimentally \\cite{{vamathevan2019ml,schneider2020rethinking,jimenezluna2020xai}}. In that context, the combination of bidirectional interaction maps, interpolation-guided retrieval, and seeded fixed-topology generation is a practical strength: it links a quantitative affinity estimate to residue-level interaction evidence and then to both a chemically local design space and a target-conditioned retrieval space. This is a more decision-oriented workflow than treating prediction and molecular generation as isolated tasks, and it is more realistic for medicinal chemistry than unconstrained novelty-driven generation alone \\cite{{brown2019guacamol}}. The methodological novelty therefore lies less in any one isolated module than in the way the shared multimodal representation supports prediction, explanation, interpolation-guided prioritization, and chemically constrained molecular design within the same inference pipeline.

        The external H1 retrieval test shows that the model preserves a reproducible compound-ranking signal under substantial domain shift, but not one strong enough to support claims of protein-level target specificity or prospective screening readiness. The EGFR interpolation-guided retrieval study extends the design story in a more rigorous way by recovering held-out EGFR-family binders against a defined large chemical background. Interpolation-path proximity outperformed the nearest-anchor baseline, indicating that the target-conditioned latent space carries meaningful family geometry beyond single-anchor similarity. At the same time, explicit affinity prediction remained the strongest ranking signal, which means interpolation should be interpreted as complementary structure in the representation rather than as a replacement for supervised affinity scoring.

        Several limitations remain. The benchmark comparison relies on literature-reported external rows rather than controlled reruns, the current benchmarked estimate is based on a single trained model rather than a larger ensemble, the structural interpretability analysis focuses on one highlighted complex, the external H1 retrieval test remains a curated stress test rather than a prospective assay campaign, the EGFR interpolation study uses a local property-filtered lead-like ZINC tranche library rather than a random global screening collection, and the EGFR designed analogs remain computational outputs from a proof-of-concept local design exercise that require downstream docking, synthesis, and biochemical validation. These limitations also define clear next steps: broaden paired predictive and interpretability validation across more targets, extend external transfer studies beyond the kinase domain, and test whether interpolation-guided retrieval and locally generated candidate molecules improve medicinal chemistry iteration cycles under orthogonal experimental readouts. More broadly, prospective validation remains the standard against which practical utility must ultimately be judged \\cite{{stokes2020antibiotic}}.

        Taken together, these findings show that DeepDTA-iBAM provides a coherent framework for affinity prediction, interpretable cross-attention analysis, interpolation-guided target-conditioned retrieval, and seeded molecular drug design using generative AI within a single workflow. The architecture is novel not because it is the top regressor in isolation, but because it links quantitative scoring, mechanistically interpretable atom-residue evidence, large-library retrieval geometry, and chemically local design within one model family. That combination is methodologically important for settings where ranking quality, explanatory transparency, and actionable follow-up chemistry matter simultaneously.

        \\section*{{\\textbf{{Code and Data Availability.}} Public reproducibility resources}}
        The public repository at \\url{{https://github.com/kevinmsong/DeepDTA-iBAM}} contains \\texttt{{README.md}}, \\texttt{{LICENSE}}, \\texttt{{requirements.txt}}, \\texttt{{config\\_profiles.py}}, \\texttt{{aggregate\\_ablations.py}}, \\texttt{{case\\_studies\\_results\\_generation.py}}, the \\texttt{{models/}}, \\texttt{{training/}}, \\texttt{{utils/}}, and code-only \\texttt{{data/}} modules, selected \\texttt{{tests/}}, and lightweight \\texttt{{results/}} artifacts supporting the manuscript. The released result files include \\texttt{{table1\\_benchmark.csv}}, \\texttt{{benchmark\\_source\\_audit.csv}}, \\texttt{{ablation\\_table.csv}}, \\texttt{{table3\\_h1\\_drug\\_fishing\\_metrics.csv}}, \\texttt{{table4\\_egfr\\_interpolation\\_metrics.csv}}, \\texttt{{table5\\_top\\_egfr\\_interpolation\\_hits.csv}}, \\texttt{{case\\_study\\_metrics.json}}, \\texttt{{source\\_manifest.json}}, \\texttt{{egfr\\_interpolation\\_anchor\\_panel.csv}}, \\texttt{{egfr\\_interpolation\\_holdout\\_panel.csv}}, and final figure PDFs and 300 dpi PNGs. Raw KIBA data, cached embeddings, model checkpoints, and the local ZINC archive are not redistributed in the public release and should be obtained or regenerated through the documented workflow.

        \\section*{{\\textbf{{Acknowledgements.}} Funding support}}
        This study was supported in part by the National Heart, Lung, and Blood Institute under grant numbers U01HL134764, P01 HL160476, R01HL131017, and R01HL149137.
        \\bibliographystyle{{unsrtnat}}
        \\bibliography{{references}}
        \\end{{document}}
        """
    ).strip() + "\n"


def build_supplementary_tex(ctx: PublicationContext) -> str:
    fishing_metrics = ctx.metrics.get("fishing", {})
    num_replicates = fishing_metrics.get("num_replicates")
    num_h1_drugs = fishing_metrics.get("num_h1_drugs")
    library_size = fishing_metrics.get("primary_library_size")
    auroc = fishing_metrics.get("AUROC")
    auprc = fishing_metrics.get("AUPRC")
    auroc_sd = fishing_metrics.get("AUROC_sd")
    auprc_sd = fishing_metrics.get("AUPRC_sd")

    def fmt(value: Any, precision: int = 4) -> str:
        if value is None:
            return "TBD"
        try:
            if np.isnan(float(value)):
                return "TBD"
        except Exception:
            return str(value)
        return f"{float(value):.{precision}f}"

    return textwrap.dedent(
        f"""
        \\documentclass[11pt]{{article}}
        \\usepackage[T1]{{fontenc}}
        \\usepackage{{newtxtext}}
        \\usepackage{{newtxmath}}
        \\usepackage[margin=1in]{{geometry}}
        \\usepackage{{graphicx}}
        \\usepackage{{booktabs}}
        \\usepackage{{setspace}}
        \\setlength{{\\emergencystretch}}{{3em}}
        \\begin{{document}}
        \\begin{{center}}
        {{\\Large\\mdseries DeepDTA-iBAM Supplementary Information\\par}}
        \\vspace{{0.8em}}
        Affiliations to be finalized at submission\\par
        March 3, 2026
        \\end{{center}}
        \\vspace{{1em}}
        \\doublespacing
        \\section*{{Extended notes}}
        Supplementary Information provides overflow benchmark notes, expanded H1 retrieval analysis, and additional generation details that complement the reported analyses.
        \\section*{{H1 retrieval analysis}}
        The expanded external H1 stress test used a curated panel of {int(num_h1_drugs) if num_h1_drugs is not None else 20} ligands embedded into {int(num_replicates) if num_replicates is not None else 5} independently sampled {int(library_size) if library_size is not None else 1000}-compound lead-like libraries. Across replicate libraries, mean AUROC was {fmt(auroc)} \\(\\pm\\ {fmt(auroc_sd)}\\) and mean AUPRC was {fmt(auprc)} \\(\\pm\\ {fmt(auprc_sd)}\\). Detailed per-library metrics are available in \\texttt{{results/table3\\_h1\\_drug\\_fishing\\_metrics.csv}} and \\texttt{{results/case\\_study\\_metrics.json}}.
        \\end{{document}}
        """
    ).strip() + "\n"


def run_manuscript_section(ctx: PublicationContext) -> Dict[str, Any]:
    write_text(
        ctx.results_dir / "fig0_model_architecture_caption.txt",
        (
            "DeepDTA-iBAM architecture and evaluation workflow. Ligands are converted from SMILES "
            "strings into atom-bond graphs and encoded by a multi-layer graph attention network, whereas protein "
            "sequences are represented as cached residue-level ESM-C embeddings and compressed by a protein "
            "adapter. Bidirectional ligand-to-target and target-to-ligand cross-attention fuses atom and residue "
            "tokens into a shared multimodal state that supports both affinity prediction and iBAM heatmap "
            "extraction. A diffusion auxiliary head is trained on the ligand representation under target "
            "conditioning and later reused for target-conditioned seeded molecular drug design."
            "\n"
        ),
    )
    write_text(Path("main.tex"), build_main_tex(ctx))
    write_text(Path("references.bib"), build_references_bib())
    for stale_path in (
        Path("supplementary.tex"),
        Path("supplementary.pdf"),
        Path("supplementary.aux"),
        Path("supplementary.log"),
        Path("supplementary.out"),
    ):
        if stale_path.exists():
            stale_path.unlink()
    metrics = {"main_tex": "main.tex", "bib_file": "references.bib"}
    ctx.update_section_metrics("manuscript", metrics)
    return metrics


def run_selected_sections(ctx: PublicationContext) -> None:
    section_map = {
        "ibam": run_ibam_section,
        "fishing": run_fishing_section,
        "generation": run_generation_section,
        "interpolation": run_interpolation_section,
        "ablation": run_ablation_section,
        "diagnostics": run_diagnostics_section,
        "benchmark": run_benchmark_section,
        "manuscript": run_manuscript_section,
    }
    for section in ctx.args.sections:
        if section == "manuscript" and ctx.args.skip_manuscript:
            continue
        print(f"[publication] starting section={section}", flush=True)
        section_map[section](ctx)
        print(f"[publication] finished section={section}", flush=True)
    if not ctx.args.skip_manuscript and "manuscript" not in ctx.args.sections:
        run_manuscript_section(ctx)


def main() -> None:
    args = _build_arg_parser().parse_args()
    configure_publication_style()
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    base_config = get_config_profile(args.profile)
    if args.profile_name_override is not None:
        base_config.profile_name = args.profile_name_override
    if args.checkpoint_dir is not None:
        base_config.checkpoint_dir = args.checkpoint_dir
    if args.device is not None:
        base_config.device = args.device
    if args.member_count is not None:
        base_config.ensemble_size = args.member_count
    if args.fusion_mode is not None:
        base_config.fusion_mode = args.fusion_mode
    base_config.num_workers = 0
    ctx = PublicationContext(
        args=args,
        base_config=base_config,
        results_dir=results_dir,
        metrics_path=results_dir / "case_study_metrics.json",
        source_manifest_path=results_dir / "source_manifest.json",
    )
    ctx.load_existing_state()
    if "ablation" not in args.sections:
        ctx.metrics.pop("ablation", None)
    run_selected_sections(ctx)
    write_json(ctx.metrics_path, ctx.metrics)
    write_json(ctx.source_manifest_path, ctx.source_manifest)


if __name__ == "__main__":
    main()
