import argparse
import base64
import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIGURE_DIR = PROJECT_ROOT / "report_assets" / "figures"
QUALITY_RESULTS_DIR = PROJECT_ROOT / "evaluation" / "results"
VIEWPORT = {"width": 1600, "height": 1000}


SCREENSHOTS = [
    {
        "page": "documents",
        "nav_label": None,
        "filename": "figure_2_documents_page.png",
        "ready_text": "remote_work_policy.md",
    },
    {
        "page": "chat",
        "nav_label": "Sohbet",
        "filename": "figure_3_chat_citations.png",
        "ready_text": "Kaynak 1:",
    },
    {
        "page": "profile",
        "nav_label": "Profil",
        "filename": "figure_4_profile_page.png",
        "ready_text": "Demo User",
    },
]


def main():
    parser = argparse.ArgumentParser(
        description="Generate report figures and Streamlit screenshots."
    )
    parser.add_argument(
        "--output-dir",
        default=FIGURE_DIR,
        type=Path,
        help="Directory where PNG figures are written.",
    )
    parser.add_argument(
        "--streamlit-url",
        help="Use an already running Streamlit app instead of starting one.",
    )
    parser.add_argument(
        "--skip-screenshots",
        action="store_true",
        help="Generate only Matplotlib figures.",
    )
    args = parser.parse_args()

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    generated = []
    generated.append(generate_retrieval_workflow(output_dir))
    chunking_figure = generate_chunking_comparison(output_dir)
    if chunking_figure:
        generated.append(chunking_figure)

    if not args.skip_screenshots:
        generated.extend(capture_streamlit_screenshots(output_dir, args.streamlit_url))

    validate_pngs(generated)
    print("Generated figures:")
    for path in generated:
        width, height = Image.open(path).size
        print(f"- {path.relative_to(PROJECT_ROOT)} ({width}x{height})")


def generate_retrieval_workflow(output_dir):
    labels = [
        "User Question",
        "Query Embedding",
        "ChromaDB Candidate Search\nexpanded candidate set: max(top_k*4, 12)",
        "Lexical Reranking\ntoken and phrase-overlap score",
        "Source Diversification\nprefer distinct source files first",
        "Top Five Retrieved Chunks",
        "Grounded Prompt\n[Kaynak n] source blocks",
    ]
    output_path = output_dir / "figure_5_retrieval_workflow.png"

    fig, ax = plt.subplots(figsize=(8, 10), dpi=300)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ax.axis("off")

    x = 0.5
    y_positions = [0.90, 0.78, 0.65, 0.51, 0.38, 0.25, 0.12]

    for index, (label, y) in enumerate(zip(labels, y_positions)):
        face = "#eaf3f8" if index not in {0, 6} else "#f4f6f8"
        ax.text(
            x,
            y,
            label,
            ha="center",
            va="center",
            fontsize=10,
            color="black",
            bbox={
                "boxstyle": "round,pad=0.42,rounding_size=0.08",
                "facecolor": face,
                "edgecolor": "#7d8a93",
                "linewidth": 1.2,
            },
        )

    for start_y, end_y in zip(y_positions, y_positions[1:]):
        ax.annotate(
            "",
            xy=(x, end_y + 0.045),
            xytext=(x, start_y - 0.045),
            arrowprops={
                "arrowstyle": "-|>",
                "color": "#444444",
                "linewidth": 1.4,
                "shrinkA": 4,
                "shrinkB": 4,
            },
        )

    ax.text(
        0.50,
        0.025,
        "Implemented in src/rag_pipeline.py: search(), rerank_search_results(), diversify_search_results(), build_rag_prompt().",
        ha="center",
        va="center",
        fontsize=9,
        color="#333333",
    )
    fig.savefig(output_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return output_path


def generate_chunking_comparison(output_dir):
    experiments, source_file = load_quality_experiments()
    if not experiments:
        print("Skipped Figure 6: no successful chunking experiments with Recall@1 and Recall@5 were found.")
        return None

    labels = [
        f"{item['chunk_size']}/{item['chunk_overlap']}"
        for item in experiments
    ]
    recall_1 = [item["reports_by_top_k"]["1"]["recall"] for item in experiments]
    recall_5 = [item["reports_by_top_k"]["5"]["recall"] for item in experiments]
    x_positions = range(len(experiments))
    output_path = output_dir / "figure_6_chunking_comparison.png"

    fig, ax = plt.subplots(figsize=(9, 5.2), dpi=300)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    width = 0.36
    ax.bar(
        [x - width / 2 for x in x_positions],
        recall_1,
        width,
        label="Recall@1",
        color="#5d7fa3",
    )
    ax.bar(
        [x + width / 2 for x in x_positions],
        recall_5,
        width,
        label="Recall@5",
        color="#9fb6c9",
    )
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Recall")
    ax.set_xlabel("Chunk size / overlap")
    ax.set_xticks(list(x_positions))
    ax.set_xticklabels(labels)
    ax.set_title("Retrieval Recall by Chunking Configuration")
    ax.grid(axis="y", color="#dddddd", linewidth=0.8)
    ax.legend(frameon=False)
    ax.text(
        0.01,
        -0.20,
        f"Source: {source_file.relative_to(PROJECT_ROOT)}",
        transform=ax.transAxes,
        fontsize=8,
        color="#333333",
    )
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return output_path


def load_quality_experiments():
    candidates = sorted(QUALITY_RESULTS_DIR.glob("quality_experiments_*.json"), reverse=True)
    for path in candidates:
        with path.open("r", encoding="utf-8") as file:
            report = json.load(file)
        experiments_by_model = {}
        for item in report.get("experiments", []):
            reports = item.get("reports_by_top_k", {})
            if item.get("status") != "ok":
                continue
            if "1" not in reports or "5" not in reports:
                continue
            if "recall" not in reports["1"] or "recall" not in reports["5"]:
                continue
            model_name = item.get("embedding_model_name", "unknown")
            experiments_by_model.setdefault(model_name, []).append(item)
        if experiments_by_model:
            _, experiments = max(
                experiments_by_model.items(),
                key=lambda entry: (
                    len(entry[1]),
                    entry[0],
                ),
            )
            experiments.sort(
                key=lambda item: (
                    item["chunk_size"],
                    item["chunk_overlap"],
                )
            )
            return experiments, path
    return [], None


def capture_streamlit_screenshots(output_dir, streamlit_url=None):
    try:
        from playwright.sync_api import sync_playwright
    except ImportError as exc:
        raise RuntimeError(
            "Playwright is required for screenshots. Install report dependencies with "
            "`python -m pip install -r requirements-report.txt`, then run "
            "`python -m playwright install chromium`."
        ) from exc

    generated = []
    if not streamlit_url:
        for item in SCREENSHOTS:
            process = None
            try:
                process, base_url = start_streamlit_process()
                generated.extend(capture_pages(output_dir, base_url, [item], sync_playwright))
            finally:
                stop_process(process)
        return generated

    return capture_pages(output_dir, streamlit_url, SCREENSHOTS, sync_playwright)


def capture_pages(output_dir, base_url, screenshots, sync_playwright):
    try:
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(
                executable_path=find_browser_executable()
            )
            generated = []
            for item in screenshots:
                page = browser.new_page(viewport=VIEWPORT, device_scale_factor=1)
                page.set_default_timeout(120_000)
                page.goto(
                    f"{base_url}/?report_page={item['page']}",
                    wait_until="commit",
                    timeout=120_000,
                )
                wait_for_body_text(page, item["ready_text"])
                hide_streamlit_chrome(page)
                output_path = output_dir / item["filename"]
                write_cdp_screenshot(page, output_path)
                generated.append(output_path)
                page.close()
            browser.close()
            return generated
    except Exception:
        raise


def start_streamlit_process():
    port = find_free_port()
    base_url = f"http://127.0.0.1:{port}"
    env = os.environ.copy()
    env["REPORT_FIGURE_MODE"] = "1"
    env.setdefault("STREAMLIT_BROWSER_GATHER_USAGE_STATS", "false")
    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "streamlit",
            "run",
            str(PROJECT_ROOT / "app.py"),
            "--server.headless=true",
            "--server.address=127.0.0.1",
            "--server.fileWatcherType=none",
            f"--server.port={port}",
        ],
        cwd=PROJECT_ROOT,
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        text=True,
    )
    wait_for_http(base_url, process)
    return process, base_url


def stop_process(process):
    if not process:
        return
    process.terminate()
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        process.kill()


def hide_streamlit_chrome(page):
    page.add_style_tag(
        content="""
        [data-testid="stToolbar"],
        [data-testid="stDecoration"],
        [data-testid="stStatusWidget"],
        #MainMenu,
        footer {
            display: none !important;
        }
        .stApp {
            background: white;
        }
        """
    )


def write_cdp_screenshot(page, output_path):
    session = page.context.new_cdp_session(page)
    result = session.send(
        "Page.captureScreenshot",
        {
            "format": "png",
            "fromSurface": True,
            "captureBeyondViewport": False,
        },
    )
    output_path.write_bytes(base64.b64decode(result["data"]))


def wait_for_body_text(page, expected_text, timeout=120):
    started = time.perf_counter()
    last_text = ""
    while time.perf_counter() - started < timeout:
        try:
            last_text = page.locator("body").inner_text(timeout=5_000)
        except Exception:
            last_text = ""
        if expected_text in last_text:
            page.wait_for_timeout(1500)
            return
        page.wait_for_timeout(1000)

    preview = last_text.encode("utf-8", errors="ignore")[:500].decode(
        "utf-8",
        errors="ignore",
    )
    raise RuntimeError(
        f"Timed out waiting for Streamlit page text {expected_text!r}. "
        f"Last body text preview: {preview!r}"
    )


def find_free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def find_browser_executable():
    configured = os.getenv("REPORT_BROWSER_EXECUTABLE")
    if configured:
        path = Path(configured)
        if not path.exists():
            raise RuntimeError(
                f"REPORT_BROWSER_EXECUTABLE does not exist: {configured}"
            )
        return str(path)

    candidates = [
        Path(r"C:\Program Files\Google\Chrome\Application\chrome.exe"),
        Path(r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe"),
        Path(r"C:\Program Files\Microsoft\Edge\Application\msedge.exe"),
        Path(r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"),
    ]
    for path in candidates:
        if path.exists():
            return str(path)

    return None


def wait_for_http(base_url, process, timeout=90):
    import urllib.error
    import urllib.request

    started = time.perf_counter()
    last_output = []
    while time.perf_counter() - started < timeout:
        if process.poll() is not None:
            if process.stdout:
                last_output.extend(process.stdout.readlines())
            raise RuntimeError(
                "Streamlit exited before becoming ready:\n"
                + "".join(last_output[-30:])
            )
        try:
            with urllib.request.urlopen(base_url, timeout=2) as response:
                if response.status < 500:
                    return
        except (urllib.error.URLError, TimeoutError):
            time.sleep(1)
    raise RuntimeError(f"Timed out waiting for Streamlit at {base_url}.")


def validate_pngs(paths):
    for path in paths:
        if not path.exists():
            raise RuntimeError(f"Expected figure was not created: {path}")
        with Image.open(path) as image:
            width, height = image.size
        if width < 800 or height < 450:
            raise RuntimeError(f"Figure is unexpectedly small: {path} ({width}x{height})")


if __name__ == "__main__":
    main()
