import io
import zipfile
import requests

from pathlib import Path

from pipeline.lib.sanitize_dataset import sanitize
from pipeline.lib.build_abalation  import build_clean_variants
from pipeline.lib.report_entry_count import report

sources = ["gpt2-output", "open-gpt-text", "open-llama-text", "open-palm-text", "open-web-text"]
from_subsets = ["test-dirty.jsonl", "train-dirty.jsonl", "valid-dirty.jsonl"]
to_subsets   = ["test.jsonl", "train.jsonl", "valid.jsonl"]

def downloadAndExtractTo(url: str, to: Path):
    print(f"Downloading: {url} => {to}")
    file = zipfile.ZipFile(io.BytesIO(requests.get(url, stream=True).content))
    file.extractall(to)

def downloadAndSaveTo(url, filename):
    print(f"Downloading: {url} => {filename}")
    with requests.get(url, stream=True) as r:
        if r.status_code == 200:
            r.raise_for_status()
            with open(filename, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192): f.write(chunk)
        else:
            print(f"Failed to download file, status code: {r.status_code}")

if __name__ == "__main__":
    from_files = [Path(source, from_subset) 
                  for from_subset in from_subsets
                  for source in sources]
    to_files   = [Path(source, to_subset) 
                  for to_subset in to_subsets
                  for source in sources]
    
    print("Download Checkpoints (models)")
    downloadAndSaveTo("https://github.com/MarkChenYutian/T5-Sentinel-public/releases/download/InitialCommit/solaiman-detector-base.pt", Path("data", "checkpoint", "solaiman-detector-base.pt"))
    downloadAndSaveTo("https://github.com/MarkChenYutian/T5-Sentinel-public/releases/download/InitialCommit/T5Hidden.0622.pt", Path("data", "checkpoint", "T5Hidden.0622.pt"))
    downloadAndSaveTo("https://github.com/MarkChenYutian/T5-Sentinel-public/releases/download/InitialCommit/T5Sentinel.0613.pt", Path("data", "checkpoint", "T5Sentinel.0613.pt"))
    
    print("Download Dataset")
    downloadAndExtractTo("https://zenodo.org/records/8285326/files/GPT2.zip?download=1", Path("data", "split", "gpt2-output"))
    downloadAndExtractTo("https://zenodo.org/records/8285326/files/ChatGPT.zip?download=1", Path("data", "split", "open-gpt-text"))
    downloadAndExtractTo("https://zenodo.org/records/8285326/files/LLaMA.zip?download=1", Path("data", "split", "open-llama-text"))
    downloadAndExtractTo("https://zenodo.org/records/8285326/files/PaLM.zip?download=1", Path("data", "split", "open-palm-text"))
    downloadAndExtractTo("https://zenodo.org/records/8285326/files/Human.zip?download=1", Path("data", "split", "open-web-text"))
    downloadAndExtractTo("https://zenodo.org/records/8285326/files/ZeroGPT-baseline-response.zip?download=1", Path("data", "baselines", "zerogpt_classifier_output"))
    downloadAndExtractTo("https://zenodo.org/records/8285326/files/OpenAI-baseline-response.zip?download=1", Path("data", "baselines", "openai_classifier_output"))
    
    # Report
    print("Download Finished!\n\nDataset Statistics:\n")
    for source in sources:
        for subset in from_subsets:
            report(source, subset)
    print("\n")
    
    # Build cleaned up dataset version
    sanitize(from_files, to_files)

    # Build clean variants for the large ablation table
    build_clean_variants(Path("data", "split", "open-palm-text"))
    build_clean_variants(Path("data", "split", "open-web-text"))
    build_clean_variants(Path("data", "split", "open-gpt-text"))
    build_clean_variants(Path("data", "split", "gpt2-output"))
    build_clean_variants(Path("data", "split", "open-llama-text"))
