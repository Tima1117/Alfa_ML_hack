"""Build a deterministic flat ZIP; never include secrets, datasets or binary packages."""
from pathlib import Path
import argparse
import hashlib
import json
import subprocess
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def build(out):
    sources = {}
    for package in ("agent", "ml_core"):
        for path in sorted((ROOT / package).glob("*.py")):
            source = path.read_text(encoding="utf-8")
            compile(source, str(path.relative_to(ROOT)), "exec")
            sources[path.relative_to(ROOT).as_posix()] = source
    for path in sorted((ROOT / "infra/runtime").glob("*.py")):
        sources[path.name] = path.read_text(encoding="utf-8")
    hashes = {name: hashlib.sha256(source.encode()).hexdigest() for name, source in sources.items()}
    bundle = ('"""Build-generated project sources; no third-party libraries or secrets."""\n'
              + "SOURCES = " + repr(sources) + "\nSOURCE_SHA256 = " + repr(hashes) + "\n")
    members = {"solution.py": (ROOT / "infra/solution.py").read_text(encoding="utf-8"), "bundle_sources.py": bundle,
               "requirements.txt": (ROOT / "Base/requirements.txt").read_text(encoding="utf-8")}
    out = Path(out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, source in sorted(members.items()):
            info = zipfile.ZipInfo(name, date_time=(2026, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, source.encode("utf-8"))
    revision = None
    try:
        top = subprocess.run(["git", "rev-parse", "--show-toplevel"], cwd=ROOT, text=True,
                             stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        if top.returncode == 0 and Path(top.stdout.strip()).resolve() == ROOT.resolve():
            head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True,
                                  stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
            if head.returncode == 0:
                revision = head.stdout.strip()
    except OSError:
        pass  # A downloaded source snapshot also builds without Git installed.
    manifest = {"base_revision": revision, "bundle_sha256": hashlib.sha256(out.read_bytes()).hexdigest(),
                "files": list(members), "source_sha256": hashes,
                "note": "Source hashes identify the included code; base_revision is optional Git HEAD and may have local changes."}
    out.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Built {out} ({out.stat().st_size} bytes)")
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("output", help="Destination ZIP")
    build(parser.parse_args().output)
