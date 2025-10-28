from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path


TEMPLATE_YAML = """INHERIT: {inherit_config}
site_dir: {output_dir}
hooks:
  - .rag_export/rag_indexing_hook.py
plugins:
  - search: false
"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Run MkDocs export for RAG indexing")
    parser.add_argument("--project-dir", required=True)
    parser.add_argument("--inherit-config", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    project_dir = Path(args.project_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    export_dir = project_dir / ".rag_export"
    export_dir.mkdir(parents=True, exist_ok=True)

    # Copy hook into project
    src_hook = Path(__file__).resolve().parents[1] / "rag_indexing_hook.py"
    dst_hook = export_dir / "rag_indexing_hook.py"
    shutil.copyfile(src_hook, dst_hook)

    # Write temporary yaml
    yaml_content = TEMPLATE_YAML.format(inherit_config=args.inherit_config, output_dir=str(output_dir))
    (export_dir / "mkdocs_for_rag_indexing.yml").write_text(yaml_content, encoding="utf-8")

    # Execute mkdocs build
    subprocess.check_call(
        [
            "mkdocs",
            "build",
            "-f",
            str(export_dir / "mkdocs_for_rag_indexing.yml"),
        ],
        cwd=str(project_dir),
    )

    print(f"Export complete → {output_dir}")


if __name__ == "__main__":
    main()


