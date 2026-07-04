import os
import re
import json

def scan_repo():
    print("[*] Starting repo-wide structural scan of proxima_ops...")
    base_dir = "proxima_ops"
    modules = {}

    for root, _, files in os.walk(base_dir):
        for file in files:
            if file.endswith(".py"):
                file_path = os.path.join(root, file)
                rel_path = os.path.relpath(file_path, base_dir).replace("\\", "/")
                mod_name = "proxima_ops." + rel_path[:-3].replace("/", ".")
                
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()
                
                # Find imports
                imports = []
                for line in content.splitlines():
                    # Match from proxima_ops.X import Y or import proxima_ops.X
                    m1 = re.search(r"from (proxima_ops|proxima_x\S*)\.(\S+) import", line)
                    m2 = re.search(r"import (proxima_ops|proxima_x\S*)\.(\S+)", line)
                    if m1:
                        imports.append(m1.group(1) + "." + m1.group(2))
                    elif m2:
                        imports.append(m2.group(1) + "." + m2.group(2))
                
                # Find classes and methods
                classes = re.findall(r"class (\w+)", content)
                methods = re.findall(r"def (\w+)", content)
                
                modules[mod_name] = {
                    "path": file_path,
                    "imports": sorted(list(set(imports))),
                    "classes": classes,
                    "methods": methods
                }

    # Generate Execution Atlas Markdown
    atlas = []
    atlas.append("# PRESENT-DAY EXECUTION ATLAS (PROXIMA COGNITIVE FLOW)\n")
    atlas.append("> **Empirical Call Dependency and Decision Gating Graph across proxima_ops**")
    atlas.append(f"> - Generated At: 2026-07-03")
    atlas.append("\n---\n")
    
    atlas.append("## 1. Execution Entry Points\n")
    atlas.append("The primary entry points of the execution loop are:")
    atlas.append("* **`run_proxima_demo.py`** (Root loop scheduler)")
    atlas.append("* **`proxima_ops.execution.wave12_executor`** (Core execution cycle driver)\n")
    
    atlas.append("## 2. Core Operational Modules and Classes\n")
    for name, data in sorted(modules.items()):
        if len(data["classes"]) > 0 or len(data["methods"]) > 0:
            clean_path = os.path.abspath(data['path']).replace("\\", "/")
            base_filename = name.split('.')[-1]
            atlas.append(f"### {name}")
            atlas.append(f"- **Path:** [{base_filename}.py](file:///{clean_path})")
            atlas.append(f"- **Classes:** {', '.join(data['classes']) if data['classes'] else 'None'}")
            atlas.append(f"- **Methods:** {', '.join(data['methods'][:10])}{'...' if len(data['methods']) > 10 else ''}")
            if data["imports"]:
                atlas.append(f"- **Dependencies:**")
                for imp in data["imports"]:
                    atlas.append(f"  - `{imp}`")
            atlas.append("")

    atlas.append("## 3. Dependency Call DAG (Mermaid)\n")
    atlas.append("```mermaid")
    atlas.append("graph TD")
    # Add nodes and edges based on import relationships
    added_edges = set()
    for name, data in modules.items():
        base_name = name.split(".")[-1]
        for imp in data["imports"]:
            imp_base = imp.split(".")[-1]
            edge = (base_name, imp_base)
            if edge not in added_edges and base_name != imp_base:
                atlas.append(f"    {base_name} --> {imp_base}")
                added_edges.add(edge)
    atlas.append("```\n")
    
    output_path = r"C:\Users\Arjun Sasi\.gemini\antigravity\brain\15f34201-5ec3-44c0-bc36-82e37095ea63\present_day_execution_atlas.md"
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(atlas))
    print(f"[+] Present-Day Execution Atlas written to {output_path}")

if __name__ == "__main__":
    scan_repo()
