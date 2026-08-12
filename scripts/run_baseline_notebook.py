import sys
import json
import io
import os
import traceback
from pathlib import Path
import matplotlib
matplotlib.use('Agg')

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

notebook_path = Path(r"c:\Users\dzaki\OneDrive\Dokumen\Bahasa Pemograman\Python\Batik\training\notebooks\03_baseline.ipynb")

print(f"Loading notebook: {notebook_path.resolve()}", flush=True)

with open(notebook_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

os.chdir(notebook_path.parent)
print(f"Working directory set to: {os.getcwd()}", flush=True)

global_env = {}
exec_count = 1

for idx, cell in enumerate(nb["cells"]):
    if cell["cell_type"] != "code":
        continue
    
    code_text = "".join(cell["source"])
    print(f"\n--- Executing Code Cell [{exec_count}] (Index {idx}) ---", flush=True)
    
    # Simple StringIO capture for cell outputs
    stdout_capture = io.StringIO()
    old_stdout = sys.stdout
    
    # Custom class that writes to real stdout AND captures string without file opening loop
    class DualOutput:
        def write(self, s):
            old_stdout.write(s)
            old_stdout.flush()
            stdout_capture.write(s)
        def flush(self):
            old_stdout.flush()

    sys.stdout = DualOutput()
    
    cell_outputs = []
    success = True
    
    try:
        exec(code_text, global_env)
    except Exception as e:
        success = False
        err_msg = traceback.format_exc()
        sys.stdout = old_stdout
        print(f"❌ Error in Cell Index {idx}:\n{err_msg}", flush=True)
        
        cell_outputs.append({
            "ename": type(e).__name__,
            "evalue": str(e),
            "output_type": "error",
            "traceback": err_msg.splitlines()
        })
    finally:
        sys.stdout = old_stdout
    
    printed_text = stdout_capture.getvalue()
    if printed_text:
        cell_outputs.insert(0, {
            "name": "stdout",
            "output_type": "stream",
            "text": printed_text.splitlines(keepends=True)
        })
        
    cell["execution_count"] = exec_count
    cell["outputs"] = cell_outputs
    exec_count += 1
    
    # Save notebook state after each cell
    with open(notebook_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
        
    if not success:
        print(f"Stopping execution due to error in cell index {idx}", flush=True)
        sys.exit(1)

print(f"\n✅ Execution completed successfully. Notebook saved to {notebook_path.resolve()}", flush=True)
