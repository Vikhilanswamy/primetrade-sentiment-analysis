"""
Convert analysis.py to a Jupyter Notebook (.ipynb) using nbformat.
Parses the percent-format Jupytext convention:
  - `# %%`          → code cell
  - `# %% [markdown]` → markdown cell (with `# ` prefix stripped from content)
"""
import nbformat as nbf
import os, re

SCRIPT = os.path.join(os.path.dirname(__file__), 'analysis.py')
NOTEBOOK = os.path.join(os.path.dirname(__file__), 'analysis.ipynb')

with open(SCRIPT, 'r', encoding='utf-8') as f:
    lines = f.readlines()

nb = nbf.v4.new_notebook()
nb.metadata['kernelspec'] = {
    'display_name': 'Python 3',
    'language': 'python',
    'name': 'python3'
}

# Skip YAML frontmatter (lines 1-11)
i = 0
while i < len(lines) and not lines[i].strip().startswith('# %%'):
    i += 1

cells = []
current_type = None
current_lines = []

def flush():
    if current_type is None:
        return
    content = ''.join(current_lines).strip()
    if not content:
        return
    if current_type == 'markdown':
        # Strip leading "# " from each line
        md_lines = []
        for line in current_lines:
            stripped = line.rstrip('\r\n')
            if stripped.startswith('# '):
                md_lines.append(stripped[2:])
            elif stripped == '#':
                md_lines.append('')
            else:
                md_lines.append(stripped)
        content = '\n'.join(md_lines).strip()
        cells.append(nbf.v4.new_markdown_cell(content))
    else:
        cells.append(nbf.v4.new_code_cell(content))

while i < len(lines):
    line = lines[i]
    stripped = line.strip()
    
    if stripped == '# %% [markdown]':
        flush()
        current_type = 'markdown'
        current_lines = []
        i += 1
        continue
    elif stripped == '# %%':
        flush()
        current_type = 'code'
        current_lines = []
        i += 1
        continue
    
    current_lines.append(line)
    i += 1

flush()

nb.cells = cells
with open(NOTEBOOK, 'w', encoding='utf-8') as f:
    nbf.write(nb, f)

print(f"✅ Created {NOTEBOOK} with {len(cells)} cells")
