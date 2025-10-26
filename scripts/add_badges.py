#!/usr/bin/env python3
"""
Script to add platform badges (Colab, Kaggle, SageMaker Studio Lab) to Jupyter notebooks.
"""

import json
import os
import sys
from pathlib import Path


def create_badge_line(notebook_path, github_repo="vuhung16au/nlp-learning-journey", branch="main"):
    """
    Create badge markdown lines for all three platforms.
    
    Args:
        notebook_path: Path to the notebook relative to repo root
        github_repo: GitHub repository in format 'username/repo'
        branch: Branch name (default: main)
    
    Returns:
        String with all badge markdown
    """
    # Normalize path separators for URLs
    url_path = notebook_path.replace(os.sep, '/')
    
    # Create badge URLs
    colab_url = f"https://colab.research.google.com/github/{github_repo}/blob/{branch}/{url_path}"
    kaggle_url = f"https://kaggle.com/kernels/welcome?src=https://github.com/{github_repo}/blob/{branch}/{url_path}"
    sagemaker_url = f"https://studiolab.sagemaker.aws/import/github/{github_repo}/blob/{branch}/{url_path}"
    
    # Create badge lines - each badge on its own line
    badge_lines = [
        f"[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]({colab_url})",
        f"[![Open In Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)]({kaggle_url})",
        f"[![Open In SageMaker Studio Lab](https://studiolab.sagemaker.aws/studiolab.svg)]({sagemaker_url})"
    ]
    
    return badge_lines


def add_badges_to_notebook(notebook_path, github_repo="vuhung16au/nlp-learning-journey", branch="main"):
    """
    Add or update platform badges in a Jupyter notebook.
    
    Args:
        notebook_path: Path to the notebook file
        github_repo: GitHub repository in format 'username/repo'
        branch: Branch name (default: main)
    
    Returns:
        True if modified, False otherwise
    """
    # Read notebook
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    
    # Find the first markdown cell
    first_markdown_idx = None
    for i, cell in enumerate(nb['cells']):
        if cell['cell_type'] == 'markdown':
            first_markdown_idx = i
            break
    
    if first_markdown_idx is None:
        print(f"  Warning: No markdown cell found in {notebook_path}")
        return False
    
    # Get the first markdown cell
    cell = nb['cells'][first_markdown_idx]
    source = ''.join(cell['source'])
    
    # Get relative path from repo root
    repo_root = Path(__file__).parent.parent
    rel_path = os.path.relpath(notebook_path, start=repo_root)
    
    # Create badge lines
    badge_lines = create_badge_line(rel_path, github_repo, branch)
    
    # Check if badges already exist
    has_colab = 'Open In Colab' in source or 'Open in Colab' in source
    has_kaggle = 'Open In Kaggle' in source or 'Open in Kaggle' in source
    has_sagemaker = 'SageMaker' in source
    
    # Split into lines
    lines = source.split('\n')
    
    # Find the title line (first line starting with #)
    title_idx = None
    for i, line in enumerate(lines):
        if line.strip().startswith('# '):
            title_idx = i
            break
    
    if title_idx is None:
        print(f"  Warning: No title found in {notebook_path}")
        return False
    
    # Remove existing badge lines
    new_lines = []
    skip_next_empty = False
    for i, line in enumerate(lines):
        # Skip existing badge lines
        if any(badge in line for badge in ['Open In Colab', 'Open in Colab', 'Open In Kaggle', 'Open in Kaggle', 'SageMaker']):
            skip_next_empty = True
            continue
        # Skip empty lines right after badges
        if skip_next_empty and line.strip() == '':
            skip_next_empty = False
            continue
        skip_next_empty = False
        new_lines.append(line)
    
    # Insert new badges after title
    result_lines = []
    for i, line in enumerate(new_lines):
        result_lines.append(line)
        if i == title_idx:
            result_lines.append('')
            # Add each badge on its own line
            result_lines.extend(badge_lines)
    
    # Reconstruct source
    new_source = '\n'.join(result_lines)
    
    # Update cell source (as list of strings with newlines)
    # Split each line into separate array elements for better readability in JSON
    cell['source'] = []
    for i, line in enumerate(result_lines):
        if i < len(result_lines) - 1:
            cell['source'].append(line + '\n')
        else:
            cell['source'].append(line)
    
    # Write back
    with open(notebook_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
        f.write('\n')  # Add final newline
    
    # Report what was done
    if has_colab and has_kaggle and has_sagemaker:
        print(f"  ✓ Updated badges in {notebook_path}")
    elif has_colab:
        print(f"  ✓ Added Kaggle and SageMaker badges to {notebook_path}")
    else:
        print(f"  ✓ Added all badges to {notebook_path}")
    
    return True


def main():
    """Main function to process notebooks."""
    if len(sys.argv) < 2:
        print("Usage: python add_badges.py <notebook_path> [<notebook_path> ...]")
        print("   or: python add_badges.py --all")
        sys.exit(1)
    
    # Get repository root
    script_dir = Path(__file__).parent
    repo_root = script_dir.parent
    
    # Get list of notebooks to process
    notebooks = []
    if sys.argv[1] == '--all':
        # Process all notebooks in examples/ directory
        examples_dir = repo_root / 'examples'
        if examples_dir.exists():
            notebooks.extend(examples_dir.glob('*.ipynb'))
            
            # Also process spaCy-Linguistic subdirectory
            spacy_dir = examples_dir / 'spaCy-Linguistic'
            if spacy_dir.exists():
                notebooks.extend(spacy_dir.glob('*.ipynb'))
    else:
        # Process specified notebooks
        notebooks = [Path(p) for p in sys.argv[1:]]
    
    if not notebooks:
        print("No notebooks found to process")
        sys.exit(1)
    
    print(f"Processing {len(notebooks)} notebooks...")
    print()
    
    modified_count = 0
    for notebook_path in sorted(notebooks):
        if add_badges_to_notebook(str(notebook_path)):
            modified_count += 1
    
    print()
    print(f"Modified {modified_count} notebooks")


if __name__ == '__main__':
    main()
