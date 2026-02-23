# How to Compile the LaTeX Report

## Option 1: Online (Easiest)

1. Go to [Overleaf](https://www.overleaf.com/)
2. Create a free account
3. Click "New Project" → "Upload Project"
4. Upload `PROJECT_REPORT.tex`
5. Click "Recompile" (or it will auto-compile)
6. Download the PDF

## Option 2: Install LaTeX Locally (Windows)

### Install MiKTeX (Recommended for Windows)

1. Download MiKTeX from: https://miktex.org/download
2. Run the installer (choose "Install missing packages on-the-fly: Yes")
3. After installation, open PowerShell in this directory
4. Run:
   ```powershell
   pdflatex PROJECT_REPORT.tex
   pdflatex PROJECT_REPORT.tex  # Run twice for TOC and references
   ```

### Alternative: Install TeX Live

1. Download TeX Live from: https://www.tug.org/texlive/
2. Run the installer (full installation ~7GB)
3. Compile as above

## Option 3: Use Docker (Cross-platform)

```bash
docker run --rm -v ${PWD}:/workdir texlive/texlive pdflatex PROJECT_REPORT.tex
docker run --rm -v ${PWD}:/workdir texlive/texlive pdflatex PROJECT_REPORT.tex
```

## Required LaTeX Packages

The document uses these packages (auto-installed by MiKTeX):
- inputenc, geometry, amsmath, amssymb
- graphicx, hyperref, booktabs, float
- listings, xcolor, caption, subcaption
- algorithm, algorithmic, enumitem

## Output

After successful compilation, you'll get:
- `PROJECT_REPORT.pdf` - The final report (30+ pages)
- `PROJECT_REPORT.aux`, `.log`, `.toc` - Auxiliary files (can delete)

## Quick Preview Without Compilation

You can view the LaTeX source in VS Code with extensions:
1. Install "LaTeX Workshop" extension
2. Open `PROJECT_REPORT.tex`
3. Press Ctrl+Alt+V for preview (if LaTeX installed)

## Troubleshooting

**Error: "pdflatex not found"**
- LaTeX is not installed or not in PATH
- Use Option 1 (Overleaf) for quickest solution

**Error: Missing packages**
- MiKTeX will auto-download them
- If it doesn't, run: `mpm --install=<package-name>`

**Error: Multiple runs needed**
- Table of contents and references require 2-3 compilation passes
- Just run `pdflatex` 2-3 times

## Report Contents

The 30+ page report includes:

1. **Abstract & Introduction** (2 pages)
2. **Methodology** - Architecture, Dataset, Training (8 pages)
3. **Results** - Performance tables, ablation studies (6 pages)
4. **Critical Issues & Resolutions** - Bug fixes (4 pages)
5. **Bottlenecks & Limitations** (3 pages)
6. **Recommendations & Future Work** (5 pages)
7. **Sanity Tests & QA** (2 pages)
8. **Conclusions** (3 pages)
9. **Appendix** - Full hyperparameter tables (2 pages)

Total: **~35 pages** of comprehensive analysis
