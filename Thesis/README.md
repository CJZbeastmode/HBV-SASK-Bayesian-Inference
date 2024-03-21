# Thesis Template

This is an unofficial thesis template to be used for arbitrary student projects at the department of informatics at TUM. It was already used in dozens of successful theses and contains a collection of useful LaTeX tips and tricks. Be aware that style regulations might change at any point so check with the official guidelines whether this template is still compliant.

## Build the PDF

Building LaTeX projects can be painful because it needs multiple passes to resolve all references. Here are multiple options to build the PDF.

### Manually:

```bash
pdflatex NAME_TITLE.tex         # fist pass to generate aux file
bibtex NAME_TITLE.aux           # create bib info
pdflatex NAME_TITLE.tex         # resolve cross references
pdflatex NAME_TITLE.tex         # resolve citations
```

### latexmk

This tool is included in every respectable LaTeX distribution and tries to resolve all necessary steps automatically.
```bash
latexmk -pdf NAME_TITLE.tex
```

### IDE

Use an IDE like [TeXStudio](https://www.texstudio.org/) to do everything for you.
```bash
texstudio NAME_TITLE.tex        # hit F5 to build and show the PDF
```

