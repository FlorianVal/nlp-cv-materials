# NLP Course Materials

This repository contains materials for a Natural Language Processing (NLP) course, including presentation slides created with LaTeX Beamer.

## Repository Structure

- `nlp_course_slides.tex`: The main LaTeX Beamer presentation file.
- `images/`: Directory containing all images used in the presentations.
  - `generated/`: Subfolder for dynamically generated illustrations (not tracked by Git).
- `generate_illustrations.py`: Python script to generate illustrations for the slides.

## Getting Started

### Prerequisites

To compile the LaTeX presentations, you'll need:
- A LaTeX distribution (e.g., TeX Live, MiKTeX)
- The Beamer package
- Python packages for running the illustration generator

### Installing Python Dependencies

```bash
pip install matplotlib numpy seaborn wordcloud scikit-learn
```

### Generating Illustrations

Run the Python script to generate illustrations for the slides:

```bash
python generate_illustrations.py
```

This will create visualization images in the `images/generated/` folder.

### Compiling the Presentation

To compile the LaTeX presentation into a PDF:

```bash
pdflatex -shell-escape nlp_course_slides.tex
```

Note: The `-shell-escape` flag is needed for the `minted` package which is used for code syntax highlighting.

## Customizing the Slides

- Change the title, author, and institution in the `nlp_course_slides.tex` file.
- Add or modify slides as needed for your specific course content.
- To use additional images, place them in the `images/` directory and reference them in your LaTeX code.

## License

[Add your chosen license here]

## Acknowledgments

- References to any third-party materials or inspirations used in creating these course materials. 