# Świadectwa

This repository stores data and scripts used for the article ["Języki religijne – potrzeby dwojakie"](https://socjolingwistyka.ijppan.pl/index.php/SOCJO/article/view/180).

## Structure
- `data/` – CSV/XLSX files and auxiliary resources.
- `data/corpus/` – place text corpora used in the analysis (not included).
- `scripts/` – Python scripts. The main one, `artykul_calcs.py`, was converted from the original notebook.
- `notebooks/` – original Jupyter notebook.

## Requirements
```
pip install -r requirements.txt
```

## Usage
Ensure that `data/stopwords.txt` contains one stopword per line and that the required corpus files are located in `data/corpus/`. Then run:
```
python scripts/artykul_calcs.py
```

The script writes its outputs back into the `data/` directory.
