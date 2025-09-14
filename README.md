# Świadectwa

This repository stores data and scripts used for the article ["Świadectwa ewangelikalne i katolickie w perspektywie korpusowej analizy dyskursu"](https://socjolingwistyka.ijp.pan.pl/index.php/SOCJO/article/view/180) published in *Socjolingwistyka*, Tom 34 (2020).

**Citation**

SOCJOLINGWISTYKA, Tom 34 (2020): Socjolingwistyka

Albert Leśniak  
Instytut Języka Polskiego PAN, Kraków  
https://orcid.org/0000-0002-7141-576X

Zbigniew Pasek  
AGH w Krakowie  
https://orcid.org/0000-0003-2580-4366

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
