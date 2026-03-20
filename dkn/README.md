# Deep Kernel Networks (DKN)

Thesis implementation: *Deep Kernel Networks — Neural Architectures with SVM Foundations*  
Joan Acero-Pousa · Supervisor: Lluís A. Belanche-Muñoz · UPC, 2025

---

## Project structure



---

## Setup

```bash
pip install -r requirements.txt
```

---

## Usage

### Sanity check (Iris, all models, ~1 min)
```bash
python train.py --dataset iris
```

### Run a single experiment from config
```bash
python train.py --config experiments/magic.json
```

### Run all experiments skipping a model (e.g dkn_align)
```bash
python run_all.py --skip dkn_align
```

### Analyse results after experiments complete
```bash
python analyse.py
```

---

## Adding a new model

1. Create `models/mymodel.py` with a class inheriting `BaseModel`.
2. Implement `fit(X, y)`, `predict(X)`, `get_params()`.
3. Register it in `train.py`'s `_CLASSES` dict.
4. Add a sweep grid to `SWEEP_GRIDS` in `train.py`.
5. Add it to the relevant experiment JSONs in `experiments/`.

## Adding a new dataset

1. Add a loader function to `data/loaders.py` returning `(X, y)`.
2. Register it in `LOADERS` at the bottom of `loaders.py`.
3. Create `experiments/<name>.json` from any existing config as template.

---