# be_predict_bystander

## Dependencies
- No strict dependencies are known, though this code was built with python 3.7 and pytorch 1.1.0.

## Installation
Clone this github repository, then set up your environment to import the `predict.py` script in however is most convenient for you. In python, for instance, you may use the following at the top of your script to import the model.

```python
import sys
sys.path.append('/directory/to/local/repo/clone/')
import predict as bystander_model
```

## Usage
```python
import predict as bystander_model
bystander_model.init_model(base_editor = 'BE4', celltype = 'mES')
```

Note: Supported cell types are `['mES', 'HEK293']` and supported base editors are `['ABE', 'ABE-CP1040', 'BE4', 'BE4-CP1028', 'AID', 'CDA', 'eA3A', 'evoAPOBEC', 'eA3A-T44DS45A', 'BE4-H47ES48A', 'eA3A-T31A', 'eA3A-T31AT44A', 'BE4-H47ES48A']`. Not all combinations of base editors and cell types are supported -- refer to `models.csv`.

If your cell type of interest is not included here, we recommend using mES. Major base editing outcomes are fairly consistent across cell-types, though rarer outcomes including cytosine transversions are known to depend on cell-type to some extent.

```python
pred_df, stats = bystander_model.predict(seq)
```

`seq` is a 50-nt string of DNA characters, spanning from positions -19 to 30 where positions 1-20 are the spacer, an NGG PAM occurs at positions 21-23, and position 0 is used to refer to the position directly upstream of position 1. 

`pred_df` is a pandas dataframe containing a row for each unique combination of base editing outcomes. The column 'Predicted frequency' sums to one.

`stats` is a dict with the following keys.
- Total predicted probability
- 50-nt target sequence
- PAM - inferred
- Protospacer sequence - inferred
- Celltype

### Example usage
```python
import inDelphi
inDelphi.init_model(celltype = 'mESC')

seq = 'TATCAGCGGGAATTCAAGCGCACCAGCCAGAGGTGTACCGTGGACGTGAG'

pred_df, stats = bystander_model.predict(seq)
```

## Additional methods
Once you have obtained `pred_df, stats`, additional methods are available for your convenience.

### Obtaining exact genotypes
```python
pred_df, stats = bystander_model.predict(seq, cutsite)
pred_df = bystander_model.add_genotype_column(pred_df, stats)
```

A new column `Genotype` will be created.

## Contact
maxwshen at mit.edu

### License
tba
