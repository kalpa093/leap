# DA Experiment

This is a page for DA experiment on NLP datasets.
Every results are averaged over five runs for each datasets (SST-2, SST-5, and IMDB).
DAs with highest F1-score within each category are experimented in our research.
The augmentations that are not categorized could be explored in future work.

## Word Editing

| DA Method | Accuracy | Precision | Recall | F1-score |
|-----------|----------|-----------|--------|----------| 
| EDA       | 91.87    | 91.10     | 93.64  | 92.30    |
| AEDA      | 95.53    | 95.61     | 95.70  | 95.64    |
| AMR-DA    | 94.31    | 95.36     | 94.76  | 95.12    |
| TAA       | 94.84    | 94.69     | 94.96  | 94.78    |

## Synonym Replacement

| DA Method | Accuracy | Precision | Recall | F1-score |
|-----------|----------|-----------|--------|----------| 
| CA        | 94.25    | 96.07     | 94.59  | 95.32    |
| C-BERT    | 94.38    | 95.97     | 95.31  | 95.28    |
| Soft CDA  | 94.32    | 94.75     | 95.19  | 94.99    |
| C-ConvNet | 94.69    | 94.17     | 95.02  | 94.76    |
| Snippext  | 93.27    | 92.19     | 95.58  | 94.86    |
| TDA       | 95.21    | 94.39     | 95.73  | 95.18    |
| SSMBA     | 95.58    | 95.69     | 95.77  | 95.68    |

## Paraphrasing

| DA Method           | Accuracy | Precision | Recall | F1-score |
|---------------------|----------|-----------|--------|----------| 
| STG                 | 96.74    | 96.99     | 95.61  | 96.30    |
| UDA                 | 95.18    | 95.24     | 95.07  | 95.11    |
| Dial-Aug            | 95.23    | 95.21     | 94.79  | 95.05    |
| Noisy Self-training | 95.86    | 94.23     | 96.18  | 95.63    |
| C2L                 | 96.57    | 96.18     | 96.64  | 96.37    |

## Interpolation

| DA Method     | Accuracy | Precision | Recall | F1-score |
|---------------|----------|-----------|--------|----------| 
| TMix          | 95.94    | 94.38     | 96.12  | 95.68    |
| SeqMix        | 95.37    | 95.74     | 95.27  | 95.46    |
| AdvAug        | 95.77    | 95.38     | 95.79  | 95.63    |
| LADA          | 96.78    | 96.28     | 96.94  | 96.75    |
| LADAM         | 96.90    | 97.01     | 96.59  | 96.79    |
| TextSmoothing | 96.82    | 96.97     | 96.98  | 97.01    |
