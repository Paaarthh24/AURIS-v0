# AURIS underwater crack dataset

Synthetic ROV-style concrete cracks with a blue-green underwater cast.

| Split | List | Contents |
| --- | --- | --- |
| train | `train.txt` | images + YOLO labels + masks |
| val | `val.txt` | same |
| test | `test.txt` | held-out evaluation |

Replace these files with real ROV captures using the same layout, or import a YOLO-seg dataset:

```bash
python scripts/prepare_dataset.py import-yolo --src path/to/yolo --out data/custom
```
