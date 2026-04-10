# TagFex Standalone for ADS-B

This project now supports two input modes.

## 1. Raw IQ mode

Use `dataset = adsb_iq` and `convnet_type = resnet1d18`.
Each sample is a `.npy` or `.npz` IQ array with shape `[2, L]`, `[L, 2]`, or flattened `[2*L]`.
Metadata fields:

- `signal_path`
- `label`
- `split`

Run with:

```bash
python main.py --config=./exps/tagfex_adsb.json
```

## 2. 3x32x32 image mode

Use `dataset = adsb_image` and `convnet_type = resnet18`.
This mode is intended for ADS-B SEI datasets already converted to image-like tensors such as `3x32x32` features.
You can load data in either of these formats:

### Split-array mode
Provide four files in the config:

- `train_data_file`
- `train_label_file`
- `test_data_file`
- `test_label_file`

Accepted shapes for data arrays:

- `[N, 3, 32, 32]`
- `[N, 32, 32, 3]`

Run with:

```bash
python main.py --config=./exps/tagfex_adsb_image.json
```

### Metadata mode
You may also provide `metadata_file` and per-sample `.npy/.npz` image arrays.
Metadata fields:

- `signal_path`
- `label`
- `split`

## Notes

- Both modes keep the TagFex training loop and return train samples as `idx, view1, view2, label`.
- The image mode uses lightweight NumPy-based augmentation and does not depend on `torchvision` transforms.
