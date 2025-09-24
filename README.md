# CAI-VISION
### Very basic vision model to identify foods around the world including a few turkish foods.

## How to Use
Prep your env and then go inside of `./scripts` directory.

**You have to be in `./scripts` directory to begin.**

#### 1. Data Prep
Place your images in `./datasets/cai-vision-dataset/new-data/` as `new-data/<class_name>/*.jpg`

Use `./data-prep/dataset_schema_tool.py` to prepare your image dataset. **You'll find instructions in the script.**

Prepare your labels txt by using `write_labels.py`

#### 2. Training
If you have CUDA, you can use `torch_check.py` to see if it is working properly.

Proceed to training with running `train_torch_lite0.py`. It will load your dataset, check for consistency and fetch the latest lite0 model to train upon. Tweak the config section for your liking.

Results will be in `torch_runs` folder as pt files.

#### 3. Evaluate
Export your torch model with `export_torchscript_int8.py`. Result model will be in `torch_runs/outputs/ts_*`.

Evaluate resulting model using `eval_from_csv.py` **You'll find instructions in the script.**

You can also check the inference with `test_torch_inference.py` 

#### 3. Export for Mobile
Final **pt** models can be used to create onnx models suitable for mobile phones.

Use `export_onnx.py`. Results will be in `torch_runs/outputs/onnx_*`.

Finally, validate exported model with `validate_onnx.py` before mobile phone usage.


