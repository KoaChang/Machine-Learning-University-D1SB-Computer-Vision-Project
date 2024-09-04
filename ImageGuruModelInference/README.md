# ImageGuruModelInference

A package that supports model inference for the Image Guru project.
Project wiki: [ImageGuru](https://w.amazon.com/bin/view/MLSciences/Berlin/Computer_Vision/Projects/ImageGuru/)

## Installation

### In a brazil environment
1. Create a brazil workspace and check out 
    the [ImageGuruModelInference](https://code.amazon.com/packages/ImageGuruModelInference/trees/mainline) package using 
    the [BrazilCLI](https://w.amazon.com/index.php/BrazilBuildSystem/Tools/BrazilCLI).
2. Set the versionset
    ```bash
    brazil workspace use --versionset ImageGuruModelInference/dev
   ```
3. On AL2_x86_64, `libgfortran.so` does not get installed. Install it using `sudo yum install libgfortran`.
4. Run `brazil-build` and `brazil-build test` on the package
    ```bash
    brazil-build
    brazil-build test
    ```

### On EC2
1. Use the Ubuntu Deep Learning AMI on a (preferably) GPU instance.
2. Activate the conda environment `source activate pytorch_p36`. The following  environments have been tested to work: 
   [`pytorch_p36`, `pytorch_p37`, `pytorch_p38`]
3. Update `pytorch` and `torchvision` by `pip install --upgrade torch torchvision`
4. Install dependencies:
    ```bash
   pip install tqdm
    ```
5. Copy over the [ImageGuruModelInference](https://code.amazon.com/packages/ImageGuruModelInference/trees/mainline) 
   package to the EC2 instance.
6. Set `PYTHONPATH`
   ```bash
   export PYTHONPATH=/<path to>/ImageGuruModelInference/src:$PYTHONPATH
   ```

## Model files
Model files can be downloaded from this [Amazon WorkDocs folder](https://amazon.awsapps.com/workdocs/index.html#/folder/5b6131b5cb4bea895915b92a1edb1b3ee7855e0ee4f65349f242bd9817e604af).
For any chosen category, you will need a model file (.pth) and a model config file (.json).

## Compute Predictions
```bash
python3 /<path to>/ImageGuruModelInference/src/image_guru_model_inference/tools/predict.py \
          --input <path to file containing image phyisical IDs (one per line)> \
          --output <path to output file to store the predictions> \
          --config <path to model config file (.json)> \
          --image_download_dir <path to directory where the code will download images> \
          --model-path <path to the model file (.pth)> \
          --device <GPU ID> \
          --batch_size 32 \
          --num_workers 16
```
