# Generate oriented box for scannet.
The box3d annotation can be directly downloaded [here](https://drive.google.com/file/d/1lGNiUMcCe3fFOS7D3Zla_LwYJuQ3Sj96/view?usp=sharing)

If you are interested in how to generate the oriented box3d annotations, keep reading:  

Run the data preparation script which parses the raw data format into the processed pickle format.
This script also generates the ground truth oriented 3d boxes.

<details>
  <summary>[Data preparation script]</summary>

## Modify the path config
Change the path config in `parse_scan2cad.py` and `generate_scannet_anno_snippet.py`.

## Parse scan2cad file and get scene-level 3D box annotation
```bash
python scripts/scannet_preprocessing/parse_scan2cad.py
```
## Form snippets and generate snippet-level 3D box annotation
```bash
python scripts/scannet_preprocessing/generate_scannet_anno_snippet.py
```
</details>
