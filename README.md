# Khmer Neural Address Parser

### Inference

```shell
python inference_onnx.py

# Input: ផ្ទះលេខ២ ផ្លូវ២៣ ភូមិប៉ុស្តិ័ចាស់ព្រះនេត្រព្រះព្រះនេត្រព្រះបន្ទាយមានជ
# {
#   "province": "ខេត្តបន្ទាយមានជ័យ",
#   "district": "ស្រុកព្រះនេត្រព្រះ",
#   "commune": "ឃុំព្រះនេត្រព្រះ",
#   "village": "ប៉ុស្ដិចាស់",
#   "house": "ផ្ទះលេខ២",
#   "road": "ផ្លូវ២៣",
# }
```


### Train

```shell
python train.py
```