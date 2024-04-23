### Conversion tool: YOLO to PASCAL VOC or PASCAL VOC to YOLO

### Usage

```
git clone https://github.com/jabborov/yolo2voc.git
cd yolo2voc
pip install -r requirements.txt
```

Edit variables in `config.py`  to align with your custom dataset.

**Convert YOLO to VOC**
```
python main.py --yolo2voc
```

**Convert VOC to YOLO**
```
python main.py --voc2yolo 
```


`<object-class> <x_center> <y_center> <width> <height>`


**where:**
- `<object-class>` - integer object number from `0` to `(classes amount -1)`
- `<x_center> <y_center> <width> <height>` - `float` values relative to width and height of image, the range equals to `(0.0 to 1.0]`
- For example: `<x> = <absolute_x> / <image_width>` or `<height> = <absolute_height> / <image_height>`
- Attention: `<x_center> <y_center>` - are center of rectangle (are not top-left corner)

**Convert COCO to YOLO**     
[Click here](https://github.com/jabborov/coco2yolo.git)

### References
1. https://github.com/yakhyo/yolo2voc/tree/main
2. https://github.com/jahongir7174/YOLO2VOC
3. https://github.com/AlexeyAB/darknet
