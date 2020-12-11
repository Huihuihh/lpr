## 依赖
- `python3.7.6`
- `openvino_2020.1.023`
- `opencv-python-3.4.2.17`
- `NCS2`
- `Ubuntu18.04`

## 生成openvino需要的模型文件

- 将得到的pb文件进行转换

```
python3 model_optimizer/mo_tf.py \
--input_model model/lpr.pb \
--input_shape [1,24,94,3] \
--output_dir lrmodels/
```

## 使用openvino进行推理（VPU）

`python3 detect.py --model lrmodels/lpr.xml --device MYRIAD`
