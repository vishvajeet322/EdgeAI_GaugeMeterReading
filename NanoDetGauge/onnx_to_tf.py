


# import onnx
# from onnx_tf.backend import prepare

# # Load ONNX model
# onnx_model = onnx.load("yolov5/runs/train/gauge_reader_yolov5n2/weights/best.onnx")

# # Convert ONNX to TensorFlow
# tf_rep = prepare(onnx_model)
# tf_rep.export_graph("yolov5/yolov5_tf_model")



import onnx
from onnx_tf.backend import prepare

onnx_model = onnx.load("nanodet/workspace/gauge_nanodet_m_0.5x/gauge_nanodet.onnx")
tf_rep = prepare(onnx_model)
tf_rep.export_graph("tf_models/nanodet_tf_model")


