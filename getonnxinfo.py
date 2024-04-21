import onnx

def print_model_io_info(onnx_file_path):
    # ONNX モデルを読み込む
    model = onnx.load(onnx_file_path)

    # モデルの入力情報を表示
    print("Model Inputs:")
    for input in model.graph.input:
        # 入力の名前と型を表示
        print(f"Name: {input.name}, Type: {input.type}")

    # モデルの出力情報を表示
    print("\nModel Outputs:")
    for output in model.graph.output:
        # 出力の名前と型を表示
        print(f"Name: {output.name}, Type: {output.type}")

# ここにONNXファイルのパスを指定します。
onnx_file_path = 'models/raft_things_iter10_240x320.onnx'

# モデルの入出力情報を表示
print_model_io_info(onnx_file_path)
