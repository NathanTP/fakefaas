import pathlib
import sys

"""This uses the bert_squad model for question/answer over plain text. Input is
   a json file with the text to parse and a set of questions to ask. The model
   predicts the whole batch of questions at the same time.

   taken from: https://github.com/onnx/models/tree/master/text/machine_comprehension/bert-squad
"""
modulePath = pathlib.Path(__file__).parent.resolve()

max_seq_length = 256
doc_stride = 128
max_query_length = 64
batch_size = 1
n_best_size = 20
max_answer_length = 30

# Python modules make no sense. I'm sure this is breaking something but
# whatever. All I want is to be able to import files in the same directory as
# this file.
sys.path.append(str(pathlib.Path(__file__).resolve().parent))

import libff as ff

class GlobalImport:
    """Allows you to import modules in a function and have them put in the global context. Good for conditional imports. We use it to abstract imports into their own function for profiling purposes.
    Taken from: Rafal Grabie https://stackoverflow.com/questions/11990556/how-to-make-global-imports-from-a-function"""

    def __enter__(self):
        return self

    def __call__(self):
        import inspect
        self.collector = inspect.getargvalues(inspect.getouterframes(inspect.currentframe())[1].frame).locals

    def __exit__(self, *args):
        globals().update(self.collector)


class Model:
    @staticmethod
    def imports():
        with GlobalImport() as gi:
            import onnxruntime, onnx, json
            import tokenization
            import run_onnx_squad as rs
            import numpy as np
            gi()


    def __init__(self, provider="CUDAExecutionProvider", profTimes=None):
        with ff.timer("imports", profTimes):
            Model.imports()

        self.tokenizer = tokenization.FullTokenizer(vocab_file=str(modulePath / 'uncased' / 'vocab.txt'), do_lower_case=True)

        opts = onnxruntime.SessionOptions()
        opts.optimized_model_filepath = str(modulePath / "optModel.onnx")
        # opts.enable_profiling = True

        modelPath = modulePath / 'bertsquad-10.onnx'

        with ff.timer("onnxruntime_session_init", profTimes):
            self.session = onnxruntime.InferenceSession(
                    str(modelPath),
                    sess_options = opts,
                    providers=[provider])


    def pre(self, raw):
        parsed = rs.read_squad_examples(input_str=raw)
        input_ids, input_mask, segment_ids, extra_data = rs.convert_examples_to_features(
                parsed, self.tokenizer, max_seq_length, doc_stride, max_query_length)

        n = len(input_ids)
        bs = batch_size
        formated_items = []
        for idx in range(0, n):
            item = parsed[idx]

            formated_items.append({"unique_ids_raw_output___9:0": np.array([item.qas_id], dtype=np.int64),
                    "input_ids:0": input_ids[idx:idx+bs],
                    "input_mask:0": input_mask[idx:idx+bs],
                    "segment_ids:0": segment_ids[idx:idx+bs]})

        return {
                "raw_inputs" : parsed, # Raw unfeaturized examples, needed by post-processor
                "inputs" : formated_items, # featurized and formatted inputs to the model
                "feature_extra" : extra_data # Needed by post-processing to interpret model results
            }


    def run(self, data):
        results = []
        for item in data['inputs']:
            results.append(self.session.run(["unique_ids:0","unstack:0", "unstack:1"], item))

        # Post processing needs everything from pre as well as the model outputs
        data['results'] = results

        return data

    def post(self, data):
        all_results = []
        for onnxRes in data['results']:
            in_batch = onnxRes[1].shape[0]
            start_logits = [float(x) for x in onnxRes[1][0].flat]
            end_logits = [float(x) for x in onnxRes[2][0].flat]
            for i in range(0, in_batch):
                unique_id = len(all_results)
                all_results.append(rs.RawResult(unique_id=unique_id, start_logits=start_logits, end_logits=end_logits))

        return rs.write_predictions(data['raw_inputs'], data['feature_extra'], all_results, n_best_size, max_answer_length, True)

    def inputs(self):
        with open(modulePath / "example.json", 'r') as f:
            raw = f.read()

        return raw


if __name__ == "__main__":
    import libff.invoke
    libff.invoke.remoteServer(Model)
