class Args(object):
    pass

class PreprocessArgs(Args):
    input_schemes = ["qr", "qrs"]
    output_schemes = ["q'r'", "sq'r'"]
    labeling_schemes = ["IO1", "IO2", "BIO1", "BIO2"] # 1: use the same class for q' and r' / 2: use different classes (e.g., I-q and I-r) for q' and r'
    
    def __init__(
        self,
        use_nltk: bool, # whether to use tokenize.word_tokenize() first before model_tokenizer
        model_tokenizer_name: str, # HuggingFace tokenizer name
        input_scheme: str, # qr | qrs
        output_scheme: str, # q'r' | sq'r'
        labeling_scheme: str, # IO1 | IO2 | BIO1 | BIO2
    ):
        self.use_nltk = use_nltk
        self.model_tokenizer_name = model_tokenizer_name
        self.set_input_scheme(input_scheme)
        self.set_output_scheme(output_scheme)
        self.set_labeling_scheme(labeling_scheme)

    def set_input_scheme(self, input_scheme: str) -> None:
        assert input_scheme in self.input_schemes
        self.input_scheme = input_scheme

    def set_output_scheme(self, output_scheme: str) -> None:
        assert output_scheme in self.output_schemes
        self.output_scheme = output_scheme

    def set_labeling_scheme(self, labeling_scheme: str) -> None:
        assert labeling_scheme in self.labeling_schemes
        self.labeling_scheme = labeling_scheme