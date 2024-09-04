import torch
import torchvision
import transformers


def get_model(model, pretrain):
    if model in ["ResNet18", "ResNet50"]:
        m = torchvision.models.resnet18(weights=("DEFAULT" if pretrain else None))
        m.fc = torch.nn.Identity()
    elif model == "Id":
        m =  torch.nn.Flatten()
    elif model == "DistilBert":
        tokenizer=transformers.DistilBertTokenizer.from_pretrained(
                'distilbert-base-uncased')
        pretrained_model = transformers.DistilBertModel.from_pretrained(
            'distilbert-base-uncased', return_dict=False)
        feature_extractor = TransformerFeatureExtractor(pretrained_model, agg_mode="avg")
        return (feature_extractor, tokenizer)
    m.eval()
    return m


class TransformerFeatureExtractor(torch.nn.Module):

    """
    Wraps a transfomer encoder in a pytorch module and makes it return a
    fixed-dimensional embedding by averaging or extracting the CLS token.

    The resulting pytorch module accepts a single input `x` which is assumed to
    contain token ids and attention mask stacked on top of each other.

    Args:
        model: The transformer model.
        agg_mode: How to aggregate to a fixed-dimensional feature
            representation. Use 'avg' to average across tokens, use 'cls' to
            use the class token.
    """

    def __init__(self, model, agg_mode: str = "avg") -> None:
        super(TransformerFeatureExtractor, self).__init__()
        self.model = model
        self.agg_mode = agg_mode

    def forward(self, X: torch.Tensor):
        input_ids = X[:, :, 0]
        attention_mask = X[:, :, 1]
        output = self.model(input_ids, attention_mask=attention_mask)[0]
        if self.agg_mode == "avg":
            return torch.mean(output, dim=1)
        elif self.agg_mode == "cls":
            return output[:, 0, :]
        else:
            raise ValueError(f"Unknown option mode={self.agg_mode}.")
