from flask_ngrok import run_with_ngrok
from flask import Flask, request
import json

# Libraries
import torch

# Preliminaries
from torchtext.data import Field, Iterator, Dataset, Example

# Models
import torch.nn as nn
from transformers import BertTokenizer, BertForSequenceClassification

app = Flask(__name__)
run_with_ngrok(app)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


# ====================== Model ========================


def load_checkpoint(load_path, model):
    if load_path == None:
        return

    state_dict = torch.load(load_path, map_location=device)
    print(f'Model loaded from <== {load_path}')

    model.load_state_dict(state_dict['model_state_dict'])
    return state_dict['valid_loss']


class BERT(nn.Module):

    def __init__(self):
        super(BERT, self).__init__()

        options_name = "bert-base-uncased"
        self.encoder = BertForSequenceClassification.from_pretrained(options_name)

    def forward(self, text, label):
        text_fea = self.encoder(text, labels=label)[:2]

        return text_fea


best_model = BERT().to(device)

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Model parameter
MAX_SEQ_LEN = 128
PAD_INDEX = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
UNK_INDEX = tokenizer.convert_tokens_to_ids(tokenizer.unk_token)

# Fields

text_field = Field(use_vocab=False, tokenize=tokenizer.encode, lower=False, include_lengths=False, batch_first=True,
                   fix_length=MAX_SEQ_LEN, pad_token=PAD_INDEX, unk_token=UNK_INDEX)
fields = [('title', text_field), ('text', text_field), ('titletext', text_field)]

load_checkpoint('model.pt', best_model)


# ====================== Routes ========================


@app.route('/')
def hello_world():
    return 'Hello, World!'


@app.route('/classify-news', methods=["POST"])
def classify():
    req_body = json.loads(request.data)
    test_1_title = req_body['title']
    test_1_text = req_body['text']
    test_1_titletext = test_1_title + ". " + test_1_text

    example_item = Example.fromlist([test_1_title, test_1_text, test_1_titletext], fields)
    eval_ds = Dataset(examples=[example_item], fields=fields, filter_pred=None)
    eval_iter = Iterator(eval_ds, batch_size=1, device=device, train=False, shuffle=False, sort=False)

    best_model.eval()
    with torch.no_grad():
        for (title, text, titletext), _ in eval_iter:
            titletext = titletext.type(torch.LongTensor)
            titletext = titletext.to(device)
            output = best_model(titletext, None)
            output = output[0]

            prediction = torch.argmax(output, 1).tolist()[0]

    return {
        "classification_result": bool(prediction)
    }

if __name__ == '__main__':
    app.run()