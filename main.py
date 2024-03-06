import os
import math
import torch
import argparse
import yaml
from transformers import GPT2Tokenizer, AdamW, BertTokenizer, BertModel
from module import ContinuousPromptLearning
from utils import rouge_score, bleu_score, DataLoader, Batchify, now_time, ids2tokens, unique_sentence_percent, \
    feature_detect, feature_matching_ratio, feature_coverage_ratio, feature_diversity
import neptune


def neptune_recoder(exp_name, tags, hyperparameters):
    run = neptune.init_run(
        project="guyelovbgu/PEPLER",
        api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiIzMDZhZTVjNC0yMmQxLTQ1MDktODJmYS0zM2Q4YWQ3MDlmMTAifQ==",
        name=exp_name,  # Optional,
        capture_stderr=True,
        capture_stdout=True,
        capture_hardware_metrics=True,
        capture_traceback=True,
    )  # your credentials

    run['hyper-parameters'] = hyperparameters
    run["sys/tags"].add(tags)

    return run


# Function to read configuration from YAML file
def read_config(config_path="config.yml"):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)


# Parser for the path of the YAML configuration file
parser = argparse.ArgumentParser(description='PErsonalized Prompt Learning for Explainable Recommendation (PEPLER)')
args = parser.parse_args()

# Read configuration from YAML file
config = read_config()

# Replace command line argument parsing with configuration from YAML
data_path = config['data_path']
index_dir = config['index_dir']
lr = config['lr']
epochs = config['epochs']
batch_size = config['batch_size']
use_cuda = config['cuda']
log_interval = config['log_interval']
checkpoint = config['checkpoint']
outf = config['outf']
endure_times = config['endure_times']
words = config['words']
dataset = config['dataset']
fold = config['fold']
amazon_data = config['amazon_data_type']
if dataset == 'Amazon':
    index_dir = f'{index_dir}{dataset}/{amazon_data}/{fold}/'
    data_path = f'{data_path}{dataset}/{amazon_data}/reviews.pickle'
else:
    index_dir = f'{index_dir}{dataset}/{fold}/'
    data_path = f'{data_path}{dataset}/reviews.pickle'
if config['data_path'] is None:
    raise ValueError('data_path should be provided for loading data in the YAML configuration file')
if config['index_dir'] is None:
    raise ValueError('index_dir should be provided for loading data splits in the YAML configuration file')
# run = neptune_recoder('PEPLER', ['Reproduce','GUY'], dict(config))
print('-' * 40 + 'ARGUMENTS' + '-' * 40)
for arg in vars(args):
    print('{:40} {}'.format(arg, getattr(args, arg)))
print('-' * 40 + 'ARGUMENTS' + '-' * 40)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)
checkpoint_path = f'{checkpoint}{dataset}'
if not os.path.exists(checkpoint_path):
    os.makedirs(checkpoint_path)
model_path = os.path.join(checkpoint_path, f'model_{dataset}_fold{fold}.pt')
prediction_path = os.path.join(checkpoint_path, outf)

###############################################################################
# Load data
###############################################################################

print(now_time() + 'Loading data')
bos = '<bos>'
eos = '<eos>'
pad = '<pad>'
tokenizer = GPT2Tokenizer.from_pretrained('gpt2', bos_token=bos, eos_token=eos, pad_token=pad)
selector_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertModel.from_pretrained('bert-base-uncased')
bert_model.to(device)
head = torch.nn.Linear(768, 1)
head.to(device)
head_mse_loss = torch.nn.MSELoss()
corpus = DataLoader(data_path, index_dir, tokenizer, words)
feature_set = corpus.feature_set
train_data = Batchify(corpus.train[:500], tokenizer, bos, eos, batch_size, shuffle=True,
                      selector_tokenizer=selector_tokenizer)
val_data = Batchify(corpus.valid[:10], tokenizer, bos, eos, batch_size, selector_tokenizer=selector_tokenizer)
test_data = Batchify(corpus.test[:10], tokenizer, bos, eos, batch_size, selector_tokenizer=selector_tokenizer)

###############################################################################
# Build the model
###############################################################################

nuser = len(corpus.user_dict)
nitem = len(corpus.item_dict)
ntoken = len(tokenizer)
model = ContinuousPromptLearning.from_pretrained('gpt2', nuser, nitem)
model.resize_token_embeddings(ntoken)  # three tokens added, update embedding table
model.to(device)
optimizer = AdamW(model.parameters(), lr=lr)


###############################################################################
# Training code
###############################################################################
def get_cls_embedding(input_ids, attention_mask):
    outputs = bert_model(input_ids, attention_mask)
    cls_embedding = outputs.last_hidden_state[:, 0, :]
    head_output = head(cls_embedding)
    return head_output


def create_selector_labels(seq, mask, seq_len, model, user, item):
    with torch.no_grad():
        sub_seq = seq[:, :seq_len]
        sub_mask = mask[:, :seq_len]
        subset_loss_list = []
        context_loss_list = []
        for i in range(sub_seq.size(0)):
            user_i = user[i].unsqueeze(0)  # Ensure user tensor is at least 1D
            item_i = item[i].unsqueeze(0)  # Same for item tensor
            sub_seq_i = sub_seq[i].unsqueeze(0)
            sub_mask_i = sub_mask[i].unsqueeze(0)

            subset_loss = model(user_i, item_i, sub_seq_i, sub_mask_i).loss
            context_loss = model(user_i, item_i, seq[i].unsqueeze(0), mask[i].unsqueeze(0)).loss

            subset_loss_list.append(subset_loss)
            context_loss_list.append(context_loss)

        ratio_loss = torch.tensor(subset_loss_list) / torch.tensor(context_loss_list)
        return ratio_loss



def modify_loss(selector_output, seq, mask, model, user, item):
    model_loss = []
    for i in range(seq.size(0)):
        user_i = user[i].unsqueeze(0)
        item_i = item[i].unsqueeze(0)
        seq_i = seq[i].unsqueeze(0)
        mask_i = mask[i].unsqueeze(0)
        model_loss.append(model(user_i, item_i, seq_i, mask_i).loss)
    model_loss = torch.tensor(model_loss, device=device)
    selector_output = 1 - selector_output
    selector_output = selector_output.to(device)
    modified_loss = model_loss * selector_output
    return modified_loss.mean()


def train(data):
    # Turn on training mode which enables dropout.
    text_loss = 0.
    total_sample = 0
    train_steps_selector = 0

    while True:
        if train_steps_selector < 2:
            model.eval()
            bert_model.train()
            head.train()
            user, item, _, seq, mask, selector_seq, selector_mask = data.next_batch()
            # print(user.shape)
            # print(item.shape)
            # print(seq.shape)
            # print(mask.shape)
            selector_seq = selector_seq.to(device)
            selector_mask = selector_mask.to(device)
            seq = seq.to(device)
            mask = mask.to(device)
            user = user.to(device)
            item = item.to(device)
            selector_embedding = get_cls_embedding(selector_seq, selector_mask)
            del selector_seq, selector_mask
            selector_seq, selector_mask = None, None
            selector_embedding = selector_embedding.to(device)
            labels = create_selector_labels(seq, mask, 5, model, user, item)
            labels = labels.to(device)
            selector_embedding = selector_embedding.squeeze(1)
            selector_loss = head_mse_loss(selector_embedding, labels)
            optimizer.zero_grad()
            selector_loss.backward()
            optimizer.step()
            # run['train/selector_loss'].log(selector_loss.item())
            train_steps_selector += 1
            if train_steps_selector == 2:
                print('selector training finished')
        else:
            model.train()
            bert_model.eval()
            head.eval()
            user, item, _, seq, mask, selector_seq, selector_mask = data.next_batch()
            user = user.to(device)  # (batch_size,)
            item = item.to(device)
            seq = seq.to(device)  # (batch_size, seq_len)
            mask = mask.to(device)
            selector_seq = selector_seq.to(device)
            selector_mask = selector_mask.to(device)
            selector_output = torch.sigmoid(get_cls_embedding(selector_seq, selector_mask))
            selector_output = selector_output.to(device)
            print(selector_seq.shape)
            print(selector_mask.shape)
            print(seq.shape)

            # data.step += 1

            # Starting each batch, we detach the hidden state from how it was previously produced.
            # If we didn't, the model would try backpropagating all the way to start of the dataset.
            optimizer.zero_grad()
            # outputs = model(user, item, seq, mask)
            loss = modify_loss(selector_output, seq, mask, model, user, item)
            loss.backward()
            optimizer.step()
            run['train/loss'].log(loss.item())

            batch_size = user.size(0)
            text_loss += batch_size * loss.item()
            total_sample += batch_size

            if data.step % log_interval == 0 or data.step == data.total_step:
                cur_t_loss = text_loss / total_sample
                print(now_time() + 'text ppl {:4.4f} | {:5d}/{:5d} batches'.format(math.exp(cur_t_loss), data.step,
                                                                                   data.total_step))
                text_loss = 0.
                total_sample = 0
            if data.step == data.total_step:
                break


def evaluate(data, train_stage='prompt_tune'):
    # Turn on evaluation mode which disables dropout.
    model.eval()
    text_loss = 0.
    total_sample = 0
    with torch.no_grad():
        while True:
            user, item, _, seq, mask = data.next_batch()  # data.step += 1
            user = user.to(device)  # (batch_size,)
            item = item.to(device)
            seq = seq.to(device)  # (batch_size, seq_len)
            mask = mask.to(device)
            outputs = model(user, item, seq, mask)
            loss = outputs.loss
            run[f'{train_stage}_val/loss'].log(loss.item())

            batch_size = user.size(0)
            text_loss += batch_size * loss.item()
            total_sample += batch_size

            if data.step == data.total_step:
                break
    return text_loss / total_sample


def generate(data):
    # Turn on evaluation mode which disables dropout.
    model.eval()
    idss_predict = []
    with torch.no_grad():
        while True:
            user, item, _, seq, _ = data.next_batch()  # data.step += 1
            user = user.to(device)  # (batch_size,)
            item = item.to(device)
            text = seq[:, :1].to(device)  # bos, (batch_size, 1)
            for idx in range(seq.size(1)):
                # produce a word at each step
                outputs = model(user, item, text, None)
                last_token = outputs.logits[:, -1, :]  # the last token, (batch_size, ntoken)
                word_prob = torch.softmax(last_token, dim=-1)
                token = torch.argmax(word_prob, dim=1,
                                     keepdim=True)  # (batch_size, 1), pick the one with the largest probability
                text = torch.cat([text, token], 1)  # (batch_size, len++)
            ids = text[:, 1:].tolist()  # remove bos, (batch_size, seq_len)
            idss_predict.extend(ids)

            if data.step == data.total_step:
                break
    return idss_predict


print(now_time() + 'Tuning Prompt Only')
# Loop over epochs.
best_val_loss = float('inf')
endure_count = 0
for epoch in range(1, epochs + 1):
    print(now_time() + 'epoch {}'.format(epoch))
    train(train_data)
    val_loss = evaluate(val_data, 'prompt_tune')
    print(now_time() + 'text ppl {:4.4f} | valid loss {:4.4f} on validation'.format(math.exp(val_loss), val_loss))
    # Save the model if the validation loss is the best we've seen so far.
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        with open(model_path, 'wb') as f:
            torch.save(model, f)
    else:
        endure_count += 1
        print(now_time() + 'Endured {} time(s)'.format(endure_count))
        if endure_count == endure_times:
            print(now_time() + 'Cannot endure it anymore | Exiting from early stop')
            break

# Load the best saved model.
with open(model_path, 'rb') as f:
    model = torch.load(f).to(device)

print(now_time() + 'Tuning both Prompt and LM')
for param in model.parameters():
    param.requires_grad = True
optimizer = AdamW(model.parameters(), lr=lr)

# Loop over epochs.
best_val_loss = float('inf')
endure_count = 0
for epoch in range(1, epochs + 1):
    print(now_time() + 'epoch {}'.format(epoch))
    train(train_data)
    val_loss = evaluate(val_data, 'both_tune')
    print(now_time() + 'text ppl {:4.4f} | valid loss {:4.4f} on validation'.format(math.exp(val_loss), val_loss))
    # Save the model if the validation loss is the best we've seen so far.
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        with open(model_path, 'wb') as f:
            torch.save(model, f)
    else:
        endure_count += 1
        print(now_time() + 'Endured {} time(s)'.format(endure_count))
        if endure_count == endure_times:
            print(now_time() + 'Cannot endure it anymore | Exiting from early stop')
            break

# Load the best saved model.
with open(model_path, 'rb') as f:
    model = torch.load(f).to(device)
    # run['best_model'].upload(model_path)

# Run on test data.
test_loss = evaluate(test_data, 'test_both_tune')
print('=' * 89)
print(now_time() + 'text ppl {:4.4f} on test | End of training'.format(math.exp(test_loss)))
print(now_time() + 'Generating text')
idss_predicted = generate(test_data)
tokens_test = [ids2tokens(ids[1:], tokenizer, eos) for ids in test_data.seq.tolist()]
tokens_predict = [ids2tokens(ids, tokenizer, eos) for ids in idss_predicted]
BLEU1 = bleu_score(tokens_test, tokens_predict, n_gram=1, smooth=False)
print(now_time() + 'BLEU-1 {:7.4f}'.format(BLEU1))
BLEU4 = bleu_score(tokens_test, tokens_predict, n_gram=4, smooth=False)
print(now_time() + 'BLEU-4 {:7.4f}'.format(BLEU4))
USR, USN = unique_sentence_percent(tokens_predict)
print(now_time() + 'USR {:7.4f} | USN {:7}'.format(USR, USN))
feature_batch = feature_detect(tokens_predict, feature_set)
DIV = feature_diversity(feature_batch)  # time-consuming
print(now_time() + 'DIV {:7.4f}'.format(DIV))
FCR = feature_coverage_ratio(feature_batch, feature_set)
print(now_time() + 'FCR {:7.4f}'.format(FCR))
FMR = feature_matching_ratio(feature_batch, test_data.feature)
print(now_time() + 'FMR {:7.4f}'.format(FMR))
text_test = [' '.join(tokens) for tokens in tokens_test]
text_predict = [' '.join(tokens) for tokens in tokens_predict]
ROUGE = rouge_score(text_test, text_predict)  # a dictionary
for (k, v) in ROUGE.items():
    print(now_time() + '{} {:7.4f}'.format(k, v))
text_out = ''
for (real, fake) in zip(text_test, text_predict):
    text_out += '{}\n{}\n\n'.format(real, fake)
with open(prediction_path, 'w', encoding='utf-8') as f:
    f.write(text_out)
print(now_time() + 'Generated text saved to ({})'.format(prediction_path))
run['BLEU1'] = BLEU1
run['BLEU4'] = BLEU4
run['USR'] = USR
run['USN'] = USN
run['DIV'] = DIV
run['FCR'] = FCR
run['FMR'] = FMR
for (k, v) in ROUGE.items():
    run[k] = v
run['generated_text'].upload(prediction_path)
run.stop()
