# from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModel
import pickle
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import copy
from tqdm import tqdm, trange

import numpy as np
import argparse
import sklearn.metrics
import shap
# import sys
# sys.path.append('../')
import icdcode_encode
import parse_xml

# class CategoricalNet(nn.Module):
#     def __init__(self, n_unique_list, embedding_size):
#         ## Assumes input is of type list e.g. [4,4,4,8]
#         ## where n_unique_list[i] is num unique features for that category
#         super().__init__()
#         self.embedding_size = embedding_size
#         self.total_embedding_size = embedding_size*len(n_unique_list)
#         self.embeddings = nn.ModuleList()

#         for i in range(len(n_unique_list)):
#             self.embeddings.append(nn.Embedding(n_unique_list[i], embedding_size))

#     def forward(self, feature_lists):
#         ## Assumes input is of type list e.g. [[1, 2, 0, 3], ... , [3, 2, 0, 2]]
#         ## where feature_lists[i][j] is the feature in i_th sample, j_th category
#         return torch.cat([self.embeddings[j](feature_lists[:,j]) for j in range(feature_lists.shape[1])], dim=1)


class CustomEncoder(nn.Module):
    def __init__(self, pretrained_model_name_or_path='dmis-lab/biobert-v1.1', device="cpu"):
        super(CustomEncoder, self).__init__()
        """Custom encoder class
            This custom encoder class allows KeyClass to use encoders beyond those 
            in Sentence Transformers. Here, we will use the BlueBert-Base (Uncased)
            language model trained on PubMed and MIMIC-III [1]. 
            
            Parameters
            ---------- 
            pretrained_model_name_or_path: str
                Is either:
                -- a string with the shortcut name of a pre-trained model configuration to load 
                   from cache or download, e.g.: bert-base-uncased.
                -- a string with the identifier name of a pre-trained model configuration that 
                   was user-uploaded to our S3, e.g.: dbmdz/bert-base-german-cased.
                -- a path to a directory containing a configuration file saved using the 
                   save_pretrained() method, e.g.: ./my_model_directory/.
                -- a path or url to a saved configuration JSON file, e.g.: ./my_model_directory/configuration.json.
            
            device: str
                Device to use for encoding. 'cpu' by default. 
            References
        """
        super(CustomEncoder, self).__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path)
        self.model = AutoModel.from_pretrained(pretrained_model_name_or_path)
        self.model.train()
        # The model is set in evaluation mode by default using model.eval()
        # (Dropout modules are deactivated) To train the model, you should
        # first set it back in training mode with model.train()
        self.device = device
        self.to(device)

    def encode(self, sentences, batch_size=32, show_progress_bar=False, normalize_embeddings=False):
        """
        Computes sentence embeddings using the forward function
        Parameters
        ---------- 
        text: the text to embed
        batch_size: the batch size used for the computation
        """
        self.model.eval()  # Set model in evaluation mode.
        with torch.no_grad():
            embeddings = self.forward(sentences, batch_size=batch_size, show_progress_bar=show_progress_bar, normalize_embeddings=normalize_embeddings).detach().cpu().numpy()
        self.model.train()
        return embeddings
    
    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0]  #First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def _text_length(self, text):
        """
        Help function to get the length for the input text. Text can be either
        a list of ints (which means a single text as input), or a tuple of list of ints
        (representing several text inputs to the model).
        Adapted from https://github.com/UKPLab/sentence-transformers/blob/40af04ed70e16408f466faaa5243bee6f476b96e/sentence_transformers/SentenceTransformer.py#L548
        """
        if isinstance(text, dict):  #{key: value} case
            return len(next(iter(text.values())))
        elif not hasattr(text, '__len__'):  #Object has no len() method
            return 1
        elif len(text) == 0 or isinstance(text[0], int):  #Empty string or list of ints
            return len(text)
        else:
            return sum([len(t) for t in text])  #Sum of length of individual strings
    
    def forward(self, sentences, batch_size=32, show_progress_bar=False, normalize_embeddings=False):
        """
        Computes sentence embeddings
        
        Parameters
        ---------- 
        sentences: the sentences to embed
        batch_size: the batch size used for the computation
        show_progress_bar: This option is not used, and primarily present due to compatibility. 
        normalize_embeddings: This option is not used, and primarily present due to compatibility. 
        """

        all_embeddings = []
        length_sorted_idx = np.argsort([-self._text_length(sen) for sen in sentences])
        sentences_sorted = [sentences[idx] for idx in length_sorted_idx]

        for start_index in trange(0, len(sentences), batch_size, desc="Batches", disable=not show_progress_bar):
            sentences_batch = sentences_sorted[start_index:start_index + batch_size]

            features = self.tokenizer(sentences_batch,
                                      return_tensors='pt',
                                      truncation=True,
                                      max_length=512,
                                      padding=True)
            features = features.to(self.device)
            out_features = self.model.forward(**features)
            embeddings = self.mean_pooling(out_features, features['attention_mask'])

            if normalize_embeddings:
                embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

            all_embeddings.extend(embeddings)

        all_embeddings = [all_embeddings[idx] for idx in np.argsort(length_sorted_idx)]
        all_embeddings = torch.stack(all_embeddings)  # Converts to tensor

        return all_embeddings
    
    
def get_linear_data(trials, drugs):
    all_x = []
    all_y = np.array([t['label'] for t in trials])
    for i in tqdm(range(len(trials))):
        trial_embeds = model.base_model.encode([trials[i]['brief_summary']] + trials[i]['brief_summary_additional'], convert_to_numpy=True)
        trial_embeds2 = model.base_model.encode(trials[i]['eligibility_text']).mean(axis=0)[np.newaxis,:]
        drug_embeds = []
        for drug in drugs[i]:
            drug_embeds.append(model.base_model.encode([drug['description'], drug['pharmacodynamics'], drug['toxicity'], drug['metabolism'], drug['absorption']]))
        drug_embeds = np.array(drug_embeds).mean(axis=0)
        # print(trial_embeds.shape, trial_embeds2.shape, drug_embeds.shape); quit()
        x = np.concatenate([trial_embeds, trial_embeds2, drug_embeds]).flatten()

        icd_embeddings = []
        for icd_code in trials[i]['icdcode_lst']:
            with torch.no_grad():
                icd_embeddings.append(model.icd_encoder.forward_code_lst3(icd_code).cpu().numpy())
        icd_embeddings = np.concatenate(icd_embeddings).mean(axis=0) # take mean of icd embeddings
        x = np.concatenate([x, icd_embeddings])
        all_x.append(x)
    return np.array(all_x), all_y



class LINT(nn.Module):
    def __init__(self, embedding_dims, icd_encoder, icd_encoder_dim, base_model_name='dmis-lab/biobert-base-cased-v1.2', f=nn.LeakyReLU(), device='cpu', 
                 d_bert=768, nhead=16, num_layers=1, loss=nn.CrossEntropyLoss()):
        super().__init__()
        assert len(embedding_dims) > 0

        self.device = torch.device(device)
        # self.tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        # self.base_model = AutoModel.from_pretrained(base_model_name).to(self.device)
        # self.base_model = SentenceTransformer(base_model_name, device=self.device)
        self.base_model = CustomEncoder(pretrained_model_name_or_path=base_model_name, device=self.device)
        self.total_encoder = nn.TransformerEncoder(encoder_layer=nn.TransformerEncoderLayer(d_model=d_bert, nhead=nhead), num_layers=num_layers)
        # self.attention_linear = nn.Linear(d_bert, 1)
        self.icd_encoder = icd_encoder
        self.loss = loss
        
        if icd_encoder is not None:
            embed_size = d_bert+icd_encoder_dim
        else:
            embed_size = d_bert

        # add final linear layer for binary classification
        self.layers = nn.ModuleList()
        # self.layers.append(nn.LayerNorm(embed_size)) 
        self.layers.append(nn.Linear(embed_size, embedding_dims[0]))
        for i in range(len(embedding_dims)-1): # only activates if len(embedding_dims) >= 2
            # self.layers.append(nn.LayerNorm(embedding_dims[i])) 
            self.layers.append(nn.Linear(embedding_dims[i], embedding_dims[i+1]))
            if i != len(embedding_dims)-2:
                self.layers.append(nn.Dropout(p=0.25))
                self.layers.append(f)

    # def focal_loss(self, y_hat, y, alpha=0.75, gamma=2):
    #     # y_hat: (N, 1)
    #     # y: (N, 1)
    #     # alpha: float
    #     # gamma: float
    #     y_hat = y_hat.view(-1, 1)
    #     y = y.view(-1, 1)
    #     y_hat = torch.clamp(y_hat, -75, 75)
    #     p = torch.sigmoid(y_hat)
    #     loss = -alpha * (1 - p) ** gamma * y * torch.log(p) - (1 - alpha) * p ** gamma * (1 - y) * torch.log(1 - p)
    #     return loss.mean()

    def forward(self, trial, drugs, use_text=False):
        if use_text:
            with torch.no_grad():
                # ====== get PLM embeddings ======
                all_text = [trial['brief_summary']] + trial['brief_summary_additional'] + trial['eligibility_text']
                for drug in drugs:
                    all_text.extend([drug['description'], drug['pharmacodynamics'], drug['toxicity'], drug['metabolism'], drug['absorption']])
                # all_text_embeds = self.base_model.encode(all_text, convert_to_tensor=True)
                all_text_embeds = self.base_model.forward(all_text)
                # print(all_text_embeds.s)
        else:
            all_text_embeds = trial['all_text_embeds'].to(self.device)
        # ====== combine embeddings ======
        # weights = self.attention_linear(self.total_encoder(all_text_embeds))
        # weights = nn.Softmax(dim=0)(weights)
        # total_embedding = (all_text_embeds * weights).sum(dim=0)
        total_embedding = self.total_encoder(all_text_embeds)[0]
        # print(total_embedding.shape)

        if self.icd_encoder is not None:
            icd_embeddings = []
            for icd_code in trial['icdcode_lst']:
                icd_embeddings.append(self.icd_encoder.forward_code_lst3(icd_code))
            icd_embeddings = torch.cat(icd_embeddings).mean(dim=0) # take mean of icd embeddings
            x = torch.cat([total_embedding, icd_embeddings])
        else:
            x = total_embedding
            
        for layer in self.layers:
            x = layer(x)

        return x

    def forward_batch(self, trials, drugs):
        return torch.stack([self.forward(trials[i], drugs[i], use_text=True) for i in range(len(trials))], dim=0)

    def test(self, data_loader):
        self.eval()

        losses = []
        outputs = []
        labels = []
        with torch.no_grad():
            for batch in tqdm(data_loader):
                # print(batch)
                trials = [b[0] for b in batch]
                drugs = [b[1] for b in batch]
                # print(criteria_lst)
                labels.extend([t['label'] for t in trials])
                label_vec = torch.Tensor([t['label'] for t in trials]).long().to(self.device)
                output = self.forward_batch(trials, drugs)
                loss = self.loss(output, label_vec)

                losses.append(loss.item())
                outputs.append(output.cpu().numpy())
        self.train()

        labels = np.array(labels)
        outputs = np.concatenate(outputs)
        score = sklearn.metrics.f1_score(y_true=labels, y_pred=outputs.argmax(axis=1))
        # score = sklearn.metrics.roc_auc_score(y_true=labels, y_score=outputs[:,1])
        # score = np.mean(losses)
        print('score', score)
        return score, outputs

    def fit(self, train_loader, valid_loader, test_loader, epochs, lr, weight_decay, verbose=True):
        self.to(self.device)
        self.train()

        optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)
        train_loss_record = [] 
        valid_score_record = []
        best_model = None

        for ep in tqdm(range(epochs)):
            # ======= Training =======
            for batch in tqdm(train_loader):
                optimizer.zero_grad() 
                trials = [b[0] for b in batch]
                drugs = [b[1] for b in batch]
                label_vec = torch.Tensor([t['label'] for t in trials]).long().to(self.device)
                output = self.forward_batch(trials, drugs)
                loss = self.loss(output, label_vec)
                # print(output.shape, label_vec.shape)
                # loss = self.focal_loss(output[:,1], label_vec.unsqueeze(dim=1))
                train_loss_record.append(loss.detach().cpu().numpy())

                loss.backward() 
                optimizer.step()

                if verbose: tqdm.write('epoch: {}, loss: {:.4f}'.format(ep, loss))        
            # ======= Validatio/home/chufan2/clinical-trial-outcome-prediction/raw_datan =======
            valid_score, _ = self.test(valid_loader)
            valid_score_record.append(valid_score)
            if valid_score >= np.max(valid_score_record): # >= will trigger for 1st epoch as well
                best_model = copy.deepcopy(self)

        # ======= Final Eval =======
        self = copy.deepcopy(best_model)
        print('test results')
        _, test_output = self.test(test_loader)
        print('train results')
        _, train_output = self.test(train_loader)
        print('valid results')
        _, valid_output = self.test(valid_loader)
        return train_output, valid_output, test_output
        # auc_score, f1score, prauc_score, precision, recall, accuracy, predict_1_ratio, label_1_ratio = self.test(test_loader, return_loss = False, validloader = valid_loader)


def explain(lint_model, trial, drugs, img_path):
    with torch.no_grad():
        lint_model.to(lint_model.device)
        print(trial, drugs, lint_model.forward(trial, drugs, use_text=True))

        def f(x):
            trial_ = copy.deepcopy(trial)
            trial_['brief_summary'] = ' '.join(x)
            out = lint_model.forward(trial_, drugs, use_text=True)
            return out.unsqueeze(0).repeat(len(x), 1)

        explainer = shap.Explainer(f, lint_model.base_model.tokenizer)
        shap_values = explainer([trial['brief_summary']])
        with open(img_path+'_brief_summary.html', 'w') as file:
            file.write(shap.plots.text(shap_values=shap_values, display=False))
        
        for i in range(len(trial['brief_summary_additional'])):
            def f(x):
                trial_ = copy.deepcopy(trial)
                trial_['brief_summary_additional'][i] = ' '.join(x)
                out = lint_model.forward(trial_, drugs, use_text=True)
                return out.unsqueeze(0).repeat(len(x), 1)


            explainer = shap.Explainer(f, lint_model.base_model.tokenizer)
            shap_values = explainer([trial['brief_summary_additional'][i]])
            with open(img_path+'_brief_summary_additional_{}.html'.format(i), 'w') as file:
                file.write(shap.plots.text(shap_values=shap_values, display=False))

        for i in range(len(trial['eligibility_text'])):
            def f(x):
                trial_ = copy.deepcopy(trial)
                trial_['eligibility_text'][i] = ' '.join(x)
                out = lint_model.forward(trial_, drugs, use_text=True)
                return out.unsqueeze(0).repeat(len(x), 1)

            explainer = shap.Explainer(f, lint_model.base_model.tokenizer)
            shap_values = explainer([trial['eligibility_text'][i]])
            with open(img_path+'_eligibility_text_{}.html'.format(i), 'w') as file:
                file.write(shap.plots.text(shap_values=shap_values, display=False))
        
        for i in range(len(drugs)):
            for var in ['description', 'pharmacodynamics', 'toxicity', 'metabolism', 'absorption']:
                def f(x):
                    drugs_ = copy.deepcopy(drugs)
                    drugs_[i][var] = ' '.join(x)
                    out = lint_model.forward(trial, drugs_, use_text=True)
                    return out.unsqueeze(0).repeat(len(x), 1)

                explainer = shap.Explainer(f, lint_model.base_model.tokenizer)
                shap_values = explainer([drugs[i][var]])
                with open(img_path+'_{}_{}.html'.format(var, i), 'w') as file:
                    file.write(shap.plots.text(shap_values=shap_values, display=False))

def ablation_measure_text(lint_model, orig_trials, orig_drugs):
    all_outputs = []
    device = lint_model.device
    lint_model.to(device)

    # ========== test no text in trial data ==========
    for text_to_remove in ['brief_summary', 'brief_summary_additional', 'eligibility_text']:
        trials = copy.deepcopy(orig_trials)
        for i in range(len(trials)):
            if text_to_remove=='brief_summary':
                trials[i][text_to_remove] = 'None'
            else:
                trials[i][text_to_remove] = ['None']

        test_loader = DataLoader(dataset=[(t,d) for t,d in zip(trials, orig_drugs)], batch_size=batch_size, shuffle=False, collate_fn=lambda x: x)
        _, test_output = lint_model.test(test_loader)
        all_outputs.append(test_output)

    # ========== test drug data ==========
    drugs = copy.deepcopy(orig_drugs)
    for i in range(len(drugs)):
        for j in range(len(drugs[i])):
            for text_to_remove in ['description', 'pharmacodynamics', 'toxicity', 'metabolism', 'absorption']:
                drugs[i][j][text_to_remove] = 'None'

    test_loader = DataLoader(dataset=[(t,d) for t,d in zip(orig_trials, drugs)], batch_size=batch_size, shuffle=False, collate_fn=lambda x: x)
    _, test_output = lint_model.test(test_loader)
    all_outputs.append(test_output)

    return all_outputs

def load_data(phase, train_mode, drug_dict_path='all_drug_dict2.pkl', trials_path='all_valid_trials2.pkl', year2split=2015):
    # assert train_mode in ["Biological", "Drug", "all"]
    # assert phase in ["1", "2", "3", "all"]
    
    all_drug_dict = pickle.load(open(drug_dict_path, 'rb'))
    all_valid_trials = pickle.load(open(trials_path, 'rb'))
    # print('len all valid files:', len(all_valid_trials))

    drug_mapping = parse_xml.get_drug_mapping(all_drug_dict=all_drug_dict)

    # phases: ['early phase 1', 'n/a', 'phase 1', 'phase 1/phase 2', 'phase 2', 'phase 2/phase 3', 'phase 3', 'phase 4'], 
    # counts: [ 110,  829, 2900, 1427, 9305,  482, 7386, 2947]
    
    processed_files = []

    for trial in all_valid_trials:
        if phase=='all':
            phase_bool = ("1" in trial['phase']) or ("2" in trial['phase']) or ("3" in trial['phase'])
        else:
            phase_bool = phase in trial['phase']
        if train_mode=='all' or 'both':
            train_mode_bool = ('Drug' in trial['intervention_types']) or ('Biological' in trial['intervention_types'])
        else:
            train_mode_bool = train_mode in trial['intervention_types']
 
        if train_mode_bool and phase_bool:
            processed_files.append(trial)
            
    # print('len valid files after selecting mode {} and phase {}: {}'.format(train_mode, phase, len(processed_files)))

    year2split = year2split-1970 # convert to standard unix time

    test_trials = [trial for trial in processed_files if trial['completion_date'] >= year2split]
    train_trials = [trial for trial in processed_files if trial['completion_date'] < year2split]
    valid_trials = train_trials
    # print('len train, valid, test:', len(train_trials), len(valid_trials), len(test_trials))

    train_drugs = parse_xml.get_drugs_from_trial(train_trials, drug_mapping=drug_mapping, all_drug_dict=all_drug_dict)
    valid_drugs = parse_xml.get_drugs_from_trial(valid_trials, drug_mapping=drug_mapping, all_drug_dict=all_drug_dict)
    test_drugs = parse_xml.get_drugs_from_trial(test_trials, drug_mapping=drug_mapping, all_drug_dict=all_drug_dict)

    return train_trials, train_drugs, valid_trials, valid_drugs, test_trials, test_drugs



if __name__=='__main__':
    np.random.seed(0)
    torch.manual_seed(0)

    # =================== parse args ===================
    parser = argparse.ArgumentParser()
    parser.add_argument('-train_mode', help='of the following ["bio", "drugs", "both"]')
    parser.add_argument('-phase', help='of the following ["1", "2", "3"]')
    parser.add_argument('-save_path', default='./test_results_f1_2/')
    parser.add_argument('-img_path', default='./test_imgs/')
    parser.add_argument('-use_icd_encoder', action='store_false', default=True)
    parser.add_argument('-explain', action='store_true', default=False)
    args = parser.parse_args()
    print(args)
    # assert args.train_mode in ["Biological", "Drug", 'all',]
    assert args.phase in ["1", "2", "3", 'all']

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    epochs = 10
    batch_size = 64
    lr = 4e-5
    weight_decay = 0.01
    print('device:', device)

    train_trials, train_drugs, \
    valid_trials, valid_drugs, \
    test_trials, test_drugs = load_data(phase=args.phase, 
                                        train_mode=args.train_mode, 
                                        drug_dict_path='all_drug_dict2.pkl', 
                                        trials_path='all_valid_trials2.pkl', 
                                        year2split=2015)
    
    # =================== model training ===================   
    train_loader = DataLoader(dataset=[(t,d) for t,d in zip(train_trials, train_drugs)], batch_size=batch_size, shuffle=True, collate_fn=lambda x: x)
    valid_loader = DataLoader(dataset=[(t,d) for t,d in zip(valid_trials, valid_drugs)], batch_size=batch_size, shuffle=False, collate_fn=lambda x: x)
    test_loader = DataLoader(dataset=[(t,d) for t,d in zip(test_trials, test_drugs)], batch_size=batch_size, shuffle=False, collate_fn=lambda x: x)

    if args.use_icd_encoder:
        icdcode2ancestor_dict = icdcode_encode.build_icdcode2ancestor_dict(pkl_file="../data/icdcode2ancestor_dict.pkl", input_file='../data/raw_data.csv')
        gram_model = icdcode_encode.GRAM(embedding_dim=50, icdcode2ancestor=icdcode2ancestor_dict, device=device)
        icd_encoder_dim = 50
    else:
        gram_model = None
        icd_encoder_dim = 0


    train_labels = np.unique([t['label'] for t in train_trials], return_counts=True)[1]
    label_weight = train_labels/np.sum(train_labels)
    print("Raw Label Weights: ", label_weight)
    label_weight = torch.Tensor(1/label_weight)
    print("Rebalanced Label Weights: ", label_weight)

    model = LINT(embedding_dims=[64, 64, 2], icd_encoder=gram_model, icd_encoder_dim=icd_encoder_dim, device=device, num_layers=1,
        loss=nn.CrossEntropyLoss(weight=label_weight))

    if args.explain:
        # train_x, train_y = get_linear_data(train_trials, train_drugs)
        # test_x, test_y = get_linear_data(test_trials, test_drugs)
        # pickle.dump([train_x, train_y, test_x, test_y], open('phase_{}_mode_{}_linear_data.pkl'.format(args.phase, args.train_mode), 'wb')); quit()

        model.load_state_dict(torch.load(args.save_path+'phase_{}_mode_{}_model.pth'.format(args.phase, args.train_mode)))
        # ind = 305
        # explain(lint_model=model, trial=test_trials[ind], drugs=test_drugs[ind], img_path=args.img_path + '{}_shap/'.format(ind))

        print('test results')
        ablation_output = ablation_measure_text(lint_model=model, orig_trials=test_trials, orig_drugs=test_drugs)
        np.save(args.save_path+'phase_{}_mode_{}_ablation_output.npy'.format(args.phase, args.train_mode), ablation_output)

    else: # run training
        train_output, valid_output, test_output = model.fit(train_loader=train_loader, \
            valid_loader=valid_loader, test_loader=test_loader, epochs=epochs, lr=lr, weight_decay=weight_decay)

        torch.save(model.state_dict(), args.save_path+'phase_{}_mode_{}_model.pth'.format(args.phase, args.train_mode))
        np.save(args.save_path+'phase_{}_mode_{}_train_output.npy'.format(args.phase, args.train_mode), train_output)
        np.save(args.save_path+'phase_{}_mode_{}_valid_output.npy'.format(args.phase, args.train_mode), valid_output)
        np.save(args.save_path+'phase_{}_mode_{}_test_output.npy'.format(args.phase, args.train_mode), test_output)
