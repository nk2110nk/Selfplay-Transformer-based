import torch
import numpy as np
from envs.inout import load_genius_domain_from_folder
import json
import os

import openai


class MyEmbedding:
    def __init__(self, scale='small') -> None:
        self.emb_dim = 256
        # Precomputed embeddings are the normal runtime path.  Constructing an
        # API client here made offline training fail even when every JSON file
        # already existed, so the client is created only if generation is used.
        self.client = None
        self.offer_emb_dict = None
        self.model_call_count = 0

    def text_embedding(self, text: str):
        if self.client is None:
            self.client = openai.OpenAI()
        embeddings = self.client.embeddings.create(input = [text], model="text-embedding-3-large").data[0].embedding[:self.emb_dim]
        embeddings = torch.as_tensor(embeddings)
        return embeddings.to(torch.float32)
    
    def list_embedding(self, inputs, save_path=None):
        embeddings = []
        if os.path.isfile(save_path):
            with open(save_path) as f:
                embeddings = json.load(f)
                return torch.stack([torch.tensor(e, dtype=torch.float32) for e in embeddings]).detach().numpy()
        else:
            for i in inputs:
                embeddings.append(self.text_embedding(i).squeeze().to(torch.float32))
            return torch.stack(embeddings).detach().numpy().tolist()
    
    def make_offer_embedding(self, scenario=None, save_path=None):
        if os.path.isfile(save_path):
            with open(save_path) as f:
                self.offer_emb_dict = json.load(f)
                self.offer_emb_dict = {k: torch.tensor(v,dtype=torch.float32) for k, v in self.offer_emb_dict.items()}
        else:
            offer_emb_dict = dict()
            for i in scenario.issues:
                for v in i.values:
                    offer_emb_dict[v] = self.text_embedding(v).squeeze().detach().numpy().tolist()
            self.offer_emb_dict = offer_emb_dict
    
    def offer_embedding(self, offer):
        embeddings = []
        for o in offer:
            embeddings.append(self.offer_emb_dict[o])
        return torch.stack(embeddings).detach().numpy()
    
    def outcomes_embedding(self, domain, batch_size=None, save_path=None):
        i_names = [i.name for i in domain]
        i_emb = self.list_embedding(i_names, save_path)
        embeddings = []
        for idx, issue in enumerate(domain):
            for v in issue:
                emb = self.offer_emb_dict[v].detach().numpy()
                emb = emb + i_emb[idx]
                embeddings.append(emb.reshape(1, self.emb_dim))
        return np.stack(embeddings)
            
    def __call__(self, text: str):
        return self.text_embedding(text)

if __name__ == '__main__':
    # 論点の選択肢を埋め込むなら"offer"，論点名なら"issue"
    emb_type = "offer"
    # emb_type = "issue"
    # ドメイン指定
    domains = ['Laptop', 'planes', 'Grocery', 'Camera', 'thompson', 'Car', 'EnergySmall_A','Coffee', 'Lunch', 'SmartPhone', 'Kitchen', 'IS_BT_Acquisition', 'ItexvsCypress']
    scale = "small"
    model = MyEmbedding()
    for domain in domains:
        scenario = load_genius_domain_from_folder('domain/' + domain)
        if emb_type == "offer":
            save_path = './embeddings/openai/' + scale + '/'
            os.makedirs(save_path, exist_ok=True)
            save_path = save_path + domain + '.json'
            model.make_offer_embedding(scenario, save_path)
            with open(save_path, 'w') as f:
                json.dump(model.offer_emb_dict, f)
        elif emb_type == "issue":
            save_path = './embeddings/openai/i_embs/' + scale + '/'
            i_names = [i.name for i in scenario.issues]
            embeddings = model.list_embedding(i_names, save_path)
            with open(save_path, 'w') as f:
                json.dump(embeddings, f)
        
