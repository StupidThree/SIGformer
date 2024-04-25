import parse
from parse import args
import torchsparsegradutils
import utils
import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.utils import structured_negative_sampling


class Attention(nn.Module):
    def __init__(self):
        super(Attention, self).__init__()
        self.lambda0 = nn.Parameter(torch.zeros(1))
        self.path_emb = nn.Embedding(2**(args.sample_hop+1)-2, 1)
        nn.init.zeros_(self.path_emb.weight)
        self.sqrt_dim = 1./torch.sqrt(torch.tensor(args.hidden_dim))
        self.sqrt_eig = 1./torch.sqrt(torch.tensor(args.eigs_dim))
        self.my_parameters = [
            {'params': self.lambda0, 'weight_decay': 1e-2},
            {'params': self.path_emb.parameters()},
        ]

    def forward(self, q, k, v,  indices, eigs, path_type):
        ni, nx, ny, nz = [], [], [], []
        for i, pt in zip(indices, path_type):
            x = torch.mul(q[i[0]], k[i[1]]).sum(dim=-1)*self.sqrt_dim
            nx.append(x)
            if 'eig' in args.model:
                if args.eigs_dim == 0:
                    y = torch.zeros(i.shape[1]).to(parse.device)
                else:
                    y = torch.mul(eigs[i[0]], eigs[i[1]]).sum(dim=-1)
                ny.append(y)
            if 'path' in args.model:
                z = self.path_emb(pt).view(-1)
                nz.append(z)
            ni.append(i)
        i = torch.concat(ni, dim=-1)
        s = []
        s.append(torch.concat(nx, dim=-1))
        if 'eig' in args.model:
            s[0] = s[0]+torch.exp(self.lambda0)*torch.concat(ny, dim=-1)
        if 'path' in args.model:
            s.append(torch.concat(nz, dim=-1))
        s = [utils.sparse_softmax(i, _, q.shape[0]) for _ in s]
        s = torch.stack(s, dim=1).mean(dim=1)
        return torchsparsegradutils.sparse_mm(torch.sparse_coo_tensor(i, s, torch.Size([q.shape[0], k.shape[0]])), v)


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.self_attention = Attention()
        self.my_parameters = self.self_attention.my_parameters

    def forward(self, x, indices, eigs, path_type):
        y = F.layer_norm(x, normalized_shape=(args.hidden_dim,))
        y = self.self_attention(
            y, y, y,
            indices,
            eigs,
            path_type)
        return y


class Model(nn.Module):
    def __init__(self, dataset):
        super(Model, self).__init__()
        self.dataset = dataset
        self.hidden_dim = args.hidden_dim
        self.n_layers = args.n_layers
        self.embedding_user = nn.Embedding(self.dataset.num_users, self.hidden_dim)
        self.embedding_item = nn.Embedding(self.dataset.num_items, self.hidden_dim)
        nn.init.normal_(self.embedding_user.weight, std=0.1)
        nn.init.normal_(self.embedding_item.weight, std=0.1)
        self.my_parameters = [
            {'params': self.embedding_user.parameters()},
            {'params': self.embedding_item.parameters()},
        ]
        self.layers = []
        for i in range(args.n_layers):
            layer = Encoder().to(parse.device)
            self.layers.append(layer)
            self.my_parameters.extend(layer.my_parameters)
        self._users, self._items = None, None
        self.optimizer = torch.optim.Adam(
            self.my_parameters,
            lr=args.learning_rate)

    def computer(self):
        users_emb = self.embedding_user.weight
        items_emb = self.embedding_item.weight
        all_emb = torch.cat([users_emb, items_emb])
        embs = [all_emb]
        for i in range(self.n_layers):
            indices, paths = self.dataset.sample()
            all_emb = self.layers[i](all_emb,
                                     indices,
                                     self.dataset.L_eigs,
                                     paths)
            embs.append(all_emb)
        embs = torch.stack(embs, dim=1)
        light_out = torch.mean(embs, dim=1)
        self._users, self._items = torch.split(light_out, [self.dataset.num_users, self.dataset.num_items])

    def evaluate(self, test_pos_unique_users, test_pos_list, test_neg_list):
        self.eval()
        if self._users is None:
            self.computer()
        user_emb, item_emb = self._users, self._items
        max_K = max(parse.topks)
        all_pre = torch.zeros(len(parse.topks))
        all_recall = torch.zeros(len(parse.topks))
        all_ndcg = torch.zeros(len(parse.topks))
        with torch.no_grad():
            users = test_pos_unique_users
            for i in range(0, users.shape[0], args.test_batch_size):
                batch_users = users[i:i+args.test_batch_size]
                user_e = user_emb[batch_users]
                rating = torch.mm(user_e, item_emb.t())
                for j, u in enumerate(batch_users):
                    rating[j, self.dataset.train_pos_list[u]] = -(1 << 10)
                    rating[j, self.dataset.train_neg_list[u]] = -(1 << 10)
                _, rating = torch.topk(rating, k=max_K)
                pre, recall, ndcg = utils.test(
                    rating,
                    test_pos_list[i:i+args.test_batch_size])
                all_pre += pre
                all_recall += recall
                all_ndcg += ndcg
            all_pre /= users.shape[0]
            all_recall /= users.shape[0]
            all_ndcg /= users.shape[0]
        return all_pre, all_recall, all_ndcg

    def valid_func(self):
        return self.evaluate(self.dataset.valid_pos_unique_users, self.dataset.valid_pos_list, self.dataset.valid_neg_list)

    def test_func(self):
        return self.evaluate(self.dataset.test_pos_unique_users, self.dataset.test_pos_list, self.dataset.valid_neg_list)

    def train_func(self):
        self.train()
        pos_u = self.dataset.train_pos_user
        pos_i = self.dataset.train_pos_item
        indices = torch.randperm(self.dataset.train_neg_user.shape[0])
        neg_u = self.dataset.train_neg_user[indices]
        neg_i = self.dataset.train_neg_item[indices]
        all_j = structured_negative_sampling(
                torch.concat([torch.stack([pos_u, pos_i]), torch.stack([neg_u, neg_i])], dim=1),
                num_nodes=self.dataset.num_items)[2]
        pos_j, neg_j = torch.split(all_j, [pos_u.shape[0], neg_u.shape[0]])
        loss = self.loss_one_batch(pos_u, pos_i, pos_j, neg_u, neg_i, neg_j)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss

    def loss_one_batch(self, pos_u, pos_i, pos_j, neg_u, neg_i, neg_j):
        self.computer()
        all_user, all_item = self._users, self._items
        pos_u_emb0, pos_u_emb = self.embedding_user(pos_u), all_user[pos_u]
        pos_i_emb0, pos_i_emb = self.embedding_item(pos_i), all_item[pos_i]
        pos_j_emb0, pos_j_emb = self.embedding_item(pos_j), all_item[pos_j]
        neg_u_emb0, neg_u_emb = self.embedding_user(neg_u), all_user[neg_u]
        neg_i_emb0, neg_i_emb = self.embedding_item(neg_i), all_item[neg_i]
        neg_j_emb0, neg_j_emb = self.embedding_item(neg_j), all_item[neg_j]
        pos_scores_ui = torch.sum(torch.mul(pos_u_emb, pos_i_emb), dim=-1)
        pos_scores_uj = torch.sum(torch.mul(pos_u_emb, pos_j_emb), dim=-1)
        neg_scores_ui = torch.sum(torch.mul(neg_u_emb, neg_i_emb), dim=-1)
        neg_scores_uj = torch.sum(torch.mul(neg_u_emb, neg_j_emb), dim=-1)
        if args.beta == 0:
            reg_loss = (1/2)*(pos_u_emb0.norm(2).pow(2) +
                              pos_i_emb0.norm(2).pow(2) +
                              pos_j_emb0.norm(2).pow(2))/float(pos_u.shape[0])
            scores = pos_scores_uj-pos_scores_ui
        else:
            reg_loss = (1/2)*(pos_u_emb0.norm(2).pow(2) +
                              pos_i_emb0.norm(2).pow(2) +
                              pos_j_emb0.norm(2).pow(2) +
                              neg_u_emb0.norm(2).pow(2) +
                              neg_i_emb0.norm(2).pow(2) +
                              neg_j_emb0.norm(2).pow(2))/float(pos_u.shape[0]+neg_u.shape[0])
            scores = torch.concat([pos_scores_uj-pos_scores_ui, args.beta*(neg_scores_uj-neg_scores_ui)], dim=0)
        loss = torch.mean(F.softplus(scores))
        return loss+args.lambda_reg*reg_loss
