import pickle
import numpy as np
from scipy.sparse import coo_matrix
from Params import args
import scipy.sparse as sp
import torch
import torch.utils.data as data
import torch.utils.data as dataloader
import dgl

def get_graph(graph):
    new_row = graph.row
    new_col = graph.col
    g = dgl.graph((new_row, new_col), num_nodes=graph.shape[0])
    return g


def get_graph_new(graph):
    new_row = torch.tensor(list(graph.row), dtype=torch.long)
    new_col = torch.tensor(list(graph.col), dtype=torch.long)
    return dgl.graph((new_row, new_col), num_nodes=graph.shape[0])


class DataHandler:
    def __init__(self):
        self.num_users = 0
        self.num_items = 0
        if args.data == 'douban-movie':
            predir = './Datasets/douban-movie/'
            # uuuuu_graph = sp.load_npz('./Datasets/douban-movie/uuuuu.npz')
            uuuuu_graph = sp.load_npz(predir + 'uuuuu.npz')
            ugu_graph = sp.load_npz(predir + 'ugu_processed.npz')
            mdm_graph = sp.load_npz(predir + 'mdm.npz')
            mam_graph = sp.load_npz(predir + 'mam.npz')
            # uuuuu_graph = get_graph(uuuuu_graph).to(device)
            uuuuu_graph = get_graph(uuuuu_graph)
            ugu_graph = get_graph(ugu_graph)
            mam_graph = get_graph(mam_graph)
            mdm_graph = get_graph(mdm_graph)
            user_mps, item_mps = [uuuuu_graph, ugu_graph], [mam_graph, mdm_graph]

        elif args.data == 'lastfm':
            predir = './Datasets/lastfm/'
            uu_graph = sp.load_npz(predir + 'uu.npz')
            uatau_graph = sp.load_npz(predir + 'uatau_processed.npz')
            aa_graph = sp.load_npz(predir + 'aa.npz')
            ata_graph = sp.load_npz(predir + 'ata_processed.npz')
            uu_graph = get_graph(uu_graph)
            uatau_graph = get_graph(uatau_graph)
            aa_graph = get_graph(aa_graph)
            ata_graph = get_graph(ata_graph)
            user_mps, item_mps = [uu_graph, uatau_graph], [aa_graph, ata_graph]

        elif args.data == 'yelp':
            predir = './Datasets/yelp/'
            uu_graph = sp.load_npz(predir + 'uu.npz')
            ubu_graph = sp.load_npz(predir + 'ubu_processed.npz')
            bcab_graph = sp.load_npz(predir + 'bcab_processed.npz')
            bcib_graph = sp.load_npz(predir + 'bcib_processed.npz')

            uu_graph = get_graph(uu_graph)
            ubu_graph = get_graph(ubu_graph)
            bcab_graph = get_graph(bcab_graph)
            bcib_graph = get_graph(bcib_graph)
            user_mps, item_mps = [uu_graph, ubu_graph], [bcib_graph, bcab_graph]  # , ubu_graph, bcib_graph

        elif args.data == 'douban-book':
            predir = './Datasets/douban-book/'
            uu_graph = sp.load_npz(predir + 'uu.npz')
            ugu_graph = sp.load_npz(predir + 'ugu_processed.npz')
            bab_graph = sp.load_npz(predir + 'bab.npz')
            byb_graph = sp.load_npz(predir + 'byb_processed.npz')
            uu_graph = get_graph(uu_graph)
            ugu_graph = get_graph(ugu_graph)
            bab_graph = get_graph(bab_graph)
            byb_graph = get_graph(byb_graph)
            user_mps, item_mps = [uu_graph, ugu_graph], [bab_graph, byb_graph]  # , ulu_graph, byb_graph

        elif args.data == 'amazon':
            predir = './Datasets/amazon/'
            uibiu_graph = sp.load_npz(predir + 'uibiu_processed.npz')
            uiu_graph = sp.load_npz(predir + 'uiu_processed.npz')
            ibi_graph = sp.load_npz(predir + 'ibi_processed.npz')
            ici_graph = sp.load_npz(predir + 'ici_processed.npz')
            uibiu_graph = get_graph(uibiu_graph)
            uiu_graph = get_graph(uiu_graph)
            ibi_graph = get_graph(ibi_graph)
            ici_graph = get_graph(ici_graph)
            user_mps, item_mps = [uibiu_graph, uiu_graph], [ibi_graph, ici_graph]  #

        elif args.data == 'movielens-1m':
            predir = './Datasets/movielens-1m/'
            uou_graph = sp.load_npz(predir + 'uou_processed.npz')
            umu_graph = sp.load_npz(predir + 'umu_processed.npz')
            mum_graph = sp.load_npz(predir + 'mum_processed.npz')
            mgm_graph = sp.load_npz(predir + 'mgm_processed.npz')

            uou_graph = get_graph(uou_graph)
            umu_graph = get_graph(umu_graph)
            mum_graph = get_graph(mum_graph)
            mgm_graph = get_graph(mgm_graph)
            user_mps, item_mps = [umu_graph, uou_graph], [mgm_graph, mum_graph]

        self.train_path = predir + "train.txt"
        self.test_path = predir + "test.txt"
        self.uu_graph = user_mps
        self.ii_graph = item_mps

    def loadOneFile_yuan(self, filename):
        with open(filename, 'rb') as fs:
            ret = (pickle.load(fs) != 0).astype(np.float32)
        if type(ret) != coo_matrix:
            ret = sp.coo_matrix(ret)
        return ret

    def loadOneFile(self, filename):
        inter_users, inter_items, unique_users = [], [], []
        inter_num = 0
        pos_length = []
        with open(filename, "r") as f:
            line = f.readline()
            while line is not None and line != "":
                temp = line.strip()
                arr = [int(i) for i in temp.split(" ")]
                user_id, pos_id = arr[0], arr[1:]
                unique_users.append(user_id)
                if len(pos_id) < 1:
                    line = f.readline()
                    continue
                self.num_users = max(self.num_users, user_id)
                self.num_items = max(self.num_items, max(pos_id))
                inter_users.extend([user_id] * len(pos_id))
                pos_length.append(len(pos_id))
                inter_items.extend(pos_id)
                inter_num += len(pos_id)
                line = f.readline()
        inter_users = np.array(inter_users)
        inter_items = np.array(inter_items)
        ret = sp.coo_matrix((np.ones_like(inter_users), (inter_users, inter_items)), dtype='float64',
                            shape=(self.num_users + 1, self.num_items + 1))
        return ret

    def normalizeAdj(self, mat):
        degree = np.array(mat.sum(axis=-1))
        dInvSqrt = np.reshape(np.power(degree, -0.5), [-1])
        dInvSqrt[np.isinf(dInvSqrt)] = 0.0
        dInvSqrtMat = sp.diags(dInvSqrt)
        return mat.dot(dInvSqrtMat).transpose().dot(dInvSqrtMat).tocoo()

    def makeTorchAdj(self, mat):
        # make ui adj
        a = sp.csr_matrix((args.user, args.user))
        b = sp.csr_matrix((args.item, args.item))
        mat = sp.vstack([sp.hstack([a, mat]), sp.hstack([mat.transpose(), b])])
        mat = (mat != 0) * 1.0
        mat = (mat + sp.eye(mat.shape[0])) * 1.0
        mat = self.normalizeAdj(mat)

        # make cuda tensor
        idxs = torch.from_numpy(np.vstack([mat.row, mat.col]).astype(np.int64))
        vals = torch.from_numpy(mat.data.astype(np.float32))
        shape = torch.Size(mat.shape)
        return torch.sparse.FloatTensor(idxs, vals, shape).cuda()

    def RelationDictBuild(self):
        relation_dict = {}
        for head in self.kg_dict:
            relation_dict[head] = {}
            for (relation, tail) in self.kg_dict[head]:
                relation_dict[head][tail] = relation
        return relation_dict

    def buildUIMatrix(self, mat):
        idxs = torch.from_numpy(np.vstack([mat.row, mat.col]).astype(np.int64))
        vals = torch.from_numpy(mat.data.astype(np.float32))
        shape = torch.Size(mat.shape)
        return torch.sparse.FloatTensor(idxs, vals, shape).cuda()

    def dgl_to_coo(self, dgl_graph):
        src, dst = dgl_graph.edges()
        num_nodes = dgl_graph.num_nodes()
        matrix = sp.csr_matrix((np.ones_like(src), (src.numpy(), dst.numpy())), dtype='float64',
                               shape=(num_nodes, num_nodes))
        return matrix

    def LoadData(self):
        trnMat = self.loadOneFile(self.train_path)
        tstMat = self.loadOneFile(self.test_path)
        self.trnMat = trnMat

        args.user, args.item = trnMat.shape
        self.torchBiAdj = self.makeTorchAdj(trnMat)

        self.ui_matrix = self.buildUIMatrix(trnMat)

        trnData = TrnData(trnMat)
        self.trnLoader = dataloader.DataLoader(trnData, batch_size=args.batch, shuffle=True, num_workers=0)
        tstData = TstData(tstMat, trnMat)
        self.tstLoader = dataloader.DataLoader(tstData, batch_size=args.tstBat, shuffle=False, num_workers=0)

        print('before uu interaction:', len(self.uu_graph[0].edges()[0]))
        print('before ii interaction:', len(self.ii_graph[0].edges()[0]))

        self.dgl_uu = self.dgl_to_coo(self.uu_graph[0])
        self.dgl_ii = self.dgl_to_coo(self.ii_graph[0])
        self.diffusionDataUU = DiffusionData(torch.FloatTensor(self.dgl_uu.toarray()))
        self.diffusionLoaderUU = dataloader.DataLoader(self.diffusionDataUU, batch_size=args.batch, shuffle=True,
                                                       num_workers=0)
        self.diffusionDataII = DiffusionData(torch.FloatTensor(self.dgl_ii.toarray()))
        self.diffusionLoaderII = dataloader.DataLoader(self.diffusionDataII, batch_size=args.batch, shuffle=True,
                                                       num_workers=0)

        self.dgl_uu1 = self.dgl_to_coo(self.uu_graph[1])
        self.dgl_ii1 = self.dgl_to_coo(self.ii_graph[1])
        self.diffusionDataUU1 = DiffusionData(torch.FloatTensor(self.dgl_uu1.toarray()))
        self.diffusionLoaderUU1 = dataloader.DataLoader(self.diffusionDataUU1, batch_size=args.batch, shuffle=True,
                                                        num_workers=0)
        self.diffusionDataII1 = DiffusionData(torch.FloatTensor(self.dgl_ii1.toarray()))
        self.diffusionLoaderII1 = dataloader.DataLoader(self.diffusionDataII1, batch_size=args.batch, shuffle=True,
                                                        num_workers=0)


class TrnData(data.Dataset):
    def __init__(self, coomat):
        self.rows = coomat.row
        self.cols = coomat.col
        self.dokmat = coomat.todok()
        self.negs = np.zeros(len(self.rows)).astype(np.int32)

    def negSampling(self):
        for i in range(len(self.rows)):
            u = self.rows[i]
            while True:
                iNeg = np.random.randint(args.item)
                if (u, iNeg) not in self.dokmat:
                    break
            self.negs[i] = iNeg

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        return self.rows[idx], self.cols[idx], self.negs[idx]


class TstData(data.Dataset):
    def __init__(self, coomat, trnMat):
        self.csrmat = (trnMat.tocsr() != 0) * 1.0

        tstLocs = [None] * coomat.shape[0]
        tstUsrs = set()
        for i in range(len(coomat.data)):
            row = coomat.row[i]
            col = coomat.col[i]
            if tstLocs[row] is None:
                tstLocs[row] = list()
            tstLocs[row].append(col)
            tstUsrs.add(row)
        tstUsrs = np.array(list(tstUsrs))
        self.tstUsrs = tstUsrs
        self.tstLocs = tstLocs

    def __len__(self):
        return len(self.tstUsrs)

    def __getitem__(self, idx):
        return self.tstUsrs[idx], np.reshape(self.csrmat[self.tstUsrs[idx]].toarray(), [-1])


class DiffusionData(data.Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, index):
        item = self.data[index]
        return item, index

    def __len__(self):
        return len(self.data)