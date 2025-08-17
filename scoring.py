import torch
import torch.nn.functional as F

class Scorer:
    def score(self, entity_emb, rel_emb, triples):
        """
        Args:
            entity_emb: [num_entities, emb_dim] Tensor
            rel_emb: [num_relations, emb_dim] Tensor
            triples: [batch_size, 3] Tensor of (head, relation, tail)
        Returns:
            scores: [batch_size] Tensor
        """
        raise NotImplementedError

class TransEScorer(Scorer):
    def __init__(self, p=2):
        self.p = p  # L1 or L2

    def score(self, entity_emb, rel_emb, triples):
        h = entity_emb[triples[:, 0]]
        r = rel_emb[triples[:, 1]]
        t = entity_emb[triples[:, 2]]
        return -torch.norm(h + r - t, p=self.p, dim=1)  # negative distance

class DistMultScorer(Scorer):
    def score(self, entity_emb, rel_emb, triples):
        h = entity_emb[triples[:, 0]]
        r = rel_emb[triples[:, 1]]
        t = entity_emb[triples[:, 2]]
        return torch.sum(h * r * t, dim=1)
