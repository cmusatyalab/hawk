from torch import nn, Tensor


class BackboneEncoder(nn.Module):

    def __init__(self, embed_dim: int):
        super(BackboneEncoder, self).__init__()
        self._embed_dim: int = embed_dim

    def forward(self, X: Tensor) -> Tensor:
        """
        :param X: (B, N, C, H, W)
        """
        pass

    @property
    def embed_dim(self):
        return self._embed_dim

    @embed_dim.setter
    def embed_dim(self, embed_dim):
        self._set_embed_dim(embed_dim)
        self._embed_dim = embed_dim

    def _set_embed_dim(self, embed_dim: int):
        pass
