"""Small standard PyTorch sequence encoders for actual physiological windows."""
import torch
from torch import nn
class SequenceRegressor(nn.Module):
    def __init__(self,input_dim,kind='gru',hidden=48,layers=2,output_dim=1):
        super().__init__()
        self.kind=kind
        if kind in ('gru','lstm'):
            cls=nn.GRU if kind=='gru' else nn.LSTM
            self.encoder=cls(input_dim,hidden,num_layers=layers,batch_first=True,dropout=.1 if layers>1 else 0)
        elif kind=='transformer':
            self.embed=nn.Linear(input_dim,hidden)
            self.encoder=nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=hidden,nhead=4,dim_feedforward=hidden*2,dropout=.1,batch_first=True),num_layers=layers)
        else:raise ValueError('kind must be gru/lstm/transformer')
        self.head=nn.Sequential(nn.LayerNorm(hidden),nn.Linear(hidden,hidden//2),nn.ReLU(),nn.Linear(hidden//2,output_dim))
    def forward(self,x,padding_mask=None):
        if self.kind in ('gru','lstm'):
            out,_=self.encoder(x);represent=out[:,-1,:]
        else:
            out=self.encoder(self.embed(x),src_key_padding_mask=padding_mask)
            represent=out[:,-1,:]
        return self.head(represent)
