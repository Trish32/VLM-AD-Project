# Faithful port of official QCNet modules/qcnet_encoder.py (HeteroData -> dict).
from typing import Dict, Optional

import torch
import torch.nn as nn

from modules.qcnet_agent_encoder import QCNetAgentEncoder
from modules.qcnet_map_encoder import QCNetMapEncoder


class QCNetEncoder(nn.Module):

    def __init__(self,
                 dataset: str,
                 input_dim: int,
                 hidden_dim: int,
                 num_historical_steps: int,
                 pl2pl_radius: float,
                 time_span: Optional[int],
                 pl2a_radius: float,
                 a2a_radius: float,
                 num_freq_bands: int,
                 num_map_layers: int,
                 num_agent_layers: int,
                 num_heads: int,
                 head_dim: int,
                 dropout: float) -> None:
        super(QCNetEncoder, self).__init__()
        self.map_encoder = QCNetMapEncoder(
            dataset=dataset,
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_historical_steps=num_historical_steps,
            pl2pl_radius=pl2pl_radius,
            num_freq_bands=num_freq_bands,
            num_layers=num_map_layers,
            num_heads=num_heads,
            head_dim=head_dim,
            dropout=dropout,
        )
        self.agent_encoder = QCNetAgentEncoder(
            dataset=dataset,
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_historical_steps=num_historical_steps,
            time_span=time_span,
            pl2a_radius=pl2a_radius,
            a2a_radius=a2a_radius,
            num_freq_bands=num_freq_bands,
            num_layers=num_agent_layers,
            num_heads=num_heads,
            head_dim=head_dim,
            dropout=dropout,
        )

    def forward(self, data: Dict) -> Dict[str, torch.Tensor]:
        # encode the static map first, then the agents (which attend to the map tokens).
        # Returns {'x_pt':[Pt,D], 'x_pl':[Pl,T,D], 'x_a':[A,T,D]} consumed by the decoder.
        map_enc = self.map_encoder(data)
        agent_enc = self.agent_encoder(data, map_enc)
        return {**map_enc, **agent_enc}
