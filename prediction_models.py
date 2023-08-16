#!/usr/bin/env python3

import torch
from torch.nn import Linear, Dropout
from torch_geometric.nn import HeteroConv
from torch_geometric.nn import SAGEConv, GATConv, GATv2Conv
from torch_geometric.nn import GCNConv, GCN2Conv, ChebConv
from copy import copy

# dictionary of convolutions valid for heterogenous edges
hetero_edge_conv_map = {
	'SAGE': lambda chans: SAGEConv((-1, -1), chans),
	'GAT': lambda chans: GATConv((-1, -1), chans, add_self_loops=False),
	'GATv2': lambda chans: GATv2Conv((-1, -1), chans, add_self_loops=False),
}

# dictionary of convolutions valid for homogenous edges
homo_edge_conv_map = copy(hetero_edge_conv_map)
homo_edge_conv_map.update({
	'GCN': lambda chans: GCNConv(-1, chans, add_self_loops=False),
	'Cheb': lambda chans: ChebConv(-1, chans, K=2),
})

class HeteroGNN(torch.nn.Module):
	# nlayers: number of convolutional layers of types hetero_edge_conv and homo_edge_conv
	# hidden_channels: number of hidden channels
	# metadata: the infromation about types of graph edges
	# hetero_edge_conv: convolution to be used on heterogeneous edges
	# homo_edge_conv: convolution to be used on homogeneous edges
	def __init__(self, nlayers, hidden_channels, metadata, hetero_edge_conv, homo_edge_conv=None):
		super().__init__()

		self.convs = torch.nn.ModuleList()

		# list of heterogenous edge types (input node type is different from output node type)
		hetero_edge_types = [ edge_type for edge_type in metadata[1] if edge_type[0] != edge_type[2] ]

		# list of homogenous edge types (if homogenous edge convolution is enabled)
		if homo_edge_conv is not None:
			homo_edge_types = [ edge_type for edge_type in metadata[1] if edge_type[0] == edge_type[2] ]
		else:
			homo_edge_types = []

		for i in range(nlayers):
			convs = { edge_type: hetero_edge_conv(hidden_channels) for edge_type in hetero_edge_types }
			convs.update({ edge_type: homo_edge_conv(hidden_channels) for edge_type in homo_edge_types })

			self.convs.append(HeteroConv(convs))

# GAE style encoder
class HeteroEncoder(HeteroGNN):
	def forward(self, x_dict, edge_index_dict):
		nlayers = len(self.convs)

		# forward throughthe heterogenous layers
		for i, conv in zip(range(nlayers), self.convs):
			x_dict = conv(x_dict, edge_index_dict)
			if i < nlayers - 1:
				x_dict = { k: x.relu() for k, x in x_dict.items() }

		return x_dict

# GAE style decoder
class EdgeDecoder(torch.nn.Module):
	def __init__(self, hidden_channels):
		super().__init__()
		self.lin1 = Linear(2 * hidden_channels, hidden_channels)
		self.lin2 = Linear(hidden_channels, 1)

	def forward(self, z_dict, edge_label_index):
		row, col = edge_label_index
		z1 = torch.cat([z_dict['author'][row], z_dict['hotel'][col]], dim=-1)

		z2 = self.lin1(z1).relu()
		z3 = self.lin2(z2)
		return z3.view(-1), (z1, z2)

# link label - review rating predictor
class LinkLabelPredModel(torch.nn.Module):
	def __init__(self, nlayers, hidden_channels, metadata, hetero_edge_conv, homo_edge_conv=None):
		super().__init__()
		self.encoder = HeteroEncoder(nlayers, hidden_channels, metadata,
					     hetero_edge_conv_map[hetero_edge_conv],
					     homo_edge_conv_map[homo_edge_conv] if homo_edge_conv is not None else None)
		self.decoder = EdgeDecoder(hidden_channels)

	def forward(self, x_dict, edge_index_dict, edge_label_index):
		z_dict = self.encoder(x_dict, edge_index_dict)
		return self.decoder(z_dict, edge_label_index)

# node label - hotel class predictor
class NodeLabelPredModel(HeteroGNN):
	def __init__(self, out_node_type, nlayers, hidden_channels, out_channels,
		     metadata, hetero_edge_conv, homo_edge_conv):
		super().__init__(nlayers, hidden_channels, metadata,
				 hetero_edge_conv_map[hetero_edge_conv],
				 homo_edge_conv_map[homo_edge_conv])

		self.out_node_type = out_node_type
		self.lin = Linear(hidden_channels, out_channels)

	def forward(self, x_dict, edge_index_dict):
		nlayers = len(self.convs)

		# forward through the layers and remember embeddings in case
		# we want to use them
		embeddings = []
		for i, conv in zip(range(nlayers), self.convs):
			x_dict = conv(x_dict, edge_index_dict)
			embeddings.append(x_dict)
			if i < nlayers - 1:
				x_dict = { k: x.relu() for k, x in x_dict.items() }

		return self.lin(x_dict[self.out_node_type]), embeddings

"""
class SAGE_VGAEEncoder(torch.nn.Module):
	def __init__(self, hidden_channels, out_channels		super().__init__()
		self._mu_encoder = SAGEEncoder(nlayers, hidden_channels, out_channels)
		self._sigma_encoder = SAGEEncoder(nlayers, hidden_channels, out_channels)
		self.conv1_m = SAGEConv((-1, -1), hidden_channels)
		self.conv2_m = SAGEConv((-1, -1), out_channels)
		self.conv1_s = SAGEConv((-1, -1), hidden_channels)
		self.conv2_s = SAGEConv((-1, -1), out_channels)

	def forward(self, x, edge_index):
		mu = self._mu_encoder(x,
		m = self.conv1_m(x, edge_index).relu()
		m = self.conv2_m(m, edge_index)
		s = self.conv1_s(x, edge_index).relu()
		s = self.conv2_s(s, edge_index)
		return m, s

class SAGE_VGAE_Model(VGAE):
	def __init__(self, hidden_channels):
		
		encoder = GNNEncoder(hidden_channels, hidden_channels)
		encoder = to_hetero(encoder, data.metadata(), aggr='sum')
		super().__init__(encoder, EdgeDecoder(hidden_channels))

	def forward(self, x_dict, edge_index_dict, edge_label_index):
		m, s = self.encoder(x_dict, edge_index_dict)
		s['hotel'] = s['hotel'].clamp(max=10)
		s['author'] = s['author'].clamp(max=10)
		self.__m__ = m
		self.__s__ = s
		z_dict = {}
		z_dict['hotel'] = self.reparametrize(m['hotel'], s['hotel'])
		z_dict['author'] = self.reparametrize(m['author'], s['author'])
		#z_dict = super().encode(x_dict, edge_index_dict)
		return self.decoder(z_dict, edge_label_index)
<------>loss += (1 / train_data['hotel'].num_nodes) * model.kl_loss(model.__m__['hotel'], model.__s__['hotel'])
<------>loss += (1 / train_data['author'].num_nodes) * model.kl_loss(model.__m__['author'], model.__s__['author'])
"""
