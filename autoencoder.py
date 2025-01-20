# autoencoder.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.nn import GINConv, global_add_pool, global_mean_pool, GATv2Conv, GCNConv
from torch_geometric.utils import to_dense_batch


# Decoder
class Decoder(nn.Module):
  def __init__(self, latent_dim, hidden_dim, n_layers, n_nodes):
    super(Decoder, self).__init__()
    self.n_layers = n_layers
    self.n_nodes = n_nodes

    mlp_layers = [nn.Linear(latent_dim,
                            hidden_dim)] + [nn.Linear(hidden_dim,
                                                      hidden_dim) for i in range(n_layers - 2)]
    mlp_layers.append(nn.Linear(hidden_dim, 2 * n_nodes * (n_nodes - 1) // 2))

    self.mlp = nn.ModuleList(mlp_layers)
    self.relu = nn.ReLU()
    self.sigmoid = nn.Sigmoid()

  def forward(self, x):
    for i in range(self.n_layers - 1):
      x = self.relu(self.mlp[i](x))

    x = self.mlp[self.n_layers - 1](x)
    x = torch.reshape(x, (x.size(0), -1, 2))
    x = F.gumbel_softmax(x, tau=1, hard=True)[:, :, 0]

    adj = torch.zeros(x.size(0), self.n_nodes, self.n_nodes, device=x.device)
    idx = torch.triu_indices(self.n_nodes, self.n_nodes, 1)
    adj[:, idx[0], idx[1]] = x
    adj = adj + torch.transpose(adj, 1, 2)
    return adj






class GraphEncoder_1(nn.Module):
  def __init__(self, num_node_features, nout, nhid, graph_hidden_channels):
    super(GraphEncoder_1, self).__init__()
    self.nhid = nhid
    self.nout = nout
    self.relu = nn.ReLU()
    self.ln = nn.LayerNorm((nout))
    self.conv1 = GCNConv(num_node_features, graph_hidden_channels)
    self.conv2 = GCNConv(graph_hidden_channels, graph_hidden_channels)
    self.conv3 = GCNConv(graph_hidden_channels, graph_hidden_channels)
    self.conv4 = GCNConv(graph_hidden_channels, graph_hidden_channels)
    self.conv5 = GCNConv(graph_hidden_channels, graph_hidden_channels)
    self.mol_hidden1 = nn.Linear(graph_hidden_channels, nhid)
    self.mol_hidden2 = nn.Linear(nhid, nout)
    self.dropout = nn.Dropout(0.05)

  def forward(self, graph_batch):
    x = graph_batch.x
    edge_index = graph_batch.edge_index
    batch = graph_batch.batch
    x = self.dropout(x)
    x = self.conv1(x, edge_index).relu()
    x = self.conv2(x, edge_index).relu()
    x = self.conv3(x, edge_index).relu()
    x = self.conv4(x, edge_index).relu() - x
    x = self.conv5(x, edge_index).relu() - x
    x = global_mean_pool(x, batch)
    x = x / x.norm(dim=1, keepdim=True)
    x = self.mol_hidden1(x).relu()
    x = x / x.norm(dim=1, keepdim=True)
    x = self.mol_hidden2(x)
    x = x / x.norm(dim=1, keepdim=True)
    return x

# Variational Autoencoder


class VariationalAutoEncoder(nn.Module):
  def __init__(self, input_dim, hidden_dim_enc, hidden_dim_dec, latent_dim, n_layers_enc, n_layers_dec, n_max_nodes, text_emb_dim=512):  # Added text_emb_dim
    super(VariationalAutoEncoder, self).__init__()
    self.n_max_nodes = n_max_nodes
    self.input_dim = input_dim
    self.encoder = GraphEncoder_1(
        input_dim,
        hidden_dim_enc,
        hidden_dim_enc,
        graph_hidden_channels=200)
    self.fc_mu = nn.Linear(hidden_dim_enc, latent_dim)
    self.fc_logvar = nn.Linear(hidden_dim_enc, latent_dim)
    self.decoder = Decoder(
        256,
        hidden_dim_dec,
        n_layers_dec,
        n_max_nodes)  # Added text_emb_dim

  def forward(self, data):
    x_g = self.encoder(data)
    mu = self.fc_mu(x_g)
    logvar = self.fc_logvar(x_g)
    x_g = self.reparameterize(mu, logvar)
    text_emb = data.stats.view(mu.size(0), 512)
    x_g = torch.cat((x_g, text_emb), dim=1)  # Concatenate text embeddings
    adj = self.decoder(x_g)
    return adj

  def encode(self, data):
    x_g = self.encoder(data)
    mu = self.fc_mu(x_g)
    logvar = self.fc_logvar(x_g)
    x_g = self.reparameterize(mu, logvar)

    return x_g, mu, logvar

  def reparameterize(self, mu, logvar, eps_scale=1.):
    if self.training:
      std = logvar.mul(0.5).exp_()
      eps = torch.randn_like(std) * eps_scale
      return eps.mul(std).add_(mu)
    else:
      return mu

  def decode(self, mu, logvar, text_emb):
    x_g = self.reparameterize(mu, logvar)
    x_g = torch.cat((x_g, text_emb), dim=1)  # Concatenate text embeddings
    adj = self.decoder(x_g)
    return adj

  def decode_mu(self, mu, text_emb):
    text_emb = text_emb.view(mu.size(0), 512)
    x_g = torch.cat((mu, text_emb), dim=1)  # Concatenate text embeddings
    adj = self.decoder(x_g)
    return adj

  def loss_function(self, data, beta=0.05):
    x_g, mu, logvar = self.encode(data)

    text_emb = data.stats
    text_emb = text_emb.view(len(data), 512)

    x_g_text = torch.cat((x_g, text_emb), dim=1)
    adj = self.decoder(x_g_text)

    recon = F.l1_loss(adj, data.A, reduction='mean')
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    loss = recon + beta * kld

    return loss, recon, kld
