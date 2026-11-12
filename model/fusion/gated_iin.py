import torch
import torch.nn as nn
from timm.models.layers import trunc_normal_

class Gate(nn.Module):
    def __init__(self, embed_dim, return_g=False):
        super().__init__()
        self.return_g = return_g
        self.inter_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.intra_proj = nn.Linear(embed_dim, embed_dim, bias=True)

    def forward(self, intra_f, inter_f, return_g=None):
        if return_g is None:
            return_g = self.return_g
        g = torch.sigmoid(self.inter_proj(inter_f) + self.intra_proj(intra_f))
        fused = g * inter_f + (1.0 - g) * intra_f
        if return_g:
            return fused, g
        return fused

# seq: [cls, wifi tokens, rfid tokens, mmwave tokens]
# len: 1 + 32 + 32 + 32 = 97
def build_mask(len_wifi, len_rfid, len_mmwave, device):
    total_len = 1 + len_wifi + len_rfid + len_mmwave
    # inter-modal attention mask
    inter_mask = torch.zeros((total_len, total_len), device=device)
    inter_mask[0, :] = 0
    inter_mask[:, 0] = 0
    inter_mask[0, 0] = float('-inf')
    inter_mask[1:1+len_wifi, 1:1+len_wifi] = float('-inf')
    inter_mask[1+len_wifi:1+len_wifi+len_rfid, 1+len_wifi:1+len_wifi+len_rfid] = float('-inf')
    inter_mask[1+len_wifi+len_rfid:, 1+len_wifi+len_rfid:] = float('-inf')
    # intra-modal attention mask
    intra_mask = torch.full((total_len, total_len), float('-inf'), device=device)
    intra_mask[0, :] = 0
    intra_mask[:, 0] = 0
    intra_mask[0, 0] = float('-inf')
    intra_mask[1:1+len_wifi, 1:1+len_wifi] = 0
    intra_mask[1+len_wifi:1+len_wifi+len_rfid, 1+len_wifi:1+len_wifi+len_rfid] = 0
    intra_mask[1+len_wifi+len_rfid:, 1+len_wifi+len_rfid:] = 0
    return inter_mask, intra_mask

class linear_projector(nn.Module):
    def __init__(self, input_dim=512, output_dim=256):
        super(linear_projector, self).__init__()
        # B,512,32 -> B,256,32 ->B,256,32
        self.wifi_linear_projection = nn.Sequential(
            nn.Conv1d(input_dim, output_dim, 1),
            nn.BatchNorm1d(output_dim),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU()
        )
        # B,512,5 -> B,256,5 ->B,256,32
        self.rfid_linear_projection = nn.Sequential(
            nn.Conv1d(input_dim, output_dim, 1),
            nn.BatchNorm1d(output_dim),
            nn.ReLU(),
            nn.Linear(5, 32),
            nn.ReLU()
        )
        # B,512,8,4 -> B,256,8,4 ->B,256,32
        self.mmwave_linear_projection = nn.Sequential(
            nn.Flatten(2),
            nn.Conv1d(input_dim, output_dim, 1),
            nn.BatchNorm1d(output_dim),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU()
        )

class GatedIIN(nn.Module):
    def __init__(self, d_l=256, sharing_num=2, return_g=False):
        super().__init__()
        self.d_l = d_l
        self.sharing_num = sharing_num
        self.return_g = return_g
        self.linear_proj = linear_projector(input_dim=512, output_dim=d_l)
        # B,512,32 -> B,256,32 ->B,256,32
        self.proj_wifi = self.linear_proj.wifi_linear_projection
        # B,512,5 -> B,256,5 ->B,256,32
        self.proj_rfid = self.linear_proj.rfid_linear_projection
        # B,512,8,4 -> B,256,8,4 ->B,256,32
        self.proj_mmwave = self.linear_proj.mmwave_linear_projection
        # modality type embedding
        self.mte = nn.Embedding(3, d_l)
        # cls token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_l))
        trunc_normal_(self.cls_token, std=0.02)
        # position embedding, seqlen = 1 + 32 + 32 + 32 = 97
        self.pos_embed = nn.Parameter(torch.zeros(1, 97, d_l))
        trunc_normal_(self.pos_embed, std=0.02)
        # transformer encoder layers
        inter_layer = nn.TransformerEncoderLayer(d_model=d_l, nhead=8, dim_feedforward=d_l*4, dropout=0.1, batch_first=True, norm_first=True, activation='gelu')
        intra_layer = nn.TransformerEncoderLayer(d_model=d_l, nhead=8, dim_feedforward=d_l*4, dropout=0.1, batch_first=True, norm_first=True, activation='gelu')
        self.inter_units = nn.ModuleList([inter_layer for _ in range(sharing_num)])
        self.intra_units = nn.ModuleList([intra_layer for _ in range(sharing_num)])
        self.gate = Gate(d_l, return_g=return_g) # 256
        # 256 -> 128 -> 55
        self.mlp_head = nn.Sequential(
            nn.Linear(d_l, d_l//2, bias=False),
            nn.GELU(),
            nn.Linear(d_l//2, 55, bias=True)
        )
    
    def forward(self, wifi_f, rfid_f, mmwave_f, return_g=None):
        if return_g is None:
            return_g = self.return_g
        B = wifi_f.size(0)
        wifi_proj = self.proj_wifi(wifi_f).transpose(1, 2) # B,32,256
        rfid_proj = self.proj_rfid(rfid_f).transpose(1, 2) # B,32,256
        mmwave_proj = self.proj_mmwave(mmwave_f).flatten(2).transpose(1, 2) # B,32,256
        
        mte_wifi = self.mte(torch.tensor(0, device=wifi_f.device)).unsqueeze(0).unsqueeze(0) # 1,1,256
        mte_rfid = self.mte(torch.tensor(1, device=wifi_f.device)).unsqueeze(0).unsqueeze(0) # 1,1,256
        mte_mmwave = self.mte(torch.tensor(2, device=wifi_f.device)).unsqueeze(0).unsqueeze(0) # 1,1,256
        wifi_embed = wifi_proj + mte_wifi # B,32,256
        rfid_embed = rfid_proj + mte_rfid # B,32,256
        mmwave_embed = mmwave_proj + mte_mmwave # B,32,256

        cls_token = self.cls_token.expand(B, -1, -1) # B,1,256
        x = torch.cat([cls_token, wifi_embed, rfid_embed, mmwave_embed], dim=1) # B,97,256
        x = x + self.pos_embed[:, :x.size(1), :]
        x_inter = x
        x_intra = x

        inter_mask, intra_mask = build_mask(
            len_wifi=wifi_proj.size(1), 
            len_rfid=rfid_proj.size(1), 
            len_mmwave=mmwave_proj.size(1), 
            device=wifi_f.device
        )

        for layer in self.inter_units:
            x_inter = layer(x_inter, src_mask=inter_mask)

        for layer in self.intra_units:
            x_intra = layer(x_intra, src_mask=intra_mask)

        cls_inter = x_inter[:, 0, :] # B,256
        cls_intra = x_intra[:, 0, :] # B,256

        if return_g:
            fused_cls, g = self.gate(cls_intra, cls_inter, return_g=True) # B,256
            logits = self.mlp_head(fused_cls) # B,55
            return logits, g
        fused_cls = self.gate(cls_intra, cls_inter, return_g=False) # B,256
        logits = self.mlp_head(fused_cls) # B,55
        return logits
