import torch
import torch.nn.functional as F
from torch import nn
import math
import numpy as np
from matplotlib import pyplot as plt


PAD = 0
UNK = 1
BOS = 2
EOS = 3

PAD_WORD = '<blank>'
UNK_WORD = '<unk>'
BOS_WORD = '<s>'
EOS_WORD = '</s>'


class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, query, key, value, mask=None):
        query = query.cuda()
        key = key.cuda()
        value = value.cuda()
        L, S = query.size(-2), key.size(-2)
        scale_factor = 1 / math.sqrt(query.size(-1))
        attn_bias = torch.zeros(L, S, dtype=query.dtype).cuda()
        if mask is not None:
            if mask.dtype == torch.bool:
                mask.masked_fill_(mask.logical_not(), float("-inf"))
            else:
                attn_bias += mask
        attn_weight = query @ key.transpose(-2, -1) * scale_factor
        attn_weight = attn_weight + attn_bias
        attn_weight = torch.softmax(attn_weight, dim=-1)
        return attn_weight @ value, attn_weight


class MultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v
        self.d_model = d_model

        self.w_qs = nn.Linear(d_model, n_head * d_k)
        self.w_ks = nn.Linear(d_model, n_head * d_k)
        self.w_vs = nn.Linear(d_model, n_head * d_v)

        self.attention = ScaledDotProductAttention(
            temperature=d_k**0.5)
        self.layer_norm = nn.LayerNorm(d_model)

        self.fc = nn.Linear(n_head * d_v, d_model)
        nn.init.xavier_normal_(self.fc.weight)

        self.dropout = nn.Dropout(dropout)

        self.reset_parameters()

    def reset_parameters(self):
        # normal distribution initialization better than kaiming(default in pytorch)
        nn.init.normal_(self.w_qs.weight, mean=0,
                        std=np.sqrt(2.0 / (self.d_model + self.d_k)))
        nn.init.normal_(self.w_ks.weight, mean=0,
                        std=np.sqrt(2.0 / (self.d_model + self.d_k)))
        nn.init.normal_(self.w_vs.weight, mean=0,
                        std=np.sqrt(2.0 / (self.d_model + self.d_v)))

    def forward(self, q, k, v, mask=None):
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head

        sz_b, len_q, _ = q.size()
        sz_b, len_k, _ = k.size()
        sz_b, len_v, _ = v.size()

        residual = q

        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        q = q.permute(2, 0, 1, 3).contiguous().view(-1,
                                                    len_q, d_k)  # (n*b) x lq x dk
        k = k.permute(2, 0, 1, 3).contiguous().view(-1,
                                                    len_k, d_k)  # (n*b) x lk x dk
        v = v.permute(2, 0, 1, 3).contiguous().view(-1,
                                                    len_v, d_v)  # (n*b) x lv x dv

        if mask is not None:
            mask = mask.repeat(n_head, 1, 1)  # (n*b) x .. x ..
        output, attn = self.attention(q, k, v, mask=mask)

        output = output.view(n_head, sz_b, len_q, d_v)
        output = output.permute(1, 2, 0, 3).contiguous().view(
            sz_b, len_q, -1)  # b x lq x (n*dv)

        output = self.dropout(self.fc(output))
        output = self.layer_norm(output + residual)

        return output, attn


class PositionwiseFeedForward(nn.Module):
    ''' A two-feed-forward-layer module '''

    def __init__(self, d_in, d_hid, dropout=0.1, fft_conv1d_kernel=(9, 1), fft_conv1d_padding=(4, 0)):
        super().__init__()

        # Use Conv1D
        # position-wise
        self.w_1 = nn.Conv1d(
            d_in, d_hid, kernel_size=fft_conv1d_kernel[0], padding=fft_conv1d_padding[0])
        # position-wise
        self.w_2 = nn.Conv1d(
            d_hid, d_in, kernel_size=fft_conv1d_kernel[1], padding=fft_conv1d_padding[1])

        self.layer_norm = nn.LayerNorm(d_in)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        output = x.transpose(1, 2)
        output = self.w_2(F.relu(self.w_1(output)))
        output = output.transpose(1, 2)
        output = self.dropout(output)
        output = self.layer_norm(output + residual)

        return output


class FFTBlock(torch.nn.Module):
    """FFT Block"""

    def __init__(self,
                 d_model,
                 d_inner,
                 n_head,
                 d_k,
                 d_v,
                 dropout=0.1):
        super(FFTBlock, self).__init__()
        self.slf_attn = MultiHeadAttention(
            n_head, d_model, d_k, d_v, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(
            d_model, d_inner, dropout=dropout)

    def forward(self, enc_input, non_pad_mask=None, slf_attn_mask=None):
        enc_output, enc_slf_attn = self.slf_attn(
            enc_input, enc_input, enc_input, mask=slf_attn_mask)

        if non_pad_mask is not None:
            enc_output *= non_pad_mask

        enc_output = self.pos_ffn(enc_output)

        if non_pad_mask is not None:
            enc_output *= non_pad_mask

        return enc_output, enc_slf_attn


def create_alignment(base_mat, duration_predictor_output):
    N, L = duration_predictor_output.shape
    for i in range(N):
        count = 0
        for j in range(L):
            for k in range(duration_predictor_output[i][j]):
                base_mat[i][count+k][j] = 1
            count = count + duration_predictor_output[i][j]
    return base_mat


class Transpose(nn.Module):
    def __init__(self, dim_1, dim_2):
        super().__init__()
        self.dim_1 = dim_1
        self.dim_2 = dim_2

    def forward(self, x):
        return x.transpose(self.dim_1, self.dim_2)


class VariancePredictor(nn.Module):
    """ Variance Predictor """

    def __init__(self, encoder_dim, duration_predictor_filter_size, duration_predictor_kernel_size, dropout):
        super(VariancePredictor, self).__init__()

        self.input_size = encoder_dim
        self.filter_size = duration_predictor_filter_size
        self.kernel = duration_predictor_kernel_size
        self.conv_output_size = duration_predictor_filter_size
        self.dropout = dropout

        self.conv_net = nn.Sequential(
            Transpose(-1, -2),
            nn.Conv1d(
                self.input_size, self.filter_size,
                kernel_size=self.kernel, padding=1
            ),
            Transpose(-1, -2),
            nn.LayerNorm(self.filter_size),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            Transpose(-1, -2),
            nn.Conv1d(
                self.filter_size, self.filter_size,
                kernel_size=self.kernel, padding=1
            ),
            Transpose(-1, -2),
            nn.LayerNorm(self.filter_size),
            nn.ReLU(),
            nn.Dropout(self.dropout)
        )

        self.linear_layer = nn.Linear(self.conv_output_size, 1)
        self.relu = nn.ReLU()

    def forward(self, encoder_output, mask=None):
        encoder_output = self.conv_net(encoder_output)
        out = self.linear_layer(encoder_output)
        out = self.relu(out)
        out = out.squeeze()
        if not self.training:
            out = out.unsqueeze(0)
        if mask is not None:
            out = out * mask
        return out


class LengthRegulator(nn.Module):
    """ Length Regulator """

    def __init__(self, encoder_dim, duration_predictor_filter_size, duration_predictor_kernel_size, dropout):
        super(LengthRegulator, self).__init__()
        self.duration_predictor = VariancePredictor(
            encoder_dim, duration_predictor_filter_size, duration_predictor_kernel_size, dropout)

    def LR(self, x, duration_predictor_output, mel_max_length=None):
        expand_max_len = torch.max(
            torch.sum(duration_predictor_output, -1), -1)[0]
        alignment = torch.zeros(duration_predictor_output.size(0),
                                expand_max_len,
                                duration_predictor_output.size(1)).numpy()
        alignment = create_alignment(alignment,
                                     duration_predictor_output.cpu().numpy())
        alignment = torch.from_numpy(alignment).to(x.device)

        output = alignment @ x
        if mel_max_length is not None:
            output = F.pad(
                output, (0, 0, 0, mel_max_length-output.size(1), 0, 0))
        return output

    def forward(self, x, alpha=1.0, target=None, mel_max_length=None):
        preds_duration = self.duration_predictor(x)
        if self.training:
            assert target is not None
            return self.LR(x, (target * alpha).int(), mel_max_length), preds_duration
        else:
            return self.LR(x, (preds_duration * alpha).int(), mel_max_length), preds_duration


class PitchEncoder(nn.Module):
    def __init__(self, encoder_dim, duration_predictor_filter_size, duration_predictor_kernel_size,
                 dropout, n_bins=256, pitch_min=1e-8, pitch_max=900.):
        super(PitchEncoder, self).__init__()
        self.pitch_predictor = VariancePredictor(
            encoder_dim, duration_predictor_filter_size, duration_predictor_kernel_size, dropout)
        self.pitch_bins = nn.Parameter(
            torch.exp(torch.linspace(np.log(pitch_min),
                      np.log(pitch_max), n_bins - 1)),
            requires_grad=False,
        )
        self.pitch_embedding = nn.Embedding(n_bins, encoder_dim)
        # self.pitch_embedding.weight.data.zero_()

    def forward(self, x, mask=None, c_pitch=1.0, target=None):
        pred_pitch = self.pitch_predictor(x, mask=mask)

        if self.training:
            assert target is not None
            pitch_embed = self.pitch_embedding(
                torch.bucketize(target, self.pitch_bins))
        else:
            pred_pitch = c_pitch * pred_pitch
            pitch_embed = self.pitch_embedding(
                torch.bucketize(pred_pitch, self.pitch_bins))
        return pitch_embed, pred_pitch


class EnergyEncoder(nn.Module):
    def __init__(self, encoder_dim, duration_predictor_filter_size, duration_predictor_kernel_size,
                 dropout, n_bins=256, energy_min=1e-8, energy_max=900.):
        super(EnergyEncoder, self).__init__()
        self.energy_predictor = VariancePredictor(
            encoder_dim, duration_predictor_filter_size, duration_predictor_kernel_size, dropout)
        self.energy_bins = nn.Parameter(
            torch.exp(torch.linspace(
                np.log(energy_min), np.log(energy_max), n_bins - 1)),
            requires_grad=False,
        )
        self.energy_embedding = nn.Embedding(n_bins, encoder_dim)
        # self.energy_embedding.weight.data.zero_()

    def forward(self, x,  mask=None,  c_energy=1.0, target=None):
        pred_energy = self.energy_predictor(x, mask=mask)
        if self.training:
            assert target is not None
            energy_embed = self.energy_embedding(
                torch.bucketize(target, self.energy_bins))
        else:
            pred_energy = c_energy * pred_energy
            energy_embed = self.energy_embedding(
                torch.bucketize(pred_energy, self.energy_bins))
        return energy_embed, pred_energy


class VarianceAdapter(nn.Module):

    def __init__(self, encoder_dim, duration_predictor_filter_size, duration_predictor_kernel_size, dropout):
        super(VarianceAdapter, self).__init__()
        self.length_regulator = LengthRegulator(
            encoder_dim, duration_predictor_filter_size, duration_predictor_kernel_size, dropout)
        self.pitch_encoder = PitchEncoder(
            encoder_dim, duration_predictor_filter_size, duration_predictor_kernel_size, dropout)
        self.energy_encoder = EnergyEncoder(
            encoder_dim, duration_predictor_filter_size, duration_predictor_kernel_size, dropout)

    def forward(self, x, mask, alpha=1.0, c_pitch=1.0, c_energy=1.0, length_target=None, pitch_target=None, energy_target=None, mel_max_length=None):
        x, pred_duration = self.length_regulator(
            x, alpha=alpha, target=length_target, mel_max_length=mel_max_length)
        pitch_embed, pred_pitch = self.pitch_encoder(
            x, mask=mask, c_pitch=c_pitch, target=pitch_target)
        x = x + pitch_embed
        energy_embed, pred_energy = self.energy_encoder(
            x, mask=mask, c_energy=c_energy, target=energy_target)
        x = x + energy_embed
        return (x, pred_duration, pred_pitch, pred_energy)


def get_non_pad_mask(seq):
    assert seq.dim() == 2
    return seq.ne(PAD).type(torch.float).unsqueeze(-1)


def get_attn_key_pad_mask(seq_k, seq_q):
    ''' For masking out the padding part of key sequence. '''

    len_q = seq_q.size(1)
    padding_mask = seq_k.eq(PAD)
    padding_mask = padding_mask.unsqueeze(
        1).expand(-1, len_q, -1)  # b x lq x lk

    return padding_mask


def get_mask_from_lengths(lengths, max_len=None):
    if max_len == None:
        max_len = torch.max(lengths).item()

    ids = torch.arange(0, max_len, 1, device=lengths.device)
    mask = (ids < lengths.unsqueeze(1)).bool()

    return mask


class Encoder(nn.Module):
    def __init__(self, vocab_size, max_seq_len, encoder_dim, encoder_n_layer, encoder_head, encoder_conv1d_filter_size, dropout):
        super(Encoder, self).__init__()

        len_max_seq = max_seq_len
        n_position = len_max_seq + 1
        n_layers = encoder_n_layer

        self.src_word_emb = nn.Embedding(
            vocab_size,
            encoder_dim,
            padding_idx=PAD
        )

        self.position_enc = nn.Embedding(
            n_position,
            encoder_dim,
            padding_idx=PAD
        )

        self.layer_stack = nn.ModuleList([FFTBlock(
            encoder_dim,
            encoder_conv1d_filter_size,
            encoder_head,
            encoder_dim // encoder_head,
            encoder_dim // encoder_head,
            dropout=dropout
        ) for _ in range(n_layers)])

    def forward(self, src_seq, src_pos, return_attns=False):

        enc_slf_attn_list = []

        # -- Prepare masks
        slf_attn_mask = get_attn_key_pad_mask(seq_k=src_seq, seq_q=src_seq)
        non_pad_mask = get_non_pad_mask(src_seq)

        # -- Forward
        enc_output = self.src_word_emb(src_seq) + self.position_enc(src_pos)

        for enc_layer in self.layer_stack:
            enc_output, enc_slf_attn = enc_layer(
                enc_output,
                non_pad_mask=non_pad_mask,
                slf_attn_mask=slf_attn_mask)
            if return_attns:
                enc_slf_attn_list += [enc_slf_attn]

        return enc_output, non_pad_mask


class Decoder(nn.Module):
    """ Decoder """

    def __init__(self, max_seq_len, decoder_dim, decoder_n_layer, decoder_head, decoder_conv1d_filter_size, dropout):

        super(Decoder, self).__init__()

        len_max_seq = max_seq_len
        n_position = len_max_seq + 1
        n_layers = decoder_n_layer

        self.position_enc = nn.Embedding(
            n_position,
            decoder_dim,
            padding_idx=PAD,
        )

        self.layer_stack = nn.ModuleList([FFTBlock(
            decoder_dim,
            decoder_conv1d_filter_size,
            decoder_head,
            decoder_dim // decoder_head,
            decoder_dim // decoder_head,
            dropout=dropout
        ) for _ in range(n_layers)])

    def forward(self, enc_seq, enc_pos, return_attns=False):

        dec_slf_attn_list = []

        # -- Prepare masks
        slf_attn_mask = get_attn_key_pad_mask(seq_k=enc_pos, seq_q=enc_pos)
        non_pad_mask = get_non_pad_mask(enc_pos)

        # -- Forward
        dec_output = enc_seq + self.position_enc(enc_pos)

        for dec_layer in self.layer_stack:
            dec_output, dec_slf_attn = dec_layer(
                dec_output,
                non_pad_mask=non_pad_mask,
                slf_attn_mask=slf_attn_mask)
            if return_attns:
                dec_slf_attn_list += [dec_slf_attn]

        return dec_output


class FastSpeech(nn.Module):
    """ FastSpeech """

    def __init__(self,
                 vocab_size=300,
                 max_seq_len=3000,
                 encoder_dim=256,
                 encoder_n_layer=4,
                 encoder_head=2,
                 encoder_conv1d_filter_size=1024,
                 decoder_dim=256,
                 decoder_n_layer=4,
                 decoder_head=2,
                 decoder_conv1d_filter_size=1024,
                 duration_predictor_filter_size=256,
                 duration_predictor_kernel_size=3,
                 dropout=0.1,
                 num_mels=80):
        super(FastSpeech, self).__init__()

        self.encoder = Encoder(vocab_size, max_seq_len,
                               encoder_dim, encoder_n_layer, encoder_head, encoder_conv1d_filter_size, dropout)
        self.var_adapter = VarianceAdapter(
            encoder_dim, duration_predictor_filter_size, duration_predictor_kernel_size, dropout)
        self.decoder = Decoder(max_seq_len, decoder_dim, decoder_n_layer,
                               decoder_head, decoder_conv1d_filter_size, dropout)

        self.mel_linear = nn.Linear(decoder_dim, num_mels)

    def mask_tensor(self, mel_output, position, mel_max_length):
        lengths = torch.max(position, -1)[0]
        mask = ~get_mask_from_lengths(lengths, max_len=mel_max_length)
        mask = mask.unsqueeze(-1).expand(-1, -1, mel_output.size(-1))
        return mel_output.masked_fill(mask, 0.)

    def forward(self, batch, alpha=1.0, c_pitch=1.0, c_energy=1.0):
        x, mask = self.encoder(batch["text"], batch["src_pos"])
        mask = get_non_pad_mask(batch["mel_pos"])[:,:,0]
        x, pred_duration, pred_pitch, pred_energy = self.var_adapter(
            x, mask, alpha=alpha, c_pitch=c_pitch, c_energy=c_energy,
            length_target=batch["duration"],
            pitch_target=batch["pitch"],
            energy_target=batch["energy"],
            mel_max_length=batch["mel_max_len"])

        if batch["mel_pos"] is None:
            batch["mel_pos"] = torch.from_numpy(
                np.arange(1, x.shape[-2]+1)).unsqueeze(0).to(x.device)

        x = self.decoder(x, batch["mel_pos"])
        x = self.mask_tensor(x, batch["mel_pos"], batch["mel_max_len"])
        mel_output = self.mel_linear(x)

        return {"mel_output": mel_output,
                "duration_predictor_output": pred_duration,
                "pitch_predictor_output": pred_pitch,
                "energy_predictor_output": pred_energy}
