import torch
from torch import nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_length, n=10000):
        super(PositionalEncoding, self).__init__()

        assert d_model % 2 == 0, 'Due to implementation limitations, please keep the value of d_model even'
        self.positional_encodings = torch.zeros(max_seq_length, d_model)  # max_seq_length × d_model

        for pos in torch.arange(0, max_seq_length, dtype=torch.int):
            i = torch.arange(0, d_model // 2)
            self.positional_encodings[pos, 0::2] = torch.sin(pos / n ** (2 * i / d_model))
            self.positional_encodings[pos, 1::2] = torch.cos(pos / n ** (2 * i / d_model))

        self.register_buffer('pe', self.positional_encodings)

    def forward(self, x):
        # Input(s)
        x  # batch_size × seq_length × d_model

        # Operation(s)
        batch_size, seq_length, d_model = x.size()
        positional_encoding_output = x + self.positional_encodings[seq_length, :]  # batch_size × seq_length × d_model

        # Output(s)
        positional_encoding_output  # batch_size × seq_length × d_model
        return positional_encoding_output


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()

        assert d_model % num_heads == 0, 'Since d_model is split across attention heads, d_model should be divisible by num_heads'

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_q = self.d_k = self.d_v = d_model // num_heads

        self.W_q = nn.Linear(in_features=self.d_model, out_features=self.d_model)
        self.W_k = nn.Linear(in_features=self.d_model, out_features=self.d_model)
        self.W_v = nn.Linear(in_features=self.d_model, out_features=self.d_model)
        self.W_o = nn.Linear(in_features=self.d_model, out_features=self.d_model)

    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        # Input(s)
        Q  # batch_size × seq_length × num_heads × d_k
        K  # batch_size × seq_length × num_heads × d_k
        V  # batch_size × seq_length × num_heads × d_k
        mask  # seq_length × seq_length

        # Operation(s)
        Q = Q.permute(0, 2, 1, 3)  # batch_size × num_heads × seq_length × d_k
        K = K.permute(0, 2, 3, 1)  # batch_size × num_heads × d_k × seq_length

        attention_scores = torch.matmul(Q, K) / (self.d_k ** 0.5)  # seq_length × seq_length
        if mask is not None:
            attention_scores = attention_scores.masked_fill(~mask, value=-1e15)  # seq_length × seq_length
        attention_probabilities = torch.softmax(attention_scores, dim=-1)  # seq_length × seq_length

        V = V.permute(0, 2, 1, 3)  # batch_size × num_heads × seq_length × d_k
        scaled_dot_product_attention_output = torch.matmul(
            attention_probabilities, V)  # batch_size × num_heads × seq_length × d_k

        scaled_dot_product_attention_output = scaled_dot_product_attention_output.permute(
            0, 2, 1, 3)  # batch_size × seq_length × num_heads × d_k

        # Output(s)
        scaled_dot_product_attention_output  # batch_size × seq_length × num_heads × d_k
        return scaled_dot_product_attention_output

    def split_heads(self, x):
        # Input(s)
        x  # batch_size × seq_length × d_model

        # Operation(s)
        batch_size, seq_length, d_model = x.size()
        x = x.view(batch_size, seq_length, self.num_heads, self.d_k)  # batch_size × seq_length × num_heads × d_k

        # Output(s)
        x  # batch_size × seq_length × num_heads × d_k
        return x

    def merge_heads(self, x):
        # Input(s)
        x  # batch_size × seq_length × num_heads × d_k

        # Operation(s)
        batch_size, seq_length, num_heads, d_k = x.size()
        x = x.contiguous().view(batch_size, seq_length, self.d_model)  # batch_size × seq_length × d_model

        # Output(s)
        x  # batch_size × seq_length × d_model
        return x

    def forward(self, for_Q, for_K, for_V, mask=None):
        # Input(s)
        for_Q  # batch_size × seq_length × input_size
        for_K  # batch_size × seq_length × input_size
        for_V  # batch_size × seq_length × input_size
        mask  # seq_length × seq_length

        # Operation(s)
        Q = self.W_q(for_Q)  # batch_size × seq_length × d_model
        Q = self.split_heads(Q)  # batch_size × seq_length × num_heads × d_k

        K = self.W_k(for_K)  # batch_size × seq_length × d_model
        K = self.split_heads(K)  # batch_size × seq_length × num_heads × d_k

        V = self.W_v(for_V)  # batch_size × seq_length × d_model
        V = self.split_heads(V)  # batch_size × seq_length × num_heads × d_k

        scaled_dot_product_attention_output = self.scaled_dot_product_attention(
            Q, K, V, mask)  # batch_size × seq_length × num_heads × d_k
        concatenated_scaled_dot_product_attention_output = self.merge_heads(
            scaled_dot_product_attention_output)  # batch_size × seq_length × d_model

        multi_head_attention_output = self.W_o(
            concatenated_scaled_dot_product_attention_output)  # batch_size × seq_length × d_model

        # Output(s)
        multi_head_attention_output  # batch_size × seq_length × d_model
        return multi_head_attention_output


class PointWiseFeedForward(nn.Module):
    def __init__(self, d_model, d_hiddens=[]):
        super(PointWiseFeedForward, self).__init__()

        linear_layers = []
        if len(d_hiddens) == 0:
            self.linear_layers.append(nn.Linear(in_features=self.d_model, out_features=self.d_model))
        else:
            in_features = d_model
            for d_hidden in d_hiddens:
                linear_layers.append(nn.Linear(in_features=in_features, out_features=d_hidden))
                linear_layers.append(nn.ReLU(inplace=True))
                in_features = d_hidden
            linear_layers.append(nn.Linear(in_features=in_features, out_features=d_model))

        self.feed_forward = nn.Sequential(*linear_layers)

    def forward(self, x):
        #  Input(s)
        x  # batch_size × seq_length × d_model

        # Operation(s)
        x = self.feed_forward(x)  # batch_size × seq_length × d_model

        # Output(s)
        x  # batch_size × seq_length × d_model
        return x


class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_hiddens, dropout_probability):
        super(EncoderLayer, self).__init__()
        self.multi_head_self_attention = MultiHeadAttention(d_model, num_heads)
        self.layer_normalization_after_self_attention = nn.LayerNorm(d_model)
        self.point_wise_feed_forward = PointWiseFeedForward(d_model, d_hiddens)
        self.layer_normalization_after_feed_forward = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout_probability)

    def forward(self, x, self_attention_mask):
        # Input(s)
        x  # batch_size × seq_length × d_model
        self_attention_mask  # seq_length × seq_length

        # Operation(s)
        multi_head_self_attention_output = self.multi_head_self_attention(
            for_Q=x, for_K=x, for_V=x,
            mask=self_attention_mask)  # batch_size × seq_length × d_model
        x = self.layer_normalization_after_self_attention(
            x + self.dropout(multi_head_self_attention_output))  # batch_size × seq_length × d_model
        point_wise_feed_forward_output = self.point_wise_feed_forward(x)  # batch_size × seq_length × d_model
        x = self.layer_normalization_after_feed_forward(
            x + self.dropout(point_wise_feed_forward_output))  # batch_size × seq_length × d_model

        # Output(s)
        x  # batch_size × seq_length × d_model
        return x


class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_hiddens, dropout_probability):
        super(DecoderLayer, self).__init__()
        self.multi_head_self_attention = MultiHeadAttention(d_model, num_heads)
        self.layer_normalization_after_self_attention = nn.LayerNorm(d_model)
        self.multi_head_cross_attention = MultiHeadAttention(d_model, num_heads)
        self.layer_normalization_after_cross_attention = nn.LayerNorm(d_model)
        self.point_wise_feed_forward = PointWiseFeedForward(d_model, d_hiddens)
        self.layer_normalization_after_feed_forward = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout_probability)

    def forward(self, x, encoder_output, self_attention_mask, cross_attention_mask):
        # Input(s)
        x  # batch_size × seq_length × d_model
        self_attention_mask  # seq_length × seq_length
        cross_attention_mask  # seq_length × seq_length

        # Operation(s)
        multi_head_self_attention_output = self.multi_head_self_attention(
            for_Q=x, for_K=x, for_V=x,
            mask=self_attention_mask)  # batch_size × seq_length × d_model
        x = self.layer_normalization_after_self_attention(
            x + self.dropout(multi_head_self_attention_output))  # batch_size × seq_length × d_model
        multi_head_cross_attention_output = self.multi_head_cross_attention(
            for_Q=x, for_K=encoder_output, for_V=encoder_output,
            mask=cross_attention_mask)  # batch_size × seq_length × d_model
        x = self.layer_normalization_after_cross_attention(
            x + self.dropout(multi_head_cross_attention_output))  # batch_size × seq_length × d_model
        point_wise_feed_forward_output = self.point_wise_feed_forward(x)  # batch_size × seq_length × d_model
        x = self.layer_normalization_after_feed_forward(
            x + self.dropout(point_wise_feed_forward_output))  # batch_size × seq_length × d_model

        # Output(s)
        x  # batch_size × seq_length × d_model
        return x
