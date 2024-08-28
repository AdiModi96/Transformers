import torch
from torch import nn
from transformer_modules import (
    PositionalEncoding,
    EncoderLayer,
    DecoderLayer
)


class EncoderDecoderTransformer(nn.Module):
    def __init__(
            self,
            encoder_vocab_size, decoder_vocab_size,
            d_model, max_seq_length, num_heads, d_hiddens, dropout_probability,
            num_encoder_layers, num_decoder_layers,
            d_decoder_output):
        super(EncoderDecoderTransformer, self).__init__()
        self.encoder_embedding = nn.Embedding(num_embeddings=encoder_vocab_size, embedding_dim=d_model)
        self.decoder_embedding = nn.Embedding(num_embeddings=decoder_vocab_size, embedding_dim=d_model)

        self.positional_encoding = PositionalEncoding(d_model=d_model, max_seq_length=max_seq_length)

        self.encoder_layers = nn.ModuleList()
        for encoder_block_idx in range(num_encoder_layers):
            self.encoder_layers.append(EncoderLayer(d_model=d_model, num_heads=num_heads, d_hiddens=d_hiddens,
                                                    dropout_probability=dropout_probability))

        self.decoder_layers = nn.ModuleList()
        for decoder_layer in range(num_decoder_layers):
            self.decoder_layers.append(DecoderLayer(d_model=d_model, num_heads=num_heads, d_hiddens=d_hiddens,
                                                    dropout_probability=dropout_probability))

        self.output_layer = nn.Linear(in_features=d_model, out_features=d_decoder_output)

    def forward(self, encoder_input, decoder_input):
        # Input(s)
        encoder_input  # batch_size × encoder_seq_length
        decoder_input  # batch_size × decoder_seq_length

        # Operation(s)
        batch_size, encoder_seq_length = encoder_input.size()
        batch_size, decoder_seq_length = decoder_input.size()

        ## Encoder
        embedded_encoder_input = self.encoder_embedding(encoder_input)  # batch_size × encoder_seq_length × d_model
        position_encoded_encoder_input = self.positional_encoding(embedded_encoder_input)  # batch_size × encoder_seq_length × d_model
        encoder_self_attention_mask = torch.full(
            size=(encoder_seq_length, encoder_seq_length),
            fill_value=True, dtype=torch.bool)  # encoder_seq_length × encoder_seq_length

        encoder_layer_output = position_encoded_encoder_input  # batch_size × encoder_seq_length × d_model
        for encoder_layer in self.encoder_layers:
            encoder_layer_output = encoder_layer(
                position_encoded_encoder_input, encoder_self_attention_mask)  # batch_size × encoder_seq_length × d_model

        ## Decoder
        embedded_decoder_input = self.decoder_embedding(decoder_input)  # batch_size × decoder_seq_length × d_model
        position_encoded_decoder_input = self.positional_encoding(
            embedded_decoder_input)  # batch_size × decoder_seq_length × d_model
        decoder_self_attention_mask = torch.tril(torch.full(
            size=(decoder_seq_length, decoder_seq_length),
            fill_value=True, dtype=torch.bool))  # decoder_seq_length × decoder_seq_length
        decoder_cross_attention_mask = torch.full(
            size=(decoder_seq_length, encoder_seq_length),
            fill_value=True, dtype=torch.bool)  # decoder_seq_length × decoder_seq_length

        decoder_layer_output = position_encoded_decoder_input  # batch_size × decoder_seq_length × d_model
        for decoder_layer in self.decoder_layers:
            decoder_layer_output = decoder_layer(
                position_encoded_decoder_input, encoder_layer_output,
                decoder_self_attention_mask, decoder_cross_attention_mask)  # batch_size × decoder_seq_length × d_model

        ## Generating final output
        transformer_output = self.output_layer(
            decoder_layer_output)  # batch_size × decoder_seq_length × d_decoder_output

        # Output(s)
        transformer_output  # batch_size × decoder_seq_length × d_decoder_output
        return transformer_output


class EncoderOnlyTransformer(nn.Module):
    def __init__(self,
                 vocab_size,
                 d_model, max_seq_length, num_heads, d_hiddens, dropout_probability,
                 num_encoder_layers, d_encoder_output):
        super(EncoderOnlyTransformer, self).__init__()

        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=d_model)
        self.positional_encoding = PositionalEncoding(d_model=d_model, max_seq_length=max_seq_length)

        self.encoder_layers = nn.ModuleList()
        for encoder_block_idx in range(num_encoder_layers):
            self.encoder_layers.append(EncoderLayer(d_model=d_model, num_heads=num_heads, d_hiddens=d_hiddens,
                                                    dropout_probability=dropout_probability))

        self.output_layer = nn.Linear(in_features=d_model, out_features=d_encoder_output)

    def forward(self, input):
        # Input(s)
        input  # batch_size × seq_length

        # Operation(s)
        batch_size, seq_length = input.size()

        embedded_inputs = self.embedding(input)  # batch_size × seq_length × d_model
        position_encoded_input = self.positional_encoding(embedded_inputs)  # batch_size × seq_length × d_model
        self_attention_mask = torch.full(
            size=(seq_length, seq_length),
            fill_value=True, dtype=torch.bool)  # seq_length × seq_length

        encoder_layer_output = position_encoded_input  # batch_size × seq_length × d_model
        for encoder_layer in self.encoder_layers:
            encoder_layer_output = encoder_layer(
                position_encoded_input, self_attention_mask)  # batch_size × seq_length × d_model

        ## Generating final output
        transformer_output = self.output_layer(encoder_layer_output)  # batch_size × seq_length × d_encoder_output

        # Output(s)
        transformer_output  # batch_size × seq_length × d_encoder_output
        return transformer_output


class DecoderOnlyTransformer(nn.Module):
    def __init__(self,
                 vocab_size,
                 d_model, max_seq_length, num_heads, d_hiddens, dropout_probability,
                 num_decoder_layers, d_decoder_output):
        super(DecoderOnlyTransformer, self).__init__()

        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=d_model)
        self.positional_encoding = PositionalEncoding(d_model=d_model, max_seq_length=max_seq_length)

        self.decoder_layers = nn.ModuleList()
        for decoder_block_idx in range(num_decoder_layers):
            self.decoder_layers.append(DecoderLayer(d_model=d_model, num_heads=num_heads, d_hiddens=d_hiddens,
                                                    dropout_probability=dropout_probability))

        self.output_layer = nn.Linear(in_features=d_model, out_features=d_decoder_output)

    def forward(self, input):
        # Input(s)
        input  # batch_size × seq_length

        # Operation(s)
        batch_size, seq_length = input.size()

        embedded_input = self.embedding(input)  # batch_size × seq_length × d_model
        position_encoded_input = self.positional_encoding(embedded_input)  # batch_size × seq_length × d_model
        self_attention_mask = torch.tril(torch.full(
            size=(seq_length, seq_length),
            fill_value=True, dtype=torch.bool))  # output_seq_length × output_seq_length

        decoder_layer_output = position_encoded_input  # batch_size × seq_length × d_model
        for decoder_layer in self.decoder_layers:
            decoder_layer_output = decoder_layer(
                decoder_layer_output, decoder_layer_output,
                self_attention_mask, self_attention_mask)  # batch_size × seq_length × d_model

        ## Generating final output
        transformer_output = self.output_layer(decoder_layer_output)  # batch_size × seq_length × d_decoder_output

        # Output(s)
        transformer_output  # batch_size × seq_length × d_decoder_output
        return transformer_output
