"""
Implements the TransFuser vision backbone.
"""

import copy
import math
import os

import timm
import torch
import torch.nn.functional as F
from torch import nn
from transformers import AutoModel

from navsim.agents.theia_score.transfuser_config import TransfuserConfig
from timm.models.resnet import _cfg

class TransfuserBackbone(nn.Module):
    """Multi-scale Fusion Transformer for image + LiDAR feature fusion."""

    def __init__(self, config: TransfuserConfig):

        super().__init__()
        self.config = config

        # Load image encoder directly from local weights (no HuggingFace download)
        if hasattr(config, 'theia_weights_path') and config.theia_weights_path and os.path.exists(config.theia_weights_path):
            print(f"Loading Theia model from local weights: {config.theia_weights_path}")

            # Load the checkpoint
            checkpoint = torch.load(config.theia_weights_path, map_location='cpu')

            # Build the model structure from checkpoint
            self.image_encoder = self._build_theia_from_checkpoint(checkpoint)

            print(f"Successfully loaded Theia model from local checkpoint")
        else:
            # Fallback: Load from HuggingFace (original behavior)
            print(f"Loading Theia model from HuggingFace: {config.theia_model_name}")
            self.image_encoder = AutoModel.from_pretrained(
                config.theia_model_name,
                trust_remote_code=True,
                feature_reduce_method=config.theia_feature_reduce_method
            )

        # Freeze the Theia model
        for param in self.image_encoder.parameters():
            param.requires_grad = True
        # self.image_encoder.eval()


        # Create a wrapper to make Theia output compatible with multi-scale fusion
        # Extract features from ViT intermediate layers [5, 7, 9, 11]
        self.theia_wrapper = TheiaFeatureWrapper(self.image_encoder, config)
        # Use the wrapper as the actual encoder
        self.image_encoder = self.theia_wrapper

        if config.use_ground_plane:
            in_channels = 2 * config.lidar_seq_len
        else:
            in_channels = config.lidar_seq_len

        if config.latent:
            self.lidar_latent = nn.Parameter(
                torch.randn(
                    (1, in_channels, config.lidar_resolution_width, config.lidar_resolution_height),
                    requires_grad=True,
                )
            )

        self.avgpool_img = nn.AdaptiveAvgPool2d((self.config.img_vert_anchors, self.config.img_horz_anchors))

        # 3. Create LiDAR encoder (ResNet34). 
        self.lidar_encoder = timm.create_model(
            config.lidar_architecture,
            pretrained=False, 
            in_chans=in_channels,
            features_only=True,
        )
        self.global_pool_lidar = nn.AdaptiveAvgPool2d(output_size=1)
        self.avgpool_lidar = nn.AdaptiveAvgPool2d((self.config.lidar_vert_anchors, self.config.lidar_horz_anchors))
        lidar_time_frames = [1, 1, 1, 1]

        self.global_pool_img = nn.AdaptiveAvgPool2d(output_size=1)
        
        # 4. Handle start_index logic
        img_start_index = 0
        
        lidar_start_index = 0
        if len(self.lidar_encoder.return_layers) > 4:
            lidar_start_index += 1

        # 5. Build fusion layers, now using the separate start_indices
        self.transformers = nn.ModuleList(
            [
                GPT(
                    n_embd=self.image_encoder.feature_info.info[img_start_index + i]["num_chs"],
                    config=config,
                    lidar_time_frames=lidar_time_frames[i],
                )
                for i in range(4)
            ]
        )
        self.lidar_channel_to_img = nn.ModuleList(
            [
                nn.Conv2d(
                    self.lidar_encoder.feature_info.info[lidar_start_index + i]["num_chs"],
                    self.image_encoder.feature_info.info[img_start_index + i]["num_chs"],
                    kernel_size=1,
                )
                for i in range(4)
            ]
        )
        self.img_channel_to_lidar = nn.ModuleList(
            [
                nn.Conv2d(
                    self.image_encoder.feature_info.info[img_start_index + i]["num_chs"],
                    self.lidar_encoder.feature_info.info[lidar_start_index + i]["num_chs"],
                    kernel_size=1,
                )
                for i in range(4)
            ]
        )

        # 6. Update final feature size calculations
        self.num_image_features = self.image_encoder.feature_info.info[img_start_index + 3]["num_chs"]
        self.perspective_upsample_factor = (
            self.image_encoder.feature_info.info[img_start_index + 3]["reduction"]
            // self.config.perspective_downsample_factor
        )

        if self.config.transformer_decoder_join:
            self.num_features = self.lidar_encoder.feature_info.info[lidar_start_index + 3]["num_chs"]
        else:
            if self.config.add_features:
                self.lidar_to_img_features_end = nn.Linear(
                    self.lidar_encoder.feature_info.info[lidar_start_index + 3]["num_chs"],
                    self.image_encoder.feature_info.info[img_start_index + 3]["num_chs"],
                )
                # Number of features the encoder produces.
                self.num_features = self.image_encoder.feature_info.info[img_start_index + 3]["num_chs"]
            else:
                # Number of features the encoder produces.
                self.num_features = (
                    self.image_encoder.feature_info.info[img_start_index + 3]["num_chs"]
                    + self.lidar_encoder.feature_info.info[lidar_start_index + 3]["num_chs"]
                )

        # 7. FPN fusion (unchanged, relies on LiDAR encoder)
        channel = self.config.bev_features_channels
        self.relu = nn.ReLU(inplace=True)
        # top down
        if self.config.detect_boxes or self.config.use_bev_semantic:
            self.upsample = nn.Upsample(
                scale_factor=self.config.bev_upsample_factor, mode="bilinear", align_corners=False
            )
            self.upsample2 = nn.Upsample(
                size=(
                    self.config.lidar_resolution_height // self.config.bev_down_sample_factor,
                    self.config.lidar_resolution_width // self.config.bev_down_sample_factor,
                ),
                mode="bilinear",
                align_corners=False,
            )

            self.up_conv5 = nn.Conv2d(channel, channel, (3, 3), padding=1)
            self.up_conv4 = nn.Conv2d(channel, channel, (3, 3), padding=1)

            # lateral
            self.c5_conv = nn.Conv2d(self.lidar_encoder.feature_info.info[lidar_start_index + 3]["num_chs"], channel, (1, 1))

    def _build_theia_from_checkpoint(self, checkpoint):
        """
        Build a minimal Theia model structure from checkpoint weights.
        This avoids using HuggingFace's from_pretrained method.

        Args:
            checkpoint: OrderedDict containing model weights

        Returns:
            A minimal model object compatible with TheiaFeatureWrapper
        """
        from transformers import ViTConfig, ViTModel, ViTImageProcessor

        # Extract backbone weights (backbone.model.* keys)
        backbone_state_dict = {}
        for key, value in checkpoint.items():
            if key.startswith('backbone.model.'):
                # Remove 'backbone.model.' prefix to get standard ViT keys
                new_key = key.replace('backbone.model.', '')
                backbone_state_dict[new_key] = value

        # Infer model configuration from checkpoint structure
        # Check embedding dimension from encoder layer weights
        sample_key = 'encoder.layer.0.attention.attention.query.weight'
        if sample_key in backbone_state_dict:
            hidden_size = backbone_state_dict[sample_key].shape[-1]
        else:
            hidden_size = 384  # Default for deit-small

        # Count number of encoder layers
        num_layers = 0
        for key in backbone_state_dict.keys():
            if key.startswith('encoder.layer.'):
                layer_idx = int(key.split('.')[2])
                num_layers = max(num_layers, layer_idx + 1)

        print(f"  Detected ViT config: hidden_size={hidden_size}, num_layers={num_layers}")

        # Create ViT configuration
        vit_config = ViTConfig(
            hidden_size=hidden_size,
            num_hidden_layers=num_layers,
            num_attention_heads=hidden_size // 64,  # Standard ratio
            intermediate_size=hidden_size * 4,  # Standard ratio for FFN
            image_size=224,
            patch_size=16,
        )

        # Create ViT model
        vit_model = ViTModel(vit_config)

        # Load weights into the model
        missing_keys, unexpected_keys = vit_model.load_state_dict(backbone_state_dict, strict=False)
        print(f"  Loaded {len(backbone_state_dict)} parameters from checkpoint")
        if missing_keys:
            print(f"  Warning: {len(missing_keys)} missing keys (e.g., {missing_keys[:3]})")
        if unexpected_keys:
            print(f"  Warning: {len(unexpected_keys)} unexpected keys")

        # Create a minimal wrapper to match Theia model structure
        # TheiaFeatureWrapper expects: model.backbone.model and model.backbone.processor
        class TheiaModel(nn.Module):
            def __init__(self, vit_model):
                super().__init__()
                self.backbone = nn.Module()
                self.backbone.model = vit_model
                # Create a simple processor with required methods
                self.backbone.processor = self._create_minimal_processor()

            def _create_minimal_processor(self):
                """Create a minimal processor for image preprocessing."""
                # Use DeiT normalization (NOT ImageNet standard!)
                # DeiT uses mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]
                # This scales images to [-1, 1] range
                processor = ViTImageProcessor(
                    do_resize=False,  # Don't resize - keep original resolution
                    do_rescale=True,  # Rescale [0, 255] -> [0, 1]
                    do_normalize=True,
                    image_mean=[0.5, 0.5, 0.5],  # DeiT mean (not ImageNet!)
                    image_std=[0.5, 0.5, 0.5],   # DeiT std (not ImageNet!)
                    # Note: size parameter is ignored when do_resize=False
                )
                return processor

        return TheiaModel(vit_model)

    def top_down(self, x):

        p5 = self.relu(self.c5_conv(x))
        p4 = self.relu(self.up_conv5(self.upsample(p5)))
        p3 = self.relu(self.up_conv4(self.upsample2(p4)))

        return p3

    def forward(self, image, lidar):
        """
        Image + LiDAR feature fusion using transformers
        Args:
            image_list (list): list of input images
            lidar_list (list): list of input LiDAR BEV
        """
        # Rename image to image_input to avoid conflict in the loop
        image_input, lidar_features = image, lidar

        if self.config.latent:
            batch_size = lidar.shape[0]
            lidar_features = self.lidar_latent.repeat(batch_size, 1, 1, 1)

        # 1. Run the ViT encoder *once*.
        # Since it's frozen and in eval() mode, this is efficient.
        # This returns a list of 4 feature maps.
        all_image_features = self.image_encoder(image_input)

        # 2. Set up iterator for the ResNet LiDAR encoder (unchanged)
        lidar_layers = iter(self.lidar_encoder.items())

        # 3. Handle LiDAR stem layer (unchanged)
        if len(self.lidar_encoder.return_layers) > 4:
            lidar_features = self.forward_layer_block(lidar_layers, self.lidar_encoder.return_layers, lidar_features)

        # 4. Loop through the 4 fusion blocks.
        for i in range(4):
            # Get the pre-computed ViT feature for this level
            image_features = all_image_features[i]
            
            # Run one block of the LiDAR (ResNet) encoder
            lidar_features = self.forward_layer_block(lidar_layers, self.lidar_encoder.return_layers, lidar_features)

            # Fuse the features from both modalities
            image_features, lidar_features = self.fuse_features(image_features, lidar_features, i)

        # 5. Post-fusion processing (unchanged)
        # The rest of the code uses the final 'image_features' and 'lidar_features'
        # from the end of the loop, which is the correct logic.
        if self.config.detect_boxes or self.config.use_bev_semantic:
            x4 = lidar_features

        image_feature_grid = None
        if self.config.use_semantic or self.config.use_depth:
            image_feature_grid = image_features

        if self.config.transformer_decoder_join:
            fused_features = lidar_features
        else:
            image_features = self.global_pool_img(image_features)
            image_features = torch.flatten(image_features, 1)
            lidar_features = self.global_pool_lidar(lidar_features)
            lidar_features = torch.flatten(lidar_features, 1)

            if self.config.add_features:
                lidar_features = self.lidar_to_img_features_end(lidar_features)
                fused_features = image_features + lidar_features
            else:
                fused_features = torch.cat((image_features, lidar_features), dim=1)

        if self.config.detect_boxes or self.config.use_bev_semantic:
            features = self.top_down(x4)
        else:
            features = None

        return features, fused_features, image_feature_grid

    def forward_layer_block(self, layers, return_layers, features):
        """
        Run one forward pass to a block of layers from a TIMM neural network and returns the result.
        Advances the whole network by just one block
        (This is now only used for the LiDAR encoder)
        :param layers: Iterator starting at the current layer block
        :param return_layers: TIMM dictionary describing at which intermediate layers features are returned.
        :param features: Input features
        :return: Processed features
        """
        for name, module in layers:
            features = module(features)
            if name in return_layers:
                break
        return features

    def fuse_features(self, image_features, lidar_features, layer_idx):
        """
        Perform a TransFuser feature fusion block using a Transformer module.
        :param image_features: Features from the image branch
        :param lidar_features: Features from the LiDAR branch
        :param layer_idx: Transformer layer index.
        :return: image_features and lidar_features with added features from the other branch.
        """
        image_embd_layer = self.avgpool_img(image_features)
        lidar_embd_layer = self.avgpool_lidar(lidar_features)

        lidar_embd_layer = self.lidar_channel_to_img[layer_idx](lidar_embd_layer)

        image_features_layer, lidar_features_layer = self.transformers[layer_idx](image_embd_layer, lidar_embd_layer)
        lidar_features_layer = self.img_channel_to_lidar[layer_idx](lidar_features_layer)

        image_features_layer = F.interpolate(
            image_features_layer,
            size=(image_features.shape[2], image_features.shape[3]),
            mode="bilinear",
            align_corners=False,
        )
        lidar_features_layer = F.interpolate(
            lidar_features_layer,
            size=(lidar_features.shape[2], lidar_features.shape[3]),
            mode="bilinear",
            align_corners=False,
        )

        image_features = image_features + image_features_layer
        lidar_features = lidar_features + lidar_features_layer

        return image_features, lidar_features


class GPT(nn.Module):
    """The full GPT language backbone, with a context size of block_size."""

    # def __init__(self, n_embd, config, lidar_video, lidar_time_frames):
    def __init__(self, n_embd, config, lidar_time_frames):
        super().__init__()
        self.n_embd = n_embd
        # We currently only support seq len 1
        self.seq_len = 1
        self.lidar_seq_len = config.lidar_seq_len
        self.config = config
        self.lidar_time_frames = lidar_time_frames

        # positional embedding parameter (learnable), image + lidar
        self.pos_emb = nn.Parameter(
            torch.zeros(
                1,
                self.seq_len * self.config.img_vert_anchors * self.config.img_horz_anchors
                + lidar_time_frames * self.config.lidar_vert_anchors * self.config.lidar_horz_anchors,
                self.n_embd,
            )
        )

        self.drop = nn.Dropout(config.embd_pdrop)

        # transformer
        self.blocks = nn.Sequential(
            *[
                Block(n_embd, config.n_head, config.block_exp, config.attn_pdrop, config.resid_pdrop)
                for layer in range(config.n_layer)
            ]
        )

        # decoder head
        self.ln_f = nn.LayerNorm(n_embd)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(
                mean=self.config.gpt_linear_layer_init_mean,
                std=self.config.gpt_linear_layer_init_std,
            )
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(self.config.gpt_layer_norm_init_weight)

    def forward(self, image_tensor, lidar_tensor):
        """
        Args:
            image_tensor (tensor): B*4*seq_len, C, H, W
            lidar_tensor (tensor): B*seq_len, C, H, W
        """

        bz = lidar_tensor.shape[0]
        lidar_h, lidar_w = lidar_tensor.shape[2:4]

        img_h, img_w = image_tensor.shape[2:4]

        assert self.seq_len == 1
        image_tensor = image_tensor.permute(0, 2, 3, 1).contiguous().view(bz, -1, self.n_embd)
        lidar_tensor = lidar_tensor.permute(0, 2, 3, 1).contiguous().view(bz, -1, self.n_embd)

        token_embeddings = torch.cat((image_tensor, lidar_tensor), dim=1)

        x = self.drop(self.pos_emb + token_embeddings)
        x = self.blocks(x)  # (B, an * T, C)
        x = self.ln_f(x)  # (B, an * T, C)

        image_tensor_out = (
            x[:, : self.seq_len * self.config.img_vert_anchors * self.config.img_horz_anchors, :]
            .view(bz * self.seq_len, img_h, img_w, -1)
            .permute(0, 3, 1, 2)
            .contiguous()
        )
        lidar_tensor_out = (
            x[
                :,
                self.seq_len * self.config.img_vert_anchors * self.config.img_horz_anchors :,
                :,
            ]
            .view(bz, lidar_h, lidar_w, -1)
            .permute(0, 3, 1, 2)
            .contiguous()
        )

        return image_tensor_out, lidar_tensor_out


class SelfAttention(nn.Module):
    """
    A vanilla multi-head masked self-attention layer with a projection at the
    end.
    """

    def __init__(self, n_embd, n_head, attn_pdrop, resid_pdrop):
        super().__init__()
        assert n_embd % n_head == 0
        # key, query, value projections for all heads
        self.key = nn.Linear(n_embd, n_embd)
        self.query = nn.Linear(n_embd, n_embd)
        self.value = nn.Linear(n_embd, n_embd)
        # regularization
        self.attn_drop = nn.Dropout(attn_pdrop)
        self.resid_drop = nn.Dropout(resid_pdrop)
        # output projection
        self.proj = nn.Linear(n_embd, n_embd)
        self.n_head = n_head

    def forward(self, x):
        b, t, c = x.size()

        # calculate query, key, values for all heads in batch and move head
        # forward to be the batch dim
        k = self.key(x).view(b, t, self.n_head, c // self.n_head).transpose(1, 2)  # (b, nh, t, hs)
        q = self.query(x).view(b, t, self.n_head, c // self.n_head).transpose(1, 2)  # (b, nh, t, hs)
        v = self.value(x).view(b, t, self.n_head, c // self.n_head).transpose(1, 2)  # (b, nh, t, hs)

        # self-attend: (b, nh, t, hs) x (b, nh, hs, t) -> (b, nh, t, t)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v  # (b, nh, t, t) x (b, nh, t, hs) -> (b, nh, t, hs)
        y = y.transpose(1, 2).contiguous().view(b, t, c)  # re-assemble all head outputs side by side

        # output projection
        y = self.resid_drop(self.proj(y))
        return y


class Block(nn.Module):
    """an unassuming Transformer block"""

    def __init__(self, n_embd, n_head, block_exp, attn_pdrop, resid_pdrop):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        self.attn = SelfAttention(n_embd, n_head, attn_pdrop, resid_pdrop)
        self.mlp = nn.Sequential(
            nn.Linear(n_embd, block_exp * n_embd),
            nn.ReLU(True),  # changed from GELU
            nn.Linear(block_exp * n_embd, n_embd),
            nn.Dropout(resid_pdrop),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))

        return x


class MultiheadAttentionWithAttention(nn.Module):
    """
    MultiheadAttention that also return attention weights
    """

    def __init__(self, n_embd, n_head, pdrop):
        super().__init__()
        assert n_embd % n_head == 0
        # key, query, value projections for all heads
        self.key = nn.Linear(n_embd, n_embd)
        self.query = nn.Linear(n_embd, n_embd)
        self.value = nn.Linear(n_embd, n_embd)
        # regularization
        self.attn_drop = nn.Dropout(pdrop)
        self.resid_drop = nn.Dropout(pdrop)
        # output projection
        self.proj = nn.Linear(n_embd, n_embd)
        self.n_head = n_head

    def forward(self, q_in, k_in, v_in):
        b, t, c = q_in.size()
        _, t_mem, _ = k_in.size()

        # calculate query, key, values for all heads in batch and move head
        # forward to be the batch dim
        q = self.query(q_in).view(b, t, self.n_head, c // self.n_head).transpose(1, 2)  # (b, nh, t, hs)
        k = self.key(k_in).view(b, t_mem, self.n_head, c // self.n_head).transpose(1, 2)  # (b, nh, t, hs)
        v = self.value(v_in).view(b, t_mem, self.n_head, c // self.n_head).transpose(1, 2)  # (b, nh, t, hs)

        # self-attend: (b, nh, t, hs) x (b, nh, hs, t) -> (b, nh, t, t)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v  # (b, nh, t, t) x (b, nh, t, hs) -> (b, nh, t, hs)
        y = y.transpose(1, 2).contiguous().view(b, t, c)  # re-assemble all head outputs side by side

        # output projection
        y = self.resid_drop(self.proj(y))
        attention = torch.mean(att, dim=1)  # Average attention over heads
        return y, attention


class TransformerDecoderLayerWithAttention(nn.Module):
    """A Transformer decoder that returns the attentions."""

    def __init__(
        self,
        d_model,
        nhead,
        dim_feedforward=2048,
        dropout=0.1,
        activation=F.relu,
        layer_norm_eps=1e-5,
    ):
        super().__init__()
        self.self_attn = MultiheadAttentionWithAttention(d_model, nhead, dropout)
        self.multihead_attn = MultiheadAttentionWithAttention(d_model, nhead, dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm3 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = activation

    def forward(self, tgt, memory):
        x = tgt
        tmp, _ = self.self_attn(x, x, x)
        x = self.norm1(x + self.dropout1(tmp))
        tmp, attention = self.multihead_attn(x, memory, memory)
        x = self.norm2(x + self.dropout2(tmp))
        tmp = self.linear2(self.dropout(self.activation(self.linear1(x))))
        x = self.norm3(x + self.dropout3(tmp))

        return x, attention


class TransformerDecoderWithAttention(nn.Module):
    """A Transformer decoder that returns the attentions."""

    def __init__(self, layers, num_layers, norm=None):
        super().__init__()
        self.layers = nn.ModuleList([copy.deepcopy(layers) for i in range(num_layers)])
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, queries, memory):
        output = queries
        attentions = []
        for mod in self.layers:
            output, attention = mod(output, memory)
            attentions.append(attention)

        if self.norm is not None:
            output = self.norm(output)

        avg_attention = torch.mean(torch.stack(attentions), dim=0)
        return output, avg_attention


class TheiaFeatureWrapper(nn.Module):
    """
    Wrapper to make Theia model output compatible with multi-scale fusion.

    This wrapper extracts features from intermediate ViT layers [5, 7, 9, 11]
    to create 4 multi-scale features like the original ViT encoder.
    """

    def __init__(self, theia_model, config):
        super().__init__()
        self.theia_model = theia_model
        self.config = config

        # Extract features from layers 5, 7, 9, 11 (matching original implementation)
        self.out_indices = [5, 7, 9, 11]

        # Get hidden size (channels) from Theia model config
        # Small: 384, Base: 768, Large: 1024, etc.
        vit_config = theia_model.backbone.model.config
        theia_channels = vit_config.hidden_size

        # All scales have the same channel size
        self.scale_channels = [theia_channels] * 4

        # Create a feature_info object to make it compatible with timm models
        class FeatureInfo:
            def __init__(self, channels_list):
                # All layers have the same spatial resolution (patch_size=16)
                self.info = [
                    {'num_chs': channels, 'reduction': 16}  
                    for channels in channels_list
                ]

        self.feature_info = FeatureInfo(self.scale_channels)

    def forward(self, x):
        """
        Forward pass through Theia model, extracting intermediate layer features.

        Args:
            x: Input image tensor [B, C, H, W] with shape [B, 3, 256, 1024]

        Returns:
            List of 4 feature tensors from layers [5, 7, 9, 11]
        """
        # Access the underlying ViT backbone and preprocessor
        backbone = self.theia_model.backbone
        vit_model = backbone.model
        preprocessor = backbone.processor

        # Convert input format for preprocessor
        # Processor expects [B, H, W, C] uint8 format
        if x.dtype != torch.uint8:
            # Convert to uint8 [0, 255]
            if x.min() < 0:
                x = ((x + 1) * 127.5).clamp(0, 255).to(torch.uint8)
            else:
                x = (x * 255).clamp(0, 255).to(torch.uint8)

        if x.shape[1] == 3:  # [B, C, H, W] -> [B, H, W, C]
            x = x.permute(0, 2, 3, 1)

        # Store original spatial dimensions
        B, H, W, C_in = x.shape

        # Process batch using Theia's built-in processor
        # Use do_resize=False to preserve original resolution
        # Use interpolate_pos_encoding=True to adapt position encodings
        # DeiT normalization: mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]
        pixel_values_list = []
        for i in range(x.shape[0]):
            img = x[i].cpu().numpy()
            processed = preprocessor(
                img,
                return_tensors="pt",
                do_resize=False,     # Keep original resolution (e.g., 256x1024)!
                do_rescale=True,     # Scale [0, 255] -> [0, 1]
                do_normalize=True    # DeiT normalization to [-1, 1]
            )["pixel_values"]
            pixel_values_list.append(processed)

        pixel_values = torch.cat(pixel_values_list, dim=0).to(x.device)

        # Get embeddings from ViT with position encoding interpolation
        embedding_output = vit_model.embeddings(
            pixel_values,
            interpolate_pos_encoding=True  # Interpolate to match input resolution
        )

        # Extract features from specified layers [5, 7, 9, 11]
        hidden_states = []
        encoder_output = embedding_output

        for i, layer_module in enumerate(vit_model.encoder.layer):
            layer_outputs = layer_module(encoder_output)
            if isinstance(layer_outputs, tuple):
                encoder_output = layer_outputs[0]
            else:
                encoder_output = layer_outputs

            # Save features from specified layers
            if i in self.out_indices:
                hidden_states.append(encoder_output)

        # Process hidden states to [B, C, H, W] format
        multi_scale_features = []
        for hidden_state in hidden_states:
            # Remove CLS token: [B, 1+H*W, C] -> [B, H*W, C]
            feature_tokens = hidden_state[:, 1:, :]

            B = feature_tokens.shape[0]
            num_patches = feature_tokens.shape[1]
            C = feature_tokens.shape[2]

            # Calculate spatial dimensions based on original input size
            # With patch_size=16: H_patches = H//16, W_patches = W//16
            H_patches = H // 16
            W_patches = W // 16

            # Verify patch count matches
            assert num_patches == H_patches * W_patches, \
                f"Patch count mismatch: {num_patches} != {H_patches}x{W_patches}"

            # Reshape to [B, C, H, W]
            features_2d = feature_tokens.transpose(1, 2).reshape(B, C, H_patches, W_patches)
            multi_scale_features.append(features_2d)

        return multi_scale_features
