import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """Double convolution block with batch normalization"""
    
    def __init__(self, in_channels, out_channels, dropout_rate=0.0):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout_rate),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.conv(x)


class EncoderBlock(nn.Module):
    """Encoder block with max pooling"""
    
    def __init__(self, in_channels, out_channels, dropout_rate=0.3):
        super(EncoderBlock, self).__init__()
        self.conv = DoubleConv(in_channels, out_channels, dropout_rate)
        self.pool = nn.MaxPool2d(2)
    
    def forward(self, x):
        conv_out = self.conv(x)
        pooled = self.pool(conv_out)
        return conv_out, pooled


class DecoderBlock(nn.Module):
    """Decoder block with upsampling and skip connections"""
    
    def __init__(self, in_channels, out_channels, dropout_rate=0.3):
        super(DecoderBlock, self).__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels, dropout_rate)
    
    def forward(self, x, skip):
        x = self.up(x)
        # Handle size mismatch
        if x.shape != skip.shape:
            x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=True)
        x = torch.cat([x, skip], dim=1)
        x = self.conv(x)
        return x


class PIDCR_UNet(nn.Module):
    """
    U-Net with Deep Ritz Method integration for phase-field fracture modeling
    
    Architecture:
    - Encoder: 4 levels with double convolution and max pooling
    - Bottleneck: Double convolution
    - Decoder: 4 levels with transpose convolution and skip connections
    - Output: Sigmoid activation for phase field in [0, 1]
    """
    
    def __init__(self, in_channels=1, out_channels=1, features=[64, 128, 256, 512],
                 dropout_rate=0.3, use_inverse_head=True):
        super(PIDCR_UNet, self).__init__()
        
        self.use_inverse_head = use_inverse_head
        self.dropout_rate = dropout_rate
        self.features = features
        
        # ===== Encoder =====
        self.enc1 = EncoderBlock(in_channels, features[0], dropout_rate)
        self.enc2 = EncoderBlock(features[0], features[1], dropout_rate)
        self.enc3 = EncoderBlock(features[1], features[2], dropout_rate)
        self.enc4 = EncoderBlock(features[2], features[3], dropout_rate)
        
        # ===== Bottleneck =====
        self.bottleneck = DoubleConv(features[3], features[3] * 2, dropout_rate)
        
        # ===== Decoder =====
        self.dec4 = DecoderBlock(features[3] * 2, features[3], dropout_rate)
        self.dec3 = DecoderBlock(features[3], features[2], dropout_rate)
        self.dec2 = DecoderBlock(features[2], features[1], dropout_rate)
        self.dec1 = DecoderBlock(features[1], features[0], dropout_rate)
        
        # ===== Output =====
        self.out_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)
        
        # ===== Inverse estimation head (for material parameters) =====
        # z = (Gc, E, v)
        if use_inverse_head:
            # Calculate bottleneck feature size
            self.inverse_head = nn.Sequential(
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
                nn.Linear(features[3] * 2, 128),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, 3)  # Gc, E, v
            )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights using Kaiming initialization"""
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x, return_features=False, mc_dropout=False):
        """
        Forward pass through U-Net
        
        Args:
            x: Input tensor [batch, channels, H, W]
            return_features: If True, return bottleneck features for inverse estimation
            mc_dropout: If True, enable dropout during inference for uncertainty
        """
        if mc_dropout:
            self.train()  # Enable dropout for MC sampling
        
        # ===== Encoder =====
        skip1, x = self.enc1(x)
        skip2, x = self.enc2(x)
        skip3, x = self.enc3(x)
        skip4, x = self.enc4(x)
        
        # ===== Bottleneck =====
        bottleneck = self.bottleneck(x)
        
        # ===== Decoder =====
        x = self.dec4(bottleneck, skip4)
        x = self.dec3(x, skip3)
        x = self.dec2(x, skip2)
        x = self.dec1(x, skip1)
        
        # ===== Output =====
        phase_field = torch.sigmoid(self.out_conv(x))  # Constrain to [0, 1]
        
        if return_features and self.use_inverse_head:
            # Inverse estimation of material parameters
            material_params = self.inverse_head(bottleneck)
            return phase_field, material_params
        
        return phase_field
    
    def forward_with_uncertainty(self, x, n_samples=50):
        """
        Monte Carlo Dropout for uncertainty quantification (Eq. 16-17)
        
        Args:
            x: Input tensor
            n_samples: Number of stochastic forward passes (T = 50)
        
        Returns:
            mean: Mean prediction
            variance: Prediction variance
        """
        predictions = []
        for _ in range(n_samples):
            with torch.no_grad():
                pred = self.forward(x, mc_dropout=True)
                predictions.append(pred)
        
        predictions = torch.stack(predictions)
        mean = predictions.mean(dim=0)
        variance = predictions.var(dim=0)
        
        return mean, variance