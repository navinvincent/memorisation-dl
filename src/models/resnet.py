"""
ResNet architecture implementation in JAX/Flax
"""
import jax.numpy as jnp
from flax import linen as nn
from typing import Any, Callable, Sequence, Tuple
from functools import partial


ModuleDef = Any


class ResNetBlock(nn.Module):
    """ResNet basic block"""
    filters: int
    conv: ModuleDef
    norm: ModuleDef
    act: Callable
    strides: Tuple[int, int] = (1, 1)
    
    @nn.compact
    def __call__(self, x):
        residual = x
        y = self.conv(self.filters, (3, 3), self.strides)(x)
        if self.norm is not None:
            y = self.norm()(y)
        y = self.act(y)
        y = self.conv(self.filters, (3, 3))(y)
        if self.norm is not None:
            y = self.norm(scale_init=nn.initializers.zeros)(y)
            
        if residual.shape != y.shape:
            residual = self.conv(self.filters, (1, 1), self.strides, name='conv_proj')(residual)
            if self.norm is not None:
                residual = self.norm(name='norm_proj')(residual)
                
        return self.act(residual + y)


class BottleneckBlock(nn.Module):
    """Bottleneck residual block"""
    filters: int
    conv: ModuleDef
    norm: ModuleDef
    act: Callable
    strides: Tuple[int, int] = (1, 1)
    
    @nn.compact
    def __call__(self, x):
        residual = x
        y = self.conv(self.filters, (1, 1))(x)
        if self.norm is not None:
            y = self.norm()(y)
        y = self.act(y)
        y = self.conv(self.filters, (3, 3), self.strides)(y)
        if self.norm is not None:
            y = self.norm()(y)
        y = self.act(y)
        y = self.conv(self.filters * 4, (1, 1))(y)
        if self.norm is not None:
            y = self.norm(scale_init=nn.initializers.zeros)(y)
            
        if residual.shape != y.shape:
            residual = self.conv(self.filters * 4, (1, 1), self.strides, name='conv_proj')(residual)
            if self.norm is not None:
                residual = self.norm(name='norm_proj')(residual)
            
        return self.act(residual + y)


class ResNet(nn.Module):
    """ResNet v1"""
    stage_sizes: Sequence[int]
    block_cls: ModuleDef
    num_classes: int = 10  # adapted to CIFAR-10
    num_filters: int = 16  # reduced number of filters to decrease training time
    dtype: Any = jnp.float32
    act: Callable = nn.relu
    
    def setup(self, enable_batch_norm=False):
        self.enable_batch_norm = enable_batch_norm
    
    @nn.compact
    def __call__(self, x, train: bool = True):
        conv = partial(nn.Conv, use_bias=not self.enable_batch_norm, dtype=self.dtype)
        if self.enable_batch_norm:
            norm = partial(nn.BatchNorm, use_running_average=not train, 
                          momentum=0.9, epsilon=1e-5, dtype=self.dtype)
        else:
            norm = None
        
        x = conv(self.num_filters, (3, 3), (1, 1),
                 padding='SAME', name='conv_init')(x)
        if self.enable_batch_norm:
            x = norm(name='bn_init')(x)
        x = nn.relu(x)
        x = nn.max_pool(x, (2, 2), strides=(2, 2), padding='SAME')
        for i, block_size in enumerate(self.stage_sizes):
            for j in range(block_size):
                strides = (2, 2) if i > 0 and j == 0 else (1, 1)
                x = self.block_cls(self.num_filters * 2 ** i,
                                   strides=strides,
                                   conv=conv,
                                   norm=norm,
                                   act=self.act)(x)
        x = jnp.mean(x, axis=(1, 2))
        x = nn.Dense(self.num_classes, dtype=self.dtype)(x)
        x = jnp.asarray(x, self.dtype)
        return x


ResNet18 = partial(ResNet, stage_sizes=[2, 2, 2, 2], block_cls=ResNetBlock)
