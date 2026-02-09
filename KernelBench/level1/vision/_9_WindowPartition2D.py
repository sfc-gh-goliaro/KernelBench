import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, NamedTuple


class WindowContext(NamedTuple):
    """Context returned by partition() needed for reverse().

    Stores the spatial metadata so that reverse() can undo padding,
    cyclic shifting, and reshape back to the original sequence layout.
    """
    batch_size: int
    height: int
    width: int
    height_pad: int
    width_pad: int
    channels: int
    pad_bottom: int
    pad_right: int


class Model(nn.Module):
    """
    2D Window Partition with Cyclic Shift

    Used by: Swin Transformer v1/v2 (window-based attention layers)

    Partitions a 2D feature map into non-overlapping windows for
    local attention, with optional cyclic shift for cross-window
    connectivity. Handles padding for inputs not divisible by the
    window size. No learnable parameters.

    Operations (partition / forward):
        1. Reshape sequence to 2D spatial layout
        2. Pad height/width to be divisible by window_size
        3. Cyclic shift by shift_size (if > 0)
        4. Partition into non-overlapping windows
        5. Compute shifted-window attention mask (if shift_size > 0)

    Operations (reverse):
        1. Merge windows back to spatial layout
        2. Reverse cyclic shift
        3. Remove padding
        4. Flatten back to sequence

    Shapes:
        partition input:  (batch, height * width, channels) + (height, width)
        partition output: (num_windows * batch, window_size^2, channels),
                          attention_mask, WindowContext
        reverse input:    (num_windows * batch, window_size^2, channels) + WindowContext
        reverse output:   (batch, height * width, channels)
    """

    def __init__(self, window_size: int, shift_size: int = 0):
        """
        Initialize window partition operator.

        Args:
            window_size: Size of each square attention window.
            shift_size: Number of pixels to cyclically shift before
                        partitioning. Set to 0 for non-shifted layers,
                        typically window_size // 2 for shifted layers.
        """
        super(Model, self).__init__()
        self.window_size = window_size
        self.shift_size = shift_size

    def _partition_windows(
        self, x: torch.Tensor, window_size: int
    ) -> torch.Tensor:
        """Partition spatial tensor into non-overlapping windows.

        Args:
            x: (batch, height, width, channels) — height and width must
               be divisible by window_size.
            window_size: Size of each window.

        Returns:
            (num_windows * batch, window_size, window_size, channels)
        """
        batch_size, height, width, channels = x.shape
        x = x.view(
            batch_size,
            height // window_size, window_size,
            width // window_size, window_size,
            channels,
        )
        windows = (
            x.permute(0, 1, 3, 2, 4, 5)
            .contiguous()
            .view(-1, window_size, window_size, channels)
        )
        return windows

    def _merge_windows(
        self,
        windows: torch.Tensor,
        window_size: int,
        height: int,
        width: int,
    ) -> torch.Tensor:
        """Merge windows back into spatial tensor.

        Args:
            windows: (num_windows * batch, window_size, window_size, channels)
            window_size: Size of each window.
            height: Padded spatial height.
            width: Padded spatial width.

        Returns:
            (batch, height, width, channels)
        """
        channels = windows.shape[-1]
        windows = windows.view(
            -1,
            height // window_size, width // window_size,
            window_size, window_size,
            channels,
        )
        x = (
            windows.permute(0, 1, 3, 2, 4, 5)
            .contiguous()
            .view(-1, height, width, channels)
        )
        return x

    def get_attn_mask(
        self, height: int, width: int, dtype: torch.dtype
    ) -> Optional[torch.Tensor]:
        """Compute attention mask for shifted windows.

        When shift_size > 0, different regions within a window come from
        different spatial neighborhoods. The mask ensures that tokens from
        different regions do not attend to each other.

        Args:
            height: Padded spatial height (divisible by window_size).
            width: Padded spatial width (divisible by window_size).
            dtype: Tensor dtype for the mask.

        Returns:
            Attention mask of shape (num_windows, window_size^2, window_size^2)
            or None if shift_size == 0.
        """
        if self.shift_size <= 0:
            return None

        img_mask = torch.zeros((1, height, width, 1), dtype=dtype)
        height_slices = (
            slice(0, -self.window_size),
            slice(-self.window_size, -self.shift_size),
            slice(-self.shift_size, None),
        )
        width_slices = (
            slice(0, -self.window_size),
            slice(-self.window_size, -self.shift_size),
            slice(-self.shift_size, None),
        )
        count = 0
        for h_slice in height_slices:
            for w_slice in width_slices:
                img_mask[:, h_slice, w_slice, :] = count
                count += 1

        mask_windows = self._partition_windows(img_mask, self.window_size)
        mask_windows = mask_windows.view(
            -1, self.window_size * self.window_size
        )
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, -100.0).masked_fill(
            attn_mask == 0, 0.0
        )
        return attn_mask

    def forward(
        self,
        hidden_states: torch.Tensor,
        input_dimensions: Tuple[int, int],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], WindowContext]:
        """Partition a sequence into shifted windows.

        Args:
            hidden_states: (batch, height * width, channels)
            input_dimensions: (height, width) spatial dimensions.

        Returns:
            Tuple of:
            - windows: (num_windows * batch, window_size^2, channels)
            - attn_mask: (num_windows, window_size^2, window_size^2) or None
            - context: WindowContext needed by reverse()
        """
        height, width = input_dimensions
        batch_size, _, channels = hidden_states.size()

        # Reshape to spatial layout
        hidden_states = hidden_states.view(batch_size, height, width, channels)

        # Pad so height and width are divisible by window_size
        pad_right = (self.window_size - width % self.window_size) % self.window_size
        pad_bottom = (self.window_size - height % self.window_size) % self.window_size
        hidden_states = F.pad(hidden_states, (0, 0, 0, pad_right, 0, pad_bottom))
        _, height_pad, width_pad, _ = hidden_states.shape

        # Cyclic shift
        if self.shift_size > 0:
            hidden_states = torch.roll(
                hidden_states,
                shifts=(-self.shift_size, -self.shift_size),
                dims=(1, 2),
            )

        # Partition into windows
        windows = self._partition_windows(hidden_states, self.window_size)
        windows = windows.view(-1, self.window_size * self.window_size, channels)

        # Compute attention mask
        attn_mask = self.get_attn_mask(
            height_pad, width_pad, dtype=hidden_states.dtype
        )
        if attn_mask is not None:
            attn_mask = attn_mask.to(windows.device)

        ctx = WindowContext(
            batch_size=batch_size,
            height=height,
            width=width,
            height_pad=height_pad,
            width_pad=width_pad,
            channels=channels,
            pad_bottom=pad_bottom,
            pad_right=pad_right,
        )

        return windows, attn_mask, ctx

    def reverse(
        self,
        windows: torch.Tensor,
        ctx: WindowContext,
    ) -> torch.Tensor:
        """Merge windows back into a sequence, reversing partition().

        Args:
            windows: (num_windows * batch, window_size^2, channels)
            ctx: WindowContext returned by forward().

        Returns:
            (batch, height * width, channels)
        """
        # Reshape to 2D windows
        windows = windows.view(
            -1, self.window_size, self.window_size, ctx.channels
        )

        # Merge windows back to spatial
        hidden_states = self._merge_windows(
            windows, self.window_size, ctx.height_pad, ctx.width_pad
        )

        # Reverse cyclic shift
        if self.shift_size > 0:
            hidden_states = torch.roll(
                hidden_states,
                shifts=(self.shift_size, self.shift_size),
                dims=(1, 2),
            )

        # Remove padding
        if ctx.pad_bottom > 0 or ctx.pad_right > 0:
            hidden_states = hidden_states[
                :, : ctx.height, : ctx.width, :
            ].contiguous()

        # Flatten back to sequence
        hidden_states = hidden_states.view(
            ctx.batch_size, ctx.height * ctx.width, ctx.channels
        )

        return hidden_states


# ============================================================================
# Benchmark Configuration
# ============================================================================
