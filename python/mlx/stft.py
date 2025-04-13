import mlx.core as mx
import numpy as np

def stft(
    x: mx.array,
    n_fft: int = 2048,
    hop_length: int = None,
    win_length: int = None,
    window: mx.array = None,
    center: bool = True,
    pad_mode: str = "reflect",
    normalized: bool = False,
    onesided: bool = True,
    return_complex: bool = True,
) -> mx.array:

    if hop_length is None:
        hop_length = n_fft // 4
        
    if win_length is None:
        win_length = n_fft
        
    if window is None:
        window = mx.ones(win_length)

    if win_length < n_fft:
        pad_left = (n_fft - win_length) // 2
        pad_right = n_fft - win_length - pad_left
        window = mx.pad(window, [(pad_left, pad_right)])

    if center:
        pad_width = n_fft // 2
        x = mx.pad(x, [(pad_width, pad_width)], mode=pad_mode)

    n_frames = 1 + (x.shape[0] - n_fft) // hop_length

    frames = mx.stack([
        x[i * hop_length : i * hop_length + n_fft] * window
        for i in range(n_frames)
    ])

    stft = mx.fft.fft(frames, n=n_fft, axis=-1)
    
    if normalized:
        stft = stft / mx.sqrt(n_fft)
        
    if onesided:
        stft = stft[..., :n_fft//2 + 1]
        
    if not return_complex:
        stft = mx.stack([stft.real, stft.imag], axis=-1)
        
    return stft

def istft(
    stft_matrix: mx.array,
    hop_length: int = None,
    win_length: int = None,
    window: mx.array = None,
    center: bool = True,
    length: int = None,
    normalized: bool = False,
    onesided: bool = True,
) -> mx.array:
    if hop_length is None:
        hop_length = stft_matrix.shape[-2] // 4
        
    if win_length is None:
        win_length = stft_matrix.shape[-2]
        
    if window is None:
        window = mx.ones(win_length)
        
    if win_length < stft_matrix.shape[-2]:
        pad_left = (stft_matrix.shape[-2] - win_length) // 2
        pad_right = stft_matrix.shape[-2] - win_length - pad_left
        window = mx.pad(window, [(pad_left, pad_right)])
        
    if stft_matrix.shape[-1] == 2:
        stft_matrix = stft_matrix[..., 0] + 1j * stft_matrix[..., 1]
        
    if onesided:
        n_fft = 2 * (stft_matrix.shape[-1] - 1)
        full_stft = mx.zeros((*stft_matrix.shape[:-1], n_fft), dtype=stft_matrix.dtype)
        full_stft[..., :stft_matrix.shape[-1]] = stft_matrix
        full_stft[..., stft_matrix.shape[-1]:] = mx.conj(stft_matrix[..., -2:0:-1])
        stft_matrix = full_stft
        
    frames = mx.fft.ifft(stft_matrix, n=stft_matrix.shape[-1], axis=-1)
    
    if normalized:
        frames = frames * mx.sqrt(frames.shape[-1])
        
    frames = frames * window
    
    signal_length = (frames.shape[0] - 1) * hop_length + frames.shape[1]
    signal = mx.zeros(signal_length, dtype=frames.dtype)
    
    for i in range(frames.shape[0]):
        signal[i * hop_length : i * hop_length + frames.shape[1]] += frames[i]
        
    window_sum = mx.zeros(signal_length, dtype=frames.dtype)
    for i in range(frames.shape[0]):
        window_sum[i * hop_length : i * hop_length + frames.shape[1]] += window
        
    signal = signal / window_sum
    
    if center:
        pad_width = frames.shape[1] // 2
        signal = signal[pad_width:-pad_width]
        
    if length is not None:
        signal = signal[:length]
        
    return signal