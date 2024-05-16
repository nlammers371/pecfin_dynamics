def _register(video: np.ndarray) -> np.ndarray:
    # registration using channel 1
    reg_channel = 1
    for t in tqdm(range(video.shape[0] - 1), "register"):
        shift, error, _ = phase_cross_correlation(
            video[t, reg_channel],
            video[t + 1, reg_channel],
            normalization=None,
            overlap_ratio=0.25,
        )
        video[t+1] = ndi.shift(video[t+1], (0, *shift), order=1)
    return video