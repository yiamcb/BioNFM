min_sample_length = min(data.shape[1] for data in all_data)
homogenized_data = np.array([data[:, :min_sample_length] for data in all_data])
print("All data shape after homogenization:", homogenized_data.shape)

eeg_data = np.array(homogenized_data)
all_labels = np.array(all_labels)

print("All data shape:", eeg_data.shape)
print("All labels:", all_labels)


sampling_rate = 512  # Hz
frame_length_seconds = 2  # in seconds
frame_length_samples = frame_length_seconds * sampling_rate  # 1024 samples per frame
overlap_percentage = 0.5  # 50% overlap
overlap_samples = int(frame_length_samples * overlap_percentage)

all_sample_frames = []

for sample in eeg_data:
    sample_frames = []

    for start in range(0, sample.shape[1] - frame_length_samples + 1, frame_length_samples - overlap_samples):
        end = start + frame_length_samples
        sample_frames.append(sample[:, start:end])

    all_sample_frames.append(np.array(sample_frames))
all_sample_frames = np.array(all_sample_frames)
print("Framed data shape:", all_sample_frames.shape)  # Expected shape: (samples, num_frames, num_channels, 1024)


