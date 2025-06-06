import mne

main_folder = "/content/drive/MyDrive/BIDA Validation/"

all_data = []
all_labels = []

def get_label(folder_name):
    if "sub-hc" in folder_name:
        return "Healthy"
    elif "sub-pd" in folder_name:
        return "PD"
    else:
        return None

for subfolder in os.listdir(main_folder):
    subfolder_path = os.path.join(main_folder, subfolder)

    if os.path.isdir(subfolder_path) and subfolder.startswith("sub-"):
        label = get_label(subfolder)

        if label:
            for sub_subfolder in os.listdir(subfolder_path):
                eeg_folder = os.path.join(subfolder_path, sub_subfolder, "eeg")

                if os.path.isdir(eeg_folder):
                    for file_name in os.listdir(eeg_folder):
                        if file_name.endswith(".bdf"):
                            file_path = os.path.join(eeg_folder, file_name)

                            raw_data = mne.io.read_raw_bdf(file_path, preload=True)
                            data = raw_data.get_data()  # Shape (n_channels, n_samples)

                            all_data.append(data)
                            all_labels.append(label)

                            print(f"Loaded: {file_path}, Label: {label}, Data Shape: {data.shape}")