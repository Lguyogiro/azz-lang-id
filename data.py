import glob
import torchaudio
from torch.utils.data import Dataset


class NahuaudioDataset(Dataset):
  def __init__(self, path_to_clips):
    self.file_list = glob.glob(f"{path_to_clips}/*wav")

  def __len__(self):
    return len(self.file_list)
  
  def __getitem__(self, index):
    path = self.file_list[index]
    label = path.split("/")[-1].split("_")[0]
    signal, sr = torchaudio.load(path)
    return signal, sr, label, path, index