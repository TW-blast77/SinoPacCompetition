import os
import torch
import pandas as pd
from torch.utils.data.dataloader import DataLoader
from sinodataset import SinoTestDataset
from dotenv import load_dotenv

if __name__ == '__main__':
    load_dotenv()
    TORCH_SEED              = int(os.getenv("TORCH_SEED", 1))
    TEST_CSV_PATH           = os.getenv("TEST_CSV_PATH", "")
    TEST_MODEL_PT_PATH      = os.getenv("TEST_MODEL_PT_PATH", "")
    TEST_MODEL_NAME         = os.getenv("TEST_MODEL_NAME", "")
    TEST_OUTPUT_CSV_PATH    = os.getenv("TEST_OUTPUT_CSV_PATH", "")

    torch.manual_seed(TORCH_SEED)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = torch.load(TEST_MODEL_PT_PATH+TEST_MODEL_NAME).to(device).eval()

    test_dataset = SinoTestDataset(TEST_CSV_PATH)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4)

    result = []
    for data, index in test_loader:
        data = data.to(device)
        pred = model(data)
        result.append([ f"PU-{index.item()+1}", pred.item() ])
    
    df = pd.DataFrame(result, columns=['ID', 'predicted_price'])
    df.to_csv(TEST_OUTPUT_CSV_PATH, index=False)