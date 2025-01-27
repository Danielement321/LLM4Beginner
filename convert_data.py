import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import datasets

def download_LLM_dataset(dataset_path):
    dataset = datasets.load_dataset(dataset_path, split='train')
    dataset = dataset.to_json(f'data/{dataset_path.split('/')[-1]}.json')
    print('The dataset from hub has been saved as json files!')
    
if __name__ == '__main__':
    download_LLM_dataset('maxtli/OpenWebText-2M')